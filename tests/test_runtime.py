import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from src import finetune, run_grid
from src.audit import completed_keys, load_jsonl, run_key
from src.models import resnet18
from src.utils import timed

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        nb = json.loads((ROOT / 'notebooks/colab_run_legacy.ipynb').read_text())
        source = next(''.join(c['source']).partition('\n')[2] for c in nb['cells']
                      if ''.join(c['source']).startswith('%%writefile /content/repo/src/finetune.py'))
        cls.legacy = types.ModuleType('src.legacy_finetune')
        cls.legacy.__package__ = 'src'
        exec(compile(source, 'legacy_finetune.py', 'exec'), cls.legacy.__dict__)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for module in [finetune, self.legacy, run_grid]:
            patcher = mock.patch.object(module, 'get_device', return_value='cpu')
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ckpt = self.root / 'dense.pt'
        torch.manual_seed(123)
        torch.save({'state_dict': resnet18().state_dict(), 'dataset': 'cifar10'}, self.ckpt)
        self.data = (torch.randint(0, 256, (20, 3, 32, 32), dtype=torch.uint8),
                     torch.arange(10).repeat(2),
                     torch.randint(0, 256, (10, 3, 32, 32), dtype=torch.uint8), torch.arange(10))

    def args(self, method='random', **changes):
        a = finetune.build_parser().parse_args(['--ckpt', str(self.ckpt), '--epochs', '2',
            '--method', method, '--frac', '.5', '--batch_size', '8',
            '--results', str(self.root / f'{method}.jsonl')])
        for key, value in changes.items():
            setattr(a, key, value)
        return a

    def test_cpu_regression_against_archived_training_for_all_methods(self):
        # Synthetic inputs test behavior, not model quality or CIFAR accuracy.
        for method in ['full', 'random', 'el2n', 'damage_topk', 'craig', 'damage_fl']:
            with self.subTest(method=method):
                a = self.args(method)
                old = self.args(method, results=str(self.root / f'legacy_{method}.jsonl'))
                models = []
                def capture(loader):
                    def load(*args, **kwargs):
                        model = loader(*args, **kwargs)
                        models.append(model)
                        return model
                    return load
                with mock.patch.object(self.legacy, 'load_dense', side_effect=capture(self.legacy.load_dense)):
                    original = self.legacy.run(old, data=self.data, verbose=False)
                old_rng = torch.get_rng_state().clone()
                with mock.patch.object(finetune, 'load_dense', side_effect=capture(finetune.load_dense)):
                    record = finetune.run(a, data=self.data, verbose=False)
                self.assertTrue(torch.equal(old_rng, torch.get_rng_state()), method)
                before, after = models[0].state_dict(), models[1].state_dict()
                self.assertEqual(before.keys(), after.keys())
                for name in before:
                    self.assertTrue(torch.equal(before[name], after[name]), (method, name))
                for field in ['curve', 'final_acc', 'best_acc', 'dense_acc', 'pruned_acc', 'steps']:
                    self.assertEqual(record[field], original[field], (method, field))
                self.assertEqual(record['unique_samples_seen'], record['n_selected'])
                self.assertEqual(record['sgd_steps'], record['steps'])
                self.assertEqual(record['samples_processed'], 2 * record['n_selected'])
                self.assertGreater(record['evaluation_time'], 0)
                self.assertGreater(record['total_wall_time'], sum(record[k] for k in
                    ['scoring_time', 'selection_time', 'train_time', 'evaluation_time']))
                if method in ['random', 'full']:
                    self.assertEqual(record['scoring_time'], 0.)
                if method in ['damage_topk', 'damage_fl']:
                    self.assertGreater(record['dense_scoring_time'], 0)
                events = load_jsonl(a.results)
                self.assertEqual([e['status'] for e in events], ['started', 'completed'])
                self.assertEqual(events[0]['attempt_id'], events[1]['attempt_id'])
                self.assertIn('commit', record['metadata']['git'])
                self.assertEqual(record['metadata']['checkpoint_sha256'], record['identity']['checkpoint_sha256'])

    def test_reselection_preserves_random_draws_and_tracks_exact_union(self):
        a = self.args(reselect_every=1)
        old = self.args(reselect_every=1, results=str(self.root / 'legacy.jsonl'))
        original = self.legacy.run(old, data=self.data, verbose=False)
        selected = []
        real_select = finetune.select
        def capture(*args, **kwargs):
            idx, weights = real_select(*args, **kwargs)
            selected.extend(idx.tolist())
            return idx, weights
        with mock.patch.object(finetune, 'per_sample_outputs', side_effect=AssertionError('random must not score')):
            with mock.patch.object(finetune, 'select', side_effect=capture):
                record = finetune.run(a, data=self.data, verbose=False)
        self.assertEqual(record['curve'], original['curve'])
        self.assertEqual(record['unique_samples_seen'], len(set(selected)))
        self.assertGreater(record['unique_samples_seen'], record['n_selected'])
        self.assertEqual(len(record['sel_stats']), 2)

    def test_failure_is_journaled_and_never_skipped(self):
        a = self.args()
        with mock.patch.object(finetune, 'load_dense', side_effect=RuntimeError('injected failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                finetune.run(a, data=self.data, verbose=False)
        self.assertEqual([r['status'] for r in load_jsonl(a.results)], ['started', 'failed'])
        self.assertFalse(completed_keys(load_jsonl(a.results)))
        record = finetune.run(a, data=self.data, verbose=False)
        raw = Path(a.results).read_bytes()
        with mock.patch.object(finetune, 'load_dense', side_effect=AssertionError('must skip')):
            self.assertIsNone(finetune.run(a, data=self.data, verbose=False))
        self.assertEqual(Path(a.results).read_bytes(), raw)
        self.assertIn(record['key'], completed_keys(load_jsonl(a.results)))

    def test_grid_dry_run_and_completed_resume_do_not_load_data(self):
        a = self.args(epochs=1)
        command = ['run_grid', '--ckpt', str(self.ckpt), '--epochs', '1', '--batch_size', '8',
                   '--results', a.results, '--methods', 'random', '--fracs', '.5',
                   '--seeds', '0', '--sparsities', '.9']
        with mock.patch.object(run_grid, 'load_cifar', side_effect=AssertionError('must not load data')):
            with mock.patch.object(sys, 'argv', command + ['--dry_run']):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_grid.main()
        finetune.run(a, data=self.data, verbose=False)
        with mock.patch.object(run_grid, 'load_cifar', side_effect=AssertionError('must skip all completed')):
            with mock.patch.object(sys, 'argv', command):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    run_grid.main()
        self.assertIn('0 to go', output.getvalue())

    def test_timing_synchronizes_both_boundaries(self):
        values = {}
        with mock.patch('src.utils.synchronize') as sync:
            with mock.patch('src.utils.time.perf_counter', side_effect=[10., 12.5]):
                with timed(values, 'train_time', 'cuda'):
                    pass
        self.assertEqual(values['train_time'], 2.5)
        self.assertEqual(sync.call_args_list, [mock.call('cuda'), mock.call('cuda')])

    def test_dense_scoring_is_once_per_attempt_not_once_per_reselection(self):
        a = self.args('damage_fl', reselect_every=1)
        with mock.patch.object(finetune, 'per_sample_outputs', wraps=finetune.per_sample_outputs) as scoring:
            r = finetune.run(a, data=self.data, verbose=False)
        self.assertEqual(scoring.call_count, 3)  # dense once, current model twice
        self.assertGreater(r['dense_scoring_time'], 0.)
        self.assertEqual(len(r['sel_stats']), 2)

    def test_notebook_generation_is_current_and_preserves_legacy(self):
        old = (ROOT / 'notebooks/colab_run_legacy.ipynb').read_bytes()
        subprocess.run([sys.executable, 'scripts/build_notebook.py', '--check'], cwd=ROOT, check=True)
        nb = json.loads((ROOT / 'notebooks/colab_run.ipynb').read_text())
        embedded = {}
        prefix = '%%writefile /content/cs260d-pruning-coreset/'
        for cell in nb['cells']:
            text = ''.join(cell['source'])
            if text.startswith(prefix):
                head, _, body = text.partition('\n')
                embedded[head.removeprefix(prefix)] = body
        for source in (ROOT / 'src').rglob('*.py'):
            relative = str(source.relative_to(ROOT))
            self.assertEqual(embedded[relative].rstrip('\n'), source.read_text().rstrip('\n'))
        self.assertEqual(old, (ROOT / 'notebooks/colab_run_legacy.ipynb').read_bytes())

    def test_imports_have_no_script_write_side_effects(self):
        for name in ['src.data', 'src.models', 'src.prune', 'src.selection', 'src.train_dense',
                     'src.run_grid', 'src.plot', 'scripts.build_notebook', 'scripts.plot_reselect']:
            importlib.import_module(name)
