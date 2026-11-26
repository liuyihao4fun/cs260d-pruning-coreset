import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import warnings

from src.audit import experiment_identity, full_config, identity_key
from src.finetune import build_parser
from src.plot import aggregate_records, figure_groups, write_outputs


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ckpt = self.root / 'dense.pt'
        self.ckpt.write_bytes(b'checkpoint')

    def record(self, seed=0, accuracy=90., **changes):
        a = build_parser().parse_args(['--ckpt', str(self.ckpt)])
        a.seed = seed
        for k, v in changes.items():
            setattr(a, k, v)
        identity = experiment_identity(a)
        return dict(schema_version=2, status='completed', key=identity_key(identity),
                    identity=identity, config=full_config(a), final_acc=accuracy,
                    best_acc=accuracy, metadata={'gpu_model': 'T4', 'device': 'cuda',
                    'timing_schema': 'v2-disjoint-scoring-selection-training-evaluation'},
                    train_time=10., selection_time=1.)

    def test_seeds_aggregate_and_identical_copies_deduplicate(self):
        a, b = self.record(), self.record(seed=1, accuracy=92.)
        table = aggregate_records([a, b, copy.deepcopy(a)])
        self.assertEqual(len(table), 1)
        self.assertEqual(table.acc_mean.iloc[0], 91.)
        self.assertAlmostEqual(table.acc_std.iloc[0], 2 ** .5)
        self.assertEqual(table.n_seeds.iloc[0], 2)

    def test_configuration_and_hardware_are_separate(self):
        records = [self.record()]
        for field, value in dict(reselect_every=5, no_weights=True, damage_alpha=2.,
                                lr=.02, batch_size=64, epochs=15, wd=0., limit_train=2000).items():
            records.append(self.record(**{field: value}))
        other_gpu = copy.deepcopy(records[0])
        other_gpu['metadata']['gpu_model'] = 'L4'
        records.append(other_gpu)
        other_source = copy.deepcopy(records[0])
        other_source['identity']['source_sha256'] = 'different'
        other_source['key'] = identity_key(other_source['identity'])
        records.append(other_source)
        self.assertEqual(len(aggregate_records(records)), len(records))
        self.assertEqual(len(figure_groups(aggregate_records(records))), len(records))

    def test_started_failed_and_interrupted_are_excluded(self):
        complete = self.record()
        records = [complete] + [{**complete, 'status': s} for s in ['started', 'failed', 'interrupted']]
        incomplete = copy.deepcopy(complete)
        del incomplete['status']
        records.append(incomplete)
        self.assertEqual(aggregate_records(records).n_seeds.iloc[0], 1)

    def test_conflicting_results_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            aggregate_records([self.record(), self.record(accuracy=91.)])

    def test_legacy_unknown_config_does_not_merge_with_v2(self):
        old = dict(dataset='cifar10', sparsity=.9, method='random', frac=.1,
                   epochs=30, seed=0, tag='', final_acc=90.)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            table = aggregate_records([old, self.record()])
        self.assertEqual(len(table), 2)
        self.assertIn('<unknown>', table.lr.tolist())

    def test_separate_epoch_figures_and_no_ambiguous_full_reference(self):
        rows = [self.record(method='full', epochs=e) for e in [3, 6, 30]]
        rows += [self.record(epochs=30), self.record(epochs=30, reselect_every=5)]
        table = aggregate_records(rows)
        out = self.root / 'plots'
        outputs = write_outputs(table, out)
        manifest = json.loads((out / 'figure_manifest.json').read_text())
        self.assertEqual(len(manifest), 4)
        self.assertEqual(len(list(out.glob('*.png'))), 4)
        old_bytes = {p: p.read_bytes() for p in outputs}
        with self.assertRaises(FileExistsError):
            write_outputs(table, out)
        self.assertEqual(old_bytes, {p: p.read_bytes() for p in outputs})

    def test_tampered_v2_key_is_rejected(self):
        r = self.record()
        r['key'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity'):
            aggregate_records([r])
