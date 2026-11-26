import argparse
import json
from pathlib import Path
import tempfile
import unittest
import warnings

from src.audit import (append_jsonl, completed_keys, experiment_identity, file_sha256,
                       identity_key, load_jsonl, run_key)


def args(checkpoint):
    return argparse.Namespace(ckpt=str(checkpoint), dataset='cifar10', sparsity=.95,
        method='craig', frac=.2, epochs=30, lr=.01, batch_size=128, wd=.0005,
        seed=0, reselect_every=0, no_weights=False, damage_alpha=1., limit_train=None,
        tag='', results='new.jsonl', data_root='./data')


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ckpt = self.root / 'dense.pt'
        self.ckpt.write_bytes(b'checkpoint-content')
        self.a = args(self.ckpt)

    def completed(self):
        identity = experiment_identity(self.a)
        return dict(schema_version=2, key=identity_key(identity), identity=identity,
                    status='completed', final_acc=90.)

    def test_all_important_parameters_change_identity(self):
        old = run_key(self.a)
        changes = dict(lr=.02, batch_size=64, wd=0., reselect_every=5, no_weights=True,
                       damage_alpha=2., limit_train=2000, dataset='cifar100', sparsity=.98,
                       method='random', frac=.1, epochs=15, seed=1, tag='repeat')
        for field, value in changes.items():
            with self.subTest(field=field):
                b = argparse.Namespace(**vars(self.a))
                setattr(b, field, value)
                self.assertNotEqual(old, run_key(b))

    def test_checkpoint_content_not_location_defines_identity(self):
        before = run_key(self.a)
        copy = self.root / 'copy.pt'
        copy.write_bytes(self.ckpt.read_bytes())
        self.a.ckpt = str(copy)
        self.assertEqual(before, run_key(self.a))
        copy.write_bytes(b'changed checkpoint at same path')
        self.assertNotEqual(before, run_key(self.a))
        self.assertEqual(len(file_sha256(copy)), 64)

    def test_source_revision_is_part_of_identity(self):
        one = experiment_identity(self.a, source_hash='a' * 64)
        two = experiment_identity(self.a, source_hash='b' * 64)
        self.assertNotEqual(identity_key(one), identity_key(two))

    def test_output_paths_do_not_change_identity_and_full_fraction_is_effective(self):
        old = run_key(self.a)
        self.a.results = 'another-journal.jsonl'
        self.a.data_root = '/relocated-official-cifar-cache'
        self.assertEqual(old, run_key(self.a))
        self.a.method = 'full'
        old = run_key(self.a)
        self.a.frac = .05
        self.assertEqual(old, run_key(self.a))

    def test_resume_requires_completed_verified_v2(self):
        complete = self.completed()
        for status in ['started', 'failed', 'interrupted']:
            self.assertEqual(completed_keys([{**complete, 'status': status}]), set())
        self.assertEqual(completed_keys([complete]), {complete['key']})
        self.assertEqual(completed_keys([{**complete, 'key': 'wrong'}]), set())
        self.assertEqual(completed_keys([{'key': 'legacy', 'final_acc': 95.}]), set())

    def test_valid_unterminated_line_remains_separate_on_append(self):
        p = self.root / 'valid.jsonl'
        raw = b'{"key":"old","final_acc":91}'
        p.write_bytes(raw)
        append_jsonl(p, {'status': 'started', 'key': 'new'})
        self.assertTrue(p.read_bytes().startswith(raw))
        self.assertEqual(len(load_jsonl(p)), 2)

    def test_partial_tail_recovery_preserves_every_original_byte(self):
        p = self.root / 'partial.jsonl'
        raw = b'{"key":"legacy","final_acc":91}\n{"status":"comple'
        p.write_bytes(raw)
        with warnings.catch_warnings(record=True) as caught:
            self.assertEqual(len(load_jsonl(p)), 1)
            append_jsonl(p, self.completed())
            records = load_jsonl(p)
        self.assertTrue(caught)
        self.assertTrue(p.read_bytes().startswith(raw))
        self.assertEqual(len(completed_keys(records)), 1)
        self.assertTrue(any(r.get('status') == 'journal_recovery' for r in records))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            append_jsonl(p, {'status': 'started'})
            self.assertEqual(len(completed_keys(load_jsonl(p))), 1)

    def test_partial_utf8_and_newline_tail(self):
        for fragment in [b'{"note":"\xe4', b'{"bad":\n', b'{"bad":\n\n']:
            with self.subTest(fragment=fragment):
                p = self.root / 'utf8.jsonl'
                p.write_bytes(fragment)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    append_jsonl(p, {'status': 'started'})
                    self.assertEqual(load_jsonl(p)[-1]['status'], 'started')
                self.assertTrue(p.read_bytes().startswith(fragment))

    def test_interior_corruption_is_not_silently_ignored(self):
        p = self.root / 'bad.jsonl'
        raw = b'{broken}\n{"final_acc":91}\n'
        p.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'interior'):
            load_jsonl(p)
        with self.assertRaises(ValueError):
            append_jsonl(p, {'status': 'started'})
        self.assertEqual(p.read_bytes(), raw)

    def test_missing_and_empty_journals(self):
        p = self.root / 'empty.jsonl'
        self.assertEqual(load_jsonl(p), [])
        p.write_text('\n')
        self.assertEqual(load_jsonl(p), [])
