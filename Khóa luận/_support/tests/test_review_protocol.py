"""Counterfactual controls, patient-safe permutations and immutable test registration."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from clindistill.artifacts import read_json
from clindistill.research.analysis import teacher_strata, weight_diagnostics
from clindistill.research.data import FeatureStore, VisitDataset, collate_visits, prepare_features
from clindistill.research.models import VisitClassifier
from clindistill.research.training import PRIMARY_COMPARISONS, dataset_for, forward_batch, model_spec, run_study
from clindistill.research.reporting import evaluate_study, freeze_test_protocol
from test_research_pipeline import config, synthetic_features


class ReviewProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_clinical_controls_do_not_modify_cache_or_cross_fold(self):
        with tempfile.TemporaryDirectory() as d:
            store = FeatureStore(synthetic_features(d), allow_debug=True)
            original = store.data['train']['clinical'].copy()
            indices = np.arange(12)
            ds = VisitDataset(store, 'train', 'teacher', role='multimodal_teacher', indices=indices,
                              clinical_mode='shuffled', shuffle_seed=42)
            self.assertEqual(set(ds.clinical_indices), set(indices))
            self.assertTrue(np.all(store.data['train']['groups'][ds.clinical_indices]
                                   != store.data['train']['groups'][indices]))
            masked = VisitDataset(store, 'train', 'teacher', role='multimodal_teacher', clinical_mode='masked')[0]
            self.assertFalse(masked['clinical_present'])
            self.assertEqual(float(masked['clinical'].abs().sum()), 0.)
            self.assertEqual(float(masked['demographics'].abs().sum()), 0.)
            np.testing.assert_array_equal(original, store.data['train']['clinical'])

    def test_clinical_modes_preserve_inputs_and_presence(self):
        with tempfile.TemporaryDirectory() as d:
            store = FeatureStore(synthetic_features(d), allow_debug=True)
            data = store.data['train']
            # Full context, text only, demographics only, and no usable context.
            data['demographics'][:4] = [[.25, 1, 1, 0, 0], [0, 0, 0, 0, 1],
                                       [.5, 1, 0, 1, 0], [0, 0, 0, 0, 1]]
            data['clinical'][2:4] = 0
            data['clinical_present'][:4] = [True, True, True, False]
            original = {key: data[key].copy() for key in
                        ('clinical', 'demographics', 'clinical_present')}
            expected_presence = {'full': [True, True, True, False],
                                 'demographics': [True, False, True, False],
                                 'masked': [False] * 4}
            for mode, presence in expected_presence.items():
                ds = VisitDataset(store, 'train', 'teacher', role='multimodal_teacher', clinical_mode=mode)
                for i in range(4):
                    with self.subTest(mode=mode, visit=i):
                        item = ds[i]
                        expected_text = original['clinical'][i] if mode == 'full' else np.zeros_like(original['clinical'][i])
                        expected_demo = np.zeros(5, np.float32) if mode == 'masked' else original['demographics'][i]
                        np.testing.assert_array_equal(item['clinical'].numpy(), expected_text)
                        np.testing.assert_array_equal(item['demographics'].numpy(), expected_demo)
                        self.assertEqual(item['clinical_present'], presence[i])
                for key, values in original.items():
                    np.testing.assert_array_equal(data[key], values)

    def test_multimodal_teacher_uses_context_but_masked_control_ignores_it(self):
        with tempfile.TemporaryDirectory() as d, torch.random.fork_rng(devices=[]):
            store = FeatureStore(synthetic_features(d), allow_debug=True)
            store.data['train']['demographics'][0] = [.25, 1, 1, 0, 0]
            cfg = config()
            for fusion in ('gated', 'concat'):
                with self.subTest(fusion=fusion):
                    torch.manual_seed(37)
                    spec, key = model_spec(store, {**cfg, 'fusion': fusion}, 'teacher_multimodal')
                    model = VisitClassifier(**spec).eval()
                    full = dataset_for(store, 'train', key, spec['role'], 'teacher_multimodal')[0]
                    batch = collate_visits([full])
                    masked = dataset_for(store, 'train', key, spec['role'], 'teacher_visual_control')[0]
                    masked_batch = collate_visits([masked])
                    with torch.inference_mode():
                        baseline = forward_batch(model, batch, 'cpu')['logits']
                        masked_baseline = forward_batch(model, masked_batch, 'cpu')['logits']
                        for field in ('clinical', 'demographics'):
                            changed = {**batch, field: batch[field].clone()}
                            changed[field] = (torch.linspace(-2, 3, batch[field].numel()).reshape_as(batch[field])
                                              if field == 'clinical' else torch.tensor([[.75, 1, 0, 1, 0]]))
                            self.assertFalse(torch.allclose(baseline, forward_batch(model, changed, 'cpu')['logits']),
                                             msg=f'{fusion} Teacher ignored {field} with full context.')
                            changed_masked = {**masked_batch, field: changed[field]}
                            torch.testing.assert_close(masked_baseline,
                                forward_batch(model, changed_masked, 'cpu')['logits'], rtol=0, atol=0)

    def test_student_path_remains_image_only_when_context_changes(self):
        with tempfile.TemporaryDirectory() as d, torch.random.fork_rng(devices=[]):
            store = FeatureStore(synthetic_features(d), allow_debug=True)
            torch.manual_seed(37)
            spec, key = model_spec(store, config(), 'student_no_kd')
            model = VisitClassifier(**spec).eval()
            before = dataset_for(store, 'train', key, spec['role'], 'student_no_kd')[0]
            data = store.data['train']
            data['clinical'][0] += 100
            data['demographics'][0] = [.75, 1, 0, 1, 0]
            data['clinical_present'][0] = False
            after = dataset_for(store, 'train', key, spec['role'], 'student_no_kd')[0]
            for item in (before, after):
                self.assertTrue({'clinical', 'demographics', 'clinical_present'}.isdisjoint(item))
            with torch.inference_mode():
                torch.testing.assert_close(forward_batch(model, collate_visits([before]), 'cpu')['logits'],
                                           forward_batch(model, collate_visits([after]), 'cpu')['logits'], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'cannot receive clinical'):
                    forward_batch(model, {**collate_visits([after]), 'clinical': torch.from_numpy(data['clinical'][:1])}, 'cpu')

    def test_mean_and_single_controls_have_distinct_evidence(self):
        x = torch.randn(2, 3, 8); mask = torch.ones(2, 3, dtype=torch.bool)
        for pooling in ('mean', 'single'):
            model = VisitClassifier(8, 2, width=8, dropout=0, pooling=pooling).eval()
            first = model(x, mask)['logits']
            changed = x.clone(); changed[:, 2] += torch.arange(8) * 20
            second = model(changed, mask)['logits']
            if pooling == 'single':
                torch.testing.assert_close(first, second)
            else:
                self.assertFalse(torch.allclose(first, second))
            padded = torch.cat([x, torch.ones(2, 1, 8) * 999], dim=1)
            torch.testing.assert_close(first, model(padded, torch.cat([mask, torch.zeros(2, 1, dtype=torch.bool)], 1))['logits'])

    def test_disagreement_reports_expose_zero_weight_rare_class(self):
        labels = np.array([0, 0, 1, 1])
        v = np.array([[3, 0], [3, 0], [3, 0], [3, 0]])
        m = np.array([[3, 0], [0, 3], [0, 3], [3, 0]])
        strata = teacher_strata(v, m, labels)
        self.assertTrue(np.all(sum(mask.astype(int) for mask in strata.values()) == 1))
        report = weight_diagnostics({'gate': np.array([1., .5, 0., 0.])}, labels, ['common', 'rare'], strata)
        self.assertEqual(report['gate']['by_class']['rare']['zero_fraction'], 1.)
        self.assertEqual(report['gate']['by_teacher_stratum']['multimodal_only_correct']['cases'], 1)
        self.assertIn(('kd_multimodal', 'kd_visual'), PRIMARY_COMPARISONS)

    def test_controls_execute_report_and_test_lock_rejects_checkpoint_change(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); cache = synthetic_features(root / 'cache'); cfg = config()
            cfg['methods'] += ['student_single', 'student_mean', 'teacher_demographics', 'teacher_shuffled', 'teacher_visual_control']
            run_study(cache, root / 'study', cfg, require_gpu=False)
            table, _ = evaluate_study(cache, root / 'study')
            self.assertEqual(len(table), len(cfg['methods']) + 2)
            strata = read_json(root / 'study/report_validation/teacher_strata.json')
            comparisons = read_json(root / 'study/report_validation/paired_comparisons.json')['comparisons']
            self.assertEqual(comparisons['kd_multimodal_minus_kd_visual']['kind'], 'primary')
            self.assertEqual(comparisons['teacher_multimodal_minus_teacher_shuffled']['kind'], 'ablation')
            self.assertIn('multimodal_only_correct', strata['results']['2026']['student_no_kd'])
            self.assertTrue((root / 'study/_oof/seed_2026/weight_diagnostics.json').is_file())
            lock = freeze_test_protocol(cache, root / 'study', 'Protocol fixed; expert annotation review remains pending.')
            with patch('clindistill.research.reporting.write_json', side_effect=PermissionError('read-only input')):
                self.assertEqual(lock, freeze_test_protocol(cache, root / 'study', lock['review_record']))
            with self.assertRaisesRegex(ValueError, 'already locked'):
                freeze_test_protocol(cache, root / 'study', 'Changed record')
            path = root / 'study/kd_reliability/seed_2026/best.pt'
            path.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                freeze_test_protocol(cache, root / 'study', lock['review_record'])

    def test_test_preparation_fails_before_opening_any_unlocked_release(self):
        cfg = config(); cfg['debug'] = False
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, 'Locked test'):
            prepare_features(Path(d) / 'does-not-exist', d, {}, Path(d) / 'cache', cfg, splits=('test',))


if __name__ == '__main__':
    unittest.main()
