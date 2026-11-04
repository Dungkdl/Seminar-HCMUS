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
from clindistill.research.data import FeatureStore, VisitDataset, prepare_features
from clindistill.research.models import VisitClassifier
from clindistill.research.training import PRIMARY_COMPARISONS, run_study
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
