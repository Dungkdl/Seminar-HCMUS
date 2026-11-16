"""Regression checks for causal controls, OOF schedule, staged CLI and review boundaries."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from clindistill.__main__ import main
from clindistill.artifacts import read_json, write_json
from clindistill.research.data import FeatureStore, collate_visits
from clindistill.research.models import VisitClassifier
from clindistill.research.reporting import evaluate_study
from clindistill.research.review import readiness, review_status
from clindistill.research.training import (dataset_for, fit, forward_batch, model_spec,
    run_study, student_method, study_plan, validate_config)
from test_research_pipeline import PROJECT, config, synthetic_features


class ResearchFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_oof_replays_selected_prefix_of_full_schedule_including_resume(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); store = FeatureStore(synthetic_features(root/'features'), allow_debug=True)
            cfg = config(); cfg.update(epochs=6, patience=6)
            fit(store, root/'full', cfg, 'teacher_visual', 19, stop_after_epoch=3)
            fit(store, root/'oof', cfg, 'teacher_visual', 19, fixed_epochs=3)
            fit(store, root/'resume', cfg, 'teacher_visual', 19, fixed_epochs=3, stop_after_epoch=1)
            fit(store, root/'resume', cfg, 'teacher_visual', 19, fixed_epochs=3)
            reference = torch.load(root/'full/last.pt', weights_only=False)
            for folder in ('oof', 'resume'):
                actual = torch.load(root/folder/'last.pt', weights_only=False)
                self.assertEqual(actual['scheduler'], reference['scheduler'])
                for key, value in reference['model'].items():
                    torch.testing.assert_close(value, actual['model'][key], rtol=0, atol=0)
                self.assertEqual(read_json(root/folder/'protocol.json')['scheduler']['horizon_epochs'], 6)

    def test_clinical_control_never_receives_visual_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            store = FeatureStore(synthetic_features(d), allow_debug=True)
            spec, key = model_spec(store, config(), 'teacher_clinical')
            model = VisitClassifier(**spec).eval()
            self.assertFalse(hasattr(model, 'mil'))
            item = dataset_for(store, 'train', key, spec['role'], 'teacher_clinical')[0]
            batch = collate_visits([item])
            with torch.inference_mode():
                before = forward_batch(model, batch, 'cpu')['logits']
                batch['visual'].fill_(12345)
                torch.testing.assert_close(before, forward_batch(model, batch, 'cpu')['logits'])
                batch['clinical'] = torch.randn_like(batch['clinical'])
                self.assertFalse(torch.allclose(before, forward_batch(model, batch, 'cpu')['logits']))
                with self.assertRaisesRegex(ValueError, 'cannot receive visual'):
                    model(batch['visual'], batch['mask'], clinical=batch['clinical'],
                        demographics=batch['demographics'], clinical_present=batch['clinical_present'])

    def test_weighted_ce_uses_oof_weights_without_teacher_soft_targets(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); store = FeatureStore(synthetic_features(root/'cache'), allow_debug=True)
            cfg = config(); cfg.update(epochs=1, batch_size=24, dropout=0)
            weights = np.array([.05, .05, .95, .95]*6, dtype=np.float32)
            targets = {'reliability': weights, 'multimodal_logits': np.zeros((24, 2), np.float32)}
            fit(store, root/'a', cfg, 'student_weighted_ce', 12, targets=targets, target_identity='weights')
            targets['multimodal_logits'].fill(999)
            fit(store, root/'b', cfg, 'student_weighted_ce', 12, targets=targets, target_identity='weights')
            fit(store, root/'plain', cfg, 'student_no_kd', 12)
            a = torch.load(root/'a/last.pt', weights_only=False)['model']
            b = torch.load(root/'b/last.pt', weights_only=False)['model']
            plain = torch.load(root/'plain/last.pt', weights_only=False)['model']
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(a[k], plain[k]) for k in a))
            with self.assertRaisesRegex(ValueError, 'collapsed'):
                fit(store, root/'zero', cfg, 'student_weighted_ce', 12,
                    targets={'reliability': np.zeros(24, np.float32)}, target_identity='zero')

    def test_staged_profiles_and_cli_match_registered_plans(self):
        for name, fits in [('research_visual.json',12), ('research_clinical.json',18),
                           ('research_transfer.json',45), ('research.json',63), ('research_controls.json',87)]:
            path = PROJECT/'configs/code'/name
            cfg = read_json(path); validate_config(cfg)
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main(['plan', '--config', str(path)])
            self.assertEqual(json.loads(stream.getvalue())['total_fits'], fits)
        bad = config(); bad['teachers'] = ['teacher_visual']
        with self.assertRaisesRegex(ValueError, 'both registered'):
            validate_config(bad)

    def test_visual_and_clinical_stages_report_without_assuming_kd_or_export(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); cache = synthetic_features(root/'cache')
            for name in ('visual','clinical'):
                cfg = read_json(PROJECT/f'configs/code/research_{name}.json')
                cfg.update(debug=True, seeds=[2026], epochs=2, patience=2,
                    feature_dim=16, bootstrap_repetitions=5)
                run_study(cache, root/name, cfg, require_gpu=False)
                table, _ = evaluate_study(cache, root/name)
                self.assertEqual(len(table), len(study_plan(cfg)['systems']))
                self.assertFalse((root/name/'_oof').exists())
                self.assertEqual(student_method(cfg), 'student_no_kd' if name == 'visual' else None)

    def test_current_cli_trains_evaluates_and_legacy_requires_explicit_command(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); cache = synthetic_features(root/'cache')
            cfg = read_json(PROJECT/'configs/code/research_visual.json')
            cfg.update(debug=True, seeds=[2026], epochs=1, patience=1,
                       feature_dim=16, bootstrap_repetitions=5)
            write_json(root/'config.json', cfg)
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                main(['train','--config',str(root/'config.json'),'--features',str(cache),
                      '--study',str(root/'study'),'--allow-cpu-debug'])
            rows = json.loads(output.getvalue())
            self.assertTrue(all(row['macro_f1_std'] is None for row in rows))
            main(['evaluate','--features',str(cache),'--study',str(root/'study')])
            self.assertTrue((root/'study/report_validation/status.json').is_file())
            with self.assertRaises(SystemExit) as stopped:
                main(['legacy','--help'])
            self.assertEqual(stopped.exception.code,0)

    def test_pending_reviews_block_confirmatory_without_claiming_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cache = synthetic_features(root/'cache')
            store = FeatureStore(cache, allow_debug=True)
            cfg = config(); cfg['study_mode'] = 'confirmatory'
            report = readiness(store.metadata, cfg)
            self.assertFalse(report['confirmatory_ready'])
            with self.assertRaisesRegex(ValueError, 'Confirmatory study blocked'):
                run_study(cache, root/'study', cfg, require_gpu=False)
            self.assertFalse((root/'study').exists())
            write_json(root/'release.json', {'release_id':'fixture','stage':'experimental','pending_reviews':['clinical']})
            columns = ['case_id','reviewer','label_mapping_correct','annotation_complete_for_six_targets',
                       'post_diagnosis_leakage','visual_grounding_review']
            with (root/'clinical_label_review_train.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
                writer.writerow({'case_id':'synthetic_private_id'})
            report = review_status(root)
            self.assertEqual(report['review_files']['clinical_label_review_train.csv']['pending_rows'],1)
            self.assertFalse(report['expert_reviews_verified'])
            self.assertNotIn('synthetic_private_id', json.dumps(report))

    def test_single_target_readiness_does_not_require_complete_multilabel_negatives(self):
        cfg = config()
        metadata = {'debug':False,'release_stage':'reviewed',
            'pending_reviews':[{'check':'label completeness for six-target multi-label training', 'status':'NOT_APPROVED'}],
            'request':{'policy':{'temporal_verified':True,'review_record':'independently documented fixture'}}}
        report = readiness(metadata, cfg)
        self.assertTrue(report['confirmatory_ready'])
        self.assertEqual(len(report['pending_reviews']),1)
        self.assertFalse(report['publication_ready'])
