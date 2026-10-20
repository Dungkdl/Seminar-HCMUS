"""Clinical distillation behavioral contract; synthetic data, no downloads."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from clindistill.artifacts import file_hash, read_json, write_json
from clindistill.clinical import FIELDS, ClinicalCache, prepare_clinical_cache
from clindistill.distillation import grouped_kd
from clindistill.distillation import representation_losses
from clindistill.engine import fit, evaluate_checkpoint
from clindistill.reporting import evaluate_suite, export_student, predict_images, paired_bootstrap
from clindistill.suite import run_suite
import test_training_pipeline as baseline_tests
from test_training_pipeline import synthetic_release


def contexts(prepared, records):
    release = read_json(prepared / 'release.json')
    for split, rows in records.items():
        path = prepared / f'teacher_context_{split}.jsonl'
        path.write_text('\n'.join(json.dumps({
            'case_id': r['case_id'], 'patient_id': r['patient_id'], 'split': split,
            'clinical_screened': {f: (f'Observed feature {i}' if i % 3 else '') for f in FIELDS},
        }) for i, r in enumerate(rows)) + '\n', encoding='utf-8')
        release['file_checksums'][path.name] = file_hash(path)
    write_json(prepared / 'release.json', release)


class ClinicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_kd_direction_temperature_and_missing_groups(self):
        s = torch.tensor([[1., -1.], [0., 2.]], requires_grad=True)
        t = torch.tensor([[[0., 2.], [2., 0.]], [[2., 0.], [0., 2.]]], requires_grad=True)
        available = torch.tensor([[True, False], [False, False]])
        loss = grouped_kd(s, t, available, [3., 1.], 2.)
        expected = F.kl_div(F.log_softmax(s[:1] / 2, -1), F.softmax(t[:1, 0].detach() / 2, -1),
                            reduction='sum') * 4 / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(t.grad)
        torch.testing.assert_close(s.grad[1], torch.zeros(2))
        self.assertEqual(float(grouped_kd(s, t, torch.zeros_like(available), [1., 1.], 2).detach()), 0.)

    def test_representation_losses_mask_missing_context_and_freeze_teacher(self):
        student = torch.randn(3, 8, requires_grad=True)
        teacher = student.detach().clone().requires_grad_()
        teacher.data[2] += 100
        feature, relation = representation_losses(student, teacher, torch.tensor([True, True, False]))
        torch.testing.assert_close(feature, torch.tensor(0.))
        torch.testing.assert_close(relation, torch.tensor(0.))
        (feature + relation).backward()
        self.assertIsNone(teacher.grad)
        torch.testing.assert_close(student.grad[2], torch.zeros(8))

    def test_cache_identity_missingness_and_no_test_access(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, prepared, records = synthetic_release(root)
            contexts(prepared, records)
            (prepared / 'student_single_test.jsonl').unlink()
            (prepared / 'teacher_context_test.jsonl').unlink()
            prepare_clinical_cache(prepared, root / 'cache', debug=True)
            cache = ClinicalCache(root / 'cache', 'synthetic-test-only', allow_debug=True)
            vectors, mask = cache.batch(['train_0', 'train_1'], 'cpu', allow_radiographs=False)
            self.assertFalse(mask[0].any())
            self.assertFalse(mask[:, 1].any())
            self.assertTrue((vectors[~mask] == 0).all())
            with self.assertRaisesRegex(ValueError, 'release'):
                ClinicalCache(root / 'cache', 'wrong-release', allow_debug=True)
            with self.assertRaisesRegex(ValueError, 'debug'):
                ClinicalCache(root / 'cache', 'synthetic-test-only')
            with self.assertRaises(KeyError):
                cache.batch(['unknown'], 'cpu')

    def test_real_transformer_backend_offline_and_adding_test_preserves_identity(self):
        # Locally initialized tiny BERT validates the production backend, not biomedical quality.
        from transformers import BertConfig, BertModel, BertTokenizerFast
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, prepared, records = synthetic_release(root)
            contexts(prepared, records)
            encoder = root / 'tiny_bert_fixture'
            encoder.mkdir()
            vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'observed', 'feature',
                     '0', '1', '2', '3', '4', '5']
            (encoder / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
            tokenizer = BertTokenizerFast(vocab_file=str(encoder / 'vocab.txt'), do_lower_case=True)
            tokenizer.save_pretrained(encoder)
            model = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=32))
            model.save_pretrained(encoder)
            cache_path = root / 'cache'
            prepare_clinical_cache(prepared, cache_path, encoder=str(encoder), device='cpu',
                                   max_length=16, local_files_only=True)
            before = ClinicalCache(cache_path, 'synthetic-test-only')
            prepare_clinical_cache(prepared, cache_path, encoder=str(encoder), device='cpu',
                                   max_length=16, local_files_only=True, splits=('test',))
            after = ClinicalCache(cache_path, 'synthetic-test-only')
            self.assertEqual(before.fingerprint, after.fingerprint)
            test = ClinicalCache(cache_path, 'synthetic-test-only', splits=('test',))
            vectors, mask = test.batch(['test_0', 'test_1'], 'cpu')
            self.assertEqual(vectors.shape, (2, 5, 16))
            self.assertTrue(torch.isfinite(vectors).all())
            self.assertGreater(float(vectors[mask].abs().sum()), 0.)

    def test_teachers_kd_resume_and_visual_only_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images, prepared, records = synthetic_release(root)
            contexts(prepared, records)
            (prepared / 'student_single_test.jsonl').unlink()
            (prepared / 'teacher_context_test.jsonl').unlink()
            cache_path = root / 'cache'
            prepare_clinical_cache(prepared, cache_path, debug=True)
            base = replace(baseline_tests.TrainingTests().config(), epochs=2)
            teachers = {}
            for group in ['clinical', 'full', 'v', 'h']:
                run = root / f'teacher_{group}'
                cfg = replace(base, method=f'teacher_{group}')
                fit(prepared, images, run, cfg, clinical_cache=cache_path, device='cpu')
                teachers[group] = run
            teacher_hashes = {g: file_hash(p / 'best.pt') for g, p in teachers.items()}
            for method in ['kd_clinical', 'kd_full', 'kd_equal', 'kd_selective']:
                cfg = replace(base, method=method)
                fit(prepared, images, root / method, cfg, clinical_cache=cache_path,
                    teacher_dirs=teachers, device='cpu', stop_after_epoch=1)
                result = fit(prepared, images, root / method, cfg, clinical_cache=cache_path,
                             teacher_dirs=teachers, device='cpu', resume=True)
                self.assertTrue(result['complete'])
                evaluated = evaluate_checkpoint(prepared, images, root / method, split='validation', device='cpu')
                self.assertEqual(evaluated['metrics']['cases'], 6)
                if method == 'kd_selective':
                    fit(prepared, images, root / 'uninterrupted', cfg, clinical_cache=cache_path,
                        teacher_dirs=teachers, device='cpu')
                    a = torch.load(root / method / 'last.pt', weights_only=False)
                    b = torch.load(root / 'uninterrupted' / 'last.pt', weights_only=False)
                    for key in a['model']:
                        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual(teacher_hashes, {g: file_hash(p / 'best.pt') for g, p in teachers.items()})

    def test_suite_report_export_and_unlabelled_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images, prepared, records = synthetic_release(root)
            contexts(prepared, records)
            cache = root / 'cache'
            prepare_clinical_cache(prepared, cache, debug=True)
            cfg = replace(baseline_tests.TrainingTests().config(), epochs=1, accumulation_steps=2)
            runs = root / 'runs'
            run_suite(prepared, images, runs, cfg, clinical_cache=cache,
                      selected=['photo_attention', 'kd_feature', 'kd_relation', 'kd_selective', 'photo_set_attention'],
                      seeds=[2026], device='cpu')
            table, _, paired = evaluate_suite(prepared, images, runs, clinical_cache=cache, device='cpu')
            self.assertEqual(len(table), 8)
            self.assertIn('kd_selective_minus_photo_attention', paired)
            export_student(runs / 'kd_selective' / 'seed_2026', root / 'student.pt')
            result = predict_images(root / 'student.pt', images, records['test'][0]['photo_paths'], device='cpu')
            self.assertAlmostEqual(sum(result['probabilities'].values()), 1., places=6)
            import pandas as pd
            frame = pd.read_csv(runs / 'photo_attention' / 'seed_2026' / 'evaluation_validation' / 'predictions.csv')
            identical = paired_bootstrap([frame], [frame.sample(frac=1, random_state=1)], 6, 10)
            self.assertEqual(identical['paired_group_bootstrap_95_ci'], [0., 0.])
            wrong = frame.copy()
            wrong.loc[0, 'label'] = 5
            with self.assertRaisesRegex(ValueError, 'identical'):
                paired_bootstrap([frame], [wrong], 6, 10)


if __name__ == '__main__':
    unittest.main()
