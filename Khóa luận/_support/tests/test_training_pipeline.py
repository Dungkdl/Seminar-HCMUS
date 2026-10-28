"""Behavioral checks: no real patient data or network calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

import numpy as np
from PIL import Image
import torch

from clindistill.artifacts import file_hash, read_json, verify_release, write_json
from clindistill.config import TrainConfig
from clindistill.data import BagDataset, check_disjoint, collate_bags
from clindistill.engine import evaluate_checkpoint, fit
from clindistill.metrics import classification_metrics, group_bootstrap
from clindistill.models import VisualBagClassifier


def synthetic_release(root):
    images, prepared = root / 'images', root / 'prepared'
    images.mkdir()
    prepared.mkdir()
    names, records = [], {}
    for split in ['train', 'validation', 'test']:
        records[split] = []
        for i in range(6):
            paths = []
            for j in range(1 + i % 2):
                name = f'{split}_{i}_{j}.png'
                rng = np.random.default_rng(i * 5 + j + {'train': 0, 'validation': 100, 'test': 200}[split])
                Image.fromarray(rng.integers(0, 255, (32, 40, 3), dtype=np.uint8)).save(images / name)
                paths.append(name)
            case = f'{split}_{i}'
            r = {'case_id': case, 'patient_id': case, 'split_group': case, 'bag_signature': case,
                 'split': split, 'photo_paths': paths, 'radiograph_paths': [], 'single_label_index': i,
                 'target_labels': [str(i)], 'labels_multihot': [int(k == i) for k in range(6)], 'eligible_single': True}
            records[split].append(r)
        name = f'student_single_{split}.jsonl'
        names.append(name)
        (prepared / name).write_text('\n'.join(json.dumps(r) for r in records[split]) + '\n', encoding='utf-8')
    write_json(prepared / 'release.json', {'schema_version': 1, 'dataset': 'COde', 'stage': 'experimental',
        'release_id': 'synthetic-test-only', 'class_names': [str(i) for i in range(6)], 'pending_reviews': [],
        'file_checksums': {n: file_hash(prepared / n) for n in names}})
    return images, prepared, records


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        return TrainConfig(backbone='tiny_debug', pretrained=False, image_size=32, feature_dim=16,
                           max_photos=2, batch_size=3, epochs=2, patience=3, workers=0, amp=False,
                           dropout=0.2, bootstrap_repetitions=10)

    def test_padding_and_permutation_do_not_change_prediction(self):
        model = VisualBagClassifier(self.config()).eval()
        images = torch.randn(1, 2, 3, 32, 32)
        with torch.no_grad():
            expected = model(images, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.long))['logits']
            padded = torch.cat([images, torch.randn(1, 2, 3, 32, 32) * 1000], dim=1)
            mask = torch.tensor([[True, True, False, False]])
            result = model(padded, mask, torch.zeros(1, 4, dtype=torch.long))
            permuted = model(images.flip(1), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.long))
        torch.testing.assert_close(expected, result['logits'])
        torch.testing.assert_close(expected, permuted['logits'])
        self.assertTrue((result['attention'][~mask] == 0).all())

    def test_set_pooling_respects_padding_and_permutation(self):
        model = VisualBagClassifier(replace(self.config(), experiment='photo_set_attention')).eval()
        images = torch.randn(2, 3, 3, 32, 32)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        modalities = torch.zeros(2, 3, dtype=torch.long)
        with torch.no_grad():
            a = model(images, mask, modalities)['logits']
            b = model(images.flip(1), mask.flip(1), modalities.flip(1))['logits']
        torch.testing.assert_close(a, b)

    def test_autocast_with_mixed_modalities_keeps_gradients_and_masks_padding(self):
        images = torch.randn(2, 3, 3, 32, 32)
        mask = torch.tensor([[True, False, False], [True, True, False]])
        modalities = torch.tensor([[0, 0, 0], [0, 1, 0]])
        devices = [('cpu', torch.bfloat16)]
        if torch.cuda.is_available():
            devices.append(('cuda', torch.float16))
        for device, dtype in devices:
            for experiment in ('photo_mean', 'photo_attention', 'photo_set_attention'):
                with self.subTest(device=device, experiment=experiment):
                    model = VisualBagClassifier(replace(self.config(), experiment=experiment)).to(device)
                    with torch.autocast(device_type=device, dtype=dtype):
                        output = model(images.to(device), mask.to(device), modalities.to(device))
                        loss = torch.nn.functional.cross_entropy(output['logits'], torch.tensor([0, 1], device=device))
                    self.assertTrue(torch.isfinite(output['logits']).all())
                    self.assertTrue((output['attention'][~mask.to(device)] == 0).all())
                    loss.backward()
                    for parameter in (model.projection[0].weight, model.modality_embedding.weight):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_dataset_is_deterministic_and_split_group_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            image_root, _, records = synthetic_release(Path(d))
            dataset = BagDataset(records['train'], image_root, self.config(), training=True)
            torch.testing.assert_close(dataset[0]['images'], dataset[0]['images'])
            batch = collate_bags([dataset[0], dataset[1]])
            self.assertEqual(batch['mask'].tolist(), [[True, False], [True, True]])
            contaminated = [dict(records['validation'][0], split_group=records['train'][0]['split_group'])]
            with self.assertRaisesRegex(ValueError, 'split_group'):
                check_disjoint(records['train'], contaminated)

    def test_metrics_and_group_bootstrap(self):
        y = np.array([0, 0, 1, 1])
        p = np.eye(2)[y]
        result = classification_metrics(y, p, ['a', 'b'])
        self.assertEqual(result['macro_f1'], 1.0)
        self.assertEqual(result['brier_multiclass'], 0.0)
        intervals = group_bootstrap(y, p, ['g1', 'g2', 'g1', 'g2'], repetitions=10)
        self.assertEqual(intervals['macro_f1_95_ci'], [1.0, 1.0])

    def test_checkpoint_resume_matches_uninterrupted_training_without_test_access(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images, prepared, _ = synthetic_release(root)
            # Train must work even when no test file is accessible.
            (prepared / 'student_single_test.jsonl').unlink()
            full = fit(prepared, images, root / 'full', self.config(), device='cpu')
            partial = fit(prepared, images, root / 'resumed', self.config(), device='cpu', stop_after_epoch=1)
            self.assertFalse(partial['complete'])
            resumed = fit(prepared, images, root / 'resumed', self.config(), device='cpu', resume=True)
            self.assertTrue(resumed['complete'])
            self.assertEqual(full['best_validation_macro_f1'], resumed['best_validation_macro_f1'])
            a = torch.load(root / 'full' / 'last.pt', weights_only=False)
            b = torch.load(root / 'resumed' / 'last.pt', weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            result = evaluate_checkpoint(prepared, images, root / 'resumed', split='validation', device='cpu')
            self.assertEqual(result['metrics']['cases'], 6)

    def test_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _, prepared, _ = synthetic_release(Path(d))
            p = prepared / 'student_single_train.jsonl'
            p.write_text(p.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_release(prepared, splits=['train'])


if __name__ == '__main__':
    unittest.main()
