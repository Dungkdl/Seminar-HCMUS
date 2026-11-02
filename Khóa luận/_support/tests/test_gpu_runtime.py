"""Software checks for flattened original photos, worker ordering and GPU cache batching."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from clindistill.artifacts import file_hash
from clindistill.research.data import (FeatureStore, VisitDataset, adaptive_image_forward,
                                     collate_visits, encode_images, resolve_image_sources)
from clindistill.research.runtime import feature_settings
from clindistill.research.training import ResidentVisitLoader
from test_research_pipeline import synthetic_features, tiny_encoders


class ThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pro6000_free_memory_policy_and_explicit_overrides(self):
        info = {'device': 'cuda', 'workers': 8, 'free_gib': 90, 'total_gib': 96}
        cfg = {'batch_size': 32, 'image_batch_size': 'auto'}
        settings = feature_settings(cfg, info)
        self.assertEqual(settings['image_batches']['teacher'], 128)
        self.assertEqual(settings['image_batches']['student'], 256)
        self.assertEqual(settings['workers'], 8)
        self.assertEqual(cfg['batch_size'], 32)
        smaller = feature_settings(cfg, {**info, 'free_gib': 20})
        self.assertEqual(smaller['image_batches']['teacher'], 8)
        explicit = feature_settings({**cfg, 'image_batch_size': 3, 'image_workers': 0}, info)
        self.assertEqual(set(explicit['image_batches'].values()), {3})
        for invalid in [0, 'fast', True]:
            with self.assertRaises(ValueError):
                feature_settings({'image_batch_size': invalid}, info)

    def test_flattened_paths_require_original_checksums_and_reject_collisions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'images').mkdir()
            photo = root / 'images/photo.jpg'; Image.new('RGB', (12, 14), 'red').save(photo)
            canonical = 'Images/Photographs/photo.jpg'
            with self.assertRaisesRegex(ValueError, 'image_index'):
                resolve_image_sources([canonical], root)
            proof = {canonical: file_hash(photo)}
            mapped = resolve_image_sources([canonical], root, proof)
            self.assertEqual(Path(mapped[canonical]), photo.resolve())
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                resolve_image_sources([canonical], root, {canonical: 'wrong'})
            other = 'Images/Radiographs/photo.jpg'
            with self.assertRaisesRegex(ValueError, 'colliding'):
                resolve_image_sources([canonical, other], root, {**proof, other: file_hash(photo)})
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                resolve_image_sources(['../escape.jpg'], root)

    def test_oom_microbatch_retry_preserves_order_and_does_not_swallow_other_errors(self):
        calls = []
        def encoder(batch):
            calls.append(len(batch))
            if len(batch) > 2:
                raise torch.cuda.OutOfMemoryError('synthetic memory limit')
            return batch * 3
        values = torch.arange(7).reshape(7, 1)
        state = {'batch_size': len(values)}
        with contextlib.redirect_stdout(io.StringIO()):
            result = adaptive_image_forward(values, encoder, device='cpu', state=state)
        torch.testing.assert_close(result, values * 3)
        self.assertGreater(len(calls), 3)
        calls.clear()
        torch.testing.assert_close(adaptive_image_forward(values, encoder, device='cpu', state=state), values * 3)
        self.assertLessEqual(max(calls), 2)
        def broken(batch):
            raise RuntimeError('invalid weights')
        with self.assertRaisesRegex(RuntimeError, 'invalid weights'):
            adaptive_image_forward(values, broken, device='cpu')

    def test_resident_batches_match_ordinary_loader_with_shuffled_clinical(self):
        with tempfile.TemporaryDirectory() as d:
            store = FeatureStore(synthetic_features(d), allow_debug=True)
            dataset = VisitDataset(store, 'train', 'teacher', role='multimodal_teacher',
                                   clinical_mode='shuffled', shuffle_seed=42)
            # Include unequal bag sizes without changing the visit/clinical mapping.
            store.data['train']['offsets'] = np.cumsum([0] + [1, 3] * 12)
            standard = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_visits,
                                  generator=torch.Generator().manual_seed(77))
            resident = ResidentVisitLoader(dataset, 5, True, torch.Generator().manual_seed(77), device='cpu')
            self.assertEqual(len(standard), len(resident))
            for expected, actual in zip(standard, resident):
                self.assertEqual(expected['ids'], actual['ids'])
                for key in expected.keys() - {'ids'}:
                    torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

    def test_parallel_encoder_and_chunk_resume_equal_sequential_original_images(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); models = tiny_encoders(root / 'models'); paths = []
            for i in range(5):
                name = f'{i}.jpg'; Image.new('RGB', (25 + i, 31), (i * 30, 60, 90)).save(root / name)
                paths.append(name)
            expected = encode_images(paths, root, models['student'], 'student', 32, 2,
                                     root / 'sequential', allow_debug=True, chunk_size=2)
            actual = encode_images(paths, root, models['student'], 'student', 32, 2,
                                   root / 'parallel', allow_debug=True, workers=1, chunk_size=2)
            np.testing.assert_array_equal(expected, actual)
            resumed = encode_images(paths, root, models['student'], 'student', 32, 2,
                                    root / 'parallel', allow_debug=True, workers=1, chunk_size=2)
            np.testing.assert_array_equal(actual, resumed)


if __name__ == '__main__':
    unittest.main()
