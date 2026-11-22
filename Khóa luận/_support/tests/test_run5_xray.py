"""Synthetic paired Run 5 exercises the real fit/report path with a stub encoder."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from clindistill.artifacts import file_hash, read_json, write_json
from test_research_pipeline import PROJECT, config, synthetic_features

spec = importlib.util.spec_from_file_location('run5', PROJECT / '_support/research-notes/run5_xray.py')
run5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run5)


class Run5Tests(unittest.TestCase):
    def test_paired_cohort_joint_fit_resume_and_report(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder = root / 'encoder'
            encoder.mkdir()
            (encoder / 'config.json').write_text('{}', encoding='utf-8')
            cache = synthetic_features(root / 'cache', encoders={'student': encoder})
            manifest = read_json(cache / 'feature_manifest.json')
            manifest['debug'] = False
            write_json(cache / 'feature_manifest.json', manifest)
            prepared = root / 'prepared'
            prepared.mkdir()
            files = {}
            from PIL import Image
            folder = root / 'Images/Radiographs'
            folder.mkdir(parents=True)
            for split in run5.SPLITS:
                rows = []
                with np.load(cache / f'{split}.npz', allow_pickle=False) as shard:
                    for i, case in enumerate(shard['ids']):
                        label = int(shard['labels'][i])
                        path = folder / f'{case}.jpg'
                        Image.new('RGB', (16, 16), (len(rows)*7, 20 if split == 'train' else 150, 42)).save(path)
                        rows.append({'case_id': str(case), 'patient_id': str(shard['patients'][i]),
                            'split_group': str(shard['groups'][i]), 'split': split,
                            'eligible_single': True, 'target_labels': [['A','B'][label]], 'labels_multihot': [1-label,label],
                            'single_label_index': label, 'photo_paths': shard['image_paths'][2*i:2*i+2].tolist(),
                            'radiograph_paths': [path.relative_to(root).as_posix()] if i != 0 else []})
                dest = prepared / f'student_single_{split}.jsonl'
                dest.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
                files[dest.name] = file_hash(dest)
            # Missing test file is intentional; this experiment must never open it.
            files['student_single_test.jsonl'] = 'unopened-synthetic-test'
            write_json(prepared / 'release.json', {'schema_version':1, 'dataset':'COde',
                'release_id':manifest['request']['release_id'], 'class_names':['A','B'],
                'pending_reviews':['fixture'], 'file_checksums':files})
            cfg = config()
            cfg.update(epochs=1, patience=1, seeds=[2026,2027], debug=False)
            out = root/'study'
            def encode(paths, *args, **kwargs):
                return np.ones((len(paths), 64), dtype=np.float16)
            with patch.object(run5, 'encode_images', side_effect=encode):
                summary, contrast = run5.run_xray(prepared, cache, root, encoder, out, cfg)
                self.assertEqual(set(summary.index), set(run5.ARMS))
                self.assertEqual(contrast['fitted_seeds'], 2)
                self.assertEqual(read_json(out/'paired_cohort.json')['validation']['visits'], 7)
                self.assertEqual(read_json(out/'completed.json')['fits'], 4)
                self.assertFalse(read_json(out/'run5_protocol.json')['test_opened'])
                best = out/'photo_mean/seed_2026/best.pt'
                checksum = file_hash(best)
                run5.run_xray(prepared, cache, root, encoder, out, cfg)
                self.assertEqual(file_hash(best), checksum)
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    run5.run_xray(prepared, cache, root, encoder, out, {**cfg, 'epochs':2})
