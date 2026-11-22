"""Synthetic paired Run 5 exercises the real fit/report path with a stub encoder."""
import importlib.util
import json
from pathlib import Path
import sys
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
spec6 = importlib.util.spec_from_file_location('run6', PROJECT / '_support/research-notes/run6_xray_fusion.py')
run6 = importlib.util.module_from_spec(spec6)
with patch.dict(sys.modules, {'run5_xray': run5}):
    spec6.loader.exec_module(run6)
spec7 = importlib.util.spec_from_file_location('run7', PROJECT / '_support/research-notes/run7_xray_kd.py')
run7 = importlib.util.module_from_spec(spec7)
with patch.dict(sys.modules, {'run6_xray_fusion': run6}):
    spec7.loader.exec_module(run7)


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

            fusion = root / 'fusion'
            with patch.object(run6, 'encode_images', side_effect=encode):
                summary, contrasts = run6.run_fusion(prepared, cache, root, encoder, fusion, cfg)
                self.assertEqual(set(summary.index), set(run6.ARMS))
                self.assertEqual(len(contrasts), 2)
                self.assertEqual(read_json(fusion/'completed.json')['fits'], 6)
                self.assertFalse(read_json(fusion/'run6_protocol.json')['test_opened'])
                self.assertEqual(read_json(fusion/'paired_cohort.json'), read_json(out/'paired_cohort.json'))
                # The reproduced early arm should match Run 5 under the same frozen inputs.
                early = read_json(fusion/'metrics.json')['run5_early_fusion/2026']
                self.assertEqual(early, read_json(out/'metrics.json')['photo_xray_mean/2026'])
                checkpoint = fusion/'two_branch_photo_xray/seed_2026/best.pt'
                before = file_hash(checkpoint)
                run6.run_fusion(prepared, cache, root, encoder, fusion, cfg)
                self.assertEqual(file_hash(checkpoint), before)

            kd = root / 'kd'
            with patch.object(run6, 'encode_images', side_effect=encode):
                summary, contrasts = run7.run_kd(prepared, cache, root, encoder, kd, cfg)
                self.assertEqual(set(summary.index), set(run7.LOSSES))
                self.assertEqual(len(contrasts), 3)
                self.assertEqual(read_json(kd/'completed.json')['total_fits'], 12)
                self.assertFalse(read_json(kd/'run7_protocol.json')['test_opened'])
                self.assertEqual(read_json(kd/'paired_cohort.json'), read_json(fusion/'paired_cohort.json'))
                self.assertEqual(read_json(kd/'metrics.json')['ce/2026'],
                                 read_json(fusion/'metrics.json')['two_branch_photo_xray/2026'])
                targets = kd/'_oof/seed_2026/targets.npz'
                with np.load(targets, allow_pickle=False) as data:
                    self.assertEqual(len(set(data['ids'])), 23)
                    self.assertEqual(set(data['fold']), {0, 1})
                    self.assertTrue(np.isfinite(data['logits']).all())
                saved = file_hash(targets)
                run7.run_kd(prepared, cache, root, encoder, kd, cfg)
                self.assertEqual(file_hash(targets), saved)

    def test_run6_modalities_and_padding(self):
        torch.manual_seed(7)
        masked = run6.TwoBranchMean(3, 8, 0., 2, use_xray=False).eval()
        full = run6.TwoBranchMean(3, 8, 0., 2, use_xray=True).eval()
        full.load_state_dict(masked.state_dict())
        bag = torch.tensor([[[0., 1., 0., 0.], [1., 0., 0., 1.], [0., 7., 9., 0.]]])
        changed = bag.clone()
        changed[:, 1, :3] = torch.tensor([0., 0., 5.])
        mask = torch.tensor([[True, True, False]])
        with torch.no_grad():
            torch.testing.assert_close(masked(bag, mask)['logits'], masked(changed, mask)['logits'])
            self.assertFalse(torch.allclose(full(bag, mask)['logits'], full(changed, mask)['logits']))
            torch.testing.assert_close(full(bag, mask)['logits'], full(bag[:, :2], mask[:, :2])['logits'])
