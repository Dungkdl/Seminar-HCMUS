import importlib.util
import json
import shutil
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from test_training_pipeline import synthetic_release
from test_clinical_distillation import contexts

spec = importlib.util.spec_from_file_location('offline', Path(__file__).resolve().parents[1] / 'kaggle_offline.py')
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)

repair_spec = importlib.util.spec_from_file_location('recovery', Path(__file__).resolve().parents[1] / 'recover_amp_bundle.py')
recovery = importlib.util.module_from_spec(repair_spec)
repair_spec.loader.exec_module(recovery)


class OfflineTests(unittest.TestCase):
    def test_oof_mini_uses_independent_groups_instead_of_first_two_visits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images, release, records = synthetic_release(root)
            for split in ('train', 'validation'):
                expanded = []
                for original in records[split]:
                    for suffix in ('a', 'b', 'c'):
                        row = dict(original, case_id=original['case_id'] + suffix,
                            patient_id=original['patient_id'] + ('x' if suffix == 'c' else ''),
                            split_group=original['split_group'] + ('x' if suffix == 'c' else ''),
                            bag_signature=original['bag_signature'] + suffix)
                        expanded.append(row)
                records[split] = expanded
                path = release / f'student_single_{split}.jsonl'
                path.write_text(''.join(json.dumps(r) + '\n' for r in expanded), encoding='utf-8')
            info = json.loads((release / 'release.json').read_text())
            info['file_checksums'] = {name: offline.digest(release / name) for name in info['file_checksums']}
            offline.write_json(release / 'release.json', info)
            contexts(release, records)
            offline.make_mini_code(release, images, root / 'mini', per_class=2, independent_groups=True)
            for split in ('train', 'validation'):
                selected = [json.loads(line) for line in (root / f'mini/prepared/student_single_{split}.jsonl').read_text().splitlines()]
                self.assertEqual(len(selected), 12)
                self.assertEqual(len({r['split_group'] for r in selected}), 12)
                self.assertTrue(all(r['case_id'].endswith(('a', 'c')) for r in selected))

    def test_amp_repair_keeps_assets_and_updates_source_wheel_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle = root / 'bundle'
            source = bundle / 'source'
            source.mkdir(parents=True)
            project = Path(__file__).resolve().parents[2]
            shutil.copy2(project / 'pyproject.toml', source / 'pyproject.toml')
            shutil.copytree(project / 'clindistill', source / 'clindistill',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            (source / '_support').mkdir()
            shutil.copy2(project / '_support/kaggle_offline.py', source / '_support/kaggle_offline.py')
            model = source / 'clindistill/models.py'
            model.write_text(model.read_text(encoding='utf-8').replace(recovery.NEW, recovery.OLD), encoding='utf-8')
            wheel = offline.build_project_wheel(source, bundle / 'wheels', root / 'initial_build')
            for name in ('requirements.txt', 'base_constraints.txt', 'models/resnet50-default.pth',
                         'models/biomedbert/config.json'):
                path = bundle / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'unchanged fixture asset')
            manifest = bundle / 'offline_manifest.json'
            offline.write_json(manifest, {'schema_version': 1, 'status': 'assets_prepared_smoke_pending',
                'checksums': {p.relative_to(bundle).as_posix(): offline.digest(p)
                    for p in bundle.rglob('*') if p.is_file()}})
            originals = [p.read_bytes() for p in (model, wheel, manifest)]
            with patch('subprocess.run', side_effect=RuntimeError('simulated build failure')):
                with self.assertRaisesRegex(RuntimeError, 'simulated build failure'):
                    recovery.repair_bundle(bundle)
            self.assertEqual(originals, [p.read_bytes() for p in (model, wheel, manifest)])
            helper, repaired = recovery.repair_bundle(bundle)
            self.assertTrue(repaired)
            self.assertIn(recovery.NEW, model.read_text(encoding='utf-8'))
            with zipfile.ZipFile(wheel) as archive:
                self.assertEqual(archive.read('clindistill/models.py'), model.read_bytes())
            self.assertEqual((bundle / 'models/resnet50-default.pth').read_bytes(), b'unchanged fixture asset')
            self.assertEqual(helper.verify_bundle(bundle)['status'], 'assets_prepared_smoke_pending')
            self.assertFalse(recovery.repair_bundle(bundle)[1])

    def test_subset_copies_real_bags_and_never_requires_test(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images, release, records = synthetic_release(root)
            contexts(release, records)
            (release / 'student_single_test.jsonl').unlink()
            (release / 'teacher_context_test.jsonl').unlink()
            result = offline.make_mini_code(release, images, root / 'mini', per_class=1)
            self.assertEqual(result['cases'], {'train': 6, 'validation': 6})
            self.assertFalse(result['contains_test'])
            mini = root / 'mini'
            info = json.loads((mini / 'prepared/release.json').read_text())
            self.assertTrue(info['debug_subset'])
            for relative, checksum in info['file_checksums'].items():
                self.assertEqual(offline.digest(mini / 'prepared' / relative), checksum)
            for split in ('train', 'validation'):
                for r in records[split]:
                    for path in r['photo_paths']:
                        self.assertEqual(offline.digest(images / path), offline.digest(mini / path))
            archive = offline.zip_directory(mini, root / 'mini.zip')
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertFalse(any('test_' in name for name in z.namelist()))

    def test_platform_or_base_package_mismatch_rejected(self):
        expected = offline.inventory()
        offline.validate_platform(expected, expected)
        for field in ['python_minor', 'system', 'machine']:
            wrong = dict(expected, **{field: 'different'})
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                offline.validate_platform(expected, wrong)
        wrong = {**expected, 'packages': dict(expected['packages'], numpy='0.0.0')}
        with self.assertRaisesRegex(ValueError, 'numpy'):
            offline.validate_platform(expected, wrong)

    def test_bundle_rejects_changed_or_missing_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = ['requirements.txt', 'base_constraints.txt', 'models/resnet50-default.pth',
                     'models/biomedbert/config.json']
            for relative in paths:
                p = root / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('fixture')
            offline.write_json(root / 'offline_manifest.json', {'schema_version': 1,
                'checksums': {p: offline.digest(root / p) for p in paths}})
            offline.verify_bundle(root)
            (root / 'requirements.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                offline.verify_bundle(root)


if __name__ == '__main__':
    unittest.main()
