import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from test_training_pipeline import synthetic_release
from test_clinical_distillation import contexts

spec = importlib.util.spec_from_file_location('offline', Path(__file__).resolve().parents[1] / 'kaggle_offline.py')
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


class OfflineTests(unittest.TestCase):
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
