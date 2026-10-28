"""Offline extension integrity; no network and no package installation in the test process."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('extension_test', Path(__file__).resolve().parents[1] / 'backbone_extension.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class BackboneExtensionTests(unittest.TestCase):
    def fixture(self, root):
        values = {'source/clindistill/models.py': b'# fixture model',
                  'source/clindistill/config.py': b'# fixture config',
                  'source/configs/code/thesis.json': b'{}',
                  'source/_support/backbone_extension.py': b'# fixture helper',
                  'source/_support/training_runtime.py': b'# fixture runtime',
                  'models/swin_b-68c6b09e.pth': b'fixture weights, not a real model'}
        for name, value in values.items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(value)
        wheel = root / 'wheels/project.whl'
        wheel.parent.mkdir()
        with zipfile.ZipFile(wheel, 'w') as z:
            for name in ('models.py', 'config.py'):
                z.writestr('clindistill/' + name, (root / 'source/clindistill' / name).read_bytes())
        manifest = {'schema_version': 1, 'backbones': ['resnet50', 'swin_b'],
                    'project_wheel': 'wheels/project.whl',
                    'checksums': {p.relative_to(root).as_posix(): extension.digest(p)
                                  for p in root.rglob('*') if p.is_file()}}
        (root / 'backbone_extension.json').write_text(json.dumps(manifest))
        return manifest['checksums']['models/swin_b-68c6b09e.pth']

    def test_tampered_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(extension, 'SWIN_SHA256', self.fixture(root)):
                extension.verify_extension(root)
                (root / 'source/clindistill/models.py').write_bytes(b'modified')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    extension.verify_extension(root)

    def test_source_wheel_disagreement_is_rejected_even_with_updated_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(extension, 'SWIN_SHA256', self.fixture(root)):
                p = root / 'source/clindistill/models.py'
                p.write_bytes(b'changed but not rebuilt')
                manifest = json.loads((root / 'backbone_extension.json').read_text())
                manifest['checksums']['source/clindistill/models.py'] = extension.digest(p)
                (root / 'backbone_extension.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'wheel/source mismatch'):
                    extension.verify_extension(root)

    def test_loaded_source_conflict_requires_restart_before_install(self):
        import clindistill
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(extension, 'SWIN_SHA256', self.fixture(root)), patch('subprocess.run') as pip:
                with self.assertRaisesRegex(RuntimeError, 'Restart'):
                    extension.activate_extension(root, {'resnet': 'unused'})
                pip.assert_not_called()
