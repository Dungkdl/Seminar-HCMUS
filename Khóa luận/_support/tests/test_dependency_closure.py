"""Offline resolution must protect Torch, fsspec and the actually used dependency closure."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dependency_bank', ROOT/'_support/research_bank.py')
bank = importlib.util.module_from_spec(spec); spec.loader.exec_module(bank)


class DependencyClosureTests(unittest.TestCase):
    def distributions(self):
        versions = {x: '1.0' for x in bank.BASE_PACKAGES}
        versions.update(transformers='4.57.1', **{'huggingface-hub':'0.36.2', 'safetensors':'0.5',
            'sentencepiece':'0.2', 'setuptools':'81.0.0', 'wheel':'0.45', 'torch':'2.11.0', 'torchvision':'0.26.0'})
        rows = {name: SimpleNamespace(metadata={'Name': name}, version=version, requires=[])
                for name, version in versions.items()}
        rows['torch'].requires = ['setuptools<82', 'fsspec>=0.8.5']
        rows['transformers'].requires = ['huggingface-hub>=0.34,<1', 'not-installed; extra == "unused"']
        rows['gradio'] = SimpleNamespace(metadata={'Name':'gradio'}, version='9.0', requires=['huggingface-hub>=1'])
        return rows

    def test_runtime_conflicts_fail_but_unrelated_image_conflicts_are_reported(self):
        rows = self.distributions()
        with patch.object(bank.importlib.metadata, 'distributions', return_value=list(rows.values())):
            report = bank.validate_dependencies()
            self.assertFalse(report['runtime_conflicts'])
            self.assertEqual(report['unrelated_image_conflicts'][0]['package'], 'gradio')
            rows['setuptools'].version = '84.2.1'
            with self.assertRaisesRegex(RuntimeError, 'setuptools<82'):
                bank.validate_dependencies()

    def test_missing_transitive_dependency_is_fatal_and_extra_is_not_required(self):
        rows = self.distributions()
        rows['fsspec'].requires = ['required-transitive>=2']
        with patch.object(bank.importlib.metadata, 'distributions', return_value=list(rows.values())):
            report = bank.dependency_report()
            self.assertEqual(report['runtime_conflicts'][0]['requires'], 'required-transitive>=2')
            self.assertNotIn('not-installed', report['used_packages'])

    def test_runtime_pins_are_compatible_with_registered_torch(self):
        from packaging.requirements import Requirement
        tools = {Requirement(x).name: Requirement(x) for x in bank.BUILD_TOOLS}
        self.assertIn('81.0.0', tools['setuptools'].specifier)
        self.assertNotIn('84.2.1', tools['setuptools'].specifier)
        self.assertIn('fsspec', bank.BASE_PACKAGES)
        self.assertIn('huggingface-hub==0.36.2', bank.ADDONS)

    def test_model_reuse_preserves_cache_fingerprints_and_rejects_tampering(self):
        from clindistill.research.data import model_assets_hash
        from clindistill.research.training import verify_feature_assets
        with tempfile.TemporaryDirectory() as d, patch.object(bank, 'CNN_URLS', {}):
            root = Path(d); old = root/'old'; models = {}; files = {}; revisions = {}
            for role, repo in bank.MODEL_IDS.items():
                folder = old/'models'/role; folder.mkdir(parents=True)
                asset = folder/'config.json'; asset.write_text('{"fixture":true}')
                files[asset.relative_to(old).as_posix()] = bank.digest(asset)
                models[role] = folder.relative_to(old).as_posix()
                revisions[role] = {'repo':repo,'revision':'synthetic-pinned-revision'}
            bank.write_json(old/'research_bank_manifest.json', {'schema_version':5,
                'models':models,'files':files,'model_revisions':revisions})
            fingerprints = {r:model_assets_hash(old/p) for r,p in models.items()}
            copied = root/'new'
            self.assertEqual(bank.reuse_model_assets(old,copied),revisions)
            store = SimpleNamespace(metadata={'request':{'model_fingerprints':fingerprints}})
            verify_feature_assets(store,{r:copied/p for r,p in models.items()})
            (copied/'models/student/config.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'assets differ'):
                verify_feature_assets(store,{r:copied/p for r,p in models.items()})
            (old/'models/student/config.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'checksum changed'):
                bank.reuse_model_assets(old,root/'rejected')
