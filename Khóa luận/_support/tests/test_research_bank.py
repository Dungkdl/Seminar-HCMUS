"""Offline bank contracts using small synthetic files, never gated downloads."""
import importlib.util
from pathlib import Path
import socket
import tempfile
import unittest
import json
import os
import shutil
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('research_bank_contracts',ROOT/'_support/research_bank.py')
bank=importlib.util.module_from_spec(spec); spec.loader.exec_module(bank)


class ResearchBankTests(unittest.TestCase):
    def fixture(self,root):
        target=root/'source/example.py'; target.parent.mkdir(); target.write_text('value = 1\n')
        info={'schema_version':bank.BANK_SCHEMA,'base_environment':bank.inventory(),'files':{'source/example.py':bank.digest(target)}}
        bank.write_json(root/'research_bank_manifest.json',info)
        return target,info

    def test_changed_source_and_changed_platform_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); target,info=self.fixture(root)
            bank.verify(root)
            target.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError,'integrity'):
                bank.verify(root)
            info['base_environment']['system']='WRONG_PLATFORM'
            bank.write_json(root/'research_bank_manifest.json',info)
            with self.assertRaisesRegex(RuntimeError,'stack differs'):
                bank.verify(root)

    def test_offline_rehearsal_blocks_outgoing_connections(self):
        with bank.block_network(), self.assertRaisesRegex(RuntimeError,'blocked'):
            socket.create_connection(('example.invalid',443))

    def test_portable_payload_has_research_source_and_no_data_or_weights(self):
        files=bank.source_files(ROOT)
        self.assertIn(ROOT/'clindistill/research/training.py',files)
        self.assertIn(ROOT/'configs/code/research.json',files)
        for path in files:
            self.assertNotIn(path.suffix,{'.npz','.pth','.pt','.csv','.safetensors'})

    def test_installed_wheel_is_used_even_when_working_directory_contains_source(self):
        """Reproduce the source-shadowing bug in an actual isolated pip installation."""
        spec=importlib.util.spec_from_file_location('bank_test_builder',ROOT/'_support/kaggle_offline.py')
        builder=importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); asset=root/'bank'; source=asset/'source'; source.mkdir(parents=True)
            for path in bank.source_files(ROOT):
                target=source/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
            wheel=builder.build_project_wheel(ROOT,asset/'wheels',root/'build')
            package_files=bank.verify_project_wheel(wheel,source)
            (asset/'requirements.lock').write_text('',encoding='utf-8')
            (asset/'base-constraints.txt').write_text('',encoding='utf-8')
            files={p.relative_to(asset).as_posix():bank.digest(p) for p in asset.rglob('*') if p.is_file()}
            bank.write_json(asset/'research_bank_manifest.json',{'schema_version':bank.BANK_SCHEMA,
                'base_environment':bank.inventory(),'files':files,'models':{},'project_files':package_files})
            environment=root/'venv'; venv.EnvBuilder(system_site_packages=True,with_pip=True).create(environment)
            python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            script=("import importlib.util,json,pathlib; "
                f"spec=importlib.util.spec_from_file_location('bank',{str(source/'_support/research_bank.py')!r}); "
                "bank=importlib.util.module_from_spec(spec); spec.loader.exec_module(bank); "
                f"info=bank.activate({str(asset)!r}); import clindistill; "
                f"assert pathlib.Path({str(environment)!r}).resolve() in pathlib.Path(clindistill.__file__).resolve().parents; "
                "assert clindistill.__version__=='0.5.0'; print('WHEEL_IMPORT_VERIFIED')")
            env=os.environ.copy(); env.pop('PYTHONPATH',None)
            result=subprocess.run([str(python),'-c',script],cwd=source,env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertIn('WHEEL_IMPORT_VERIFIED',result.stdout)
            target=source/'clindistill/research/losses.py'; target.write_text('# changed\n')
            with self.assertRaisesRegex(ValueError,'differs'):
                bank.verify_project_wheel(wheel,source)
