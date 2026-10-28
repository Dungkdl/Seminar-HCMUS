"""Offline bank contracts using small synthetic files, never gated downloads."""
import importlib.util
from pathlib import Path
import socket
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('research_bank_contracts',ROOT/'_support/research_bank.py')
bank=importlib.util.module_from_spec(spec); spec.loader.exec_module(bank)


class ResearchBankTests(unittest.TestCase):
    def fixture(self,root):
        target=root/'source/example.py'; target.parent.mkdir(); target.write_text('value = 1\n')
        info={'schema_version':4,'base_environment':bank.inventory(),'files':{'source/example.py':bank.digest(target)}}
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
