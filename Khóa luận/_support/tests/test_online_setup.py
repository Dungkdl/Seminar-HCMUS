"""Network/Secrets failures must not be misreported as invalid HF credentials."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import types
import unittest
import zipfile
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('online_bank', ROOT / '_support/research_bank.py')
bank = importlib.util.module_from_spec(spec); spec.loader.exec_module(bank)


class OnlineSetupTests(unittest.TestCase):
    def test_connectivity_probe_has_no_credentials_and_distinguishes_http_failure(self):
        response = Mock(); response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
        opener = Mock(return_value=response)
        bank.check_online(opener)
        request = opener.call_args.args[0]
        self.assertEqual(request.get_method(), 'HEAD')
        self.assertFalse(request.has_header('Authorization'))
        self.assertEqual(opener.call_args.kwargs['timeout'], 10)
        for error in [URLError('network unavailable'), TimeoutError('timeout')]:
            with self.assertRaisesRegex(RuntimeError, 'HF_TOKEN has not been read or checked'):
                bank.check_online(Mock(side_effect=error))
        with self.assertRaisesRegex(RuntimeError, 'HTTP 503'):
            bank.check_online(Mock(side_effect=HTTPError(request.full_url, 503, 'unavailable', {}, None)))

    def test_secret_network_error_never_exposes_value_or_blames_hf_authentication(self):
        hidden = 'hf_synthetic_private_value_do_not_log'
        with patch.dict(os.environ, {'HF_TOKEN': ''}):
            try:
                bank.read_hf_token(Mock(side_effect=ConnectionError(hidden)))
            except RuntimeError as exc:
                message = ''.join(traceback.format_exception(exc))
                self.assertIn('Secrets service is unreachable', message)
                self.assertIn('ConnectionError', message)
                self.assertNotIn(hidden, message)
                self.assertTrue(exc.__suppress_context__)
            else:
                self.fail('Expected a redacted Secrets connection error')

    def test_secret_access_and_empty_values_have_distinct_messages(self):
        CredentialError = type('CredentialError', (Exception,), {})
        with patch.dict(os.environ, {'HF_TOKEN': ''}):
            with self.assertRaisesRegex(RuntimeError, 'Kaggle denied access'):
                bank.read_hf_token(Mock(side_effect=CredentialError('private detail')))
            with self.assertRaisesRegex(RuntimeError, 'empty'):
                bank.read_hf_token(lambda name: '   ')

    def test_existing_and_kaggle_tokens_are_read_without_output(self):
        with patch.dict(os.environ, {'HF_TOKEN': ' hf_synthetic ' + ' '}), contextlib.redirect_stdout(io.StringIO()) as output:
            reader = Mock()
            bank.read_hf_token(reader)
            reader.assert_not_called()
            self.assertEqual(os.environ['HF_TOKEN'], 'hf_synthetic')
            self.assertEqual(output.getvalue(), '')
        with patch.dict(os.environ, {'HF_TOKEN': ''}):
            reader = Mock(return_value='hf_synthetic')
            bank.read_hf_token(reader)
            reader.assert_called_once_with('HF_TOKEN')

    def test_target_environment_checks_cuda_build_and_python_without_overriding_them(self):
        expected = bank.inventory()
        bank.require_environment(expected, expected)
        current = {**expected, 'packages': {**expected['packages'], 'torch': 'different+cu130'}}
        with self.assertRaisesRegex(RuntimeError, 'torch: target='):
            bank.require_environment(expected, current)
        with self.assertRaisesRegex(RuntimeError, 'python_minor: target='):
            bank.require_environment(expected, {**expected, 'python_minor': 'wrong'})

    def test_cpu_preparation_bridge_allows_only_matching_release_and_other_packages(self):
        actual = bank.inventory()
        actual = {**actual, 'packages': {**actual['packages'], 'torch': '2.11.0+cpu', 'torchvision': '0.26.0+cpu'}}
        target = {**actual, 'packages': {**actual['packages'], 'torch': '2.11.0+cu130', 'torchvision': '0.26.0+cu130'}}
        self.assertTrue(bank.require_preparation_environment(target, actual))
        self.assertEqual(actual['packages']['torch'], '2.11.0+cpu')
        for change in [{'torch': '2.10.0+cpu'}, {'numpy': 'wrong'}, {'torchvision': '0.26.0+cu128'}, {'torch': '2.11.0+rocm7.2'}]:
            with self.assertRaisesRegex(RuntimeError, 'stack differs'):
                bank.require_preparation_environment(target, {**actual, 'packages': {**actual['packages'], **change}})
        with self.assertRaisesRegex(RuntimeError, 'stack differs'):
            bank.require_preparation_environment(target, {**actual, 'python_minor': 'wrong'})
        with self.assertRaisesRegex(RuntimeError, 'stack differs'):
            bank.require_environment(target, actual)  # GPU activation remains exact.

    def test_cpu_prepared_bank_records_target_and_still_rejects_cpu_activation(self):
        actual = bank.inventory()
        actual = {**actual, 'system': 'Linux', 'packages': {**actual['packages'], 'torch': '2.11.0+cpu', 'torchvision': '0.26.0+cpu'}}
        target = {**actual, 'packages': {**actual['packages'], 'torch': '2.11.0+cu130', 'torchvision': '0.26.0+cu130'}}
        include_torch = False
        def fake_run(args, **kwargs):
            args = list(map(str, args))
            if 'download' in args:
                dest = Path(args[args.index('--dest') + 1])
                (dest/'synthetic_addon-1.0-py3-none-any.whl').write_bytes(b'fixture only')
                if include_torch:
                    (dest/'torch-2.11.0-py3-none-any.whl').write_bytes(b'forbidden target runtime')
            elif 'wheel' in args:
                dest = Path(args[args.index('--wheel-dir') + 1])
                staging = Path(args[-1])
                with zipfile.ZipFile(dest/'clindistill_code-0.5.1-py3-none-any.whl', 'w') as z:
                    for p in (staging/'clindistill').rglob('*.py'):
                        z.write(p, p.relative_to(staging).as_posix())
            elif 'install' not in args:
                self.fail('Unexpected subprocess')
        def snapshot(repo, **kwargs):
            folder = Path(kwargs['local_dir']); folder.mkdir(parents=True)
            (folder/'config.json').write_text('{}')
        def encoder(folder, **kwargs):
            kind = Path(folder).name
            model_type, width = {'student': ('dinov3_vit', 768), 'teacher': ('dinov3_vit', 1024),
                                 'clinical': ('xlm-roberta', 1024)}[kind]
            return types.SimpleNamespace(config=types.SimpleNamespace(model_type=model_type, hidden_size=width))
        hf = types.SimpleNamespace(HfApi=lambda **k: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(sha='synthetic')),
                                   snapshot_download=snapshot)
        transformers = types.SimpleNamespace(AutoModel=types.SimpleNamespace(from_pretrained=encoder),
                                              AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: None))
        with tempfile.TemporaryDirectory() as d, patch.object(bank.platform, 'system', return_value='Linux'), \
             patch.object(bank, 'inventory', return_value=actual), patch.object(bank, 'run', side_effect=fake_run), \
             patch.object(bank, 'CNN_URLS', {}), patch.dict(sys.modules, {'huggingface_hub': hf, 'transformers': transformers,
                                                                        'torch': types.SimpleNamespace()}), \
             contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)/'bank'
            info = bank.prepare(ROOT, root, target_environment=target)
            self.assertEqual(info['base_environment'], target)
            self.assertEqual(info['preparation_environment'], actual)
            self.assertTrue(info['cpu_to_cuda_preparation'])
            self.assertIn('CPU model loading only', info['backend_validation'])
            self.assertIn('torch==2.11.0+cu130', (root/'base-constraints.txt').read_text())
            self.assertEqual(bank.prepare(ROOT, root, target_environment=target), info)
            with self.assertRaisesRegex(RuntimeError, 'stack differs'):
                bank.verify(root)
            with patch.object(bank, 'inventory', return_value=target):
                self.assertEqual(bank.verify(root)['base_environment'], target)
            include_torch = True
            with self.assertRaisesRegex(ValueError, 'must not contain Torch/CUDA'):
                bank.prepare(ROOT, Path(d)/'forbidden', target_environment=target)

    def test_replacement_notebook_supports_old_addon_and_inspect_skips_network_and_secrets(self):
        doc = json.loads((ROOT / 'COde_00_Kaggle_Offline_Setup.ipynb').read_text(encoding='utf-8'))
        source = ''.join(doc['cells'][1]['source'])
        node = ast.parse(source)
        helpers = [n for n in node.body if isinstance(n, ast.FunctionDef)
                   and n.name in {'require_environment', 'require_preparation_environment', 'check_online', 'read_hf_token'}]
        self.assertEqual(len(helpers), 4)
        tail = next(n for n in node.body if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "MODE == 'prepare'")
        code = compile(ast.Module(body=helpers + [tail], type_ignores=[]), '<online notebook>', 'exec')
        # Older addon has inventory/prepare, but none of the new helpers.
        old = types.SimpleNamespace(inventory=bank.inventory, prepare=Mock())
        namespace = {'os': os, 'MODE': 'inspect', 'bank': old}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)
        old.prepare.assert_not_called()
        # A mismatched CPU stack must stop before Secrets or any downloads.
        namespace.update(MODE='prepare', TARGET_ENVIRONMENT={'python_minor': 'wrong'}, source='source', BANK_ROOT='bank')
        with self.assertRaisesRegex(RuntimeError, 'stack differs'):
            exec(code, namespace)
        old.prepare.assert_not_called()
        namespace.update(TARGET_ENVIRONMENT=bank.inventory(),
                         check_online=Mock(side_effect=RuntimeError('network blocked')),
                         read_hf_token=Mock())
        tail_code = compile(ast.Module(body=[tail], type_ignores=[]), '<prepare tail>', 'exec')
        with self.assertRaisesRegex(RuntimeError, 'network blocked'):
            exec(tail_code, namespace)
        namespace['read_hf_token'].assert_not_called()
        old.prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
