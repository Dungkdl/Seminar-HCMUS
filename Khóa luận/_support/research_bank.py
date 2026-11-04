"""Standard-library entry point for the COde v4 offline research asset bank.

Run preparation on Linux/Kaggle with Internet. HF_TOKEN is read from the
environment only and is never copied into the bank or reports.
"""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import uuid
import zipfile
from contextlib import contextmanager

BASE_PACKAGES = ('numpy', 'pandas', 'Pillow', 'matplotlib', 'scipy', 'scikit-learn', 'torch', 'torchvision')
ADDONS = ('transformers==4.57.1', 'safetensors>=0.4', 'sentencepiece>=0.2')
BUILD_TOOLS = ('setuptools>=68', 'wheel>=0.43')
BANK_SCHEMA = 5
MODEL_IDS = {'student': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
             'teacher': 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'clinical': 'BAAI/bge-m3'}
CNN_URLS = {'resnet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
            'swin_b': 'https://download.pytorch.org/models/swin_b-68c6b09e.pth'}


@contextmanager
def block_network():
    """Rehearsal fails on any attempted outgoing connection, even with Internet on."""
    from unittest.mock import patch
    def denied(*args, **kwargs):
        raise RuntimeError('Network access is blocked during the offline rehearsal.')
    with patch('socket.socket.connect', denied), patch('socket.create_connection', denied), patch('socket.getaddrinfo', denied):
        yield


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def inventory():
    return {'python_minor': '.'.join(map(str, sys.version_info[:2])), 'system': platform.system(),
            'machine': platform.machine(), 'packages': {x: importlib.metadata.version(x) for x in BASE_PACKAGES}}


def require_environment(expected, actual):
    """CPU online preparation must match the GPU image; never silently replace CUDA."""
    differences = []
    for key in ('python_minor', 'system', 'machine'):
        if expected.get(key) != actual.get(key):
            differences.append(f'{key}: target={expected.get(key)!r}, current={actual.get(key)!r}')
    for key in sorted(set(expected.get('packages', {})) | set(actual.get('packages', {}))):
        wanted, current = expected.get('packages', {}).get(key), actual.get('packages', {}).get(key)
        if wanted != current:
            differences.append(f'{key}: target={wanted!r}, current={current!r}')
    if differences:
        raise RuntimeError('Python/platform/base scientific stack differs from the target GPU image. '
                           'Prepare on a matching online image; do not use this bank on a different stack.\n'
                           + '\n'.join(differences))


def require_preparation_environment(target, actual):
    """Allow only CPU -> CUDA wheel variants of the same Torch/vision releases."""
    import re
    checked = {**actual, 'packages': dict(actual.get('packages', {}))}
    variants = []
    for key in ('torch', 'torchvision'):
        wanted = target.get('packages', {}).get(key, '')
        current = actual.get('packages', {}).get(key, '')
        wanted_release, _, wanted_build = wanted.partition('+')
        current_release, _, current_build = current.partition('+')
        variants.append((key, wanted_release == current_release, wanted_build, current_build))
    bridge = (all(same and cpu == 'cpu' and re.fullmatch(r'cu\d+', cuda)
                  for _, same, cuda, cpu in variants)
              and variants[0][2] == variants[1][2])
    if bridge:
        for key, *_ in variants:
            checked['packages'][key] = target['packages'][key]
    require_environment(target, checked)
    return bool(bridge)


def check_online(opener=None):
    """Public, unauthenticated request: network failure is not a rejected HF token."""
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    request = Request('https://huggingface.co', method='HEAD')
    try:
        with (opener or urlopen)(request, timeout=10):
            pass
    except HTTPError as exc:
        status = exc.code
        exc.close()
        raise RuntimeError(f'Online setup: Hugging Face returned HTTP {status} to an unauthenticated request. '
                           'HF_TOKEN has not been read or checked yet. Retry when the service is reachable.') from None
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(f'Online setup cannot reach huggingface.co ({type(exc).__name__}). '
                           'HF_TOKEN has not been read or checked yet. If Kaggle blocks Internet with RTX PRO 6000, '
                           'stop the session, select Accelerator=None (CPU), enable Internet, and restart notebook00. '
                           'Use Pro 6000 offline for notebooks01/02 after saving the bank.') from None


def read_hf_token(secret_reader=None):
    """Read a secret without logging its value or an exception that may contain it."""
    token = os.environ.get('HF_TOKEN')
    if not token or not token.strip():
        try:
            if secret_reader is None:
                from kaggle_secrets import UserSecretsClient
                secret_reader = UserSecretsClient().get_secret
            token = secret_reader('HF_TOKEN')
        except Exception as exc:
            kind = type(exc).__name__
            if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
                hint = 'Kaggle Secrets service is unreachable. Check Internet/session connectivity; this does not mean the HF token is invalid.'
            elif kind == 'CredentialError':
                hint = 'Kaggle denied access to Secrets. Grant this notebook access to HF_TOKEN and restart the session; HF has not validated the token.'
            else:
                hint = 'Cannot retrieve Kaggle Secret HF_TOKEN. Check its exact name and notebook access; HF has not validated the token.'
            raise RuntimeError(f'{hint} Error type: {kind}.') from None
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError('HF_TOKEN is empty. Set a nonempty Kaggle Secret and grant notebook access.')
    os.environ['HF_TOKEN'] = token.strip()


def source_files(root):
    root = Path(root)
    files = [root / 'pyproject.toml', root / 'README.md'] + sorted(root.glob('COde_0*.ipynb'))
    files += sorted((root / 'clindistill').rglob('*.py')) + sorted((root / 'configs/code').glob('*.json'))
    files += [root / '_support' / name for name in ('research_bank.py', 'image_layout.py', 'kaggle_offline.py',
              'backbone_extension.py', 'training_runtime.py', 'recover_amp_bundle.py', 'TECHNICAL_GUIDE.md',
              'KAGGLE_RUNBOOK.md', 'VERIFICATION.md')]
    return [p for p in files if p.is_file()]


def verify_project_wheel(wheel, source):
    """Every packaged Python module must match the reviewed source, including subpackages."""
    expected = {p.relative_to(source).as_posix(): p.read_bytes()
                for p in (Path(source) / 'clindistill').rglob('*.py')}
    with zipfile.ZipFile(wheel) as archive:
        actual = {name: archive.read(name) for name in archive.namelist()
                  if name.startswith('clindistill/') and name.endswith('.py')}
    if actual != expected:
        raise ValueError('Project wheel differs from reviewed source; rebuild the package.')
    return {name: digest(Path(source) / name) for name in sorted(expected)}


def installed_project_matches(files):
    """Verify the installed wheel; importing source must not silently bypass it."""
    spec = importlib.util.find_spec('clindistill')
    if spec is None or not spec.origin:
        raise RuntimeError('Project wheel is not installed.')
    root = Path(spec.origin).resolve().parent.parent
    actual = {p.relative_to(root).as_posix(): digest(p) for p in (root / 'clindistill').rglob('*.py')}
    if actual != files:
        raise ValueError('Installed project differs from the bank wheel.')
    return root / 'clindistill'


def prepare(source, destination, *, target_environment=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if platform.system() != 'Linux':
        raise RuntimeError('Prepare binary wheels on Linux/Kaggle, matching the target Python environment.')
    preparation = inventory()
    base = target_environment if target_environment is not None else preparation
    cpu_bridge = require_preparation_environment(base, preparation)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'research_bank_manifest.json'
    if manifest_path.exists():
        existing = verify(destination, check_environment=False)
        require_environment(base, existing['base_environment'])
        expected_source = {p.relative_to(source).as_posix(): digest(p) for p in source_files(source)}
        saved = {p.removeprefix('source/'): h for p, h in read_json(manifest_path)['files'].items()
                 if p.startswith('source/')}
        if expected_source != saved:
            raise ValueError('Prepared bank belongs to different source. Choose a new bank directory.')
        return read_json(manifest_path)
    constraints = destination / 'base-constraints.txt'
    constraints.write_text(''.join(f'{k}=={v}\n' for k, v in base['packages'].items()), encoding='utf-8')
    wheelhouse = destination / 'wheels'; wheelhouse.mkdir(exist_ok=True)
    online_env = os.environ.copy()
    for key in ('PIP_NO_INDEX', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        online_env.pop(key, None)
    run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--constraint', constraints,
         '--dest', wheelhouse, *ADDONS, *BUILD_TOOLS], env=online_env)
    # Target Torch/CUDA is supplied by Kaggle; addons must not replace the CPU preparer's
    # Torch or the GPU runtime. No compiled Torch extension is supported by this bridge.
    forbidden = [p.name for p in wheelhouse.glob('*.whl')
                 if p.name.split('-')[0].replace('_', '-').lower() in {'torch', 'torchvision', 'torchaudio', 'triton'}
                 or p.name.lower().startswith(('nvidia-', 'nvidia_'))]
    if forbidden:
        raise ValueError('Research addon wheelhouse must not contain Torch/CUDA wheels or extensions: ' + ', '.join(forbidden))
    # Freeze the complete downloaded dependency closure, not just the three top-level addons.
    locked = sorted({f'{p.name.split("-")[0].replace("_", "-")}=={p.name.split("-")[1]}'
                     for p in wheelhouse.glob('*.whl') if not p.name.startswith('clindistill_code-')})
    lock = destination / 'requirements.lock'
    lock.write_text('\n'.join(locked) + '\n', encoding='utf-8')
    run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', wheelhouse,
         '--constraint', constraints, '--requirement', lock])
    payload = destination / 'source'
    for path in source_files(source):
        target = payload / path.relative_to(source); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(path):
            raise ValueError('Partially built bank contains different source. Choose a fresh bank name.')
        shutil.copy2(path, target)
    # A fresh build staging folder avoids stale egg-info and duplicate project versions.
    staging = destination / ('_build_' + uuid.uuid4().hex[:8]); staging.mkdir()
    shutil.copy2(payload / 'pyproject.toml', staging / 'pyproject.toml')
    shutil.copytree(payload / 'clindistill', staging / 'clindistill')
    run([sys.executable, '-m', 'pip', 'wheel', '--no-index', '--no-deps', '--no-build-isolation',
         '--wheel-dir', wheelhouse, staging])
    project = list(wheelhouse.glob('clindistill_code-*-py3-none-any.whl'))
    if len(project) != 1:
        raise ValueError('Expected exactly one fresh project wheel.')
    project_files = verify_project_wheel(project[0], payload)
    run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--force-reinstall', project[0]])
    # Do not download gated weights until explicit access has been granted by the model owner.
    from huggingface_hub import HfApi, snapshot_download
    token = os.environ.get('HF_TOKEN') or None
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ.pop(key, None)
    api = HfApi(token=token); revisions, models = {}, {}
    revision_path = destination / 'model_revisions.json'
    pinned = read_json(revision_path) if revision_path.exists() else {}
    for role, repo in MODEL_IDS.items():
        if role in pinned and pinned[role]['repo'] != repo:
            raise ValueError('Bank model ID changed. Choose a new bank directory.')
        revision = pinned[role]['revision'] if role in pinned else api.model_info(repo).sha
        pinned[role] = {'repo': repo, 'revision': revision}
        write_json(revision_path, pinned)
        target = destination / 'models' / role
        try:
            snapshot_download(repo, revision=revision, token=token, local_dir=target,
                allow_patterns=['*.json', '*.safetensors', 'pytorch_model*.bin', '*.txt', '*.model'],
                ignore_patterns=['onnx/*', 'onnx/**', '1_Pooling/*', 'imgs/*'])
        except Exception as exc:
            # Avoid echoing an authenticated request or token in a notebook traceback.
            raise RuntimeError(f'Cannot download {repo}. Accept its access terms and set private HF_TOKEN; '
                               f'original error type: {type(exc).__name__}.') from None
        revisions[role] = {'repo': repo, 'revision': revision}
        models[role] = target.relative_to(destination).as_posix()
    import torch
    for role, url in CNN_URLS.items():
        target = destination / 'models' / Path(url).name; target.parent.mkdir(exist_ok=True)
        if not target.exists():
            torch.hub.download_url_to_file(url, target, hash_prefix=Path(url).stem.split('-')[-1])
        if not digest(target).startswith(Path(url).stem.split('-')[-1]):
            raise ValueError('Official CNN checkpoint checksum mismatch.')
        models[role] = target.relative_to(destination).as_posix()
    # Instantiate each backend locally before marking the bank complete (no network fallback).
    from transformers import AutoModel, AutoTokenizer
    for role in MODEL_IDS:
        folder = destination / models[role]
        encoder = AutoModel.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        expected = {'student': ('dinov3_vit', 768), 'teacher': ('dinov3_vit', 1024), 'clinical': ('xlm-roberta', 1024)}[role]
        if (encoder.config.model_type, encoder.config.hidden_size) != expected:
            raise ValueError(f'Unexpected {role} architecture.')
        if role == 'clinical':
            AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True)
        del encoder
    require_preparation_environment(base, inventory())
    files = {p.relative_to(destination).as_posix(): digest(p)
             for folder in ('source', 'models', 'wheels') for p in sorted((destination / folder).rglob('*'))
             if p.is_file() and not {'.cache', '__pycache__'} & set(p.relative_to(destination).parts)}
    files['base-constraints.txt'] = digest(constraints)
    files['model_revisions.json'] = digest(revision_path)
    files['requirements.lock'] = digest(lock)
    manifest = {'schema_version': BANK_SCHEMA, 'status': 'assets_verified_backends_loaded', 'base_environment': base,
                'preparation_environment': preparation, 'cpu_to_cuda_preparation': cpu_bridge,
                'backend_validation': 'Local CPU model loading only; GPU execution and full experiments remain unverified.',
                'models': models, 'model_revisions': revisions, 'files': files,
                'project_wheel': project[0].relative_to(destination).as_posix(), 'project_files': project_files,
                'source_files': sorted(p for p in files if p.startswith('source/')),
                'note': 'Backend loading passed; run the small end-to-end research rehearsal before a full study.'}
    write_json(manifest_path, manifest)
    print('BANK READY:', destination, flush=True)
    return manifest


def verify(root, *, check_environment=True):
    root = Path(root).resolve(); info = read_json(root / 'research_bank_manifest.json')
    if info.get('schema_version') != BANK_SCHEMA:
        raise ValueError('Expected research bank v5; prepare a new bank instead of mixing old source/wheels.')
    if check_environment:
        require_environment(info['base_environment'], inventory())
    print('Verifying research bank checksums...', flush=True)
    for name, expected in info['files'].items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Research bank integrity failure: {name}')
    return info


def activate(root):
    root = Path(root).resolve()
    if 'clindistill' in sys.modules:
        raise RuntimeError('Restart the kernel and run setup before importing clindistill.')
    info = verify(root)
    wheels = root / 'wheels'
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PIP_NO_INDEX='1',
                      HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    # Installing a different Transformers after importing it leaves stale Python classes.
    if any(x in sys.modules for x in ('transformers', 'tokenizers')):
        raise RuntimeError('Restart the kernel; activate research wheels before Transformers imports.')
    run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', wheels,
         '--constraint', root / 'base-constraints.txt', '--requirement', root / 'requirements.lock'])
    project = list(wheels.glob('clindistill_code-*-py3-none-any.whl'))
    if len(project) != 1:
        raise ValueError('Expected exactly one research project wheel.')
    run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--force-reinstall', project[0]])
    verify_project_wheel(project[0], root / 'source')
    # Remove source checkouts from import search paths; configs/docs still live in source_root.
    installed_roots = {Path(sysconfig.get_path(key)).resolve() for key in ('purelib', 'platlib')}
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() in installed_roots
                  or not (Path(p or os.getcwd()).resolve() / 'clindistill').is_dir()]
    installed = installed_project_matches(info['project_files'])
    print('Installed project wheel:', installed, flush=True)
    return {'root': root, 'source_root': root / 'source',
            'models': {k: root / v for k, v in info['models'].items()}, 'manifest': info,
            'installed_project': installed}


BOOTSTRAP_CELL = '''from pathlib import Path
import os, sys, json, importlib.util
RESEARCH_BANK_OVERRIDE = globals().get('RESEARCH_BANK_OVERRIDE')
RESEARCH_ASSETS = None

def research_find(predicate, *, include_runs=False):
    found = set()
    for base in (Path.cwd(), Path.cwd().parent / 'Data', Path('/kaggle/input')):
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base):
            here = Path(current)
            if predicate(here, dirs, files):
                found.add(here.resolve())
            excluded = {'Images', 'images', '.git', '.cache', '__pycache__', '_archive', '_support',
                        'models', 'wheels', 'wheelhouse', 'official_code', '_chunks', 'student_export'}
            dirs[:] = [d for d in dirs if not d.lower().startswith('qwen')]
            if not include_runs:
                excluded.add('runs')
            dirs[:] = [d for d in dirs if d not in excluded]
            if len(here.relative_to(base).parts) >= 8:
                dirs[:] = []
    return sorted(found, key=str)

def research_one(override, found, name):
    if override is not None:
        return Path(override).resolve()
    if len(found) != 1:
        raise RuntimeError(f'{name}: found {len(found)} folders: {found}. Set the matching *_OVERRIDE.')
    return found[0]

if Path('/kaggle').exists():
    bank_root = research_one(RESEARCH_BANK_OVERRIDE,
        research_find(lambda p, d, f: 'research_bank_manifest.json' in f), 'Research bank')
    helper = bank_root / 'source/_support/research_bank.py'
    spec = importlib.util.spec_from_file_location('research_bank_setup', helper)
    bank = importlib.util.module_from_spec(spec); spec.loader.exec_module(bank)
    RESEARCH_ASSETS = bank.activate(bank_root)
    PROJECT_ROOT = RESEARCH_ASSETS['source_root']
    # Compatibility for existing data-audit publishing cells only.
    OFFLINE_ASSETS = {'root': bank_root, 'source_root': PROJECT_ROOT}
else:
    PROJECT_ROOT = research_one(globals().get('PROJECT_ROOT_OVERRIDE'),
        research_find(lambda p, d, f: (p / 'clindistill/research').is_dir() and 'configs' in d), 'Research source')
    sys.path.insert(0, str(PROJECT_ROOT))
print('Research source:', PROJECT_ROOT)
'''
