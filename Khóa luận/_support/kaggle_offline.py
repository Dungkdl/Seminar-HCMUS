"""Offline preparation/bootstrap. Importable using only the Python standard library."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

BASE_PACKAGES = ('numpy', 'pandas', 'Pillow', 'matplotlib', 'scipy', 'scikit-learn', 'torch', 'torchvision')
ADDONS = ('transformers==4.57.1', 'safetensors>=0.4')
TEXT_MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def inventory():
    packages = {}
    for name in BASE_PACKAGES + ('transformers', 'huggingface-hub', 'tokenizers', 'safetensors', 'setuptools'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'python_minor': '.'.join(map(str, sys.version_info[:2])),
            'system': platform.system(), 'machine': platform.machine(), 'packages': packages}


def probe(output=None):
    """Report package versions and real import failures, without installing anything."""
    report = inventory()
    checks = {}
    for name, code in {
        'scientific': 'import numpy,pandas,PIL,matplotlib,scipy,sklearn',
        'torchvision': ('import torch,torchvision; from torchvision.ops import nms; '
                        'nms(torch.tensor([[0.,0.,1.,1.]]),torch.tensor([1.]),0.5); '
                        'print(torch.__version__,torchvision.__version__,torch.cuda.is_available()); '
                        'print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'),
        'clinical': 'from transformers import AutoModel,AutoTokenizer,BertModel; import safetensors',
    }.items():
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, encoding='utf-8', errors='replace')
        checks[name] = {'ok': result.returncode == 0, 'details': (result.stdout + result.stderr)[-5000:]}
    report['imports'] = checks
    if output:
        write_json(output, report)
    return report


def run(arguments, cwd=None, env=None):
    subprocess.run([str(x) for x in arguments], cwd=cwd, env=env, check=True)


def build_project_wheel(project_root, wheelhouse, staging):
    """Pure Python project wheel is portable. Third-party binary wheels are a separate step."""
    project_root, wheelhouse, staging = map(Path, (project_root, wheelhouse, staging))
    if staging.exists() and any(staging.iterdir()):
        raise FileExistsError(f'Use an empty build staging directory: {staging}')
    staging.mkdir(parents=True, exist_ok=True)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project_root / 'pyproject.toml', staging / 'pyproject.toml')
    shutil.copytree(project_root / 'clindistill', staging / 'clindistill',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    run([sys.executable, '-m', 'pip', 'wheel', '--no-index', '--no-deps', '--no-build-isolation',
         '--wheel-dir', wheelhouse.resolve(), staging.resolve()])
    wheels = list(wheelhouse.glob('clindistill_code-*-py3-none-any.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected exactly one portable project wheel.')
    return wheels[0]


def make_mini_code(release_dir, image_root, destination, per_class=2, *, independent_groups=False):
    """Copy a class-balanced train/validation subset with full original bags. Never read test."""
    release_dir, image_root, destination = map(Path, (release_dir, image_root, destination))
    image_root = image_root.resolve()
    if per_class < 1:
        raise ValueError('per_class must be positive.')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Mini dataset already exists. Use it or choose a new destination.')
    release = json.loads((release_dir / 'release.json').read_text(encoding='utf-8'))
    prepared = destination / 'prepared'
    prepared.mkdir(parents=True)
    selected, image_paths, checksums = {}, set(), {}
    for split in ('train', 'validation'):
        records_path = release_dir / f'student_single_{split}.jsonl'
        contexts_path = release_dir / f'teacher_context_{split}.jsonl'
        for path in (records_path, contexts_path):
            if digest(path) != release['file_checksums'][path.name]:
                raise ValueError(f'Source release checksum mismatch: {path.name}')
        rows = [json.loads(line) for line in records_path.read_text(encoding='utf-8').splitlines() if line]
        selected[split] = []
        used_groups = set()
        for label in range(len(release['class_names'])):
            candidates = sorted((r for r in rows if r['photo_paths'] and r['single_label_index'] == label),
                                key=lambda r: r['case_id'])
            chosen = []
            for row in candidates:
                if independent_groups and row['split_group'] in used_groups:
                    continue
                chosen.append(row)
                used_groups.add(row['split_group'])
                if len(chosen) == per_class:
                    break
            if len(chosen) < per_class:
                raise ValueError(f'Not enough cases for {split}, class {label}.')
            selected[split].extend(chosen)
        ids = {r['case_id'] for r in selected[split]}
        contexts = [json.loads(line) for line in contexts_path.read_text(encoding='utf-8').splitlines() if line]
        contexts = [r for r in contexts if r['case_id'] in ids]
        if {r['case_id'] for r in contexts} != ids:
            raise ValueError('Missing clinical contexts in subset.')
        for kind, data in [('student_single', selected[split]), ('teacher_context', contexts)]:
            path = prepared / f'{kind}_{split}.jsonl'
            path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data), encoding='utf-8')
            checksums[path.name] = digest(path)
        for r in selected[split]:
            image_paths.update(r['photo_paths'] + r['radiograph_paths'])
    for key in ('case_id', 'patient_id', 'split_group', 'bag_signature'):
        if {r[key] for r in selected['train']} & {r[key] for r in selected['validation']}:
            raise ValueError(f'Mini split overlaps in {key}.')
    for relative in sorted(image_paths):
        source = (image_root / relative).resolve()
        target = (destination / relative).resolve()
        if image_root not in source.parents or destination.resolve() not in target.parents:
            raise ValueError('Unsafe image path.')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    identity = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()
    write_json(prepared / 'release.json', {'schema_version': 1, 'dataset': 'COde',
        'release_id': 'mini-debug-' + identity, 'stage': 'experimental', 'class_names': release['class_names'],
        'file_checksums': checksums, 'pending_reviews': ['Subset for software checks only; not research evaluation.'],
        'source_release_id': release['release_id'], 'debug_subset': True})
    summary = {'cases': {s: len(r) for s, r in selected.items()}, 'images': len(image_paths),
               'bytes': sum((destination / p).stat().st_size for p in image_paths), 'contains_test': False,
               'purpose': 'Small real-data rehearsal. Never report its metrics as thesis results.'}
    write_json(destination / 'mini_dataset.json', summary)
    return summary


def zip_directory(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Archive must be outside the source directory.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file() and not {'.cache', '__pycache__'} & set(path.parts):
                archive.write(path, path.relative_to(source).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity check failed.')
    return destination


def validate_platform(expected, actual):
    for key in ('system', 'machine', 'python_minor'):
        if expected[key].lower() != actual[key].lower():
            raise ValueError(f'Wheel target mismatch: {key}: prepared={expected[key]}, runtime={actual[key]}')
    # Do not replace the Kaggle scientific/CUDA stack. CUDA local suffix may differ.
    for name in BASE_PACKAGES:
        a, b = expected['packages'][name], actual['packages'][name]
        if name in ('torch', 'torchvision') and a and b:
            a, b = a.split('+')[0], b.split('+')[0]
        if a != b:
            raise ValueError(f'Base environment differs for {name}: prepared={a}, runtime={b}. '
                             'Prepare wheels on a matching Kaggle image; do not overwrite the CUDA stack.')


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'offline_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unknown offline bundle schema.')
    for relative, expected in manifest['checksums'].items():
        path = (bundle / relative).resolve()
        if bundle not in path.parents or digest(path) != expected:
            raise ValueError(f'Offline asset checksum mismatch: {relative}')
    for relative in ('requirements.txt', 'base_constraints.txt', 'models/resnet50-default.pth',
                     'models/biomedbert/config.json'):
        if relative not in manifest['checksums']:
            raise ValueError(f'Bundle incomplete: {relative}')
    return manifest


def activate_bundle(bundle, *, install=True):
    """Called before importing torch/transformers. Never contacts package indexes or model hubs."""
    bundle = Path(bundle).resolve()
    manifest = verify_bundle(bundle)
    validate_platform(manifest['environment'], inventory())
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      PIP_NO_INDEX='1', TOKENIZERS_PARALLELISM='false')
    wheelhouse = bundle / 'wheels'
    if install:
        run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', wheelhouse,
             '--constraint', bundle / 'base_constraints.txt', '--requirement', bundle / 'requirements.txt'])
        project_wheels = list(wheelhouse.glob('clindistill_code-*-py3-none-any.whl'))
        if len(project_wheels) != 1:
            raise ValueError('Missing/ambiguous project wheel.')
        run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--force-reinstall', project_wheels[0]])
    checked = probe()
    if not all(c['ok'] for c in checked['imports'].values()):
        raise RuntimeError('Offline imports failed: ' + json.dumps(checked['imports'], ensure_ascii=False))
    return {'root': bundle, 'resnet': str(bundle / 'models/resnet50-default.pth'),
            'text_encoder': str(bundle / 'models/biomedbert'), 'environment': checked}


def prepare_bundle(project_root, destination):
    """Run on ONLINE Linux Kaggle before opening the internet-disabled GPU session."""
    project_root, destination = Path(project_root).resolve(), Path(destination).resolve()
    environment = inventory()
    if environment['system'] != 'Linux':
        raise ValueError('Prepare third-party wheels in ONLINE Linux Kaggle, not Windows.')
    if any(environment['packages'][p] is None for p in BASE_PACKAGES):
        raise ValueError('Use the Kaggle image with its scientific/torch/torchvision base packages.')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Use a new empty bundle directory to avoid mixing package versions.')
    destination.mkdir(parents=True)
    wheelhouse = destination / 'wheels'
    wheelhouse.mkdir()
    (destination / 'requirements.txt').write_text('\n'.join(ADDONS) + '\n', encoding='utf-8')
    # Keep existing NumPy/scientific versions when resolving transformer dependencies.
    (destination / 'base_constraints.txt').write_text(
        '\n'.join(f'{p}=={v}' for p, v in environment['packages'].items() if p in BASE_PACKAGES) + '\n',
        encoding='utf-8')
    online_env = os.environ.copy()
    for key in ('PIP_NO_INDEX', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        online_env.pop(key, None)
    run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--dest', wheelhouse,
         '--constraint', destination / 'base_constraints.txt', '--requirement', destination / 'requirements.txt'],
        env=online_env)
    existing_wheels = list((project_root / '_support/dist/project_wheel').glob('clindistill_code-*-py3-none-any.whl'))
    if len(existing_wheels) == 1:
        with zipfile.ZipFile(existing_wheels[0]) as archive:
            for path in (project_root / 'clindistill').glob('*.py'):
                if archive.read('clindistill/' + path.name) != path.read_bytes():
                    raise ValueError('Prebuilt project wheel differs from source; rebuild the source bundle.')
        shutil.copy2(existing_wheels[0], wheelhouse / existing_wheels[0].name)
    else:
        build_project_wheel(project_root, wheelhouse, destination.parent / (destination.name + '_build'))
    # Install only from the downloaded wheelhouse; this also tests dependency closure.
    run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', wheelhouse,
         '--constraint', destination / 'base_constraints.txt', '--requirement', destination / 'requirements.txt'])
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ.pop(key, None)
    import torch
    from torchvision.models import ResNet50_Weights
    from huggingface_hub import snapshot_download
    models = destination / 'models'
    models.mkdir()
    url = ResNet50_Weights.DEFAULT.url
    prefix = Path(url).stem.rsplit('-', 1)[-1]
    torch.hub.download_url_to_file(url, str(models / 'resnet50-default.pth'), hash_prefix=prefix)
    snapshot_download(TEXT_MODEL, local_dir=models / 'biomedbert', allow_patterns=[
        'config.json', 'model.safetensors', 'pytorch_model.bin', 'vocab.txt', 'tokenizer*.json',
        'special_tokens_map.json', 'added_tokens.json'])
    source = destination / 'source'
    source.mkdir()
    for folder in ('clindistill', 'configs'):
        shutil.copytree(project_root / folder, source / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / '_support').mkdir()
    shutil.copy2(project_root / '_support/kaggle_offline.py', source / '_support/kaggle_offline.py')
    shutil.copy2(project_root / '_support/TECHNICAL_GUIDE.md', source / '_support/TECHNICAL_GUIDE.md')
    for path in list(project_root.glob('COde_0*.ipynb')) + [project_root / 'README.md', project_root / 'pyproject.toml']:
        shutil.copy2(path, source / path.name)
    checksums = {p.relative_to(destination).as_posix(): digest(p) for p in sorted(destination.rglob('*'))
                 if p.is_file() and '.cache' not in p.parts and '__pycache__' not in p.parts}
    write_json(destination / 'offline_manifest.json', {'schema_version': 1, 'environment': environment,
        'checksums': checksums, 'status': 'assets_prepared_smoke_pending', 'resnet_url': url, 'text_model': TEXT_MODEL})
    verify_bundle(destination)
    return destination


def smoke_bundle(bundle, mini_root, output):
    """Real ResNet/BiomedBERT smoke with sockets blocked; not a quality experiment."""
    from unittest.mock import patch
    assets = activate_bundle(bundle)
    mini_root, output = Path(mini_root), Path(output)
    sys.path.insert(0, str(Path(bundle) / 'source'))
    import torch
    from clindistill.clinical import prepare_clinical_cache
    from clindistill.config import TrainConfig
    from clindistill.suite import run_suite
    from clindistill.reporting import evaluate_suite
    torch.set_num_threads(2)
    with patch('socket.socket.connect', side_effect=RuntimeError('Network forbidden during offline smoke')), \
         patch('socket.create_connection', side_effect=RuntimeError('Network forbidden during offline smoke')):
        cache = output / 'clinical_cache'
        prepare_clinical_cache(mini_root / 'prepared', cache, encoder=assets['text_encoder'],
                               local_files_only=True, max_length=64, batch_size=8)
        config = TrainConfig(backbone='resnet50', pretrained=True, backbone_weights_path=assets['resnet'],
            freeze_backbone=True, image_size=64, max_photos=2, batch_size=2, epochs=1, patience=1,
            feature_dim=32, workers=0, debug_max_cases=12, bootstrap_repetitions=10)
        run_suite(mini_root / 'prepared', mini_root, output / 'runs', config, clinical_cache=cache,
                  selected=['photo_attention', 'kd_full', 'kd_selective'], seeds=[2026], resume=True)
        evaluate_suite(mini_root / 'prepared', mini_root, output / 'runs', clinical_cache=cache)
    report = {'status': 'passed', 'network_blocked_during_models_and_training': True,
              'full_pretrained_models': True, 'debug_budget': True, 'research_result': False,
              'environment': inventory()}
    write_json(output / 'offline_smoke.json', report)
    write_json(Path(bundle) / 'offline_smoke.json', report)
    manifest = verify_bundle(bundle)
    manifest['checksums']['offline_smoke.json'] = digest(Path(bundle) / 'offline_smoke.json')
    manifest['status'] = 'offline_smoke_passed'
    write_json(Path(bundle) / 'offline_manifest.json', manifest)
    return report


BOOTSTRAP_CELL = r'''
# Chạy trước khi import torch/transformers/pandas.
from pathlib import Path
import os, sys, importlib.util
OFFLINE_MODE = Path('/kaggle').exists()
OFFLINE_BUNDLE_OVERRIDE = None  # folder chứa offline_manifest.json
OFFLINE_ASSETS = None
if OFFLINE_MODE:
    candidates = []
    for base in [Path('/kaggle/input'), Path('/kaggle/working')]:
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base):
            here = Path(current)
            if 'offline_manifest.json' in files:
                candidates.append(here)
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in {'Images', '.cache', '__pycache__', 'runs', 'models', 'wheels'}]
            if len(here.relative_to(base).parts) >= 7:
                dirs[:] = []
    if OFFLINE_BUNDLE_OVERRIDE is not None:
        bundle = Path(OFFLINE_BUNDLE_OVERRIDE)
    elif len(candidates) == 1:
        bundle = candidates[0]
    else:
        raise RuntimeError(f'Attach offline bundle prepared by notebook 00; found: {candidates}. Set OFFLINE_BUNDLE_OVERRIDE.')
    helper = bundle / 'source' / '_support' / 'kaggle_offline.py'
    spec = importlib.util.spec_from_file_location('kaggle_offline', helper)
    offline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(offline)
    OFFLINE_ASSETS = offline.activate_bundle(bundle)
    sys.path.insert(0, str(bundle / 'source'))
    # The optional source/weights add-on is separate from the original dependency bundle.
    extension_candidates = []
    for base in [Path('/kaggle/input'), Path('/kaggle/working')]:
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base):
            here = Path(current)
            if 'backbone_extension.json' in files:
                extension_candidates.append(here)
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in {'Images', '.cache', '__pycache__', 'runs', 'models', 'wheels'}]
            if len(here.relative_to(base).parts) >= 8:
                dirs[:] = []
    extension_override = globals().get('BACKBONE_EXTENSION_OVERRIDE')
    if extension_override:
        extension_candidates = [Path(extension_override)]
    if len(extension_candidates) > 1:
        raise RuntimeError('Có nhiều backbone add-on; đặt BACKBONE_EXTENSION_OVERRIDE ở cell đầu.')
    if extension_candidates:
        extension_root = extension_candidates[0]
        extension_spec = importlib.util.spec_from_file_location('backbone_extension',
            extension_root / 'source/_support/backbone_extension.py')
        extension_setup = importlib.util.module_from_spec(extension_spec)
        extension_spec.loader.exec_module(extension_setup)
        OFFLINE_ASSETS = extension_setup.activate_extension(extension_root, OFFLINE_ASSETS)
    elif globals().get('REQUIRE_BACKBONE_EXTENSION', False):
        raise RuntimeError('Gắn dataset code-swin-addon có backbone_extension.json rồi Run All lại.')
    print('Offline setup OK. Local pretrained weights:', OFFLINE_ASSETS['resnet'])
else:
    print('Local/online mode: no offline installation. Kaggle offline uses notebook 00 bundle.')
'''
