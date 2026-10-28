"""Portable source/wheel + official Swin-B weights; reuses the original offline dependencies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

SWIN_URL = 'https://download.pytorch.org/models/swin_b-68c6b09e.pth'
SWIN_SHA256 = '68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_extension(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'backbone_extension.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('backbones') != ['resnet50', 'swin_b']:
        raise ValueError('Unsupported backbone extension.')
    required = {'source/clindistill/models.py', 'source/clindistill/config.py', 'source/_support/training_runtime.py',
                'source/configs/code/thesis.json', 'source/_support/backbone_extension.py',
                'models/swin_b-68c6b09e.pth', manifest['project_wheel']}
    if not required <= set(manifest['checksums']):
        raise ValueError('Incomplete backbone extension.')
    for relative, expected in manifest['checksums'].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Backbone extension checksum mismatch: {relative}')
    if manifest['checksums']['models/swin_b-68c6b09e.pth'] != SWIN_SHA256:
        raise ValueError('Swin weights do not match the pinned official ImageNet-1K checkpoint.')
    with zipfile.ZipFile(root / manifest['project_wheel']) as archive:
        modules = list((root / 'source/clindistill').glob('*.py'))
        if not modules:
            raise ValueError('Missing project source.')
        for path in modules:
            if archive.read('clindistill/' + path.name) != path.read_bytes():
                raise ValueError('Extension wheel/source mismatch.')
    return manifest


def activate_extension(directory, base_assets, *, install=True):
    root = Path(directory).resolve()
    manifest = verify_extension(root)
    source = root / 'source'
    loaded = sys.modules.get('clindistill')
    if loaded and Path(loaded.__file__).resolve().parent != source / 'clindistill':
        raise RuntimeError('Source đã đổi. Restart kernel/session rồi Run All notebook mới từ đầu.')
    if install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--force-reinstall', str(root / manifest['project_wheel'])], check=True)
    sys.path.insert(0, str(source))
    assets = dict(base_assets)
    assets.update(source_root=source, backbone_extension=root,
                  backbone_weights={'resnet50': str(base_assets['resnet']),
                                    'swin_b': str(root / 'models/swin_b-68c6b09e.pth')})
    print('Backbone extension OK: ResNet-50 + Swin-B; no network installation.', flush=True)
    return assets


def build_extension(project_root, weights_file, destination):
    project, weights, destination = map(lambda p: Path(p).resolve(), (project_root, weights_file, destination))
    if digest(weights) != SWIN_SHA256:
        raise ValueError('Download the pinned official Swin-B weights before building the extension.')
    spec = importlib.util.spec_from_file_location('extension_offline', project / '_support/kaggle_offline.py')
    offline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(offline)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='swin_extension_', dir=destination.parent) as temp:
        root = Path(temp) / 'extension'
        source = root / 'source'
        source.mkdir(parents=True)
        for folder in ('clindistill', 'configs'):
            shutil.copytree(project / folder, source / folder,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (source / '_support').mkdir()
        for name in ('backbone_extension.py', 'kaggle_offline.py', 'training_runtime.py'):
            shutil.copy2(project / '_support' / name, source / '_support' / name)
        for path in [project / 'pyproject.toml', project / 'README.md',
                     project / 'COde_02_Train_Baselines.ipynb', project / 'COde_03_Evaluate.ipynb']:
            shutil.copy2(path, source / path.name)
        wheel = offline.build_project_wheel(project, root / 'wheels', Path(temp) / 'build')
        (root / 'models').mkdir()
        shutil.copy2(weights, root / 'models/swin_b-68c6b09e.pth')
        manifest = {'schema_version': 1, 'backbones': ['resnet50', 'swin_b'],
                    'project_wheel': wheel.relative_to(root).as_posix(),
                    'swin_url': SWIN_URL, 'swin_weights': 'IMAGENET1K_V1',
                    'checksums': {p.relative_to(root).as_posix(): digest(p)
                        for p in sorted(root.rglob('*')) if p.is_file()}}
        offline.write_json(root / 'backbone_extension.json', manifest)
        verify_extension(root)
        offline.zip_directory(root, destination)
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(build_extension(Path(__file__).resolve().parents[1], args.weights, args.output))
