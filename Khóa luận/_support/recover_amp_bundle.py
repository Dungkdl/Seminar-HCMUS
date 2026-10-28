"""Repair the initial offline bundle's AMP assignment bug without redownloading assets."""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile


OLD = '''        valid_features = self.projection(self.encoder(images[mask]))
        features = valid_features.new_zeros((*mask.shape, valid_features.shape[-1]))
        features[mask] = valid_features + self.modality_embedding(modalities[mask])'''
NEW = '''        valid_features = self.projection(self.encoder(images[mask]))
        # Embeddings can promote autocast features to float32; allocate after the addition.
        valid_features = valid_features + self.modality_embedding(modalities[mask])
        features = valid_features.new_zeros((*mask.shape, valid_features.shape[-1]))
        features[mask] = valid_features'''


def repair_bundle(bundle):
    bundle = Path(bundle).resolve()
    spec = importlib.util.spec_from_file_location('amp_recovery_offline',
        bundle / 'source/_support/kaggle_offline.py')
    offline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(offline)
    manifest = offline.verify_bundle(bundle)
    model_path = bundle / 'source/clindistill/models.py'
    original = model_path.read_bytes()
    source = original.decode('utf-8').replace('\r\n', '\n')
    if NEW in source:
        return offline, False
    if source.count(OLD) != 1:
        raise ValueError('Unexpected models.py; refusing to modify this bundle.')
    wheels = list((bundle / 'wheels').glob('clindistill_code-*-py3-none-any.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected one existing project wheel.')
    wheel_path = wheels[0]
    # Build first in a temporary copy, preserving the downloaded bundle on build failure.
    with tempfile.TemporaryDirectory(prefix='code_amp_repair_', dir=bundle.parent) as temp:
        temp = Path(temp)
        project = temp / 'project'
        project.mkdir()
        shutil.copy2(bundle / 'source/pyproject.toml', project / 'pyproject.toml')
        shutil.copytree(bundle / 'source/clindistill', project / 'clindistill',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        patched = source.replace(OLD, NEW).encode('utf-8')
        (project / 'clindistill/models.py').write_bytes(patched)
        rebuilt = offline.build_project_wheel(project, temp / 'wheels', temp / 'build')
        rebuilt_bytes = rebuilt.read_bytes()
    old_wheel = wheel_path.read_bytes()
    manifest_path = bundle / 'offline_manifest.json'
    old_manifest = manifest_path.read_bytes()
    try:
        model_path.write_bytes(patched)
        wheel_path.write_bytes(rebuilt_bytes)
        for path in (model_path, wheel_path):
            manifest['checksums'][path.relative_to(bundle).as_posix()] = offline.digest(path)
        manifest['status'] = 'assets_prepared_smoke_pending'
        manifest['repairs'] = sorted(set(manifest.get('repairs', [])) | {'amp_modality_dtype_v1'})
        offline.write_json(manifest_path, manifest)
        offline.verify_bundle(bundle)
    except Exception:
        model_path.write_bytes(original)
        wheel_path.write_bytes(old_wheel)
        manifest_path.write_bytes(old_manifest)
        raise
    return offline, True


def recover_and_smoke(bundle, mini_root):
    bundle, mini_root = Path(bundle).resolve(), Path(mini_root).resolve()
    if not (mini_root / 'mini_dataset.json').is_file():
        raise FileNotFoundError('mini_root must contain mini_dataset.json.')
    offline, repaired = repair_bundle(bundle)
    print('AMP source/wheel repair:', 'applied' if repaired else 'already applied', flush=True)
    # Discard classes imported before the failure; the repaired bundle becomes the source.
    for name in list(sys.modules):
        if name == 'clindistill' or name.startswith('clindistill.'):
            del sys.modules[name]
    importlib.invalidate_caches()
    # Previous smoke outputs include implementation fingerprints and must not be resumed.
    suffix = 1
    output = bundle.parent / f'{bundle.name}_smoke_amp_{suffix}'
    while output.exists():
        suffix += 1
        output = bundle.parent / f'{bundle.name}_smoke_amp_{suffix}'
    report = offline.smoke_bundle(bundle, mini_root, output)
    if offline.verify_bundle(bundle)['status'] != 'offline_smoke_passed':
        raise RuntimeError('Smoke did not pass; no archive created.')
    archive = offline.zip_directory(bundle, bundle.parent / (bundle.name + '.zip'))
    print('UPLOAD/ATTACH:', archive, flush=True)
    return report
