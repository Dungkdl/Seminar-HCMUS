"""Build the portable project wheel and Kaggle source ZIP; never upload artifacts."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def package(root, destination, prepared=None, image_root=None):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    offline = helper('package_wheel_builder', root / '_support/kaggle_offline.py')
    bank = helper('package_research_bank', root / '_support/research_bank.py')
    with tempfile.TemporaryDirectory(dir=destination, prefix='_wheel_build_') as temporary:
        staging = Path(temporary)
        generated = offline.build_project_wheel(root, staging / 'wheels', staging / 'source')
        bank.verify_project_wheel(generated, root)
        wheel = destination / 'project_wheel' / generated.name
        wheel.parent.mkdir(exist_ok=True); shutil.copy2(generated, wheel)
    files = bank.source_files(root) + [wheel]
    files += sorted((root / '_support/tests').glob('*.py'))
    files += sorted((root / '_support/research-notes').glob('*.py'))
    archive_path = destination / 'clindistill-code-source.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            relative = path.relative_to(root).as_posix() if root in path.parents else '_support/dist/project_wheel/' + path.name
            if path.suffix == '.ipynb':
                notebook = json.loads(path.read_text(encoding='utf-8'))
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        cell['outputs'], cell['execution_count'] = [], None
                        cell.get('metadata', {}).pop('execution', None)
                archive.writestr(relative, json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
            else:
                archive.write(path, relative)
    artifacts = [archive_path, wheel]
    for path in sorted(root.glob('COde_0*.ipynb')):
        target = destination / 'notebooks' / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        artifacts.append(target)
    if prepared is not None:
        prepared = Path(prepared).resolve()
        info = json.loads((prepared / 'release.json').read_text(encoding='utf-8'))
        image_index = None
        if image_root is not None:
            # Public path keys + original checksums only; no images or records are copied.
            image_root = Path(image_root).resolve()
            paths = set()
            for path in prepared.glob('student_single_*.jsonl'):
                for line in path.read_text(encoding='utf-8').splitlines():
                    paths.update(json.loads(line)['photo_paths'])
            images = {}
            for name in sorted(paths):
                path = (image_root / name).resolve()
                if image_root not in path.parents:
                    raise ValueError('Unsafe image path in prepared release.')
                images[name] = bank.digest(path)
            image_index = {'release_id': info['release_id'], 'images': images,
                'purpose': 'Verify flattened Qwen photos against original audited CODe bytes; keep canonical paths.'}
        prepared_zip = destination / 'code_v1_prepared.zip'
        with zipfile.ZipFile(prepared_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in ['release.json'] + sorted(info['file_checksums']):
                if Path(name).name != name:
                    raise ValueError('Unsafe path in prepared release.')
                path = prepared / name
                if name != 'release.json' and bank.digest(path) != info['file_checksums'][name]:
                    raise ValueError('Prepared release checksum mismatch.')
                archive.write(path, name)
            if image_index is not None:
                archive.writestr('image_index.json', json.dumps(image_index, ensure_ascii=False, indent=1) + '\n')
        artifacts.append(prepared_zip)
        # Overlay onto the user's existing assets without copying Qwen/data images/wheelhouse.
        addon = destination / 'clindistill-assets-addon.zip'
        with zipfile.ZipFile(addon, 'w', zipfile.ZIP_DEFLATED) as target:
            for source_zip, prefix in [(archive_path, 'clindistill_source/'),
                                       (prepared_zip, 'data/COde_processed/code_v1/')]:
                with zipfile.ZipFile(source_zip) as source:
                    for name in source.namelist():
                        target.writestr(prefix + name, source.read(name))
        artifacts.append(addon)
    for path in artifacts:
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError('Generated ZIP failed integrity verification.')
    manifest = {'project_wheel': wheel.name, 'third_party_wheels': 'Prepare on Linux/Kaggle with notebook00.',
                'contains_raw_images': False, 'files': {str(p.relative_to(destination)): bank.digest(p) for p in artifacts}}
    bank.write_json(destination / 'package_manifest.json', manifest)
    for path in artifacts:
        print(path, path.stat().st_size, 'bytes')
    return manifest


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / '_support/dist')
    parser.add_argument('--prepared-root', type=Path, help='Optional immutable prepared release; no raw images.')
    parser.add_argument('--image-root', type=Path, help='Original CODe root; include checksums for flattened Qwen photos.')
    args = parser.parse_args()
    if args.image_root and not args.prepared_root:
        parser.error('--image-root requires --prepared-root')
    package(root, args.output, args.prepared_root, args.image_root)
