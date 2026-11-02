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


def package(root, destination, prepared=None):
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
    if prepared is not None:
        prepared = Path(prepared).resolve()
        info = json.loads((prepared / 'release.json').read_text(encoding='utf-8'))
        prepared_zip = destination / 'code_v1_prepared.zip'
        with zipfile.ZipFile(prepared_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in ['release.json'] + sorted(info['file_checksums']):
                if Path(name).name != name:
                    raise ValueError('Unsafe path in prepared release.')
                path = prepared / name
                if name != 'release.json' and bank.digest(path) != info['file_checksums'][name]:
                    raise ValueError('Prepared release checksum mismatch.')
                archive.write(path, name)
        artifacts.append(prepared_zip)
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
    args = parser.parse_args()
    package(root, args.output, args.prepared_root)
