"""Create local Kaggle upload bundles; no upload or raw-image copy."""
from pathlib import Path
import json
import zipfile

import nbformat

root = Path(__file__).resolve().parents[2]
destination = root / '_support' / 'dist'
destination.mkdir(exist_ok=True)
files = [root / 'README.md', root / 'pyproject.toml']
files += [root / '_support/kaggle_offline.py', root / '_support/recover_amp_bundle.py']
files += sorted((root / '_support/dist/project_wheel').glob('*.whl'))
files += [root / '_support' / 'TECHNICAL_GUIDE.md',
          root / '_support' / 'research-notes' / 'KE_HOACH_KHOA_LUAN.md']
files += sorted((root / 'clindistill').glob('*.py'))
files += sorted((root / 'configs' / 'code').glob('*.json'))
files += sorted((root / '_support' / 'tests').glob('*.py'))
files += sorted((root / '_support' / 'smoke_runs').glob('verification_*/verification.json'))
files += sorted((root / '_support' / 'smoke_runs' / 'offline_wheel_check').glob('verification.json'))
with zipfile.ZipFile(destination / 'clindistill-code-source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root).as_posix())
    for path in sorted(root.glob('COde_0*.ipynb')):
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                cell.outputs, cell.execution_count = [], None
                cell.metadata.pop('execution', None)
        archive.writestr(path.name, nbformat.writes(nb))
prepared = root.parent / 'Data' / 'COde_processed' / 'code_v1'
with zipfile.ZipFile(destination / 'code_v1_prepared.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(prepared.glob('*')):
        if path.is_file():
            archive.write(path, path.name)
for name in ['clindistill-code-source.zip', 'code_v1_prepared.zip']:
    with zipfile.ZipFile(destination / name) as archive:
        assert archive.testzip() is None
    print(name, (destination / name).stat().st_size, 'bytes')
