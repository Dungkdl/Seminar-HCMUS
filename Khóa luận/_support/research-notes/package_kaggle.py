"""Create local Kaggle upload bundles; no upload or raw-image copy."""
from pathlib import Path
import importlib.util
import json
import re
import shutil
import tempfile
import zipfile

import nbformat

root = Path(__file__).resolve().parents[2]
destination = root / '_support' / 'dist'
destination.mkdir(exist_ok=True)
project_version = re.search(r'^version\s*=\s*"([^"]+)"', (root / 'pyproject.toml').read_text(encoding='utf-8'), re.M).group(1)
wheel_path = destination / 'project_wheel' / f'clindistill_code-{project_version}-py3-none-any.whl'
# Build in isolation on every package run so unchanged version numbers cannot hide stale source.
spec = importlib.util.spec_from_file_location('wheel_builder', root / '_support/kaggle_offline.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
with tempfile.TemporaryDirectory(dir=destination, prefix='_wheel_build_') as temporary:
    build_root = Path(temporary)
    generated = helper.build_project_wheel(root, build_root / 'wheels', build_root / 'source')
    wheel_path.parent.mkdir(exist_ok=True)
    shutil.copy2(generated, wheel_path)
files = [root / 'README.md', root / 'pyproject.toml']
files += [root / '_support' / name for name in
          ['kaggle_offline.py', 'recover_amp_bundle.py', 'backbone_extension.py', 'training_runtime.py', 'research_bank.py']]
files += [wheel_path]
files += [root / '_support' / 'TECHNICAL_GUIDE.md',
          root / '_support' / 'research-notes' / 'KE_HOACH_KHOA_LUAN.md']
files += sorted((root / 'clindistill').rglob('*.py'))
files += sorted((root / 'configs' / 'code').glob('*.json'))
files += sorted((root / '_support' / 'tests').glob('*.py'))
files += sorted((root / '_support' / 'research-notes').glob('*.py'))
# Verification summaries are documentation; exclude old experiment outputs from source distribution.
verification = root / '_support/smoke_runs/research_v4/verification.json'
if verification.is_file():
    files.append(verification)
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
artifacts = ['clindistill-code-source.zip']
if (prepared / 'release.json').exists():
    with zipfile.ZipFile(destination / 'code_v1_prepared.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(prepared.glob('*')):
            if path.is_file():
                archive.write(path, path.name)
    artifacts.append('code_v1_prepared.zip')
for name in artifacts:
    with zipfile.ZipFile(destination / name) as archive:
        assert archive.testzip() is None
    print(name, (destination / name).stat().st_size, 'bytes')
