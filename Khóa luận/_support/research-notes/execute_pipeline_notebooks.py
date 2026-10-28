"""Execute real notebook cells on COde/CPU debug without overwriting user notebooks.

Default: existing release -> 01 section 14 -> all 16 training branches -> evaluation.
Use --full-audit to execute every preparation cell too.
"""
import argparse
import os
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import nbformat

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
os.chdir(root)
os.environ['CLINDISTILL_SMOKE_TEST'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import torch
torch.set_num_threads(2)
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config
from clindistill.artifacts import code_hash, write_json
from clindistill.suite import CORE, ABLATIONS

parser = argparse.ArgumentParser()
parser.add_argument('--full-audit', action='store_true')
args = parser.parse_args()
run_name = 'verification_' + code_hash()[:12]
support = root / '_support' / 'smoke_runs'
cache = support / 'clinical_debug_v2'
output = support / run_name
output.mkdir(parents=True, exist_ok=True)
shell_config = Config()
shell_config.HistoryManager.enabled = False
shell = InteractiveShell.instance(config=shell_config)
log = []
started = time.perf_counter()
for filename in ['COde_01_Data_Audit_and_Preparation.ipynb',
                 'COde_02_Train_Baselines.ipynb', 'COde_03_Evaluate.ipynb']:
    notebook = nbformat.read(root / filename, as_version=4)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    if filename.startswith('COde_01') and not args.full_audit:
        cells = cells[-1:]
    for i, cell in enumerate(cells):
        source = cell.source.replace('CLINICAL_OUTPUT_OVERRIDE = None', f'CLINICAL_OUTPUT_OVERRIDE = {str(cache)!r}')
        source = source.replace('CLINICAL_CACHE_OVERRIDE = None', f'CLINICAL_CACHE_OVERRIDE = {str(cache)!r}')
        source = source.replace("RUN_NAME = 'code_clindistill_v2'", f'RUN_NAME = {run_name!r}')
        source = source.replace('SELECTED = CORE  #', 'SELECTED = CORE + ABLATIONS  #')
        source = source.replace("EXPERIMENTS = ['kd_selective']", f'EXPERIMENTS = {CORE + ABLATIONS!r}')
        print(f'{filename}: code cell {i + 1}/{len(cells)}', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(source, store_history=False)
        log.append(f'\n{filename}: cell {i + 1}\n{captured.stdout}\n{captured.stderr}')
        (output / 'notebook_execution.log').write_text(''.join(log), encoding='utf-8')
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Notebook failed: {filename}, cell {i + 1}. See {output}') from (
                result.error_before_exec or result.error_in_exec)
write_json(output / 'verification.json', {'status': 'passed', 'code_sha256': code_hash(),
    'elapsed_seconds': time.perf_counter() - started, 'device': 'cpu', 'debug': True,
    'data': 'real COde release, deterministic 12-case subsets per partition',
    'experiments': 16, 'seeds': [2026], 'epochs': 2,
    'biomedical_encoder_trained_or_evaluated': False,
    'scope': '01 clinical cache cell, all code cells of 02 and 03; no full GPU training'})
print('PASS:', output, flush=True)
