"""Run synthetic software verification and save a non-patient verification summary."""
import ast
from importlib.metadata import version
import os
from pathlib import Path
import re
import subprocess
import sys

import nbformat

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from clindistill.artifacts import code_hash, file_hash, write_json

output=ROOT/'_support/smoke_runs/research_v4'; output.mkdir(parents=True,exist_ok=True)
notebooks={}
for path in sorted(ROOT.glob('COde_0*.ipynb')):
    doc=nbformat.read(path,as_version=4); nbformat.validate(doc)
    for cell in doc.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
            if cell.outputs or cell.execution_count is not None:
                raise ValueError('Published notebook must have clear outputs.')
    notebooks[path.name]=file_hash(path)
env=os.environ.copy(); env['HF_HUB_OFFLINE']='1'; env['MPLBACKEND']='Agg'
result=subprocess.run([sys.executable,'-m','unittest','discover','-s','_support/tests','-p','test_*.py','-v'],
    cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
log=result.stdout+'\n'+result.stderr
(output/'tests.log').write_text(log,encoding='utf-8')
count=re.search(r'Ran (\d+) tests',log)
passed=result.returncode==0 and count and int(count.group(1))>0
info={'status':'passed' if passed else 'failed','tests':int(count.group(1)) if count else 0,
      'source_sha256':code_hash(),'notebooks':notebooks,'config_sha256':file_hash(ROOT/'configs/code/research.json'),
      'python':sys.version.split()[0],
      'packages':{p:version(p) for p in ('torch','torchvision','transformers','huggingface-hub')},
      'data':'synthetic visits/features; tiny random local foundation models; no patient data',
      'coverage':['notebook02 Run All','group OOF','KD gradients','feature checksums','epoch resume',
                  'validation evaluation','image-only export/inference','legacy regressions'],
      'limitations':['No gated pretrained DINO download or full COde training in this verification.',
                     'No Kaggle GPU performance claim; actual bank rehearsal remains notebook00.']}
write_json(output/'verification.json',info)
print(info)
if not passed:
    print(log[-12000:]); raise SystemExit(1)
