"""Generate the standalone Run 7 Kaggle notebook using the installed v6 bank."""
import ast
from pathlib import Path

from package_run4_run5 import ROOT, write_notebook


def package(destination):
    destination.mkdir(parents=True, exist_ok=True)
    tree = ast.parse((ROOT / '_support/research_bank.py').read_text(encoding='utf-8'))
    bootstrap = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'BOOTSTRAP_CELL' for t in node.targets))
    names = ('run5_xray.py', 'run6_xray_fusion.py', 'run7_xray_kd.py')
    helpers = {name: Path(__file__).with_name(name).read_text(encoding='utf-8') for name in names}
    write_notebook(destination / 'Run07_Xray_KD_Validation.ipynb',
        '# Run 7 — CE and two KD losses on the same image Student\n\n'
        'Three Student losses and one multimodal Teacher with group OOF. Validation only.',
        """
        from pathlib import Path
        RESEARCH_BANK_OVERRIDE = '/kaggle/input/notebooks/messi1231/notebook308b8c1b62/clindistill_research_v6'
        FEATURE_ROOT = Path('/kaggle/input/notebooks/messi1231/code-01-features-only-v051/data/COde_processed/research_features_v3')
        PREPARED_ROOT = Path('/kaggle/input/datasets/messi1231/code-v1')
        IMAGE_ROOT = Path('/kaggle/input/datasets/messi1231/radiograph-images')
        IMAGE_BATCH_SIZE = 32
        """, bootstrap, """
        import runpy
        import torch
        from clindistill.artifacts import read_json

        if not torch.cuda.is_available():
            raise RuntimeError('Select the GPU for Run 7.')
        if not (IMAGE_ROOT / 'Images/Radiographs').is_dir():
            raise ValueError('IMAGE_ROOT must contain Images/Radiographs.')
        source_dir = Path('/kaggle/working/run7_source')
        source_dir.mkdir(exist_ok=True)
        for name, source in HELPER_SOURCES.items():
            (source_dir / name).write_text(source, encoding='utf-8', newline='\\n')
        sys.path.insert(0, str(source_dir))
        run = runpy.run_path(str(source_dir / 'run7_xray_kd.py'))['run_kd']
        cfg = read_json(PROJECT_ROOT / 'configs/code/research_visual.json')
        summary, comparisons = run(PREPARED_ROOT, FEATURE_ROOT, IMAGE_ROOT, RESEARCH_ASSETS['models']['student'],
            '/kaggle/working/runs/code_run7_xray_kd_v1', cfg, image_batch=IMAGE_BATCH_SIZE)
        display(summary)
        print(json.dumps(comparisons, indent=2))
        """.replace('HELPER_SOURCES', repr(helpers)))
    (destination / 'RUN7.md').write_bytes((ROOT / '_support/RUN7.md').read_bytes())


if __name__ == '__main__':
    package(ROOT / '_support/dist/run7')
