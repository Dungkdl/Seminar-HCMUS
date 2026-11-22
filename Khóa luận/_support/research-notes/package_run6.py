"""Generate the standalone Run 6 Kaggle notebook without rebuilding the v6 bank."""
import ast
from pathlib import Path

from package_run4_run5 import ROOT, write_notebook


def package(destination):
    destination.mkdir(parents=True, exist_ok=True)
    tree = ast.parse((ROOT / '_support/research_bank.py').read_text(encoding='utf-8'))
    bootstrap = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'BOOTSTRAP_CELL' for t in node.targets))
    helpers = {name: Path(__file__).with_name(name).read_text(encoding='utf-8')
               for name in ('run5_xray.py', 'run6_xray_fusion.py')}
    write_notebook(destination / 'Run06_Xray_Fusion_Validation.ipynb',
        '# Run 6 — photo and radiograph fusion\n\nThree architectures/controls, three seeds, paired validation. Use the same four inputs as Run 5.',
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
            raise RuntimeError('Select the GPU for Run 6.')
        if not (IMAGE_ROOT / 'Images/Radiographs').is_dir():
            raise ValueError('IMAGE_ROOT must contain Images/Radiographs.')
        work = Path('/kaggle/working/run6_source')
        work.mkdir(exist_ok=True)
        for name, source in HELPER_SOURCES.items():
            (work / name).write_text(source, encoding='utf-8', newline='\\n')
        sys.path.insert(0, str(work))
        run = runpy.run_path(str(work / 'run6_xray_fusion.py'))['run_fusion']
        cfg = read_json(PROJECT_ROOT / 'configs/code/research_visual.json')
        summary, comparisons = run(PREPARED_ROOT, FEATURE_ROOT, IMAGE_ROOT, RESEARCH_ASSETS['models']['student'],
            '/kaggle/working/runs/code_run6_xray_fusion_v1', cfg, image_batch=IMAGE_BATCH_SIZE)
        display(summary)
        print(json.dumps(comparisons, indent=2))
        """.replace('HELPER_SOURCES', repr(helpers)))
    (destination / 'RUN6.md').write_bytes((ROOT / '_support/RUN6.md').read_bytes())


if __name__ == '__main__':
    package(ROOT / '_support/dist/run6')
