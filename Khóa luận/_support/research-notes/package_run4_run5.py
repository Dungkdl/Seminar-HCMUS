"""Generate Kaggle notebooks for the frozen exploratory Run 4 test and paired Run 5."""
import ast
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]


def write_notebook(path, intro, config, bootstrap, body):
    cells = []
    for i, (kind, source) in enumerate([('markdown', intro), ('code', config), ('code', bootstrap), ('code', body)]):
        cell = {'cell_type': kind, 'id': f'run-cell-{i}', 'metadata': {},
                'source': dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code':
            ast.parse(''.join(cell['source']))
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    path.write_text(json.dumps({'cells': cells, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 5}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def package(destination):
    destination.mkdir(parents=True, exist_ok=True)
    tree = ast.parse((ROOT / '_support/research_bank.py').read_text(encoding='utf-8'))
    bootstrap = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'BOOTSTRAP_CELL' for t in node.targets))
    common = """
        from pathlib import Path
        RESEARCH_BANK_OVERRIDE = '/kaggle/input/notebooks/messi1231/notebook308b8c1b62/clindistill_research_v6'
        FEATURE_ROOT = Path('/kaggle/input/notebooks/messi1231/code-01-features-only-v051/data/COde_processed/research_features_v3')
    """
    write_notebook(destination / 'Run04_Locked_Test.ipynb',
        '# Run 4 — locked exploratory test\n\nRun one mode per saved version: lock → features → evaluate. See RUN4_RUN5.md.',
        common + """
        MODE = 'lock'
        STUDY_ROOT = None  # Path to runs/code_controls_v6 in the training or locked output
        PREPARED_ROOT = None  # Path to code_v1 for MODE='features'
        IMAGE_ROOT = None  # CODe root containing Images/Photographs
        TEST_PROTOCOL_PATH = None  # test_protocol.json from MODE='lock'
        REVIEW_RECORD = (
            'Run 4 exploratory held-out evaluation. Freeze all registered models, seeds and checkpoints. '
            'Primary metric: visit-level macro-F1. Label, near-duplicate and clinical timing reviews remain unresolved.'
        )
        """, bootstrap, """
        import shutil
        from clindistill.artifacts import read_json
        from clindistill.research.data import FeatureStore, prepare_features
        from clindistill.research.reporting import evaluate_study, freeze_test_protocol
        from clindistill.research.training import study_plan

        working = Path('/kaggle/working')
        if MODE in {'lock', 'evaluate'}:
            if STUDY_ROOT is None:
                raise ValueError('Set STUDY_ROOT to the Run 4 study directory.')
            source = Path(STUDY_ROOT)
            study = read_json(source / 'study.json')
            if study['config']['run_name'] != 'code_controls_v6':
                raise ValueError('This notebook expects Run 4 code_controls_v6.')
        if MODE == 'lock':
            target = working / 'runs/code_controls_v6'
            names = ['study.json', 'completed.json', 'research_readiness.json']
            for method in study_plan(study['config'])['systems']:
                for seed in study['config']['seeds']:
                    names += [f'{method}/seed_{seed}/{name}' for name in ('best.pt', 'result.json')]
            for name in names:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, path)
            protocol = freeze_test_protocol(FEATURE_ROOT, target, REVIEW_RECORD)
            print('Frozen protocol:', target / 'test_protocol.json', protocol['identity'])
        elif MODE == 'features':
            if None in (PREPARED_ROOT, IMAGE_ROOT, TEST_PROTOCOL_PATH):
                raise ValueError('Set PREPARED_ROOT, IMAGE_ROOT and TEST_PROTOCOL_PATH.')
            if not (Path(IMAGE_ROOT) / 'Images/Photographs').is_dir():
                raise ValueError('Attach the original canonical CODe photo tree.')
            protocol = read_json(TEST_PROTOCOL_PATH)
            if protocol['config']['run_name'] != 'code_controls_v6':
                raise ValueError('Attach the frozen Run 4 protocol.')
            original = FeatureStore(FEATURE_ROOT)
            target = working / 'run4_features_with_test'
            target.mkdir(exist_ok=True)
            if not (target / 'feature_manifest.json').exists():
                for name in ('train.npz', 'validation.npz', 'feature_manifest.json'):
                    shutil.copy2(FEATURE_ROOT / name, target / name)
            if FeatureStore(target).identity != original.identity:
                raise ValueError('Working train/validation does not match the original cache.')
            cfg = read_json(PROJECT_ROOT / 'configs/code/features.json')
            prepare_features(PREPARED_ROOT, IMAGE_ROOT, RESEARCH_ASSETS['models'], target, cfg,
                             splits=('test',), test_protocol=protocol)
            FeatureStore(target, splits=('train', 'validation', 'test'))
            print('Test cache:', target)
        elif MODE == 'evaluate':
            table, summary = evaluate_study(FEATURE_ROOT, source, working / 'reports/code_controls_v6/test',
                                            split='test', test_review_record=REVIEW_RECORD)
            display(summary)
        else:
            raise ValueError('MODE must be lock, features or evaluate.')
        """)
    experiment = Path(__file__).with_name('run5_xray.py').read_text(encoding='utf-8')
    write_notebook(destination / 'Run05_Xray_Validation.ipynb',
        '# Run 5 — photo vs photo + X-ray\n\nSix fits on paired visits. Attach the old bank, photo cache, prepared release and canonical radiographs.',
        common + """
        PREPARED_ROOT = None  # Path to code_v1 (release.json)
        IMAGE_ROOT = None  # CODe root containing Images/Radiographs
        IMAGE_BATCH_SIZE = 32
        """, bootstrap, """
        import runpy
        import torch
        from clindistill.artifacts import read_json

        if PREPARED_ROOT is None or IMAGE_ROOT is None:
            raise ValueError('Set PREPARED_ROOT and IMAGE_ROOT.')
        if not (Path(IMAGE_ROOT) / 'Images/Radiographs').is_dir():
            raise ValueError('IMAGE_ROOT must contain Images/Radiographs.')
        if not torch.cuda.is_available():
            raise RuntimeError('Select the GPU for Run 5.')
        script = Path('/kaggle/working/run5_xray.py')
        script.write_text(EXPERIMENT_SOURCE, encoding='utf-8', newline='\\n')
        run = runpy.run_path(str(script))['run_xray']
        cfg = read_json(PROJECT_ROOT / 'configs/code/research_visual.json')
        summary, contrast = run(PREPARED_ROOT, FEATURE_ROOT, IMAGE_ROOT, RESEARCH_ASSETS['models']['student'],
            '/kaggle/working/runs/code_run5_xray_v1', cfg, image_batch=IMAGE_BATCH_SIZE)
        display(summary)
        print(contrast)
        """.replace('EXPERIMENT_SOURCE', repr(experiment)))
    (destination / 'RUN4_RUN5.md').write_bytes((ROOT / '_support/RUN4_RUN5.md').read_bytes())


if __name__ == '__main__':
    package(ROOT / '_support/dist/run4-run5')
