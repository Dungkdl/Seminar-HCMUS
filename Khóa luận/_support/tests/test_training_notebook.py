"""Execute the current notebook cells on synthetic cached features, CPU only."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import torch

from clindistill.artifacts import file_hash, read_json
from test_research_pipeline import config, synthetic_features, tiny_encoders


class TrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.project=Path(__file__).resolve().parents[2]
        cls.notebook=nbformat.read(cls.project/'COde_02_Train_Baselines.ipynb',as_version=4)

    def execute(self, root, cache, *, previous=None):
        namespace={'__name__':'__main__','display':lambda *a,**k:None,'DEBUG_CONFIG':config()}
        replacements={'PROJECT_ROOT_OVERRIDE':self.project,'FEATURE_ROOT_OVERRIDE':cache,
                      'OUTPUT_ROOT_OVERRIDE':root/'output','PREVIOUS_STUDY_OVERRIDE':previous}
        with patch.dict('os.environ',{'MPLBACKEND':'Agg'}), contextlib.redirect_stdout(io.StringIO()):
            for cell in self.notebook.cells:
                if cell.cell_type!='code': continue
                source=cell.source
                for key,value in replacements.items():
                    if value is not None:
                        source=source.replace(f'{key} = None',f'{key} = {str(value)!r}')
                source=source.replace("CONFIG = read_json(PROJECT_ROOT / 'configs/code/research.json')","CONFIG = DEBUG_CONFIG.copy()")
                exec(compile(source,'COde_02_Train_Baselines.ipynb','exec'),namespace)
        import matplotlib.pyplot as plt
        plt.close('all')
        return namespace

    def test_default_registers_complete_three_seed_oof_study(self):
        from clindistill.research.training import study_plan
        cfg=read_json(self.project/'configs/code/research.json')
        self.assertEqual(cfg['seeds'],[2026,2027,2028]); self.assertEqual(cfg['folds'],5)
        self.assertEqual(study_plan(cfg)['total_fits'],57)
        self.assertIn('student_no_kd',cfg['methods']); self.assertIn('kd_reliability',cfg['methods'])
        namespace={}; exec(next(c.source for c in self.notebook.cells if c.cell_type=='code'),namespace)
        self.assertTrue(namespace['RUN_TRAIN'])
        # Entry point 02 does not perform raw-data preparation or open test shards.
        sources='\n'.join(c.source for c in self.notebook.cells if c.cell_type=='code')
        self.assertNotIn('prepare_features(',sources); self.assertNotIn('complete_dataset.csv',sources)

    def test_run_all_export_resume_and_new_session_without_test_files(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); encoders=tiny_encoders(root/'encoders')
            cache=synthetic_features(root/'cache',encoders=encoders)
            ns=self.execute(root,cache); study=ns['STUDY_ROOT']
            self.assertTrue((study/'completed.json').is_file())
            self.assertEqual(len(read_json(study/'completed.json')['results']),9)
            self.assertFalse(read_json(study/'completed.json')['test_opened'])
            self.assertTrue((study/'learning_curves.png').is_file())
            self.assertTrue((root/'output/student_export/code_research_v1/student_manifest.json').is_file())
            hashes={p.relative_to(study):file_hash(p) for p in study.rglob('*.pt')}
            self.execute(root,cache)
            self.assertEqual(hashes,{p.relative_to(study):file_hash(p) for p in study.rglob('*.pt')})
            saved=root/'saved_input'
            self.assertEqual(saved.parent.resolve(),root.resolve())
            self.assertEqual(study.parents[1].resolve(),(root/'output').resolve())
            (root/'output').rename(saved)
            restored=self.execute(root,cache,previous=saved/'runs/code_research_v1')['STUDY_ROOT']
            self.assertEqual(hashes,{p.relative_to(restored):file_hash(p) for p in restored.rglob('*.pt')})

    def test_corrupt_cache_fails_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cache=synthetic_features(root/'cache'); (cache/'train.npz').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):
                self.execute(root,cache)
            self.assertFalse((root/'output/runs').exists())

    def test_legacy_plan_still_separates_backbones_and_seeds(self):
        spec=importlib.util.spec_from_file_location('legacy_study',self.project/'_support/training_runtime.py')
        runtime=importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
        plan=runtime.execution_plan(['resnet50','swin_b'],[2026,2027,2028],['kd_selective'])
        self.assertEqual(len(plan),18); self.assertEqual(len({tuple(r.values()) for r in plan}),18)
