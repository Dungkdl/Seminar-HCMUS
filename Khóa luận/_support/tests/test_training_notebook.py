"""Execute notebook 02 cells against an isolated synthetic release, never real data."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import torch

from clindistill.artifacts import file_hash
from clindistill.clinical import prepare_clinical_cache
from test_training_pipeline import synthetic_release
from test_clinical_distillation import contexts


class TrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.project = Path(__file__).resolve().parents[2]
        cls.notebook = nbformat.read(cls.project / 'COde_02_Train_Baselines.ipynb', as_version=4)

    def fixture(self, root):
        images, prepared, records = synthetic_release(root)
        contexts(prepared, records)
        (prepared / 'student_single_test.jsonl').unlink()
        (prepared / 'teacher_context_test.jsonl').unlink()
        cache = root / 'clinical'
        prepare_clinical_cache(prepared, cache, debug=True)
        return images, prepared, cache

    def execute(self, root, images, prepared, cache, stop_before_training=False):
        namespace = {'__name__': '__main__'}
        replacements = {'PROJECT_ROOT_OVERRIDE': self.project, 'PREPARED_ROOT_OVERRIDE': prepared,
                        'IMAGE_ROOT_OVERRIDE': images, 'CLINICAL_CACHE_OVERRIDE': cache,
                        'OUTPUT_ROOT_OVERRIDE': root / 'output'}
        with patch.dict(os.environ, {'CLINDISTILL_SMOKE_TEST': '1', 'MPLBACKEND': 'Agg'}), \
                patch('IPython.display.display'), contextlib.redirect_stdout(io.StringIO()):
            for cell in self.notebook.cells:
                if cell.cell_type != 'code':
                    continue
                if stop_before_training and cell.source.startswith('def collect_progress'):
                    break
                source = cell.source
                source = source.replace('/kaggle/input', (root / 'saved_input').as_posix())
                for name, value in replacements.items():
                    source = source.replace(f'{name} = None', f'{name} = {str(value)!r}')
                exec(compile(source, 'COde_02_Train_Baselines.ipynb', 'exec'), namespace)
        return namespace

    def test_default_is_three_seed_proposed_method_without_baselines(self):
        source = next(c.source for c in self.notebook.cells if c.cell_type == 'code')
        namespace = {}
        with patch.dict(os.environ, {'CLINDISTILL_SMOKE_TEST': ''}), contextlib.redirect_stdout(io.StringIO()):
            exec(source, namespace)
        self.assertEqual(namespace['SEEDS'], [2026, 2027, 2028])
        self.assertEqual(namespace['EXPERIMENTS'], ['kd_selective'])
        self.assertTrue(namespace['RUN_TRAIN'])
        self.assertTrue(namespace['RESUME'])

    def test_run_all_exports_proposed_student_and_completed_runs_resume_without_test_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images, prepared, cache = self.fixture(root)
            namespace = self.execute(root, images, prepared, cache)
            report = namespace['REPORT_ROOT']
            status = json.loads((report / 'suite_status.json').read_text())
            self.assertTrue(status['complete'])
            self.assertEqual(status['completed_runs'], 3)
            self.assertFalse(status['test_evaluated'])
            self.assertEqual(namespace['PLAN'], ['teacher_v', 'teacher_h', 'kd_selective'])
            exports = json.loads((report / 'student_exports.json').read_text())
            self.assertEqual(len(exports), 1)
            self.assertEqual(exports[0]['experiment'], 'kd_selective')
            state = torch.load(exports[0]['path'], weights_only=True)
            self.assertNotIn('optimizer', state)
            self.assertNotIn('patient_id', state)
            self.assertTrue((report / 'validation_mean_std.csv').is_file())
            self.assertTrue((report / 'learning_curves.png').is_file())
            checkpoints = {str(p): file_hash(p) for p in namespace['RUNS_ROOT'].glob('*/seed_*/*.pt')}
            self.execute(root, images, prepared, cache)
            self.assertEqual(checkpoints, {p: file_hash(p) for p in checkpoints})
            # A new session attaches the old output as read-only input; restore into fresh output.
            old_output = root / 'output'
            saved_input = root / 'saved_input'
            self.assertEqual(old_output.parent.resolve(), root.resolve())
            self.assertEqual(saved_input.parent.resolve(), root.resolve())
            old_output.rename(saved_input)
            restored = self.execute(root, images, prepared, cache)
            self.assertEqual(checkpoints, {p: file_hash(p) for p in checkpoints})
            self.assertTrue(json.loads((restored['REPORT_ROOT'] / 'suite_status.json').read_text())['complete'])

    def test_invalid_cache_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images, prepared, cache = self.fixture(root)
            (cache / 'train.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.execute(root, images, prepared, cache, stop_before_training=True)
            self.assertFalse((root / 'output/_support/smoke_runs/code_clindistill_v2/protocol.json').exists())
