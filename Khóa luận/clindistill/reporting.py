"""Held-out tables, paired cluster bootstrap, risk/coverage and image-only export."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .artifacts import code_hash, file_hash, read_json, write_json
from .config import TrainConfig
from .data import BagDataset, collate_bags
from .engine import evaluate_checkpoint
from .models import VisualBagClassifier
from .metrics import classification_metrics
from .data import load_records


def paired_bootstrap(a, b, classes, repetitions=1000, seed=2026):
    """A/B lists contain the same fitted seeds; resample groups, never images independently."""
    if len(a) != len(b) or not a:
        raise ValueError('Paired comparison needs matching seeds.')
    reference = a[0].sort_values('case_id').reset_index(drop=True)
    aligned_a, aligned_b = [], []
    for left, right in zip(a, b):
        left, right = (x.sort_values('case_id').reset_index(drop=True) for x in [left, right])
        columns = ['case_id', 'patient_id', 'split_group', 'label']
        if not left[columns].equals(reference[columns]) or not right[columns].equals(reference[columns]):
            raise ValueError('Paired predictions must have identical cases, labels and groups.')
        aligned_a.append(left.prediction.to_numpy())
        aligned_b.append(right.prediction.to_numpy())
    labels, groups = reference.label.to_numpy(), reference.split_group.to_numpy()
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    def delta(index):
        return float(np.mean([f1_score(labels[index], right[index], labels=range(classes), average='macro', zero_division=0) -
                              f1_score(labels[index], left[index], labels=range(classes), average='macro', zero_division=0)
                              for left, right in zip(aligned_a, aligned_b)]))
    values = [delta(np.concatenate([members[g] for g in rng.choice(unique, len(unique), replace=True)]))
              for _ in range(repetitions)]
    return {'delta_macro_f1_b_minus_a': delta(np.arange(len(labels))),
            'paired_group_bootstrap_95_ci': np.percentile(values, [2.5, 97.5]).tolist(),
            'groups': len(unique), 'fitted_seeds': len(a), 'repetitions': repetitions,
            'interpretation': 'Conditional on these fitted models; not a CI for retraining variability.'}


def risk_coverage(frame):
    p = frame[[c for c in frame if c.startswith('p_')]].to_numpy()
    confidence = p.max(1)
    # Stable tie-breaking by case ID; no threshold selected against test labels.
    order = np.argsort(-confidence, kind='stable')
    errors = (frame.prediction.to_numpy() != frame.label.to_numpy())[order]
    coverage = np.arange(1, len(frame) + 1) / len(frame)
    risk = errors.cumsum() / np.arange(1, len(frame) + 1)
    return pd.DataFrame({'coverage': coverage, 'risk': risk, 'confidence_cutoff': confidence[order]})


def evaluate_suite(release_dir, image_root, runs_root, *, split='validation',
                   allow_experimental_test=False, clinical_cache=None, device=None):
    rows, predictions, run_dirs = [], {}, {}
    root = Path(runs_root)
    for path in sorted(root.glob('*/seed_*/summary.json')):
        folder = path.parent
        if not read_json(path)['complete']:
            continue
        info = read_json(folder / 'run_config.json')
        config = TrainConfig(**info['config'])
        result = evaluate_checkpoint(release_dir, image_root, folder, split=split,
            allow_experimental_test=allow_experimental_test, clinical_cache=clinical_cache, device=device)
        name, seed = folder.parent.name, config.seed
        frame = pd.read_csv(folder / f'evaluation_{split}' / 'predictions.csv',
                            dtype={'case_id': str, 'patient_id': str, 'split_group': str, 'bag_signature': str})
        predictions[(name, seed)], run_dirs[(name, seed)] = frame, folder
        risk = risk_coverage(frame)
        risk.to_csv(folder / f'evaluation_{split}' / 'risk_coverage.csv', index=False)
        records = load_records(release_dir, split, config.experiment, config.debug_max_cases)
        counts = {r['case_id']: (len(r['photo_paths']), len(r['radiograph_paths'])) for r in records}
        cases = frame.copy()
        cases['n_photos'] = [counts[c][0] for c in cases.case_id]
        cases['has_radiograph'] = [counts[c][1] > 0 for c in cases.case_id]
        probabilities = cases[[f'p_{i}' for i in range(len(info['class_names']))]].to_numpy()
        cases['confidence'] = probabilities.max(1)
        cases['entropy'] = -(probabilities * np.log(probabilities.clip(1e-12))).sum(1)
        cases['correct'] = cases.label == cases.prediction
        cases.sort_values(['correct', 'confidence'], ascending=[True, False]).to_csv(
            folder / f'evaluation_{split}' / 'case_analysis.csv', index=False)
        subgroups = {}
        for label, mask in [('one_photo', cases.n_photos == 1), ('multiple_photos', cases.n_photos > 1),
                            ('with_radiograph', cases.has_radiograph), ('without_radiograph', ~cases.has_radiograph)]:
            if mask.any():
                subgroups[label] = classification_metrics(cases.loc[mask, 'label'], probabilities[mask], info['class_names'])
        write_json(folder / f'evaluation_{split}' / 'subgroup_metrics.json', subgroups)
        m = result['metrics']
        rows.append({'experiment': name, 'seed': seed, 'method': config.method, 'debug_run': result['debug_run'],
                     **{k: m[k] for k in ['accuracy', 'macro_f1', 'balanced_accuracy_observed_classes',
                                         'macro_auroc_defined_classes', 'brier_multiclass', 'ece_10_bins']},
                     'aurc_max_probability': float(risk.risk.mean()), 'cases': m['cases'],
                     'release_id': result['release_id'], 'release_stage': result['release_stage']})
    if not rows:
        raise ValueError('No completed runs found under RUNS_ROOT.')
    if len({r['release_id'] for r in rows}) != 1 or len({r['debug_run'] for r in rows}) != 1:
        raise ValueError('Cannot aggregate different releases or mix debug and research runs.')
    output = root / f'report_{split}'
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'per_seed.csv', index=False)
    metrics = ['accuracy', 'macro_f1', 'balanced_accuracy_observed_classes', 'macro_auroc_defined_classes', 'brier_multiclass',
               'ece_10_bins', 'aurc_max_probability']
    summary = table.groupby('experiment')[metrics].agg(['mean', 'std', 'count'])
    summary.to_csv(output / 'mean_std.csv')
    comparisons = {}
    for left, right in [('photo_attention', 'kd_full'), ('photo_attention', 'kd_selective'),
                        ('kd_full', 'kd_equal'), ('kd_equal', 'kd_selective'), ('kd_selective', 'kd_higher_h')]:
        left_seeds = {s for n, s in predictions if n == left}
        right_seeds = {s for n, s in predictions if n == right}
        if not left_seeds or left_seeds != right_seeds:
            continue  # Do not silently use an optimistic subset of seeds.
        seeds = sorted(left_seeds)
        cfg = TrainConfig(**read_json(run_dirs[(left, seeds[0])] / 'run_config.json')['config'])
        comparisons[f'{right}_minus_{left}'] = paired_bootstrap(
            [predictions[(left, s)] for s in seeds], [predictions[(right, s)] for s in seeds],
            len(read_json(run_dirs[(left, seeds[0])] / 'run_config.json')['class_names']),
            cfg.bootstrap_repetitions, cfg.seed)
    write_json(output / 'paired_comparisons.json', comparisons)
    dependence = []
    for (name, seed), teacher in predictions.items():
        if name != 'teacher_full' or ('photo_attention', seed) not in predictions:
            continue
        student = predictions[('photo_attention', seed)]
        joined = student.merge(teacher, on='case_id', suffixes=('_visual', '_teacher'), validate='one_to_one')
        if len(joined) != len(student) or len(joined) != len(teacher):
            raise ValueError('Clinical-dependency analysis requires the same cohort.')
        visual_correct = joined.prediction_visual == joined.label_visual
        teacher_correct = joined.prediction_teacher == joined.label_teacher
        dependence.append({'seed': seed, 'cases': len(joined),
            'both_correct': int((visual_correct & teacher_correct).sum()),
            'teacher_only_correct': int((~visual_correct & teacher_correct).sum()),
            'visual_only_correct': int((visual_correct & ~teacher_correct).sum()),
            'both_wrong': int((~visual_correct & ~teacher_correct).sum())})
    write_json(output / 'clinical_dependency_descriptive.json', {
        'results': dependence,
        'interpretation': 'Model disagreement, not ground truth that a case requires clinical information.'})
    write_json(output / 'status.json', {'split': split, 'debug_run': rows[0]['debug_run'],
        'release_stage': rows[0]['release_stage'], 'publication_ready': False,
        'seeds_by_experiment': {n: sorted(g.seed.tolist()) for n, g in table.groupby('experiment')},
        'note': 'Validation is model-selection evidence. One seed has undefined sample standard deviation.'})
    return table, summary, comparisons


def export_student(run_dir, destination):
    """Small, weights-only artifact without optimizer, patient IDs, clinical cache or teachers."""
    run_dir, destination = Path(run_dir), Path(destination)
    info = read_json(run_dir / 'run_config.json')
    if info['code_sha256'] != code_hash():
        raise ValueError('Export requires the implementation used for training.')
    if info['config']['method'].startswith('teacher_'):
        raise ValueError('Only a visual student can be exported.')
    if not read_json(run_dir / 'summary.json')['complete']:
        raise ValueError('Complete training before exporting.')
    state = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'schema_version': 1, 'config': state['config'], 'class_names': state['class_names'],
                'model': state['model'], 'release_id': state['release_id'],
                'code_sha256': code_hash(), 'source_checkpoint_sha256': file_hash(run_dir / 'best.pt')}, destination)
    return destination


def predict_images(artifact, image_root, photo_paths, radiograph_paths=(), device=None):
    """No labels/clinical fields in the public inference interface; paths relative to image_root."""
    state = torch.load(artifact, map_location='cpu', weights_only=True)
    if state['schema_version'] != 1 or state['code_sha256'] != code_hash():
        raise ValueError('Use the matching source distribution with this student artifact.')
    config = TrainConfig(**state['config'])
    if config.method.startswith('teacher_'):
        raise ValueError('Artifact is not a visual student.')
    if not photo_paths:
        raise ValueError('At least one photograph is required by this COde cohort.')
    record = dict(case_id='inference', patient_id='inference', split_group='inference',
                  bag_signature='inference', single_label_index=0, photo_paths=list(photo_paths),
                  radiograph_paths=list(radiograph_paths))
    item = BagDataset([record], image_root, config)[0]
    batch = collate_bags([item])
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = VisualBagClassifier(config, len(state['class_names']), load_pretrained=False).to(device).eval()
    with torch.inference_mode():
        result = model(*[batch[k].to(device) for k in ['images', 'mask', 'modalities']])
        p = result['logits'].softmax(-1)[0].cpu()
    return {'prediction': state['class_names'][int(p.argmax())],
            'probabilities': dict(zip(state['class_names'], p.tolist())),
            'bag_attention': result['attention'][0, batch['mask'][0].to(device)].cpu().tolist(),
            'note': 'Recorded-condition research classifier; attention is not lesion localization or clinical certainty.'}
