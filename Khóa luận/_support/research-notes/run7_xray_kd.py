"""Matched image-only Student losses with patient-group OOF multimodal targets."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from clindistill.artifacts import file_hash, write_json
from clindistill.metrics import classification_metrics
from clindistill.reporting import paired_bootstrap
from clindistill.research.training import dataset_for, grouped_folds, predict, probabilities

from run6_xray_fusion import TwoBranchMean, branch_store, fit_branches, prepare_paired_inputs

LOSSES = {'ce': None, 'kd_vanilla': 'vanilla', 'kd_correctness': 'correctness'}


class MultimodalTeacher(TwoBranchMean):
    role = 'multimodal_teacher'

    def __init__(self, dimension, width, dropout, classes, clinical_dim):
        super().__init__(dimension, width, dropout, classes, use_xray=True)
        self.clinical = nn.Sequential(nn.Linear(clinical_dim + 5, 2 * width), nn.LayerNorm(2 * width), nn.GELU())
        self.gate = nn.Linear(4 * width, 2 * width)
        self.fusion_norm = nn.LayerNorm(2 * width)

    def forward(self, visual, mask, *, clinical, demographics, clinical_present):
        image = self.pooled_features(visual, mask)
        present = clinical_present.unsqueeze(-1)
        context = self.clinical(torch.cat((clinical, demographics), dim=-1)) * present
        gate = self.gate(torch.cat((image, context), dim=-1)).sigmoid()
        fused = self.fusion_norm(image + gate * context)
        return {'logits': self.head(fused)}


def add_clinical_context(store, source):
    """Join cached clinical context to the already release-checked, paired visits."""
    for split, data in store.data.items():
        cached = source.data[split]
        by_id = {case: i for i, case in enumerate(cached['ids'])}
        indices = [by_id[case] for case in data['ids']]
        for key in ('clinical', 'demographics', 'clinical_present'):
            data[key] = cached[key][indices]


def teacher_targets(store, out, config, seed, device, teacher_factory, plan, best_epoch):
    """Fit each teacher without any held-out patient group or validation selection."""
    train = store.data['train']
    logits = np.empty((len(train['ids']), len(store.metadata['class_names'])), dtype=np.float32)
    assigned_fold = np.full(len(train['ids']), -1, dtype=np.int64)
    proof = []
    for fold, (training, heldout) in enumerate(plan):
        directory = out / f'fold_{fold}'
        model, _, _ = fit_branches(store, directory, config, seed + 1000 + fold, True, device,
                                    teacher_factory=teacher_factory, indices=training, fixed_epochs=best_epoch)
        ds = dataset_for(store, 'train', 'student', model.role, 'student_mean', seed, heldout)
        prediction, ids = predict(model, ds, config, device)
        if ids != train['ids'][heldout].tolist() or np.any(assigned_fold[heldout] != -1):
            raise ValueError('OOF target order or one-prediction-per-visit contract violated.')
        logits[heldout], assigned_fold[heldout] = prediction, fold
        proof.append({'fold': fold, 'train_groups': len(set(train['groups'][training])),
                      'heldout_groups': len(set(train['groups'][heldout])), 'heldout_cases': len(heldout),
                      'checkpoint_sha256': file_hash(directory / 'best.pt')})
    if (assigned_fold < 0).any() or not np.isfinite(logits).all():
        raise ValueError('OOF targets are incomplete or non-finite.')
    correct = (logits.argmax(1) == train['labels']).astype(np.float32)
    mean = float(correct.mean())
    if mean <= 0:
        raise ValueError('No correct OOF Teacher targets. Correctness gate is undefined.')
    cache = out / 'targets.npz'
    # Write deterministic bytes for resumed Student identity even when rebuilt from folds.
    tmp = cache.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, logits=logits, correct=correct, ids=train['ids'], fold=assigned_fold)
    tmp.replace(cache)
    sha = file_hash(cache)
    write_json(out / 'targets.json', {'sha256': sha, 'folds': proof, 'cases': len(correct),
        'mean_correct': mean, 'best_epoch_from_full_teacher': best_epoch,
        'correct_by_class': {name: {'cases': int(np.sum(train['labels'] == i)),
                                   'correct': int(correct[train['labels'] == i].sum())}
                             for i, name in enumerate(store.metadata['class_names'])}})
    return {'logits': logits, 'correct': correct, 'mean_correct': mean}, sha


def run_kd(prepared_root, feature_root, image_root, encoder_root, output, config, *, image_batch=32):
    output = Path(output)
    source, records, xrays, early, protocol = prepare_paired_inputs(
        prepared_root, feature_root, image_root, encoder_root, output, config, image_batch=image_batch,
        run_name='code_run7_xray_kd_v1', protocol_extra={
            'student_losses': LOSSES, 'teacher': 'Run 6 two branches plus cached clinical gated residual',
            'run7_sha256': file_hash(__file__)})
    config = protocol['config']
    store = branch_store(early, records, xrays, protocol, 'photo_xray')
    del early, xrays
    add_clinical_context(store, source)
    classes = store.metadata['class_names']
    train, val = store.data['train'], store.data['validation']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plans = {seed: grouped_folds(train['labels'], train['groups'], config['folds'], seed) for seed in config['seeds']}
    print(f'Run 7: {len(config["seeds"])*(4+config["folds"])} fits; test remains closed.', flush=True)

    def teacher_factory(d, cfg, n):
        return MultimodalTeacher(d, cfg['feature_dim'], cfg['dropout'], n, train['clinical'].shape[-1])

    rows, frames, details = [], {}, {}
    for seed in config['seeds']:
        directory = output / 'teacher_multimodal' / f'seed_{seed}'
        model, ds, best_epoch = fit_branches(store, directory, config, seed, True, device,
                                              teacher_factory=teacher_factory)
        logits, ids = predict(model, ds, config, device)
        if ids != val['ids'].tolist():
            raise ValueError('Teacher validation order differs from paired cohort.')
        details[f'teacher_multimodal/{seed}'] = classification_metrics(val['labels'], probabilities(logits), classes)
        targets, sha = teacher_targets(store, output / '_oof' / f'seed_{seed}', config, seed,
                                       device, teacher_factory, plans[seed], best_epoch)
        for arm, mode in LOSSES.items():
            model, ds, epoch = fit_branches(store, output / arm / f'seed_{seed}', config, seed, True, device,
                                             kd_mode=mode, targets=targets if mode else None,
                                             target_sha256=sha if mode else None)
            logits, ids = predict(model, ds, config, device)
            if ids != val['ids'].tolist():
                raise ValueError('Student validation order differs from paired cohort.')
            prob = probabilities(logits)
            result = classification_metrics(val['labels'], prob, classes)
            details[f'{arm}/{seed}'] = result
            rows.append({'method': arm, 'seed': seed, 'best_epoch': epoch, 'cases': len(ids),
                         'accuracy': result['accuracy'], 'macro_f1': result['macro_f1']})
            frames[arm, seed] = pd.DataFrame({'case_id': ids, 'patient_id': val['patients'],
                'split_group': val['groups'], 'label': val['labels'], 'prediction': prob.argmax(1)})
    comparisons = {}
    for reference, candidate in (('ce','kd_vanilla'), ('ce','kd_correctness'), ('kd_vanilla','kd_correctness')):
        a, b = [[frames[arm, seed] for seed in config['seeds']] for arm in (reference, candidate)]
        comparisons[f'{candidate}_minus_{reference}'] = paired_bootstrap(
            a, b, len(classes), config['bootstrap_repetitions'], config['seeds'][0])
    table = pd.DataFrame(rows)
    summary = table.groupby('method')[['macro_f1', 'accuracy']].agg(['mean', 'std', 'count'])
    table.to_csv(output / 'per_seed.csv', index=False)
    summary.to_csv(output / 'mean_std.csv')
    write_json(output / 'metrics.json', details)
    write_json(output / 'paired_comparisons.json', comparisons)
    write_json(output / 'completed.json', {'student_fits': len(table),
        'total_fits': len(config['seeds'])*(4+config['folds']), 'test_opened': False})
    return summary, comparisons
