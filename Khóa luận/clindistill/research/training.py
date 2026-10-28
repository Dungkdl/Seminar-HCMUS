"""Frozen-foundation MIL experiments with group-OOF privileged distillation.

No test partition is opened here. OOF teachers train for a fixed epoch count
chosen using the independent validation partition, never their held-out fold.
"""
import csv
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..artifacts import code_hash, file_hash, object_hash, read_json, write_json
from ..engine import seed_everything, rng_state, restore_rng, save_checkpoint
from ..metrics import classification_metrics
from .data import FeatureStore, VisitDataset, collate_visits
from .losses import reliability_weights, weighted_kd, per_case_kl
from .models import VisitClassifier

METHODS = ('resnet50_mil', 'swin_b_mil', 'student_no_kd', 'kd_visual',
           'kd_multimodal', 'kd_confidence', 'kd_reliability', 'kd_feature')
PRIMARY_COMPARISONS = [('teacher_multimodal', 'teacher_visual'),
    ('kd_reliability', 'student_no_kd'), ('kd_reliability', 'kd_multimodal'),
    ('kd_reliability', 'kd_confidence')]


def validate_config(config):
    for name in ('folds', 'epochs', 'patience', 'batch_size', 'accumulation_steps', 'feature_dim'):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if config['folds'] < 2 or config['encoder_training'] != 'frozen_features':
        raise ValueError('Need >=2 group folds and the frozen_features protocol.')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Need distinct experiment seeds.')
    if not config['methods'] or not set(config['methods']) <= set(METHODS):
        raise ValueError('Unknown research method.')
    if len(set(config['methods'])) != len(config['methods']):
        raise ValueError('Duplicate research methods.')
    for name in ('temperature', 'gate_temperature', 'learning_rate'):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'Invalid {name}.')
    if not 0 < config['visual_support_epsilon'] <= 1 or not 0 <= config['dropout'] < 1:
        raise ValueError('Invalid epsilon/dropout.')
    if any(not math.isfinite(config[x]) or config[x] < 0 for x in ('kd_weight', 'weight_decay', 'feature_kd_weight')):
        raise ValueError('Invalid loss or optimizer coefficient.')
    for key, expected in {'student_model': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                          'teacher_model': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
                          'clinical_model': 'BAAI/bge-m3'}.items():
        if config[key] != expected:
            raise ValueError('This protocol supports the registered DINOv3-B/L and BGE-M3 model IDs only.')


def grouped_folds(labels, groups, folds, seed):
    labels, groups = np.asarray(labels), np.asarray(groups)
    if len(np.unique(groups)) < folds:
        raise ValueError('Too few independent patient/duplicate groups for OOF folds.')
    for label in np.unique(labels):
        if len(np.unique(groups[labels == label])) < folds:
            raise ValueError('A class has fewer independent groups than OOF folds; reduce folds explicitly.')
    coverage = np.zeros(len(labels), int)
    plan = []
    for training, heldout in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(labels, labels, groups):
        if set(groups[training]) & set(groups[heldout]):
            raise ValueError('OOF patient group leakage.')
        if set(np.unique(labels[training])) != set(np.unique(labels)):
            raise ValueError('OOF teacher training fold lacks a class.')
        coverage[heldout] += 1
        plan.append((training, heldout))
    if not np.all(coverage == 1):
        raise ValueError('Every train visit must have exactly one held-out prediction.')
    return plan


def model_spec(store, config, name):
    role = ('multimodal_teacher' if name == 'teacher_multimodal' else
            'visual_teacher' if name == 'teacher_visual' else 'student')
    key = ('teacher' if name.startswith('teacher_') else
           name.removesuffix('_mil') if name in {'resnet50_mil', 'swin_b_mil'} else 'student')
    return {'visual_dim': store.metadata['dimensions'][key], 'classes': len(store.metadata['class_names']),
            'width': config['feature_dim'], 'dropout': config['dropout'], 'role': role,
            'clinical_dim': store.metadata['dimensions']['clinical'] if role == 'multimodal_teacher' else 0,
            'fusion': config.get('fusion', 'gated'),
            'alignment_dim': store.metadata['dimensions']['teacher'] if name == 'kd_feature' else 0}, key


def forward_batch(model, batch, device):
    args = {key: batch[key].to(device) for key in ('clinical', 'demographics', 'clinical_present') if key in batch}
    return model(batch['visual'].to(device), batch['mask'].to(device), **args)


def loader(dataset, config, *, shuffle=False, epoch=0, seed=0):
    return DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_visits,
                      shuffle=shuffle, num_workers=0, generator=torch.Generator().manual_seed(seed + epoch))


def predict(model, dataset, config, device):
    model.eval()
    values, ids = [], []
    with torch.inference_mode():
        for batch in loader(dataset, config):
            values.append(forward_batch(model, batch, device)['logits'].float().cpu().numpy())
            ids.extend(batch['ids'])
    return np.concatenate(values), ids


def probabilities(logits):
    return torch.from_numpy(np.asarray(logits)).float().softmax(-1).numpy()


def write_predictions(path, data, logits, *, indices=None):
    ix = np.arange(len(data['ids'])) if indices is None else np.asarray(indices)
    p = probabilities(logits)
    frame = pd.DataFrame({'case_id': data['ids'][ix], 'patient_id': data['patients'][ix],
        'split_group': data['groups'][ix], 'label': data['labels'][ix], 'prediction': p.argmax(-1)})
    for k in range(p.shape[1]):
        frame[f'p_{k}'] = p[:, k]
    frame.to_csv(path, index=False)


def fit(store, directory, config, name, seed, *, indices=None, fixed_epochs=None, targets=None,
        target_identity=None, stop_after_epoch=None):
    """Full-fit uses validation selection; OOF fit has no validation dataset at all."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    spec, key = model_spec(store, config, name)
    train_data = store.data['train']
    indices = np.arange(len(train_data['ids'])) if indices is None else np.asarray(indices)
    protocol = {'source': code_hash(), 'features': store.identity, 'config': config, 'name': name,
                'seed': seed, 'training_ids': train_data['ids'][indices].tolist(),
                'fixed_epochs': fixed_epochs, 'targets': target_identity, 'spec': spec}
    identity = object_hash(protocol)
    last_path, best_path = directory / 'last.pt', directory / 'best.pt'
    if (directory / 'protocol.json').exists():
        if read_json(directory / 'protocol.json')['identity'] != identity:
            raise ValueError('Run fingerprint changed; use a new study directory.')
    elif any(directory.iterdir()):
        raise ValueError('Nonempty run has no protocol. Use a new directory.')
    write_json(directory / 'protocol.json', {'identity': identity, **protocol})
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisitClassifier(**spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    epochs = fixed_epochs if fixed_epochs is not None else config['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start, best, bad, history = 0, -1.0, 0, []
    if last_path.exists():
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Checkpoint fingerprint mismatch.')
        if state['complete']:
            if not best_path.exists():
                raise ValueError('Completed run is missing its selected checkpoint.')
            result = read_json(directory / 'result.json')
            if result['identity'] != identity or file_hash(best_path) != result['checkpoint_sha256']:
                raise ValueError('Completed checkpoint/result integrity mismatch.')
            return result
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        # Move restored Adam moments with the model when resuming on CUDA.
        for group in optimizer.state.values():
            for k, v in group.items():
                if torch.is_tensor(v):
                    group[k] = v if k == 'step' else v.to(device)
        scheduler.load_state_dict(state['scheduler']); restore_rng(state['rng'])
        start, best, bad, history = state['epoch'], state['best'], state['bad'], state['history']
    training = VisitDataset(store, 'train', key, role=spec['role'], indices=indices)
    validation = VisitDataset(store, 'validation', key, role=spec['role']) if fixed_epochs is None else None
    classes = store.metadata['class_names']
    for epoch in range(start, epochs):
        tick = time.monotonic(); model.train()
        batches = loader(training, config, shuffle=True, epoch=epoch, seed=seed)
        optimizer.zero_grad(set_to_none=True)
        total, count, running_kd = 0., 0, 0.
        for step, batch in enumerate(batches):
            out = forward_batch(model, batch, device)
            labels = batch['labels'].to(device)
            ce = F.cross_entropy(out['logits'], labels)
            kd = ce.new_zeros(())
            if name.startswith('kd_'):
                if targets is None:
                    raise ValueError('KD requires verified train-only OOF targets.')
                ix = batch['indices'].numpy()
                t = torch.from_numpy(targets['visual_logits' if name == 'kd_visual' else 'multimodal_logits'][ix]).to(device)
                if name in {'kd_confidence', 'kd_reliability', 'kd_feature'}:
                    weight_key = 'confidence' if name == 'kd_confidence' else 'reliability'
                    raw = torch.from_numpy(targets[weight_key][ix]).to(device)
                    kd = weighted_kd(out['logits'], t, raw, float(targets[weight_key].mean()), config['temperature'])
                else:
                    kd = per_case_kl(out['logits'], t, config['temperature']).mean()
                if name == 'kd_feature':
                    teacher_vectors = []
                    for i in ix:
                        photos = train_data['image_indices'][train_data['offsets'][i]:train_data['offsets'][i + 1]]
                        teacher_vectors.append(train_data['teacher'][photos].astype(np.float32).mean(0))
                    frozen_descriptor = torch.from_numpy(np.stack(teacher_vectors)).to(device)
                    per_case = 1 - F.cosine_similarity(out['alignment'], frozen_descriptor, dim=-1)
                    kd = kd + config['feature_kd_weight'] * (per_case * raw / float(targets['reliability'].mean())).mean()
            loss = ce + config['kd_weight'] * kd
            # Normalize by the actual examples in this accumulation window (including final remainder).
            window_start = step // config['accumulation_steps'] * config['accumulation_steps']
            window_cases = min(config['accumulation_steps'] * config['batch_size'],
                               len(training) - window_start * config['batch_size'])
            n = len(labels)
            (loss * n / window_cases).backward()
            if (step + 1) % config['accumulation_steps'] == 0 or step + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * n; running_kd += float(kd.detach()) * n; count += n
        scheduler.step()
        metrics = None
        if validation is not None:
            logits, _ = predict(model, validation, config, device)
            metrics = classification_metrics(store.data['validation']['labels'], probabilities(logits), classes)
            score = metrics['macro_f1']
            improved = score > best
        else:
            score, improved = float(epoch + 1), True  # Fixed-epoch OOF: final checkpoint, no held-out selection.
        if improved:
            best, bad = score, 0
            save_checkpoint(best_path, {'identity': identity, 'model': model.state_dict(), 'spec': spec,
                'visual_key': key, 'epoch': epoch + 1, 'metrics': metrics, 'feature_identity': store.identity})
        else:
            bad += 1
        row = {'epoch': epoch + 1, 'loss': total / count, 'kd_loss': running_kd / count,
               'validation_macro_f1': metrics['macro_f1'] if metrics else None,
               'seconds': time.monotonic() - tick, 'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(row); pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        complete = epoch + 1 == epochs or (validation is not None and bad >= config['patience'])
        if complete:
            chosen = torch.load(best_path, map_location='cpu', weights_only=False)
            result = {'name': name, 'seed': seed, 'best_epoch': chosen['epoch'],
                      'metrics': chosen['metrics'], 'identity': identity, 'fixed_epochs': fixed_epochs,
                      'checkpoint_sha256': file_hash(best_path), 'epochs_run': epoch + 1}
            write_json(directory / 'result.json', result)
        save_checkpoint(last_path, {'identity': identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'epoch': epoch + 1, 'best': best, 'bad': bad,
            'history': history, 'rng': rng_state(), 'complete': complete})
        print(f'{name} seed={seed} epoch={epoch+1}/{epochs} loss={total/count:.4f}' +
              (f' val_macro_f1={score:.4f}' if metrics else ' [OOF fixed epochs]'), flush=True)
        if complete:
            break
        if stop_after_epoch and epoch + 1 >= stop_after_epoch:
            return {'interrupted': True, 'identity': identity}
    return read_json(directory / 'result.json')


def load_model(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = VisitClassifier(**state['spec']).to(device)
    model.load_state_dict(state['model'], strict=True)
    return model.eval(), state


def build_oof(store, root, config, seed, teachers):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    data = store.data['train']; folds = grouped_folds(data['labels'], data['groups'], config['folds'], seed)
    request = {'source': code_hash(), 'features': store.identity, 'config': config, 'seed': seed,
               'teachers': teachers, 'ids': data['ids'].tolist()}
    identity = object_hash(request)
    manifest_path, path = root / 'targets.json', root / 'targets.npz'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['identity'] != identity or file_hash(path) != manifest['sha256']:
            raise ValueError('OOF targets changed or belong to another experiment.')
        with np.load(path, allow_pickle=False) as values:
            targets = {k: values[k].copy() for k in values.files}
        if not np.array_equal(targets['ids'], data['ids']):
            raise ValueError('OOF target/visit alignment mismatch.')
        return targets, manifest
    n, k = len(data['ids']), len(store.metadata['class_names'])
    values = {'visual_logits': np.empty((n, k), np.float32), 'multimodal_logits': np.empty((n, k), np.float32),
              'fold': np.full(n, -1, int), 'ids': data['ids']}
    proof = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold, (training, heldout) in enumerate(folds):
        proof.append({'fold': fold, 'train_groups': sorted(set(map(str, data['groups'][training]))),
                      'heldout_groups': sorted(set(map(str, data['groups'][heldout]))),
                      'train_ids': data['ids'][training].tolist(), 'heldout_ids': data['ids'][heldout].tolist()})
        values['fold'][heldout] = fold
        for role, key in [('visual', 'visual_logits'), ('multimodal', 'multimodal_logits')]:
            name = f'teacher_{role}'; destination = root / f'fold_{fold}' / name
            fit(store, destination, config, name, seed + 1000 + fold * 17 + (role == 'multimodal'),
                indices=training, fixed_epochs=teachers[name]['best_epoch'])
            model, state = load_model(destination / 'best.pt', device)
            dataset = VisitDataset(store, 'train', state['visual_key'], role=state['spec']['role'], indices=heldout)
            logits, ids = predict(model, dataset, config, device)
            if ids != data['ids'][heldout].tolist():
                raise ValueError('OOF prediction order changed.')
            values[key][heldout] = logits
            proof[-1][name] = {'checkpoint_sha256': file_hash(destination / 'best.pt'),
                               'fixed_epochs': teachers[name]['best_epoch']}
            del model
    weights = reliability_weights(torch.from_numpy(values['visual_logits']),
        torch.from_numpy(values['multimodal_logits']), torch.from_numpy(data['labels']),
        epsilon=config['visual_support_epsilon'], gate_temperature=config['gate_temperature'])
    values.update({key: value.cpu().numpy() for key, value in weights.items()})
    for key in ('confidence', 'reliability'):
        if key == 'reliability' and not any(m in config['methods'] for m in ('kd_reliability', 'kd_feature')):
            continue
        if values[key].mean() <= 1e-12:
            raise ValueError(f'OOF {key} weights collapse to zero. Inspect Teacher quality; no silent CE fallback.')
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)
    stats = {key: {'mean_train': float(values[key].mean()), 'zero_fraction': float((values[key] <= 1e-12).mean()),
             'mean_by_class': {name: float(values[key][data['labels'] == j].mean())
                for j, name in enumerate(store.metadata['class_names'])}}
             for key in ('confidence', 'reliability')}
    manifest = {'identity': identity, 'request': request, 'sha256': file_hash(path), 'folds': proof,
                'weight_statistics': stats, 'weight_normalization': 'fixed_global_OOF_train_mean',
                'OOF_selection': 'fixed epochs selected on independent validation; no fold-heldout selection'}
    write_json(manifest_path, manifest)
    return values, manifest


def study_plan(config):
    main = len(config['seeds']) * (2 + len(config['methods']))
    kd = any(x.startswith('kd_') for x in config['methods'])
    oof = len(config['seeds']) * 2 * config['folds'] if kd else 0
    return {'main_fits': main, 'OOF_fits': oof, 'total_fits': main + oof,
            'encoder_training': 'frozen_features', 'seeds': config['seeds'],
            'systems': ['teacher_visual', 'teacher_multimodal'] + config['methods'],
            'primary_comparisons': PRIMARY_COMPARISONS, 'test_opened': False}


def run_study(feature_root, study_root, config, *, require_gpu=True):
    validate_config(config)
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU for the full study. CPU is permitted only for explicit debug runs.')
    store = FeatureStore(feature_root, allow_debug=bool(config.get('debug')))
    if store.metadata['debug'] != bool(config.get('debug')):
        raise ValueError('Research/debug cache and training mode disagree.')
    policy = store.metadata['request']['policy']
    if policy['fields'] != config['clinical_fields'] or policy['language'] != config['language']:
        raise ValueError('Clinical cache policy does not match the training protocol.')
    if policy['temporal_verified'] != config['clinical_temporal_verified']:
        raise ValueError('Clinical review status does not match the training protocol.')
    for key in ('image_size', 'max_photos', 'encoder_training'):
        if store.metadata['request'].get(key) != config[key]:
            raise ValueError(f'Feature preparation/train protocol mismatch: {key}.')
    if store.metadata['request'].get('model_ids') != {k: config[k + '_model'] for k in ('student', 'teacher', 'clinical')}:
        raise ValueError('Feature encoder model IDs do not match the study.')
    if set(store.data['train']['labels']) != set(range(len(store.metadata['class_names']))):
        raise ValueError('Training cohort is missing a registered class.')
    root = Path(study_root); root.mkdir(parents=True, exist_ok=True)
    protocol = {'source': code_hash(), 'features': store.identity, 'config': config, 'plan': study_plan(config),
                'feature_root_hint': str(Path(feature_root).resolve()), 'clinical_policy': policy,
                'release_stage': store.metadata['release_stage'], 'debug': store.metadata['debug']}
    # Paths are hints and may change when attaching Kaggle output in the next session.
    identity = object_hash({k: v for k, v in protocol.items() if k != 'feature_root_hint'})
    if (root / 'study.json').exists() and read_json(root / 'study.json')['identity'] != identity:
        raise ValueError('Study inputs/config/source changed. Choose a new run_name; no checkpoint mixing.')
    write_json(root / 'study.json', {'identity': identity, **protocol})
    print(study_plan(config), flush=True)
    results = []
    for seed in config['seeds']:
        teachers = {}
        for name in ('teacher_visual', 'teacher_multimodal'):
            directory = root / name / f'seed_{seed}'
            teachers[name] = fit(store, directory, config, name, seed)
            results.append(teachers[name])
        targets, manifest = (build_oof(store, root / '_oof' / f'seed_{seed}', config, seed, teachers)
                             if any(m.startswith('kd_') for m in config['methods']) else (None, None))
        for name in config['methods']:
            result = fit(store, root / name / f'seed_{seed}', config, name, seed,
                targets=targets if name.startswith('kd_') else None,
                target_identity=manifest['identity'] if name.startswith('kd_') else None)
            results.append(result)
        # Persist predictions after each completed seed, useful when a Kaggle session expires.
        for result in results:
            directory = root / result['name'] / f"seed_{result['seed']}"
            if not (directory / 'validation_predictions.csv').exists():
                model, state = load_model(directory / 'best.pt')
                logits, _ = predict(model, VisitDataset(store, 'validation', state['visual_key'],
                    role=state['spec']['role']), config, torch.device('cpu'))
                write_predictions(directory / 'validation_predictions.csv', store.data['validation'], logits)
    rows = [{'method': r['name'], 'seed': r['seed'], 'best_epoch': r['best_epoch'],
             **{k: r['metrics'][k] for k in ('macro_f1', 'accuracy', 'macro_auroc_defined_classes', 'brier_multiclass', 'ece_10_bins')}} for r in results]
    frame = pd.DataFrame(rows); frame.to_csv(root / 'validation_results.csv', index=False)
    summary = frame.groupby('method').agg(macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), seeds=('seed', 'count')).reset_index()
    summary.to_csv(root / 'validation_summary.csv', index=False)
    write_json(root / 'completed.json', {'identity': identity, 'results': results, 'test_opened': False})
    return summary
