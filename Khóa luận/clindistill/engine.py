"""Training and held-out evaluation are separate entry points."""
import csv
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import code_hash, file_hash, object_hash, read_json, verify_release, write_json
from .config import TrainConfig
from .data import BagDataset, check_disjoint, collate_bags, load_records
from .metrics import classification_metrics, group_bootstrap
from .models import VisualBagClassifier
from .distillation import TrainingRuntime
from .clinical import ClinicalCache


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def make_loader(dataset, config, training=False, epoch=0):
    generator = torch.Generator().manual_seed(config.seed + epoch)
    # Spawn workers in interactive Windows notebooks is fragile; Kaggle/Linux uses configured workers.
    workers = 0 if os.name == 'nt' else config.workers
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training,
                      num_workers=workers, collate_fn=collate_bags,
                      generator=generator, persistent_workers=False,
                      pin_memory=torch.cuda.is_available())


def predict(model, loader, device, class_names, runtime=None):
    model.eval()
    probabilities, labels, metadata = [], [], []
    with torch.inference_mode():
        for batch in loader:
            output = (runtime.forward(batch) if runtime else
                      model(batch['images'].to(device), batch['mask'].to(device), batch['modalities'].to(device)))
            probabilities.extend(output['logits'].float().softmax(-1).cpu().tolist())
            labels.extend(batch['labels'].tolist())
            for i in range(len(batch['labels'])):
                metadata.append({key: batch[key][i] for key in ['case_id', 'patient_id', 'split_group', 'bag_signature']})
    metrics = classification_metrics(labels, probabilities, class_names)
    return metrics, np.asarray(labels), np.asarray(probabilities), metadata


def write_predictions(path, labels, probabilities, metadata, class_names):
    names = ['case_id', 'patient_id', 'split_group', 'bag_signature', 'label', 'prediction'] + [f'p_{i}' for i in range(len(class_names))]
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for y, p, meta in zip(labels, probabilities, metadata):
            writer.writerow({**meta, 'label': int(y), 'prediction': int(p.argmax()),
                             **{f'p_{i}': float(v) for i, v in enumerate(p)}})


def fit(release_dir, image_root, run_dir, config, *, resume=False, device=None, stop_after_epoch=None,
        clinical_cache=None, teacher_dirs=None):
    """Only train/validation are loaded. stop_after_epoch simulates an epoch-boundary interruption."""
    release_dir, image_root, run_dir = Path(release_dir), Path(image_root), Path(run_dir)
    release = verify_release(release_dir, splits=['train', 'validation'])
    config = config if isinstance(config, TrainConfig) else TrainConfig(**config)
    classes = release['class_names']
    train = load_records(release_dir, 'train', config.experiment, config.debug_max_cases)
    validation = load_records(release_dir, 'validation', config.experiment, config.debug_max_cases)
    check_disjoint(train, validation)
    counts = np.bincount([r['single_label_index'] for r in train], minlength=len(classes))
    if (counts == 0).any() and not config.debug_max_cases:
        raise ValueError('Training cohort lacks a class; return to data preparation.')
    seed_everything(config.seed)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    runtime = TrainingRuntime(config, release, device, clinical_cache=clinical_cache,
                              teacher_dirs=teacher_dirs, resume=resume)
    identity = {'config': config.to_dict(), 'release_id': release['release_id'], 'code_sha256': code_hash(),
                'dependencies': runtime.dependencies}
    if config.backbone_weights_path:
        identity['backbone_weights_sha256'] = file_hash(config.backbone_weights_path)
    fingerprint = object_hash(identity)
    checkpoint_path = run_dir / 'last.pt'
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError('Resume requires last.pt from this training pipeline.')
        if read_json(run_dir / 'run_config.json')['run_fingerprint'] != fingerprint:
            raise ValueError('Config, data release or implementation changed. Start a new run.')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Run directory already has artifacts. Use resume=True or a new run_dir.')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'Run={run_dir.name} | {device} | train={len(train)} validation={len(validation)} | release={release["stage"]}', flush=True)
    model = runtime.model
    # Teacher construction must not change student's dropout/augmentation RNG stream.
    seed_everything(config.seed)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_amp = config.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    weights = torch.tensor(len(train) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights if config.class_weighted_loss else None)
    train_dataset = BagDataset(train, image_root, config, training=True)
    val_dataset = BagDataset(validation, image_root, config)
    val_loader = make_loader(val_dataset, config)
    start_epoch, best, bad_epochs, history = 0, -1.0, 0, []
    if resume:
        # Trusted local checkpoints contain optimizer/RNG objects. Never load untrusted .pt files here.
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state['run_fingerprint'] != fingerprint:
            raise ValueError('Checkpoint identity mismatch.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
        start_epoch, best, bad_epochs, history = state['epoch'] + 1, state['best'], state['bad_epochs'], state['history']
        if (run_dir / 'summary.json').exists() and read_json(run_dir / 'summary.json').get('complete'):
            print('This run is already complete; returning its stored summary.')
            return read_json(run_dir / 'summary.json')
    else:
        write_json(run_dir / 'run_config.json', {**identity, 'run_fingerprint': fingerprint,
            'release_stage': release['stage'], 'class_names': classes, 'actual_train_class_counts': counts.tolist(),
            'train_cases': [r['case_id'] for r in train], 'validation_cases': [r['case_id'] for r in validation],
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'cuda_runtime': torch.version.cuda, 'device': str(device),
                            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                            'vram_gib': torch.cuda.get_device_properties(device).total_memory / 2**30 if device.type == 'cuda' else None,
                            'numpy': np.__version__, 'torchvision': version('torchvision'),
                            'pillow': version('Pillow'), 'scikit_learn': version('scikit-learn')},
            'clinical_cache_path': str(clinical_cache) if clinical_cache else None,
            'backbone_weights_sha256': file_hash(config.backbone_weights_path) if config.backbone_weights_path else None})
        write_json(run_dir / 'readiness.json', {'data_release_stage': release['stage'], 'pending_reviews': release['pending_reviews'],
            'research_result': not bool(config.debug_max_cases or config.backbone == 'tiny_debug'),
            'publication_ready': False})
    session_start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    interrupted = False
    for epoch in range(start_epoch, config.epochs):
        if bad_epochs >= config.patience:
            break
        train_dataset.set_epoch(epoch)
        epoch_start = time.perf_counter()
        loader = make_loader(train_dataset, config, training=True, epoch=epoch)
        model.train()
        loss_sum, ce_sum, kd_sum, seen = 0.0, 0.0, 0.0, 0
        current_lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            count = len(batch['labels'])
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = runtime.forward(batch)
                ce = criterion(output['logits'], batch['labels'].to(device))
                kd = runtime.distillation_loss(batch, output)
                loss = ce + kd
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss. Last complete epoch remains resumable.')
            window_start = (step // config.accumulation_steps) * config.accumulation_steps * config.batch_size
            window_cases = min(config.accumulation_steps * config.batch_size, len(train_dataset) - window_start)
            scaler.scale(loss * (count / window_cases)).backward()
            if (step + 1) % config.accumulation_steps == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * count
            ce_sum += float(ce.detach()) * count
            kd_sum += float(kd.detach()) * count
            seen += count
        val_metrics, y, probabilities, metadata = predict(model, val_loader, device, classes, runtime)
        improved = val_metrics['macro_f1'] > best
        bad_epochs = 0 if improved else bad_epochs + 1
        best = max(best, val_metrics['macro_f1'])
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / seen, 'learning_rate': current_lr,
               'train_ce': ce_sum / seen, 'train_kd': kd_sum / seen,
               'epoch_seconds': time.perf_counter() - epoch_start,
               'validation_macro_f1': val_metrics['macro_f1'], 'validation_accuracy': val_metrics['accuracy']}
        history.append(row)
        scheduler.step()
        payload = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(),
                   'best': best, 'bad_epochs': bad_epochs, 'history': history,
                   'run_fingerprint': fingerprint, 'release_id': release['release_id'],
                   'config': config.to_dict(), 'class_names': classes}
        if improved:
            # Evaluation/teacher checkpoints need no optimizer/RNG; keep those only in last.pt.
            save_checkpoint(run_dir / 'best.pt', {key: payload[key] for key in
                ['epoch', 'model', 'best', 'run_fingerprint', 'release_id', 'config', 'class_names']})
            write_json(run_dir / 'best_validation_metrics.json', val_metrics)
            write_predictions(run_dir / 'validation_predictions.csv', y, probabilities, metadata, classes)
        save_checkpoint(checkpoint_path, payload)
        write_json(run_dir / 'history.json', history)
        print(f'Epoch {epoch+1}/{config.epochs} loss={row["train_loss"]:.4f} '
              f'val_macro_f1={val_metrics["macro_f1"]:.4f} best={best:.4f}', flush=True)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            interrupted = True
            break
    summary = {'complete': not interrupted, 'epochs_completed': len(history), 'best_validation_macro_f1': best,
               'run_fingerprint': fingerprint, 'release_id': release['release_id'], 'release_stage': release['stage'],
               'test_evaluated': False, 'session_seconds': time.perf_counter() - session_start,
               'debug_run': bool(config.debug_max_cases or config.backbone == 'tiny_debug'),
               'peak_gpu_memory_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
               'model_parameters': sum(p.numel() for p in model.parameters()), 'checkpoint': 'best.pt'}
    write_json(run_dir / 'summary.json', summary)
    return summary


def evaluate_checkpoint(release_dir, image_root, run_dir, *, split='test', allow_experimental_test=False,
                        device=None, clinical_cache=None):
    """Explicit held-out evaluation; never called by fit()."""
    if split not in {'validation', 'test'}:
        raise ValueError('Evaluation partition must be validation or test.')
    release = verify_release(release_dir, splits=[split])
    run_dir = Path(run_dir)
    if split == 'test' and release['stage'] != 'reviewed' and not allow_experimental_test:
        raise ValueError('This data release still has pending reviews. Test is disabled by default; '
                         'experimental evaluation must be explicit and cannot be reported as final evidence.')
    if not read_json(run_dir / 'summary.json')['complete']:
        raise ValueError('Training is incomplete; resume it before held-out evaluation.')
    checkpoint_path = run_dir / 'best.pt'
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if state['release_id'] != release['release_id']:
        raise ValueError('Checkpoint and prepared data release differ.')
    config = TrainConfig(**state['config'])
    run_info = read_json(run_dir / 'run_config.json')
    if run_info['code_sha256'] != code_hash():
        raise ValueError('Implementation differs from training; use the saved code version.')
    records = load_records(release_dir, split, config.experiment, config.debug_max_cases)
    trained = set(run_info['train_cases'])
    if trained & {r['case_id'] for r in records}:
        raise ValueError('Evaluation overlaps training cases.')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if config.method.startswith('teacher_') and clinical_cache is None:
        clinical_cache = run_info['clinical_cache_path']
    runtime = TrainingRuntime(config, release, device, clinical_cache=clinical_cache,
                              resume=True, splits=(split,), training=False)
    if config.method.startswith('teacher_'):
        # A test shard may be added later; the frozen encoder must remain identical.
        trained_cache = ClinicalCache(run_info['clinical_cache_path'] if clinical_cache is None else clinical_cache,
                                      release['release_id'], allow_debug=config.backbone == 'tiny_debug')
        if trained_cache.fingerprint != run_info['dependencies']['clinical_cache']:
            raise ValueError('Evaluation clinical encoder/cache differs from teacher training.')
    model = runtime.model
    model.load_state_dict(state['model'])
    loader = make_loader(BagDataset(records, image_root, config), config)
    metrics, labels, probabilities, metadata = predict(model, loader, device, release['class_names'], runtime)
    intervals = group_bootstrap(labels, probabilities, [m['split_group'] for m in metadata],
                                config.bootstrap_repetitions, config.seed)
    output = run_dir / f'evaluation_{split}'
    output.mkdir(exist_ok=True)
    write_predictions(output / 'predictions.csv', labels, probabilities, metadata, release['class_names'])
    result = {'split': split, 'release_stage': release['stage'], 'release_id': release['release_id'],
              'checkpoint_sha256': file_hash(checkpoint_path), 'debug_run': bool(config.debug_max_cases or config.backbone == 'tiny_debug'),
              'metrics': metrics, 'bootstrap': intervals}
    write_json(output / 'metrics.json', result)
    return result
