"""Compare Run 5 fusion with separate photo and radiograph branches on validation."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from clindistill.artifacts import code_hash, file_hash, object_hash, read_json, verify_release, write_json
from clindistill.engine import save_checkpoint, seed_everything
from clindistill.metrics import classification_metrics
from clindistill.reporting import paired_bootstrap
from clindistill.research.data import FeatureStore, encode_images, model_assets_hash, resolve_image_sources
from clindistill.research.losses import per_case_kl, weighted_kd
from clindistill.research.training import dataset_for, fit, forward_batch, load_model, loader, predict, probabilities

from run5_xray import MAX_XRAYS, SPLITS, matched_store, paired_records, selected_xrays

ARMS = ('run5_early_fusion', 'two_branch_xray_masked', 'two_branch_photo_xray')


class TwoBranchMean(nn.Module):
    role = 'student'

    def __init__(self, dimension, width, dropout, classes, *, use_xray):
        super().__init__()
        self.use_xray = use_xray
        self.photo = nn.Sequential(nn.Linear(dimension, width), nn.LayerNorm(width), nn.GELU())
        self.xray = nn.Sequential(nn.Linear(dimension, width), nn.LayerNorm(width), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 2, classes))

    @staticmethod
    def project_mean(features, mask, projection):
        return (projection(features) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def pooled_features(self, visual, mask):
        features, is_xray = visual[..., :-1], visual[..., -1].bool()
        photo = self.project_mean(features, mask & ~is_xray, self.photo)
        xray = (self.project_mean(features, mask & is_xray, self.xray)
                if self.use_xray else torch.zeros_like(photo))
        return self.dropout(torch.cat((photo, xray), dim=-1))

    def forward(self, visual, mask):
        return {'logits': self.head(self.pooled_features(visual, mask))}


def branch_store(early, records, xrays, protocol, arm):
    """Keep individual radiograph descriptors alongside exactly the original photo bag."""
    data = {}
    for split, rows in records.items():
        source = early.data[split]
        width = source['student'].shape[1] // 2
        vectors, lengths = [], []
        for i, row in enumerate(rows):
            start, end = source['offsets'][i:i+2]
            photos = source['student'][start:end, :width]
            films = np.stack([xrays[path] for path in selected_xrays(row)])
            bag = np.concatenate((photos, films), axis=0)
            modality = np.concatenate((np.zeros(len(photos)), np.ones(len(films))))[:, None]
            vectors.append(np.concatenate((bag, modality.astype(bag.dtype)), axis=1))
            lengths.append(len(bag))
        data[split] = {key: source[key] for key in ('ids', 'patients', 'groups', 'labels', 'clinical')}
        data[split].update(student=np.concatenate(vectors), offsets=np.cumsum([0, *lengths]),
                           image_indices=np.arange(sum(lengths)))
    return SimpleNamespace(data=data, identity=object_hash([protocol, arm]), metadata={
        'dimensions': {'student': data['train']['student'].shape[-1]},
        'class_names': early.metadata['class_names']})


def fit_branches(store, directory, config, seed, use_xray, device, *, teacher_factory=None,
                 indices=None, fixed_epochs=None, kd_mode=None, targets=None, target_sha256=None):
    """Use the same CE, AdamW, cosine schedule and validation stopping as Run 5."""
    if config['accumulation_steps'] != 1:
        raise ValueError('Run 6 uses the Run 5 setting accumulation_steps=1.')
    directory.mkdir(parents=True, exist_ok=True)
    classes = store.metadata['class_names']
    dimension = store.data['train']['student'].shape[-1] - 1
    seed_everything(seed)
    model = (TwoBranchMean(dimension, config['feature_dim'], config['dropout'], len(classes), use_xray=use_xray)
             if teacher_factory is None else teacher_factory(dimension, config, len(classes)))
    model = model.to(device)
    if kd_mode not in (None, 'vanilla', 'correctness'):
        raise ValueError('Unknown KD loss.')
    if kd_mode and (targets is None or not target_sha256 or model.role != 'student'):
        raise ValueError('KD requires verified OOF targets and an image-only Student.')
    epochs = config['epochs'] if fixed_epochs is None else fixed_epochs
    if not 1 <= epochs <= config['epochs']:
        raise ValueError('Fixed epochs must fit within the registered cosine schedule.')
    identity = object_hash([store.identity, seed, use_xray, code_hash(), model.role,
                            None if indices is None else np.asarray(indices).tolist(),
                            fixed_epochs, kd_mode, target_sha256])
    best_path, result_path = directory / 'best.pt', directory / 'result.json'
    train = dataset_for(store, 'train', 'student', model.role, 'student_mean', seed, indices)
    val = (dataset_for(store, 'validation', 'student', model.role, 'student_mean', seed)
           if fixed_epochs is None else None)
    if result_path.is_file():
        result = read_json(result_path)
        if result['identity'] != identity or result['checkpoint_sha256'] != file_hash(best_path):
            raise ValueError('Run 6 completed checkpoint fingerprint mismatch.')
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        return model, val, result['best_epoch']

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    best, best_epoch, bad, history = -1.0, 0, 0, []
    for epoch in range(epochs):
        model.train()
        for batch in loader(train, config, shuffle=True, epoch=epoch, seed=seed, device=device):
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch, device)['logits']
            loss = F.cross_entropy(logits, batch['labels'].to(device))
            if kd_mode:
                ix = batch['indices'].numpy()
                teacher = torch.from_numpy(targets['logits'][ix]).to(device)
                if kd_mode == 'correctness':
                    weights = torch.from_numpy(targets['correct'][ix]).to(device)
                    kd = weighted_kd(logits, teacher, weights, targets['mean_correct'], config['temperature'])
                else:
                    kd = per_case_kl(logits, teacher, config['temperature']).mean()
                loss = loss + config['kd_weight'] * kd
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        scheduler.step()
        score = float(epoch + 1)
        if val is not None:
            logits, ids = predict(model, val, config, device)
            if ids != store.data['validation']['ids'].tolist():
                raise ValueError('Validation prediction order differs from paired cohort.')
            score = classification_metrics(store.data['validation']['labels'], probabilities(logits), classes)['macro_f1']
        history.append({'epoch': epoch+1, 'validation_macro_f1': score if val is not None else None})
        validation_log = f'val_macro_f1={score:.4f}' if val is not None else 'OOF fixed epochs'
        print(f'{directory.name} role={model.role} kd={kd_mode} seed={seed} epoch={epoch+1}/{epochs} {validation_log}',
              flush=True)
        if score > best:
            best, best_epoch, bad = score, epoch+1, 0
            save_checkpoint(best_path, model.state_dict())
        else:
            bad += 1
            if bad >= config['patience']:
                break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
    write_json(result_path, {'identity': identity, 'best_epoch': best_epoch,
                             'checkpoint_sha256': file_hash(best_path)})
    return model, val, best_epoch


def prepare_paired_inputs(prepared_root, feature_root, image_root, encoder_root, output, config, *,
                          image_batch=32, run_name='code_run6_xray_fusion_v1', protocol_extra=None):
    output = Path(output)
    release = verify_release(prepared_root, splits=SPLITS)
    photos = FeatureStore(feature_root, splits=SPLITS)
    request = photos.metadata['request']
    encoder_hash = model_assets_hash(encoder_root)
    if (release['release_id'] != request['release_id'] or release['class_names'] != photos.metadata['class_names']
            or encoder_hash != request['model_fingerprints']['student']):
        raise ValueError('Attach the same prepared release, photo cache and DINO-B used in Run 5.')
    records = paired_records(prepared_root)
    paths_by_split = {s: {p for row in records[s] for p in selected_xrays(row)} for s in SPLITS}
    paths = sorted(set.union(*paths_by_split.values()))
    sources = resolve_image_sources(paths, image_root)
    hashes = {path: file_hash(sources[path]) for path in paths}
    if {hashes[p] for p in paths_by_split['train']} & {hashes[p] for p in paths_by_split['validation']}:
        raise ValueError('Identical X-ray bytes cross train and validation.')
    config = {**config, 'run_name': run_name}
    runtime = [torch.__version__, torch.version.cuda,
               torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu', image_batch]
    protocol = {'photo_cache': photos.identity, 'release': release['release_id'], 'encoder': encoder_hash,
                'xray_bytes': object_hash(hashes), 'max_xrays': MAX_XRAYS, 'max_photos': request['max_photos'],
                'config': config, 'runtime': runtime, 'test_opened': False, 'source': code_hash(),
                'experiment_sha256': file_hash(__file__), 'run5_helper_sha256': file_hash(Path(__file__).with_name('run5_xray.py')),
                'pending_reviews': release['pending_reviews']}
    protocol.update(protocol_extra or {'arms': list(ARMS)})
    path = output / ('run6_protocol.json' if protocol_extra is None else 'run7_protocol.json')
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Paired study protocol changed. Select a fresh output directory.')
    write_json(path, protocol)
    write_json(output / 'paired_cohort.json', {s: {'visits': len(rows),
        'per_class': dict(Counter(row['target_labels'][0] for row in rows))} for s, rows in records.items()})
    chunk_key = object_hash([encoder_hash, hashes, request, 1, runtime])
    encoded = encode_images(paths, image_root, encoder_root, 'student', request['image_size'], image_batch,
                            output / '_chunks' / chunk_key[:24], image_sources=sources, workers=0, chunk_size=256)
    if encoded.shape != (len(paths), photos.metadata['dimensions']['student']) or not np.isfinite(encoded).all():
        raise ValueError('Unexpected frozen X-ray descriptors.')
    xrays = dict(zip(paths, encoded))
    early = matched_store(photos, records, xrays, protocol, 'photo_xray_mean')
    return photos, records, xrays, early, protocol


def run_fusion(prepared_root, feature_root, image_root, encoder_root, output, config, *, image_batch=32):
    output = Path(output)
    photos, records, xrays, early, protocol = prepare_paired_inputs(
        prepared_root, feature_root, image_root, encoder_root, output, config, image_batch=image_batch)
    config = protocol['config']
    classes = photos.metadata['class_names']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, frames, details = [], {}, {}
    for arm in ARMS:
        store = early if arm == ARMS[0] else branch_store(early, records, xrays, protocol, arm)
        val = store.data['validation']
        for seed in config['seeds']:
            directory = output / arm / f'seed_{seed}'
            if arm == ARMS[0]:
                result = fit(store, directory, config, 'student_mean', seed)
                model, _ = load_model(directory / 'best.pt', device)
                ds = dataset_for(store, 'validation', 'student', 'student', 'student_mean', seed)
                best_epoch = result['best_epoch']
            else:
                model, ds, best_epoch = fit_branches(store, directory, config, seed, arm == ARMS[2], device)
            logits, ids = predict(model, ds, config, device)
            if ids != val['ids'].tolist():
                raise ValueError('Prediction order differs from paired validation cohort.')
            p = probabilities(logits)
            metrics = classification_metrics(val['labels'], p, classes)
            details[f'{arm}/{seed}'] = metrics
            rows.append({'method': arm, 'seed': seed, 'best_epoch': best_epoch, 'cases': len(ids),
                         'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']})
            frames[arm, seed] = pd.DataFrame({'case_id': ids, 'patient_id': val['patients'],
                'split_group': val['groups'], 'label': val['labels'], 'prediction': p.argmax(1)})
        del store
    contrasts = {}
    for reference in ARMS[:2]:
        a, b = [[frames[arm, seed] for seed in config['seeds']] for arm in (reference, ARMS[2])]
        contrasts[f'{ARMS[2]}_minus_{reference}'] = paired_bootstrap(
            a, b, len(classes), config['bootstrap_repetitions'], config['seeds'][0])
    table = pd.DataFrame(rows)
    summary = table.groupby('method')[['macro_f1', 'accuracy']].agg(['mean', 'std', 'count'])
    table.to_csv(output / 'per_seed.csv', index=False)
    summary.to_csv(output / 'mean_std.csv')
    write_json(output / 'metrics.json', details)
    write_json(output / 'paired_comparisons.json', contrasts)
    write_json(output / 'completed.json', {'fits': len(table), 'protocol': object_hash(protocol)})
    return summary, contrasts
