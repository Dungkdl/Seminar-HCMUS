"""Paired validation experiment using the Run 4 bank and frozen photo cache."""
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from clindistill.artifacts import file_hash, object_hash, read_json, verify_release, write_json
from clindistill.data import deterministic_paths, load_records
from clindistill.metrics import classification_metrics
from clindistill.reporting import paired_bootstrap
from clindistill.research.data import FeatureStore, encode_images, model_assets_hash, resolve_image_sources
from clindistill.research.training import dataset_for, fit, load_model, predict, probabilities

SPLITS = ('train', 'validation')
ARMS = ('photo_mean', 'photo_xray_mean')
MAX_XRAYS = 2


def paired_records(prepared_root):
    records = {}
    for split in SPLITS:
        rows = load_records(prepared_root, split, 'photo_attention')
        records[split] = [row for row in rows if row['radiograph_paths']]
        if not records[split]:
            raise ValueError(f'No paired photo/radiograph visits in {split}.')
    return records


def selected_xrays(row):
    paths = deterministic_paths(set(row['radiograph_paths']), MAX_XRAYS)
    for path in paths:
        parts = PurePosixPath(path).parts
        if len(parts) != 3 or parts[:2] != ('Images', 'Radiographs') or '\\' in path or '..' in parts:
            raise ValueError('Expected canonical Images/Radiographs paths.')
    return paths


def matched_store(photo_store, records, xrays, protocol, arm):
    data = {}
    for split, rows in records.items():
        source = photo_store.data[split]
        positions = {visit: i for i, visit in enumerate(source['ids'])}
        visits, vectors, lengths = [], [], []
        for row in rows:
            i = positions[row['case_id']]
            start, end = source['offsets'][i:i+2]
            ix = source['image_indices'][start:end]
            expected_paths = deterministic_paths(row['photo_paths'], protocol['max_photos'])
            if (source['patients'][i] != row['patient_id'] or source['groups'][i] != row['split_group']
                    or source['labels'][i] != row['single_label_index']
                    or source['image_paths'][ix].tolist() != expected_paths):
                raise ValueError(f'Prepared and cached {split} visit metadata differ.')
            photos = source['student'][ix]
            xray = np.mean([xrays[p] for p in selected_xrays(row)], axis=0, dtype=np.float32)
            if arm == 'photo_mean':
                xray = np.zeros_like(xray)
            vectors.append(np.concatenate((photos, np.broadcast_to(xray, photos.shape)), axis=1))
            visits.append(i)
            lengths.append(len(ix))
        data[split] = {key: source[key][visits] for key in ('ids', 'patients', 'groups', 'labels')}
        data[split].update(student=np.concatenate(vectors), offsets=np.cumsum([0, *lengths]),
                           image_indices=np.arange(sum(lengths)), clinical=np.empty((len(visits), 0)))
    return SimpleNamespace(data=data, identity=object_hash([protocol, arm]), metadata={
        'dimensions': {'student': data['train']['student'].shape[-1]},
        'class_names': photo_store.metadata['class_names']})


def run_xray(prepared_root, feature_root, image_root, encoder_root, output, config, *, image_batch=32,
             experiment_file=None):
    """Run six fitted heads on paired visits. Open train/validation only."""
    output = Path(output)
    release = verify_release(prepared_root, splits=SPLITS)
    photo_store = FeatureStore(feature_root, splits=SPLITS)
    request = photo_store.metadata['request']
    if release['release_id'] != request['release_id'] or release['class_names'] != photo_store.metadata['class_names']:
        raise ValueError('Attach the prepared release that generated the photo cache.')
    encoder_hash = model_assets_hash(encoder_root)
    if encoder_hash != request['model_fingerprints']['student']:
        raise ValueError('DINO-B differs from the encoder used for the supplied photo cache.')
    records = paired_records(prepared_root)
    paths = sorted({path for rows in records.values() for row in rows for path in selected_xrays(row)})
    sources = resolve_image_sources(paths, image_root)
    hashes = {path: file_hash(sources[path]) for path in paths}
    split_hashes = [{hashes[path] for row in records[split] for path in selected_xrays(row)} for split in SPLITS]
    if split_hashes[0] & split_hashes[1]:
        raise ValueError('Identical selected X-ray bytes cross train and validation.')
    config = {**config, 'run_name': 'code_run5_xray_v1'}
    protocol = {'photo_cache': photo_store.identity, 'release': release['release_id'], 'encoder': encoder_hash,
                'xray_bytes': object_hash(hashes), 'max_xrays': MAX_XRAYS, 'max_photos': request['max_photos'],
                'xray_selection': 'sorted unique canonical paths; fixed evenly spaced positions',
                'fusion': 'concat each photo with visit mean X-ray or zeros; shared mean MIL architecture',
                'experiment_sha256': file_hash(experiment_file or __file__), 'config': config, 'test_opened': False,
                'evidence': 'exploratory paired train/validation only', 'pending_reviews': release['pending_reviews']}
    protocol_path = output / 'run5_protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError('Run 5 protocol changed. Select a fresh output directory.')
    write_json(protocol_path, protocol)
    write_json(output / 'paired_cohort.json', {s: {'visits': len(rows),
        'per_class': dict(Counter(row['target_labels'][0] for row in rows))} for s, rows in records.items()})
    # The chunk identity includes byte hashes, the encoder and the preprocessing contract.
    encoded = encode_images(paths, image_root, encoder_root, 'student', request['image_size'], image_batch,
        output / '_chunks' / object_hash([encoder_hash, hashes, request, 1,
            torch.__version__, torch.version.cuda, torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'])[:24],
        image_sources=sources, workers=0, chunk_size=256)
    if encoded.shape != (len(paths), photo_store.metadata['dimensions']['student']) or not np.isfinite(encoded).all():
        raise ValueError('Unexpected frozen X-ray descriptor shape or values.')
    xrays = dict(zip(paths, encoded))
    rows, frames, details = [], {}, {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for arm in ARMS:
        store = matched_store(photo_store, records, xrays, protocol, arm)
        val = store.data['validation']
        for seed in config['seeds']:
            directory = output / arm / f'seed_{seed}'
            result = fit(store, directory, config, 'student_mean', seed)
            model, state = load_model(directory / 'best.pt', device)
            logits, ids = predict(model, dataset_for(store, 'validation', state['visual_key'],
                'student', 'student_mean', seed), config, device)
            if ids != val['ids'].tolist():
                raise ValueError('Validation prediction order differs from paired cohort.')
            prob = probabilities(logits)
            metrics = classification_metrics(val['labels'], prob, store.metadata['class_names'])
            details[f'{arm}/{seed}'] = metrics
            rows.append({'method': arm, 'seed': seed, 'cases': len(ids), 'best_epoch': result['best_epoch'],
                         'macro_f1': metrics['macro_f1'], 'accuracy': metrics['accuracy']})
            frames[arm, seed] = pd.DataFrame({'case_id': ids, 'patient_id': val['patients'],
                'split_group': val['groups'], 'label': val['labels'], 'prediction': prob.argmax(axis=1)})
    table = pd.DataFrame(rows)
    summary = table.groupby('method')[['macro_f1', 'accuracy']].agg(['mean', 'std', 'count'])
    contrast = paired_bootstrap(*[[frames[arm, seed] for seed in config['seeds']] for arm in ARMS],
        len(photo_store.metadata['class_names']), config['bootstrap_repetitions'], config['seeds'][0])
    table.to_csv(output / 'per_seed.csv', index=False)
    summary.to_csv(output / 'mean_std.csv')
    write_json(output / 'metrics.json', details)
    write_json(output / 'paired_comparisons.json', {'photo_xray_minus_photo': contrast})
    write_json(output / 'completed.json', {'fits': len(table), 'protocol': object_hash(protocol)})
    return summary, contrast
