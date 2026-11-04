"""Research tables and deployable, image-only Student export."""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from ..artifacts import code_hash, file_hash, object_hash, read_json, write_json
from ..metrics import classification_metrics, group_bootstrap
from ..reporting import paired_bootstrap, risk_coverage
from .data import FeatureStore, VisitDataset, model_assets_hash, encode_images
from .training import PRIMARY_COMPARISONS, ABLATION_COMPARISONS, load_model, predict, probabilities, write_predictions, dataset_for
from .analysis import teacher_strata, stratum_metrics


def freeze_test_protocol(feature_root, study_root, review_record):
    """Register fitted checkpoint identities before preparing/opening the test shard."""
    root = Path(study_root); study = read_json(root / 'study.json')
    if not isinstance(review_record, str) or not review_record.strip():
        raise ValueError('Locked test: supply a documented protocol/review record.')
    if not (root / 'completed.json').is_file() or study['source'] != code_hash():
        raise ValueError('Complete the registered study with matching source before locking test.')
    store = FeatureStore(feature_root, allow_debug=study['debug'])
    if store.identity != study['features']:
        raise ValueError('Cannot lock a different train/validation feature release.')
    checkpoints = {}
    for name in ['teacher_visual', 'teacher_multimodal'] + study['config']['methods']:
        for seed in study['config']['seeds']:
            relative = f'{name}/seed_{seed}/best.pt'
            expected = read_json(root / name / f'seed_{seed}/result.json')['checkpoint_sha256']
            if file_hash(root / relative) != expected:
                raise ValueError('Selected checkpoint changed before test lock.')
            checkpoints[relative] = expected
    protocol = {'study_identity': study['identity'], 'source': study['source'],
                'feature_request': store.metadata['request'], 'config': study['config'],
                'checkpoints': checkpoints, 'review_record': review_record,
                'pending_reviews': study.get('pending_reviews', ['Unresolved review status.'])}
    protocol['identity'] = object_hash(protocol)
    path = root / 'test_protocol.json'
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError('Test protocol already locked; cannot change models, config or review record.')
        return protocol  # Attached Kaggle Notebook Outputs are read-only.
    write_json(path, protocol)
    return protocol


def evaluate_study(feature_root, study_root, output_root=None, *, split='validation', test_review_record=None):
    root = Path(study_root); study = read_json(root / 'study.json'); cfg = study['config']
    if split not in {'validation', 'test'}:
        raise ValueError('Evaluation supports validation or locked test.')
    if study['source'] != code_hash():
        raise ValueError('Evaluation source differs from the fitted study.')
    if not (root / 'completed.json').exists():
        raise ValueError('Finish all registered systems and seeds before evaluation.')
    training_store = FeatureStore(feature_root, allow_debug=bool(cfg.get('debug')))
    if training_store.identity != study['features']:
        raise ValueError('Train/validation cache differs from the fitted study.')
    if split == 'test':
        if not isinstance(test_review_record, str) or not test_review_record.strip():
            raise ValueError('Locked test: record the frozen protocol and unresolved review limitations first.')
        if not (root / 'test_protocol.json').is_file():
            raise ValueError('Locked test: call freeze_test_protocol before preparing/opening test features.')
        freeze_test_protocol(feature_root, root, test_review_record)
        store = FeatureStore(feature_root, splits=('train', 'validation', 'test'), allow_debug=bool(cfg.get('debug')))
    else:
        store = training_store
    out = Path(output_root) if output_root else root / f'report_{split}'
    out.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], {}
    classes, data = store.metadata['class_names'], store.data[split]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name in ['teacher_visual', 'teacher_multimodal'] + cfg['methods']:
        for seed in cfg['seeds']:
            directory = root / name / f'seed_{seed}'
            result = read_json(directory / 'result.json')
            if file_hash(directory / 'best.pt') != result['checkpoint_sha256']:
                raise ValueError('Selected checkpoint checksum changed.')
            model, state = load_model(directory / 'best.pt', device)
            logits, ids = predict(model, dataset_for(store, split, state['visual_key'], state['spec']['role'], name, seed), cfg, device)
            if ids != data['ids'].tolist():
                raise ValueError('Evaluation visit order mismatch.')
            p = probabilities(logits); metrics = classification_metrics(data['labels'], p, classes)
            target = out / name / f'seed_{seed}'; target.mkdir(parents=True, exist_ok=True)
            write_predictions(target / 'predictions.csv', data, logits)
            frame = pd.read_csv(target / 'predictions.csv', dtype={'case_id': str, 'patient_id': str, 'split_group': str})
            predictions[(name, seed)] = frame
            risk = risk_coverage(frame); risk.to_csv(target / 'risk_coverage.csv', index=False)
            frame['confidence'] = p.max(-1); frame['correct'] = frame.label == frame.prediction
            frame['n_photos'] = np.diff(data['offsets'])
            frame.sort_values(['correct', 'confidence'], ascending=[True, False]).to_csv(target / 'case_analysis.csv', index=False)
            subgroups = {}
            for tag, mask in [('one_photo', np.diff(data['offsets']) == 1), ('multiple_photos', np.diff(data['offsets']) > 1),
                              ('clinical_present', data['clinical_present']), ('clinical_missing', ~data['clinical_present'])]:
                if mask.any():
                    subgroups[tag] = classification_metrics(data['labels'][mask], p[mask], classes)
            write_json(target / 'metrics.json', {'metrics': metrics, 'subgroups': subgroups,
                'cluster_bootstrap': group_bootstrap(data['labels'], p, data['groups'], cfg['bootstrap_repetitions'], seed)})
            rows.append({'method': name, 'seed': seed, **{k: metrics[k] for k in
                ('accuracy', 'macro_f1', 'macro_auroc_defined_classes', 'brier_multiclass', 'ece_10_bins')},
                'aurc': float(risk.risk.mean()), 'cases': len(ids)})
    table = pd.DataFrame(rows); table.to_csv(out / 'per_seed.csv', index=False)
    summary = table.groupby('method')[['macro_f1', 'accuracy', 'macro_auroc_defined_classes', 'brier_multiclass', 'ece_10_bins', 'aurc']].agg(['mean', 'std', 'count'])
    summary.to_csv(out / 'mean_std.csv')
    comparisons = {}
    for right, left in PRIMARY_COMPARISONS + ABLATION_COMPARISONS:
        if not all((method, seed) in predictions for method in (left, right) for seed in cfg['seeds']):
            continue
        comparisons[f'{right}_minus_{left}'] = paired_bootstrap(
            [predictions[(left, s)] for s in cfg['seeds']], [predictions[(right, s)] for s in cfg['seeds']],
            len(classes), cfg['bootstrap_repetitions'], cfg['seeds'][0])
        comparisons[f'{right}_minus_{left}']['kind'] = 'primary' if (right, left) in PRIMARY_COMPARISONS else 'ablation'
    write_json(out / 'paired_comparisons.json', {'comparisons': comparisons,
        'multiplicity': 'Pre-registered contrasts; intervals are unadjusted/descriptive, not confirmatory family-wise tests.'})
    disagreements = {}
    for seed in cfg['seeds']:
        visual = predictions[('teacher_visual', seed)]
        mm = predictions[('teacher_multimodal', seed)]
        if not visual.case_id.equals(mm.case_id):
            raise ValueError('Teacher strata require aligned visits.')
        strata = teacher_strata(visual[[f'p_{i}' for i in range(len(classes))]].to_numpy(),
                               mm[[f'p_{i}' for i in range(len(classes))]].to_numpy(), data['labels'])
        disagreements[str(seed)] = {name: stratum_metrics(data['labels'],
            predictions[(name, seed)][[f'p_{i}' for i in range(len(classes))]].to_numpy(), classes, strata)
            for name in ['teacher_visual', 'teacher_multimodal'] + cfg['methods']}
    write_json(out / 'teacher_strata.json', {'results': disagreements,
        'interpretation': 'Descriptive groups defined by model correctness, not clinical ground truth. '
                          'Subgroup scores are conditional and do not establish clinical dependency.'})
    write_json(out / 'status.json', {'split': split, 'study_identity': study['identity'], 'debug': study['debug'],
        'test_review_record': test_review_record, 'clinical_policy': study['clinical_policy'],
        'release_stage': study['release_stage'], 'publication_ready': False,
        'pending_reviews': study.get('pending_reviews', ['Unresolved review status.']),
        'annotation_note': study.get('annotation_note'),
        'limitations': ['Frozen foundation encoders; MIL/fusion/head training only.',
            'Single recorded target; clinical temporal alignment and annotation reviews may remain unresolved.',
            'COde internal evaluation only; no external dataset validation.']})
    return table, summary


def export_student(feature_root, study_root, encoder_dir, destination, *, method='kd_reliability'):
    """Choose a seed using validation only; include the local DINO-B encoder once."""
    root, destination = Path(study_root), Path(destination)
    study = read_json(root / 'study.json'); store = FeatureStore(feature_root, allow_debug=study['debug'])
    if study['source'] != code_hash() or study['features'] != store.identity:
        raise ValueError('Export protocol/source mismatch.')
    if method not in study['config']['methods'] or method in {'resnet50_mil', 'swin_b_mil'} or method.startswith('teacher_'):
        raise ValueError('Select a DINO-B Student method registered in this study.')
    if model_assets_hash(encoder_dir) != store.metadata['request']['model_fingerprints']['student']:
        raise ValueError('Student image encoder differs from the feature encoder.')
    candidates = [(read_json(root / method / f'seed_{s}' / 'result.json')['metrics']['macro_f1'], s)
                  for s in study['config']['seeds']]
    _, seed = max(candidates, key=lambda x: (x[0], -x[1]))
    source = root / method / f'seed_{seed}' / 'best.pt'
    if file_hash(source) != read_json(source.parent / 'result.json')['checkpoint_sha256']:
        raise ValueError('Student checkpoint checksum changed.')
    state = torch.load(source, map_location='cpu', weights_only=False)
    if state['spec']['role'] != 'student' or state['visual_key'] != 'student':
        raise ValueError('Export must be an image-only DINO-B Student.')
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'student_manifest.json'
    expected = {'study_identity': study['identity'], 'method': method, 'seed': seed,
                'encoder_sha256': model_assets_hash(encoder_dir)}
    if manifest_path.exists() and any(read_json(manifest_path)[k] != v for k, v in expected.items()):
        raise ValueError('Destination belongs to another exported Student.')
    from ..engine import save_checkpoint
    save_checkpoint(destination / 'student_head.pt', {'spec': state['spec'], 'model': state['model'],
                                                    'visual_key': 'student'})
    target_encoder = destination / 'image_encoder'
    try:
        existing_encoder = model_assets_hash(target_encoder)
    except FileNotFoundError:
        existing_encoder = None
    if existing_encoder != expected['encoder_sha256']:
        # A session may have stopped midway through copying weights; safely finish this generated export.
        shutil.copytree(encoder_dir, target_encoder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    if model_assets_hash(target_encoder) != expected['encoder_sha256']:
        raise ValueError('Exported image encoder integrity mismatch.')
    manifest = {**expected, 'head_sha256': file_hash(destination / 'student_head.pt'),
        'source': code_hash(), 'class_names': store.metadata['class_names'],
        'image_size': study['config']['image_size'],
        'max_photos': 1 if state['spec'].get('pooling') == 'single' else study['config']['max_photos'],
        'debug': study['debug'], 'input': 'photos only; no clinical data or labels',
        'selection': 'registered method; highest validation macro-F1 seed, tie -> smaller seed'}
    write_json(manifest_path, manifest)
    return manifest


def predict_student(export_root, photo_paths, image_root, work_dir):
    """Single-visit raw-photo inference; there is no clinical/label argument."""
    from ..data import deterministic_paths
    root = Path(export_root); info = read_json(root / 'student_manifest.json')
    if info['source'] != code_hash() or file_hash(root / 'student_head.pt') != info['head_sha256']:
        raise ValueError('Student source/head integrity mismatch.')
    if model_assets_hash(root / 'image_encoder') != info['encoder_sha256']:
        raise ValueError('Student encoder integrity mismatch.')
    paths = deterministic_paths(photo_paths, info['max_photos'])
    if not paths:
        raise ValueError('Provide at least one photo.')
    # Hash bytes as well as names: inference cache cannot reuse a changed photograph.
    identity = object_hash({p: file_hash(Path(image_root) / p) for p in paths})
    array = encode_images(paths, image_root, root / 'image_encoder', 'student', info['image_size'], 8,
        Path(work_dir) / identity / info['encoder_sha256'][:16], allow_debug=info['debug'])
    model, _ = load_model(root / 'student_head.pt')
    with torch.inference_mode():
        p = model(torch.from_numpy(array.astype(np.float32)).unsqueeze(0),
                  torch.ones(1, len(paths), dtype=torch.bool))['logits'].softmax(-1)[0].numpy()
    return {'label': info['class_names'][int(p.argmax())],
            'probabilities': dict(zip(info['class_names'], map(float, p))), 'photos_used': paths}
