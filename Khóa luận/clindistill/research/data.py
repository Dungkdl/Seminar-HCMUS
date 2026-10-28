"""Whitelisted clinical data and frozen image/text feature preparation; never fits on labels."""
import csv
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from ..artifacts import file_hash, object_hash, read_json, verify_release, write_json
from ..data import load_records, check_disjoint, deterministic_paths

ALLOWED_FIELDS = {'age', 'gender', 'chief_complaint', 'present_illness', 'past_medical_record'}
DIAGNOSTIC = re.compile(r'\b(?:diagnos(?:is|ed|tic)|treatment\s+plan|ICD[- ]?\d*|caries|gingivitis|malocclusion|pulpitis|tooth\s+loss|tooth\s+structure\s+loss)\b', re.I)
VISUAL_KEYS = {'student', 'teacher', 'resnet50', 'swin_b'}


def model_assets_hash(folder):
    folder = Path(folder)
    values = {p.relative_to(folder).as_posix(): file_hash(p) for p in sorted(folder.rglob('*'))
              if p.is_file() and not {'.cache', '__pycache__'} & set(p.relative_to(folder).parts)}
    if not values:
        raise FileNotFoundError(f'Missing model assets: {folder}')
    return object_hash(values)


def clinical_input(row, fields):
    if not set(fields) <= ALLOWED_FIELDS:
        raise ValueError('Clinical whitelist contains forbidden diagnostic/examination fields.')
    age = None
    if 'age' in fields:
        try:
            value = float(row.get('age_raw', ''))
            age = value if np.isfinite(value) and 0 <= value <= 120 else None
        except (ValueError, TypeError):
            pass
    raw_gender = str(row.get('gender_raw', '')).strip().lower() if 'gender' in fields else ''
    male, female = raw_gender in {'male', 'm', 'man'}, raw_gender in {'female', 'f', 'woman'}
    demographics = np.array([0 if age is None else age / 120, age is not None,
                              male, female, not (male or female)], dtype=np.float32)
    source = row.get('clinical_raw_allowlist', row.get('clinical_screened', {}))
    parts, blocked = [], []
    for field in fields:
        if field in {'age', 'gender'}:
            continue
        text = str(source.get(field, '') or '').strip()
        # Do not rewrite symptoms into artificial sentences: omit flagged fields pending review.
        if DIAGNOSTIC.search(text):
            blocked.append(field)
        elif text:
            parts.append(field.replace('_', ' ') + ': ' + text)
    text = '\n'.join(parts)
    present = bool(text or age is not None or male or female)
    return text, demographics, present, blocked


def encode_images(paths, image_root, model_dir, kind, image_size, batch_size, chunk_root, *, allow_debug=False):
    from transformers import AutoModel
    from torchvision import models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if kind in {'student', 'teacher'}:
        model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if model.config.model_type != 'dinov3_vit':
            raise ValueError('Expected a DINOv3 ViT checkpoint.')
        if not allow_debug and model.config.hidden_size != {'student': 768, 'teacher': 1024}[kind]:
            raise ValueError('Expected DINOv3-B Student and DINOv3-L Teacher dimensions.')
        if image_size % model.config.patch_size:
            raise ValueError('Image size must be divisible by the DINOv3 patch size.')
        registers = model.config.num_register_tokens
        processor = read_json(Path(model_dir) / 'preprocessor_config.json')
        mean, std = processor['image_mean'], processor['image_std']
    else:
        builder, head = {'resnet50': (models.resnet50, 'fc'), 'swin_b': (models.swin_b, 'head')}[kind]
        model = builder(weights=None)
        model.load_state_dict(torch.load(model_dir, map_location='cpu', weights_only=True), strict=True)
        setattr(model, head, torch.nn.Identity())
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    model.eval().requires_grad_(False).to(device)
    precision = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    chunk_root = Path(chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(paths), 256):
        names = paths[start:start + 256]
        path = chunk_root / f'{start:08d}.npz'
        proof = path.with_suffix('.json')
        if path.exists() and proof.exists():
            if file_hash(path) != read_json(proof)['sha256']:
                raise ValueError('Feature chunk checksum mismatch.')
            with np.load(path, allow_pickle=False) as data:
                if data['paths'].tolist() != names:
                    raise ValueError('Feature chunk image order mismatch.')
                chunks.append(data['values'].copy())
            continue
        values = []
        for offset in range(0, len(names), batch_size):
            tensors = []
            for relative in names[offset:offset + batch_size]:
                source = (Path(image_root) / relative).resolve()
                if Path(image_root).resolve() not in source.parents:
                    raise ValueError('Unsafe image path.')
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                image = ImageOps.pad(image, (image_size, image_size), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
                tensors.append(TF.normalize(TF.to_tensor(image), mean, std))
            batch = torch.stack(tensors).to(device)
            with torch.inference_mode(), torch.autocast('cuda', enabled=device.type == 'cuda', dtype=precision):
                if kind in {'student', 'teacher'}:
                    hidden = model(pixel_values=batch).last_hidden_state.float()
                    encoded = torch.cat([hidden[:, 0], hidden[:, 1 + registers:].mean(1)], -1)
                else:
                    encoded = model(batch).float()
                values.append(F.normalize(encoded, dim=-1).cpu().numpy().astype(np.float16))
        array = np.concatenate(values)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, paths=np.asarray(names), values=array)
        temporary.replace(path)
        write_json(proof, {'sha256': file_hash(path)})
        chunks.append(array)
        print(f'Frozen {kind}: {start + len(names):,}/{len(paths):,} photos', flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(chunks)


def encode_clinical(texts, model_dir, batch_size=16, max_length=256, *, allow_debug=False):
    from transformers import AutoModel, AutoTokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False).eval().requires_grad_(False).to(device)
    if not allow_debug and (model.config.model_type != 'xlm-roberta' or model.config.hidden_size != 1024):
        raise ValueError('Expected the BGE-M3 dense text encoder.')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False, use_fast=True)
    output = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    valid = [i for i, text in enumerate(texts) if text]
    truncated = 0
    for start in range(0, len(valid), batch_size):
        indices = valid[start:start + batch_size]
        sentences = [texts[i] for i in indices]
        lengths = tokenizer(sentences, truncation=False, padding=False)['input_ids']
        truncated += sum(len(x) > max_length for x in lengths)
        tokens = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True, max_length=max_length).to(device)
        with torch.inference_mode():
            # BGE-M3 dense representation: normalized CLS, not the old BiomedBERT mean pool.
            values = F.normalize(model(**tokens).last_hidden_state[:, 0].float(), dim=-1)
        output[indices] = values.cpu().numpy()
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return output, truncated


def prepare_features(prepared_root, image_root, asset_models, destination, config, *, splits=('train', 'validation')):
    prepared, destination = Path(prepared_root), Path(destination)
    if not set(splits) <= {'train', 'validation', 'test'} or not splits:
        raise ValueError('Invalid feature splits.')
    release = verify_release(prepared, splits=list(splits))
    if release.get('debug_subset') and not config.get('debug'):
        raise ValueError('Mini/debug release cannot produce a production research cache.')
    fields = config['clinical_fields']
    if not set(fields) <= ALLOWED_FIELDS:
        raise ValueError('Forbidden clinical whitelist.')
    if config['encoder_training'] != 'frozen_features':
        raise ValueError('This research protocol is frozen-foundation MIL, not full fine-tuning.')
    if config['language'] != 'en':
        raise ValueError('Primary implementation uses the audited English fields; Chinese needs a separate protocol.')
    roles = ['student', 'teacher'] + [r for r in ('resnet50', 'swin_b') if r + '_mil' in config['methods']]
    fingerprints = {role: (file_hash(asset_models[role]) if Path(asset_models[role]).is_file()
                           else model_assets_hash(asset_models[role])) for role in roles + ['clinical']}
    if config.get('clinical_temporal_verified') and not config.get('clinical_review_record'):
        raise ValueError('A reviewed clinical protocol needs a documented clinical_review_record.')
    if set(fields) & {'present_illness', 'past_medical_record'} and not config.get('clinical_temporal_verified'):
        raise ValueError('History fields require documented pre-diagnostic clinical review.')
    policy = {'fields': fields, 'language': 'en', 'temporal_verified': config['clinical_temporal_verified'],
              'review_record': config.get('clinical_review_record'),
              'screening': DIAGNOSTIC.pattern, 'source': 'existing allowlist; diagnosis/treatment/examination excluded',
              'status': 'reviewed' if config['clinical_temporal_verified'] else 'screened_exploratory'}
    request = {'release_id': release['release_id'], 'policy': policy, 'model_fingerprints': fingerprints,
               'model_ids': {k: config[k + '_model'] for k in ('student', 'teacher', 'clinical')},
               'image_size': config['image_size'], 'max_photos': config['max_photos'],
               'preprocessing': 'RGB_EXIF_resize_pad_bilinear_v1',
               'image_pooling': 'DINO_cls_concat_mean_patch_excluding_registers_L2',
               'text_pooling': 'BGE_dense_CLS_L2', 'encoder_training': 'frozen_features',
               'text_max_length': 256, 'demographics': 'age_div120_and_present;male_female_unknown',
               'bag_selection': 'fixed_even_filename_order_for_all_train_validation_and_OOF',
               'builder_sha256': file_hash(__file__)}
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / 'feature_manifest.json'
    previous = read_json(metadata_path) if metadata_path.exists() else None
    if previous and previous['request'] != request:
        raise ValueError('Feature/policy/model protocol changed. Use a new feature directory.')
    all_rows, selected, contexts = {}, {}, {}
    for split in splits:
        all_rows[split] = load_records(prepared, split, 'photo_attention')
        selected[split] = [deterministic_paths(r['photo_paths'], config['max_photos']) for r in all_rows[split]]
        with (prepared / f'teacher_context_{split}.jsonl').open(encoding='utf-8') as stream:
            contexts[split] = {row['case_id']: row for line in stream if line.strip() for row in [json.loads(line)]}
    if {'train', 'validation'} <= set(splits):
        check_disjoint(all_rows['train'], all_rows['validation'])
    paths = sorted({p for bags in selected.values() for bag in bags for p in bag})
    image_hashes = {}
    for p in paths:
        path = (Path(image_root) / p).resolve()
        if Path(image_root).resolve() not in path.parents:
            raise ValueError('Unsafe image path in prepared manifest.')
        image_hashes[p] = file_hash(path)
    image_identity = object_hash(image_hashes)
    split_image_ids = {split: object_hash({p: image_hashes[p] for bag in selected[split] for p in bag}) for split in splits}
    if previous:
        for split in splits:
            if split in previous['splits'] and previous['image_identities'][split] != split_image_ids[split]:
                raise ValueError('Image bytes changed; use the original files or a new feature release.')
    pending = [s for s in splits if not previous or s not in previous['splits']]
    encode_paths = sorted({p for split in pending for bag in selected[split] for p in bag})
    # Cache checkpointed image chunks; labels are never given to foundation encoders.
    arrays = {role: encode_images(encode_paths, image_root, asset_models[role], role,
              config['image_size'], config['image_batch_size'],
              destination / '_chunks' / role / object_hash([fingerprints[role], image_identity, request])[:16],
              allow_debug=bool(config.get('debug'))) for role in roles} if pending else {}
    index = {name: i for i, name in enumerate(encode_paths)}
    manifest = previous or {'schema_version': 4, 'request': request, 'release_stage': release['stage'],
        'class_names': release['class_names'], 'debug': bool(config.get('debug')), 'image_identities': {},
        'model_locations': {k: str(v) for k, v in asset_models.items()}, 'splits': {}}
    for split, rows in all_rows.items():
        if split in manifest['splits']:
            if file_hash(destination / f'{split}.npz') != manifest['splits'][split]['sha256']:
                raise ValueError('Feature shard checksum mismatch.')
            continue
        texts, demographics, present, audits = [], [], [], []
        for row in rows:
            context = contexts[split][row['case_id']]
            if context['split'] != split or context['patient_id'] != row['patient_id']:
                raise ValueError('Clinical/visual visit mapping mismatch.')
            text, demo, available, blocked = clinical_input(context, fields)
            texts.append(text); demographics.append(demo); present.append(available)
            audits.append({'case_id': row['case_id'], 'split': split, 'blocked_fields': '|'.join(blocked),
                           'clinical_present': available, 'text_present': bool(text),
                           'age_present': bool(demo[1]), 'gender_present': bool(demo[2] or demo[3]),
                           'temporal_verified': policy['temporal_verified']})
        clinical, truncated = encode_clinical(texts, asset_models['clinical'], allow_debug=bool(config.get('debug')))
        with (destination / f'clinical_audit_{split}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(audits[0])); writer.writeheader(); writer.writerows(audits)
        local_paths = sorted({p for bag in selected[split] for p in bag})
        local_index = {p: i for i, p in enumerate(local_paths)}
        offsets = np.cumsum([0] + [len(bag) for bag in selected[split]])
        values = {'ids': np.asarray([r['case_id'] for r in rows]),
                  'patients': np.asarray([r['patient_id'] for r in rows]),
                  'groups': np.asarray([r['split_group'] for r in rows]),
                  'bags': np.asarray([r['bag_signature'] for r in rows]),
                  'labels': np.asarray([r['single_label_index'] for r in rows]),
                  'offsets': offsets, 'image_indices': np.asarray([local_index[p] for bag in selected[split] for p in bag]),
                  'image_paths': np.asarray(local_paths), 'clinical': clinical,
                  'demographics': np.asarray(demographics), 'clinical_present': np.asarray(present)}
        values.update({role: arrays[role][[index[p] for p in local_paths]] for role in roles})
        target = destination / f'{split}.npz'
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **values)
        temporary.replace(target)
        manifest['dimensions'] = {role: int(values[role].shape[-1]) for role in roles + ['clinical']}
        manifest['splits'][split] = {'sha256': file_hash(target), 'cases': len(rows), 'truncated_fields': truncated,
            'context_sha256': release['file_checksums'][f'teacher_context_{split}.jsonl']}
        manifest['image_identities'][split] = split_image_ids[split]
        write_json(metadata_path, manifest)
    write_json(destination / 'clinical_whitelist.yaml', policy)  # JSON is valid YAML 1.2.
    combined = []
    for path in sorted(destination.glob('clinical_audit_*.csv')):
        with path.open(encoding='utf-8', newline='') as stream:
            combined.extend(csv.DictReader(stream))
    if combined:
        with (destination / 'clinical_leakage_audit.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(combined[0])); writer.writeheader(); writer.writerows(combined)
        summary = {}
        for split in sorted({r['split'] for r in combined}):
            rows = [r for r in combined if r['split'] == split]
            summary[split] = {'visits': len(rows),
                'blocked_record_fraction': sum(bool(r['blocked_fields']) for r in rows) / len(rows),
                **{f'{key}_missing_fraction': sum(str(r[key]).lower() != 'true' for r in rows) / len(rows)
                   for key in ('clinical_present', 'text_present', 'age_present', 'gender_present')}}
        write_json(destination / 'clinical_audit_summary.json', summary)
    return manifest


class FeatureStore:
    def __init__(self, directory, *, splits=('train', 'validation'), allow_debug=False):
        self.root = Path(directory)
        self.metadata = read_json(self.root / 'feature_manifest.json')
        if self.metadata.get('schema_version') != 4:
            raise ValueError('Need research feature cache v4, not the old BiomedBERT cache.')
        if self.metadata.get('debug') and not allow_debug:
            raise ValueError('Debug feature cache cannot be used for research training.')
        if self.metadata['request'].get('builder_sha256') != file_hash(__file__):
            raise ValueError('Feature builder source changed; use a matching source bank/cache.')
        self.data = {}
        seen = {key: set() for key in ('ids', 'patients', 'groups', 'bags')}
        for split in splits:
            info = self.metadata['splits'][split]
            if file_hash(self.root / f'{split}.npz') != info['sha256']:
                raise ValueError('Research feature checksum mismatch.')
            with np.load(self.root / f'{split}.npz', allow_pickle=False) as values:
                data = {key: values[key].copy() for key in values.files}
            for key in seen:
                values = set(map(str, data[key]))
                if values & seen[key]:
                    raise ValueError(f'Feature partitions overlap in {key}.')
                seen[key].update(values)
            if len(set(data['ids'])) != len(data['ids']):
                raise ValueError('Duplicate feature visit ID.')
            if not np.all(np.diff(data['offsets']) > 0):
                raise ValueError('Empty visual bag.')
            n = len(data['ids'])
            if (data['offsets'].shape != (n + 1,) or data['offsets'][0] != 0 or
                data['offsets'][-1] != len(data['image_indices']) or
                np.any(data['image_indices'] < 0) or np.any(data['image_indices'] >= len(data['image_paths'])) or
                np.any(data['labels'] < 0) or np.any(data['labels'] >= len(self.metadata['class_names']))):
                raise ValueError('Invalid feature bag offsets, image indexes or labels.')
            for key in ('patients', 'groups', 'bags', 'labels', 'clinical', 'demographics', 'clinical_present'):
                if len(data[key]) != n:
                    raise ValueError('Feature visit alignment mismatch.')
            for key in (VISUAL_KEYS & set(data)) | {'clinical', 'demographics'}:
                if not np.isfinite(data[key]).all():
                    raise ValueError('Non-finite features.')
                expected = n if key in {'clinical', 'demographics'} else len(data['image_paths'])
                if len(data[key]) != expected:
                    raise ValueError('Feature vector/image alignment mismatch.')
                width = 5 if key == 'demographics' else self.metadata['dimensions'][key]
                if data[key].ndim != 2 or data[key].shape[1] != width:
                    raise ValueError('Feature dimension does not match its manifest.')
            self.data[split] = data
        self.identity = object_hash({'request': self.metadata['request'],
                                    'shards': {s: self.metadata['splits'][s] for s in splits}})


class VisitDataset(Dataset):
    def __init__(self, store, split, visual_key, *, role='student', indices=None):
        if visual_key not in VISUAL_KEYS:
            raise ValueError('A clinical vector cannot be used as visual input.')
        self.values, self.key, self.role = store.data[split], visual_key, role
        self.indices = np.arange(len(self.values['ids'])) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        i = int(self.indices[offset]); data = self.values
        images = data['image_indices'][data['offsets'][i]:data['offsets'][i + 1]]
        item = {'visual': torch.from_numpy(data[self.key][images].astype(np.float32)),
                'label': int(data['labels'][i]), 'index': i, 'case_id': str(data['ids'][i])}
        if self.role == 'multimodal_teacher':
            item.update(clinical=torch.from_numpy(data['clinical'][i].astype(np.float32)),
                        demographics=torch.from_numpy(data['demographics'][i].astype(np.float32)),
                        clinical_present=bool(data['clinical_present'][i]))
        return item


def collate_visits(items):
    features = torch.zeros(len(items), max(len(x['visual']) for x in items), items[0]['visual'].shape[-1])
    mask = torch.zeros(features.shape[:2], dtype=torch.bool)
    for i, item in enumerate(items):
        features[i, :len(item['visual'])] = item['visual']; mask[i, :len(item['visual'])] = True
    result = {'visual': features, 'mask': mask, 'labels': torch.tensor([x['label'] for x in items]),
              'indices': torch.tensor([x['index'] for x in items]), 'ids': [x['case_id'] for x in items]}
    if 'clinical' in items[0]:
        result.update(clinical=torch.stack([x['clinical'] for x in items]),
                      demographics=torch.stack([x['demographics'] for x in items]),
                      clinical_present=torch.tensor([x['clinical_present'] for x in items]))
    return result
