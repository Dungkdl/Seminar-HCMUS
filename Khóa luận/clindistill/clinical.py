"""Frozen, per-field biomedical text embeddings. Preparation only; no label input."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import file_hash, object_hash, read_json, verify_release, write_json

FIELDS = ('examination', 'radiographs_examination', 'chief_complaint',
          'present_illness', 'past_medical_record')
DEFAULT_ENCODER = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'


def prepare_clinical_cache(release_dir, destination, *, encoder=DEFAULT_ENCODER, revision='main',
                           splits=('train', 'validation'), max_length=192, batch_size=32,
                           device=None, debug=False, local_files_only=False):
    """Encode screened English only. Test is never opened unless explicitly requested.

    An immutable metadata file and separate checksummed split files allow adding test later
    without changing the training cache identity. Debug vectors are never research features.
    """
    if not splits or not set(splits) <= {'train', 'validation', 'test'}:
        raise ValueError('Invalid cache splits.')
    if max_length < 4 or batch_size < 1:
        raise ValueError('Invalid text encoding size.')
    release = verify_release(release_dir, splits=list(splits))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    request = {'release_id': release['release_id'], 'encoder': str(encoder), 'revision': revision,
               'fields': list(FIELDS), 'max_length': max_length, 'debug': debug,
               'pooling': 'attention_mask_mean_including_special_tokens', 'schema_version': 1}
    meta_path = destination / 'metadata.json'
    previous = read_json(meta_path) if meta_path.exists() else None
    if previous and previous['request'] != request:
        raise ValueError('Cache configuration/release changed. Choose a new cache directory.')
    pending = []
    for split in splits:
        manifest = destination / f'{split}.json'
        if manifest.exists():
            info = read_json(manifest)
            if file_hash(destination / f'{split}.npz') != info['sha256']:
                raise ValueError('Clinical cache checksum mismatch.')
        else:
            pending.append(split)
    if not pending:
        return previous
    tokenizer, model = None, None
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if debug:
        width, weights_identity = 16, 'SHA256-debug-vectors-not-a-language-model'
    else:
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        local = Path(encoder)
        if not local.is_dir():
            local = Path(snapshot_download(str(encoder), revision=revision, local_files_only=local_files_only,
                allow_patterns=['config.json', '*.safetensors', 'pytorch_model.bin', 'vocab.txt',
                                'tokenizer*.json', 'special_tokens_map.json', 'added_tokens.json']))
        assets = {p.name: file_hash(p) for p in sorted(local.iterdir()) if p.is_file()}
        weights_identity = object_hash(assets)
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(local, local_files_only=True, trust_remote_code=False).eval().to(device)
        model.requires_grad_(False)
        width = model.config.hidden_size
    metadata = {'request': request, 'dimension': width, 'encoder_assets_sha256': weights_identity}
    if previous and previous != metadata:
        raise ValueError('Text encoder assets changed. Use the original encoder snapshot or a new cache.')
    write_json(meta_path, metadata)
    for split in pending:
        path = Path(release_dir) / f'teacher_context_{split}.jsonl'
        with path.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        ids = [r['case_id'] for r in rows]
        if len(set(ids)) != len(ids) or any(r['split'] != split for r in rows):
            raise ValueError('Clinical context partition/ID mismatch.')
        vectors = np.zeros((len(rows), len(FIELDS), width), dtype=np.float32)
        mask = np.zeros((len(rows), len(FIELDS)), dtype=bool)
        entries = []
        for i, row in enumerate(rows):
            for j, field in enumerate(FIELDS):
                text = row['clinical_screened'].get(field, '').strip()
                if text:
                    entries.append((i, j, text))
                    mask[i, j] = True
        truncated = 0
        for offset in range(0, len(entries), batch_size):
            items = entries[offset:offset + batch_size]
            if debug:
                values = np.stack([np.random.default_rng(int.from_bytes(
                    hashlib.sha256(text.encode()).digest()[:8], 'little')).standard_normal(width)
                    for _, _, text in items]).astype(np.float32)
            else:
                texts = [text for _, _, text in items]
                lengths = tokenizer(texts, truncation=False, padding=False)['input_ids']
                truncated += sum(len(x) > max_length for x in lengths)
                tokens = tokenizer(texts, return_tensors='pt', padding=True,
                                   truncation=True, max_length=max_length).to(device)
                with torch.inference_mode():
                    hidden = model(**tokens).last_hidden_state.float()
                    attention = tokens['attention_mask'].unsqueeze(-1)
                    values = ((hidden * attention).sum(1) / attention.sum(1).clamp_min(1)).cpu().numpy()
            for (i, j, _), value in zip(items, values):
                vectors[i, j] = value
            if offset % (batch_size * 50) == 0:
                print(f'Clinical cache {split}: {offset + len(items)}/{len(entries)} fields', flush=True)
        target = destination / f'{split}.npz'
        temporary = destination / f'{split}.tmp'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, ids=np.asarray(ids), vectors=vectors, mask=mask)
        temporary.replace(target)
        write_json(destination / f'{split}.json', {
            'sha256': file_hash(target), 'metadata_sha256': object_hash(metadata),
            'context_sha256': file_hash(path), 'cases': len(rows), 'nonempty_fields': len(entries),
            'truncated_fields': truncated, 'split': split})
    return metadata


class ClinicalCache:
    def __init__(self, directory, release_id, *, splits=('train', 'validation'), allow_debug=False):
        directory = Path(directory)
        self.metadata = read_json(directory / 'metadata.json')
        if self.metadata['request']['release_id'] != release_id:
            raise ValueError('Clinical cache release mismatch.')
        if self.metadata['request']['debug'] and not allow_debug:
            raise ValueError('A debug cache cannot be used for research training.')
        self.dimension = self.metadata['dimension']
        self.lookup, self.split_info = {}, {}
        for split in splits:
            info = read_json(directory / f'{split}.json')
            if info['metadata_sha256'] != object_hash(self.metadata):
                raise ValueError('Clinical cache metadata mismatch.')
            if file_hash(directory / f'{split}.npz') != info['sha256']:
                raise ValueError('Clinical cache checksum mismatch.')
            self.split_info[split] = info
            with np.load(directory / f'{split}.npz', allow_pickle=False) as data:
                for case, vector, mask in zip(data['ids'], data['vectors'], data['mask']):
                    if str(case) in self.lookup:
                        raise ValueError('Duplicate clinical case across splits.')
                    self.lookup[str(case)] = (torch.from_numpy(vector.copy()), torch.from_numpy(mask.copy()))
        self.fingerprint = object_hash({'metadata': self.metadata, 'splits': self.split_info})

    def batch(self, case_ids, device, *, allow_radiographs=True, xray_available=None):
        pairs = [self.lookup[case] for case in case_ids]  # Missing case is an error, not missing text.
        vectors = torch.stack([p[0] for p in pairs]).to(device)
        mask = torch.stack([p[1] for p in pairs]).to(device)
        if not allow_radiographs:
            mask[:, 1] = False
        elif xray_available is not None:
            mask[:, 1] &= xray_available
        return vectors * mask.unsqueeze(-1), mask
