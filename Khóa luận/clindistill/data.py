"""Training consumes prepared manifests only; never reads or re-splits raw CSV."""
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def load_records(release_dir, split, experiment, debug_max_cases=0):
    if split not in {'train', 'validation', 'test'}:
        raise ValueError('Invalid partition.')
    with (Path(release_dir) / f'student_single_{split}.jsonl').open(encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    ids = set()
    for r in records:
        if r['split'] != split or r['case_id'] in ids or not r['eligible_single']:
            raise ValueError('Manifest partition/ID/cohort mismatch.')
        ids.add(r['case_id'])
        if len(r['target_labels']) != 1 or sum(r['labels_multihot']) != 1:
            raise ValueError('This trainer accepts the single-target cohort only.')
    # Same photo-available cohort in all four ablations, including photo+X-ray.
    records = sorted((r for r in records if r['photo_paths']), key=lambda r: r['case_id'])
    if debug_max_cases:
        # Deterministic round-robin by class gives more useful debug coverage than a file prefix.
        by_class = {}
        for r in records:
            by_class.setdefault(r['single_label_index'], []).append(r)
        selected, offset = [], 0
        while len(selected) < min(debug_max_cases, len(records)):
            for label in sorted(by_class):
                if offset < len(by_class[label]) and len(selected) < debug_max_cases:
                    selected.append(by_class[label][offset])
            offset += 1
        records = selected
    if not records:
        raise ValueError(f'No photo-available cases in {split}.')
    return records


def check_disjoint(train, validation):
    for key in ['case_id', 'patient_id', 'split_group', 'bag_signature']:
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f'Train/validation overlap in {key}.')


def deterministic_paths(paths, count):
    """Fixed evenly spaced filename order, not a clinically defined canonical view."""
    paths = sorted(paths)
    if len(paths) <= count:
        return paths
    if count == 1:
        return paths[:1]
    indexes = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[i] for i in indexes]


class BagDataset(Dataset):
    def __init__(self, records, image_root, config, training=False):
        self.records = records
        self.root = Path(image_root).resolve()
        self.config = config
        self.training = training
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        r = self.records[index]
        seed = int.from_bytes(hashlib.sha256(f'{self.config.seed}:{self.epoch}:{r["case_id"]}'.encode()).digest()[:8], 'little')
        rng = random.Random(seed)
        photos = sorted(r['photo_paths'])
        if self.config.experiment == 'single_photo':
            photos = deterministic_paths(photos, 1)
        elif self.training and len(photos) > self.config.max_photos:
            photos = sorted(rng.sample(photos, self.config.max_photos))
        else:
            photos = deterministic_paths(photos, self.config.max_photos)
        xrays = []
        if self.config.experiment == 'photo_xray_attention':
            paths = sorted(r['radiograph_paths'])
            xrays = (sorted(rng.sample(paths, self.config.max_radiographs))
                     if self.training and len(paths) > self.config.max_radiographs
                     else deterministic_paths(paths, self.config.max_radiographs))
        images, modalities = [], []
        for modality, paths in [(0, photos), (1, xrays)]:
            for relative in paths:
                path = (self.root / relative).resolve()
                if self.root not in path.parents:
                    raise ValueError('Image path escapes DATA_ROOT.')
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                # Preserve the field of view; no random crop that can remove the lesion.
                image = ImageOps.pad(image, (self.config.image_size, self.config.image_size),
                                     method=Image.Resampling.BILINEAR, color=(0, 0, 0))
                if self.training and modality == 0:
                    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
                    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
                tensor = TF.to_tensor(image)
                images.append(TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
                modalities.append(modality)
        if not images:
            raise ValueError(f'Empty bag: {r["case_id"]}')
        return {'images': torch.stack(images), 'modalities': torch.tensor(modalities),
                'label': int(r['single_label_index']), 'case_id': r['case_id'],
                'patient_id': r['patient_id'], 'split_group': r['split_group'], 'bag_signature': r['bag_signature']}


def collate_bags(items):
    b, n = len(items), max(len(x['images']) for x in items)
    image_shape = items[0]['images'].shape[1:]
    images = torch.zeros((b, n, *image_shape), dtype=torch.float32)
    mask = torch.zeros((b, n), dtype=torch.bool)
    modalities = torch.zeros((b, n), dtype=torch.long)
    for i, item in enumerate(items):
        count = len(item['images'])
        images[i, :count] = item['images']
        mask[i, :count] = True
        modalities[i, :count] = item['modalities']
    return {'images': images, 'mask': mask, 'modalities': modalities,
            'labels': torch.tensor([x['label'] for x in items]),
            **{key: [x[key] for x in items] for key in ['case_id', 'patient_id', 'split_group', 'bag_signature']}}
