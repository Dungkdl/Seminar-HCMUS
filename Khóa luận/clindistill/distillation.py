"""Clinical teachers and frozen-teacher KD; student remains VisualBagClassifier."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import code_hash, file_hash, read_json
from .clinical import ClinicalCache
from .config import TrainConfig
from .models import VisualBagClassifier


def grouped_kd(student_logits, teacher_logits, available, weights, temperature):
    """Mean over the whole batch; missing groups get zero weight, no-context cases CE only."""
    s = student_logits.float().unsqueeze(1) / temperature
    t = teacher_logits.detach().float() / temperature
    divergence = F.kl_div(F.log_softmax(s.expand_as(t), -1), F.softmax(t, -1),
                          reduction='none').sum(-1) * temperature ** 2
    w = torch.as_tensor(weights, dtype=torch.float32, device=s.device).unsqueeze(0) * available
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    return (divergence * w).sum(1).mean()


def representation_losses(student, teacher, available):
    """Normalized feature MSE and off-diagonal cosine-relation MSE on available cases."""
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.detach().float(), dim=-1)
    feature = ((student - teacher).square().mean(-1) * available).mean()
    pairs = available[:, None] & available[None, :]
    pairs &= ~torch.eye(len(student), dtype=torch.bool, device=student.device)
    relation = (((student @ student.T - teacher @ teacher.T).square()) * pairs).sum() / pairs.sum().clamp_min(1)
    return feature, relation


class ClinicalTeacher(nn.Module):
    def __init__(self, config, dimension, classes, load_pretrained=True):
        super().__init__()
        self.group = config.method.removeprefix('teacher_')
        self.field_dropout = config.field_dropout
        self.visual = (None if self.group == 'clinical' else
                       VisualBagClassifier(config, classes, load_pretrained=load_pretrained))
        # Field-specific projections preserve the meaning of each field.
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(dimension, config.feature_dim),
                                                        nn.ReLU()) for _ in range(5)])
        width = config.feature_dim * (1 if self.visual is None else 2) + 5
        self.head = nn.Sequential(nn.Linear(width, config.feature_dim), nn.ReLU(),
                                  nn.Dropout(config.dropout), nn.Linear(config.feature_dim, classes))

    def forward(self, images, image_mask, modalities, vectors, field_mask):
        field_mask = field_mask.clone()
        if self.group == 'v':
            field_mask[:, 2:] = False
        elif self.group == 'h':
            field_mask[:, :2] = False
        available = field_mask.any(1)
        if self.training and self.field_dropout:
            field_mask &= torch.rand_like(field_mask, dtype=torch.float32) >= self.field_dropout
        features = torch.stack([layer(vectors[:, i]) for i, layer in enumerate(self.projections)], dim=1)
        pooled = (features * field_mask.unsqueeze(-1)).sum(1) / field_mask.sum(1, keepdim=True).clamp_min(1)
        inputs = [pooled, field_mask.to(pooled.dtype)]
        if self.visual is not None:
            inputs.insert(0, self.visual(images, image_mask, modalities)['features'])
        fused = self.head[:2](torch.cat(inputs, -1))
        return {'logits': self.head[2:](fused), 'available': available, 'features': fused}


class TrainingRuntime:
    """Owns cache/teachers separately so KD checkpoints contain only the visual student."""
    def __init__(self, config, release, device, *, clinical_cache=None, teacher_dirs=None,
                 resume=False, splits=('train', 'validation'), training=True):
        self.config, self.device = config, device
        self.cache, self.teachers, self.dependencies = None, {}, {}
        needs_cache = config.method.startswith('teacher_') or (training and config.method.startswith('kd_'))
        if needs_cache:
            if clinical_cache is None:
                raise ValueError('Run notebook 01 clinical encoding first and supply clinical_cache.')
            self.cache = ClinicalCache(clinical_cache, release['release_id'], splits=splits,
                                       allow_debug=config.backbone == 'tiny_debug')
            for split, info in self.cache.split_info.items():
                if info['context_sha256'] != release['file_checksums'][f'teacher_context_{split}.jsonl']:
                    raise ValueError('Clinical context differs from prepared release.')
            self.dependencies['clinical_cache'] = self.cache.fingerprint
        self.model = (ClinicalTeacher(config, self.cache.dimension, len(release['class_names']),
                                     load_pretrained=not resume) if config.method.startswith('teacher_')
                      else VisualBagClassifier(config, len(release['class_names']), load_pretrained=not resume)).to(device)
        if training and config.method.startswith('kd_'):
            groups = {'kd_clinical': ['clinical'], 'kd_full': ['full'],
                      'kd_equal': ['v', 'h'], 'kd_selective': ['v', 'h']}[config.method]
            for group in groups:
                if group not in (teacher_dirs or {}):
                    raise ValueError(f'Missing independently trained teacher: {group}')
                folder = Path(teacher_dirs[group])
                info = read_json(folder / 'run_config.json')
                if not read_json(folder / 'summary.json')['complete']:
                    raise ValueError('Teacher training is incomplete.')
                if info['release_id'] != release['release_id'] or info['code_sha256'] != code_hash():
                    raise ValueError('Teacher data/code identity mismatch.')
                if info.get('dependencies', {}).get('clinical_cache') != self.cache.fingerprint:
                    raise ValueError('Teacher clinical cache mismatch.')
                teacher_config = TrainConfig(**info['config'])
                if teacher_config.method != f'teacher_{group}':
                    raise ValueError('Wrong teacher method.')
                # Same cohort, visual inputs, augmentation seed and architecture in controlled comparisons.
                for key in ['experiment', 'backbone', 'image_size', 'max_photos', 'max_radiographs',
                            'feature_dim', 'seed', 'debug_max_cases']:
                    if getattr(config, key) != getattr(teacher_config, key):
                        raise ValueError(f'Teacher/student protocol mismatch: {key}')
                state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
                if state['run_fingerprint'] != info['run_fingerprint']:
                    raise ValueError('Teacher checkpoint identity mismatch.')
                teacher = ClinicalTeacher(teacher_config, self.cache.dimension, len(release['class_names']),
                                          load_pretrained=False).to(device)
                teacher.load_state_dict(state['model'])
                self.teachers[group] = teacher.eval().requires_grad_(False)
                self.dependencies[f'teacher_{group}'] = file_hash(folder / 'best.pt')

    def clinical(self, batch):
        xray = ((batch['modalities'] == 1) & batch['mask']).any(1).to(self.device)
        return self.cache.batch(batch['case_id'], self.device,
                                allow_radiographs=self.config.experiment == 'photo_xray_attention',
                                xray_available=xray)

    def forward(self, batch):
        args = [batch[k].to(self.device, non_blocking=True) for k in ['images', 'mask', 'modalities']]
        if self.config.method.startswith('teacher_'):
            return self.model(*args, *self.clinical(batch))
        return self.model(*args)

    def distillation_loss(self, batch, output):
        if not self.teachers:
            return output['logits'].sum() * 0.
        args = [batch[k].to(self.device, non_blocking=True) for k in ['images', 'mask', 'modalities']]
        vectors, mask = self.clinical(batch)
        with torch.no_grad():
            outputs = [teacher(*args, vectors, mask) for teacher in self.teachers.values()]
        weights = ([self.config.v_weight, self.config.h_weight] if self.config.method == 'kd_selective'
                   else [1.] * len(outputs))
        result = grouped_kd(output['logits'], torch.stack([x['logits'] for x in outputs], 1),
                            torch.stack([x['available'] for x in outputs], 1), weights, self.config.temperature)
        result = self.config.kd_weight * result
        if self.config.feature_kd_weight or self.config.relation_kd_weight:
            feature, relation = representation_losses(output['features'], outputs[0]['features'],
                                                      outputs[0]['available'])
            result = result + self.config.feature_kd_weight * feature + self.config.relation_kd_weight * relation
        return result
