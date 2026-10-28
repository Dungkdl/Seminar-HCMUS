import math
import torch
from torch.nn import functional as F


def per_case_kl(student, teacher, temperature):
    if temperature <= 0:
        raise ValueError('Temperature must be positive.')
    return F.kl_div(F.log_softmax(student.float() / temperature, -1),
                    F.softmax(teacher.detach().float() / temperature, -1),
                    reduction='none').sum(-1) * temperature ** 2


def reliability_weights(visual_logits, multimodal_logits, labels, *, epsilon=0.1, gate_temperature=1.0):
    if not 0 < epsilon <= 1 or gate_temperature <= 0:
        raise ValueError('Invalid reliability parameters.')
    with torch.no_grad():
        pv = (visual_logits.float() / gate_temperature).softmax(-1)
        pm = (multimodal_logits.float() / gate_temperature).softmax(-1)
        confidence = (1 + (pm * pm.clamp_min(1e-12).log()).sum(-1) / math.log(pm.shape[-1])).clamp(0, 1)
        support = pv.gather(1, labels.long().unsqueeze(1)).squeeze(1)
        correct = pm.argmax(-1).eq(labels)
        proposed = correct * confidence * (epsilon + (1 - epsilon) * support)
    return {'confidence': confidence, 'reliability': proposed, 'multimodal_correct': correct, 'visual_support': support}


def weighted_kd(student, teacher, raw_weight, train_mean_weight, temperature):
    # The normalizer is fixed from OOF TRAIN, not recomputed on each microbatch.
    # Thus batch_size=1 and gradient accumulation preserve the reliability magnitude.
    if train_mean_weight <= 1e-12:
        raise ValueError('Reliability weights collapsed; inspect OOF Teacher quality.')
    weight = raw_weight.detach().float() / float(train_mean_weight)
    return (weight * per_case_kl(student, teacher, temperature)).mean()
