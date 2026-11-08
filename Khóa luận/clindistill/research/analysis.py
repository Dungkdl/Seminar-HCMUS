"""Descriptive Teacher disagreement and gate diagnostics, not clinical ground truth."""
import numpy as np

from ..metrics import classification_metrics


def teacher_strata(visual, multimodal, labels):
    v = np.asarray(visual).argmax(-1) == np.asarray(labels)
    m = np.asarray(multimodal).argmax(-1) == np.asarray(labels)
    return {'both_correct': v & m, 'multimodal_only_correct': ~v & m,
            'visual_only_correct': v & ~m, 'both_wrong': ~v & ~m}


def weight_diagnostics(weights, labels, class_names, strata):
    labels = np.asarray(labels)
    def describe(values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return {'cases': 0, 'mean': None, 'zero_fraction': None, 'effective_cases': 0.}
        squares = float((values ** 2).sum())
        return {'cases': len(values), 'mean': float(values.mean()),
                'zero_fraction': float((values <= 1e-12).mean()),
                'effective_cases': float(values.sum() ** 2 / squares) if squares else 0.}
    return {key: {'all': describe(value),
                  'by_class': {name: describe(value[labels == i]) for i, name in enumerate(class_names)},
                  'by_teacher_stratum': {name: describe(value[mask]) for name, mask in strata.items()}}
            for key, value in weights.items()}


def stratum_metrics(labels, probabilities, class_names, strata):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    return {name: classification_metrics(labels[mask], probabilities[mask], class_names) if mask.any()
            else {'cases': 0} for name, mask in strata.items()}
