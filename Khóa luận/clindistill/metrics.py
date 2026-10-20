import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(labels, probabilities, class_names):
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    classes = np.arange(len(class_names))
    if len(y) == 0 or p.shape != (len(y), len(classes)) or not np.isfinite(p).all():
        raise ValueError('Invalid predictions.')
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=classes)
    support = cm.sum(axis=1)
    recalls = np.divide(cm.diagonal(), support, out=np.zeros(len(classes), dtype=float), where=support > 0)
    aucs = {name: (float(roc_auc_score(y == c, p[:, c])) if 0 < (y == c).sum() < len(y) else None)
            for c, name in enumerate(class_names)}
    confidence, correct = p.max(axis=1), (pred == y)
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (confidence >= left) & ((confidence < right) if right < 1 else (confidence <= right))
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {'cases': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)),
            'balanced_accuracy_observed_classes': float(recalls[support > 0].mean()),
            'per_class_support': dict(zip(class_names, map(int, support))),
            'per_class_recall': {name: float(recalls[c]) if support[c] else None for c, name in enumerate(class_names)},
            'per_class_auroc': aucs,
            'macro_auroc_defined_classes': float(np.mean([v for v in aucs.values() if v is not None])) if any(v is not None for v in aucs.values()) else None,
            'brier_multiclass': float(np.mean(np.sum((p - np.eye(len(classes))[y]) ** 2, axis=1))),
            'ece_10_bins': float(ece), 'confusion_matrix': cm.tolist(),
            'macro_f1_includes_absent_classes_as_zero': True}


def group_bootstrap(labels, probabilities, group_ids, repetitions=1000, seed=2026):
    """Cluster bootstrap retains all cases of each sampled duplicate-linked patient group."""
    y, p, groups = np.asarray(labels), np.asarray(probabilities), np.asarray(group_ids)
    unique = np.unique(groups)
    if len(unique) < 2:
        return {'unit': 'split_group', 'groups': len(unique), 'macro_f1_95_ci': None}
    rng = np.random.default_rng(seed)
    indexes = {g: np.flatnonzero(groups == g) for g in unique}
    scores, classes = [], np.arange(p.shape[1])
    predictions = p.argmax(axis=1)
    for _ in range(repetitions):
        ix = np.concatenate([indexes[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        scores.append(f1_score(y[ix], predictions[ix], labels=classes, average='macro', zero_division=0))
    return {'unit': 'split_group', 'groups': len(unique), 'repetitions': repetitions,
            'macro_f1_95_ci': np.quantile(scores, [0.025, 0.975]).tolist()}
