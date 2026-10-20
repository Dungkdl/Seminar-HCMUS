"""Ordered experiment plan and resumable execution; no data preparation or test access."""
import gc
from dataclasses import replace
from pathlib import Path

import torch

from .artifacts import code_hash, object_hash, read_json, write_json
from .engine import fit

CORE = ['single_photo', 'photo_mean', 'photo_attention', 'photo_set_attention', 'photo_xray_attention',
        'teacher_clinical', 'teacher_full', 'teacher_v', 'teacher_h',
        'kd_clinical', 'kd_full', 'kd_equal', 'kd_selective']
ABLATIONS = ['kd_higher_h', 'kd_feature', 'kd_relation']


def experiment_config(base, name, seed):
    if name in CORE[:5]:
        return replace(base, seed=seed, experiment=name, method='supervised',
                       feature_kd_weight=0., relation_kd_weight=0.)
    changes = {'seed': seed, 'experiment': 'photo_attention', 'method': name,
               'feature_kd_weight': 0., 'relation_kd_weight': 0.}
    if name == 'kd_higher_h':
        changes.update(method='kd_selective', v_weight=1., h_weight=3.)
    elif name == 'kd_feature':
        changes.update(method='kd_full', feature_kd_weight=1.)
    elif name == 'kd_relation':
        changes.update(method='kd_full', relation_kd_weight=1.)
    elif name not in CORE:
        raise ValueError(f'Unknown experiment: {name}')
    return replace(base, **changes)


def expand_plan(selected):
    required = set(selected)
    unknown = required - set(CORE + ABLATIONS)
    if unknown:
        raise ValueError(f'Unknown experiments: {unknown}')
    for name in selected:
        if name == 'kd_clinical':
            required.add('teacher_clinical')
        elif name in {'kd_full', 'kd_feature', 'kd_relation'}:
            required.add('teacher_full')
        elif name in {'kd_equal', 'kd_selective', 'kd_higher_h'}:
            required.update(['teacher_v', 'teacher_h'])
    return [name for name in CORE + ABLATIONS if name in required]


def run_suite(release_dir, image_root, runs_root, base_config, *, clinical_cache=None,
              selected=None, seeds=(2026, 2027, 2028), resume=True, device=None):
    """Stages can be selected across sessions; each existing run keeps a strict identity."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique.')
    plan = expand_plan(CORE if selected is None else selected)
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    config_lock = {'base_config': base_config.to_dict(), 'code_sha256': code_hash(),
                   'release_id': read_json(Path(release_dir) / 'release.json')['release_id']}
    lock = root / 'protocol.json'
    if lock.exists() and object_hash(read_json(lock)) != object_hash(config_lock):
        raise ValueError('Suite protocol changed. Choose a new RUN_NAME, preserving previous results.')
    write_json(lock, config_lock)
    results = []
    for seed in seeds:
        for name in plan:
            directory = root / name / f'seed_{seed}'
            config = experiment_config(base_config, name, seed)
            teacher_dirs = {g: root / f'teacher_{g}' / f'seed_{seed}' for g in ['clinical', 'full', 'v', 'h']}
            print(f'\n=== {name} | seed {seed} ===', flush=True)
            result = fit(release_dir, image_root, directory, config,
                         clinical_cache=clinical_cache, teacher_dirs=teacher_dirs,
                         resume=resume and (directory / 'last.pt').exists(), device=device)
            results.append({'experiment': name, 'seed': seed, **result})
            write_json(root / 'latest_session.json', results)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results
