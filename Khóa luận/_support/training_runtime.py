"""Shared notebook orchestration for controlled, separately resumable backbone studies."""
import gc
import json
import os
from pathlib import Path
import shutil
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from clindistill.artifacts import code_hash, file_hash, object_hash, read_json, verify_release, write_json
from clindistill.clinical import ClinicalCache
from clindistill.config import TrainConfig
from clindistill.data import load_records, check_disjoint
from clindistill.models import VisualBagClassifier
from clindistill.reporting import export_student
from clindistill.suite import expand_plan, run_suite


def execution_plan(backbones, seeds, experiments):
    if not backbones or len(backbones) != len(set(backbones)):
        raise ValueError('Backbones rỗng hoặc trùng.')
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Seeds rỗng hoặc trùng.')
    return [{'backbone': backbone, 'experiment': method, 'seed': seed}
            for backbone in backbones for seed in seeds for method in expand_plan(experiments)]


def restore_study(root, protocol, *, previous=None, search_root='/kaggle/input'):
    root = Path(root)
    if (root / 'study_protocol.json').exists():
        if object_hash(read_json(root / 'study_protocol.json')) != object_hash(protocol):
            raise ValueError('Study đã có protocol khác; đặt RUN_NAME mới.')
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError('Study folder không có protocol hoàn chỉnh; giữ nguyên và dùng RUN_NAME mới.')
    if previous is None:
        matches = []
        if Path(search_root).exists():
            for current, dirs, files in os.walk(search_root):
                here = Path(current)
                if 'study_protocol.json' in files and here.name == root.name:
                    if object_hash(read_json(here / 'study_protocol.json')) == object_hash(protocol):
                        matches.append(here)
                    dirs[:] = []
                else:
                    dirs[:] = [d for d in dirs if d not in {'Images', '.cache', '__pycache__', 'models', 'wheels'}]
                if len(here.relative_to(search_root).parts) >= 9:
                    dirs[:] = []
        if len(matches) > 1:
            raise ValueError('Có nhiều Output resume phù hợp; đặt PREVIOUS_RUNS_OVERRIDE.')
        previous = matches[0] if matches else None
    if previous is not None:
        previous = Path(previous)
        if object_hash(read_json(previous / 'study_protocol.json')) != object_hash(protocol):
            raise ValueError('Output cũ khác source/config/release/cache. Dùng đúng protocol hoặc RUN_NAME mới.')
        print('Khôi phục study:', previous, flush=True)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = root.with_name(root.name + '.restoring')
        shutil.copytree(previous, staging, dirs_exist_ok=True)
        if root.exists():
            # Only an empty destination is allowed here; never replace existing progress.
            root.rmdir()
        staging.rename(root)


def prepare_study(project_root, prepared_root, image_root, clinical_cache, output_root, *,
                  run_name, backbones, seeds, experiments, assets=None, resume=True, previous=None,
                  smoke=False, check_images=True, batch_size=1, accumulation_steps=16,
                  search_root='/kaggle/input'):
    project, prepared, images, output = map(Path, (project_root, prepared_root, image_root, output_root))
    if not run_name or Path(run_name).name != run_name or run_name in {'.', '..'}:
        raise ValueError('RUN_NAME phải là tên một folder.')
    plan = execution_plan(backbones, seeds, experiments)
    release = verify_release(prepared, splits=['train', 'validation'])
    if release.get('debug_subset') and not smoke:
        raise ValueError('Đang gắn bộ mini; cần prepared release đầy đủ.')
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError('Chưa nhận GPU; chọn Accelerator rồi Run All lại.')
    device = torch.device('cuda' if torch.cuda.is_available() and not smoke else 'cpu')
    template = TrainConfig.load(project / 'configs/code' / ('smoke_cpu.json' if smoke else 'thesis.json'))
    configs = {}
    weights = dict((assets or {}).get('backbone_weights', {}))
    if assets and assets.get('resnet'):
        weights.setdefault('resnet50', assets['resnet'])
    for backbone in backbones:
        config = replace(template, backbone=backbone, backbone_weights_path=weights.get(backbone),
                         batch_size=batch_size, accumulation_steps=accumulation_steps)
        if not smoke:
            if config.debug_max_cases or not config.pretrained or config.freeze_backbone:
                raise ValueError('Full run cần pretrained, fine-tuning và debug_max_cases=0.')
            if not config.backbone_weights_path or not Path(config.backbone_weights_path).is_file():
                raise FileNotFoundError(f'Thiếu weights local cho {backbone}. Gắn code-swin-addon và bundle gốc.')
        configs[backbone] = config
    rows = {split: load_records(prepared, split, 'photo_attention', template.debug_max_cases)
            for split in ('train', 'validation')}
    check_disjoint(rows['train'], rows['validation'])
    counts = np.bincount([r['single_label_index'] for r in rows['train']], minlength=len(release['class_names']))
    if (counts == 0).any():
        raise ValueError('Train thiếu lớp; kiểm tra prepared release.')
    methods = expand_plan(experiments)
    cache_fingerprint = None
    if any(name.startswith(('teacher_', 'kd_')) for name in methods):
        if clinical_cache is None:
            raise FileNotFoundError('Gắn Output notebook 01 có clinical cache.')
        cache = ClinicalCache(clinical_cache, release['release_id'], allow_debug=smoke)
        for split, records in rows.items():
            if any(r['case_id'] not in cache.lookup for r in records):
                raise ValueError(f'Clinical cache thiếu ca {split}.')
            if cache.split_info[split]['context_sha256'] != release['file_checksums'][f'teacher_context_{split}.jsonl']:
                raise ValueError('Clinical context không khớp prepared release.')
        cache_fingerprint = cache.fingerprint
        del cache
    required_images = {p for records in rows.values() for r in records for p in r['photo_paths']}
    if 'photo_xray_attention' in methods:
        required_images.update(p for records in rows.values() for r in records for p in r['radiograph_paths'])
    if check_images:
        print(f'Kiểm tra đường dẫn {len(required_images):,} ảnh...', flush=True)
        image_base = images.resolve()
        for relative in sorted(required_images):
            path = (image_base / relative).resolve()
            if image_base not in path.parents or not path.is_file():
                raise FileNotFoundError(f'Ảnh thiếu hoặc đường dẫn không hợp lệ: {relative}')
    parameters = {}
    for backbone, config in configs.items():
        print('Kiểm tra pretrained forward/AMP:', backbone, flush=True)
        model = VisualBagClassifier(config, len(release['class_names'])).to(device).eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=config.amp and device.type == 'cuda'):
            result = model(torch.zeros(1, 2, 3, config.image_size, config.image_size, device=device),
                torch.ones(1, 2, dtype=torch.bool, device=device), torch.zeros(1, 2, dtype=torch.long, device=device))
        if not torch.isfinite(result['logits']).all():
            raise FloatingPointError(f'{backbone}: forward có giá trị không hữu hạn.')
        parameters[backbone] = sum(p.numel() for p in model.parameters())
        del model, result
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    root = output / ('_support/smoke_runs' if smoke else 'runs') / run_name
    protocol = {'code_sha256': code_hash(), 'orchestration_sha256': file_hash(__file__),
                'release_id': release['release_id'], 'cache_fingerprint': cache_fingerprint,
                'configs': {name: config.to_dict() for name, config in configs.items()}}
    if resume:
        restore_study(root, protocol, previous=previous, search_root=search_root)
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError('Study đã tồn tại. Bật RESUME hoặc đặt RUN_NAME mới.')
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'study_protocol.json', protocol)
    report = root / 'training_report'
    report.mkdir(exist_ok=True)
    preflight = {'release_id': release['release_id'], 'release_stage': release['stage'],
        'pending_reviews': release.get('pending_reviews', []), 'code_sha256': code_hash(),
        'cache_fingerprint': cache_fingerprint, 'debug_run': smoke,
        'train_cases': len(rows['train']), 'validation_cases': len(rows['validation']),
        'checked_image_paths': len(required_images) if check_images else 0,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        'torch': torch.__version__, 'device': str(device), 'configs': protocol['configs'],
        'backbones': backbones, 'requested_experiments': experiments, 'expanded_plan': methods,
        'seeds': seeds, 'planned_runs': len(plan), 'student_parameters': parameters,
        'test_evaluated': False, 'publication_ready': False}
    write_json(report / 'preflight.json', preflight)
    write_json(report / 'execution_plan.json', plan)
    print(f"PREFLIGHT PASS | train={len(rows['train']):,}, val={len(rows['validation']):,} | {len(plan)} lượt")
    print('Mỗi lượt dùng một GPU; backbones và seeds chạy tuần tự. Kết quả:', root)
    print('Release:', release['stage'], '| pending reviews:', len(release.get('pending_reviews', [])))
    return {'root': root, 'report': report, 'plan': plan, 'methods': methods, 'configs': configs,
            'prepared': prepared, 'images': images, 'cache': clinical_cache, 'release': release,
            'seeds': seeds, 'experiments': experiments, 'device': str(device), 'resume': resume,
            'smoke': smoke, 'parameters': parameters, 'started': time.monotonic()}


def collect_progress(study):
    rows = []
    for item in study['plan']:
        folder = study['root'] / item['backbone'] / item['experiment'] / f"seed_{item['seed']}"
        row = {**item, 'complete': False, 'epochs_completed': 0, 'best_validation_macro_f1': None}
        if (folder / 'summary.json').exists():
            row.update(read_json(folder / 'summary.json'))
        elif (folder / 'history.json').exists():
            history = read_json(folder / 'history.json')
            row['epochs_completed'] = len(history)
            row['best_validation_macro_f1'] = max((h['validation_macro_f1'] for h in history), default=None)
        if (folder / 'best_validation_metrics.json').exists():
            row.update(read_json(folder / 'best_validation_metrics.json'))
        rows.append(row)
    return rows


def train_study(study, *, run_train=True):
    error = None
    try:
        if run_train:
            for backbone, config in study['configs'].items():
                for seed in study['seeds']:
                    pending = any(not r['complete'] for r in collect_progress(study)
                                  if r['backbone'] == backbone and r['seed'] == seed)
                    free = shutil.disk_usage(study['root']).free
                    if pending and not study['smoke'] and free < max(1024**3, study['parameters'][backbone] * 32):
                        raise OSError('Không đủ chỗ ghi checkpoint nguyên tử; lưu Output và resume trên phiên có đủ chỗ.')
                    print(f'\n===== {backbone} | SEED {seed} | {len(study["methods"])} phương pháp =====', flush=True)
                    run_suite(study['prepared'], study['images'], study['root'] / backbone, config,
                              clinical_cache=study['cache'], selected=study['experiments'], seeds=[seed],
                              resume=study['resume'], device=study['device'])
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        progress = collect_progress(study)
        pd.DataFrame(progress).to_csv(study['report'] / 'run_progress.csv', index=False)
        completed = sum(bool(r['complete']) for r in progress)
        write_json(study['report'] / 'suite_status.json', {'complete': completed == len(study['plan']),
            'completed_runs': completed, 'planned_runs': len(study['plan']), 'error': error,
            'session_seconds': time.monotonic() - study['started'], 'test_evaluated': False,
            'debug_run': study['smoke'], 'release_stage': study['release']['stage'], 'publication_ready': False})
        print(f'Đã hoàn thành {completed}/{len(study["plan"])} lượt. Output:', study['root'], flush=True)


def report_study(study):
    import matplotlib.pyplot as plt
    progress = pd.DataFrame(collect_progress(study))
    progress.to_csv(study['report'] / 'per_seed_validation.csv', index=False)
    complete = progress[progress.complete.astype(bool)].copy()
    columns = ['macro_f1', 'accuracy', 'balanced_accuracy_observed_classes',
               'macro_auroc_defined_classes', 'ece_10_bins', 'brier_multiclass']
    table = complete.groupby(['backbone', 'experiment'])[columns].agg(['mean', 'std', 'count']) if not complete.empty else pd.DataFrame()
    table.to_csv(study['report'] / 'validation_mean_std.csv')
    students = complete[complete.experiment.isin(study['experiments'])]
    exports = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for row in students.to_dict('records'):
        if row['experiment'].startswith('teacher_'):
            continue
        folder = study['root'] / row['backbone'] / row['experiment'] / f"seed_{row['seed']}"
        target = study['root'] / 'exports' / row['backbone'] / row['experiment'] / f"seed_{row['seed']}" / 'student.pt'
        artifact = export_student(folder, target)
        exports.append({'backbone': row['backbone'], 'experiment': row['experiment'], 'seed': row['seed'],
                        'validation_macro_f1': row['best_validation_macro_f1'], 'path': str(artifact), 'sha256': file_hash(artifact)})
        history = pd.DataFrame(read_json(folder / 'history.json'))
        label = f"{row['backbone']} / {row['experiment']} / {row['seed']}"
        for ax, column in zip(axes, ['train_ce', 'train_kd', 'validation_macro_f1']):
            ax.plot(history.epoch, history[column], label=label)
    write_json(study['report'] / 'student_exports.json', exports)
    for ax, title in zip(axes, ['Train CE', 'Train KD (weighted)', 'Validation macro-F1']):
        ax.set(xlabel='Epoch', title=title)
    if not students.empty:
        axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(study['report'] / 'learning_curves.png', dpi=160)
    plt.show()
    plt.close(fig)
    if not students.empty:
        comparison = students.groupby(['backbone', 'experiment']).agg(
            macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
            seeds=('seed', 'count'), parameters=('model_parameters', 'mean'),
            mean_peak_gpu_gib=('peak_gpu_memory_gib', 'mean'))
        comparison.to_csv(study['report'] / 'backbone_comparison.csv')
    else:
        comparison = pd.DataFrame()
    status = read_json(study['report'] / 'suite_status.json')
    print('TRAINING COMPLETE' if status['complete'] else 'TRAINING INCOMPLETE — giữ Output rồi resume')
    print('Báo cáo:', study['report'], '| Student:', study['root'] / 'exports')
    return {'per_seed': progress, 'mean_std': table, 'backbone_comparison': comparison, 'exports': exports}
