"""Generate the two thin, Kaggle-ready training/evaluation entry points."""
from pathlib import Path
import textwrap
import nbformat as nbf
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('offline_setup', ROOT / '_support/kaggle_offline.py')
offline_setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline_setup)
def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())

BOOTSTRAP = r'''
from pathlib import Path
import os, sys, json

PROJECT_ROOT_OVERRIDE = None  # folder chứa clindistill/ + configs/
PREPARED_ROOT_OVERRIDE = None  # folder chứa release.json
IMAGE_ROOT_OVERRIDE = None  # folder chứa Images/Photographs
CLINICAL_CACHE_OVERRIDE = None  # folder chứa metadata.json + train.npz + validation.npz

def find_roots(predicate):
    results = set()
    bases = [Path.cwd(), Path.cwd().parent / 'Data', Path('/kaggle/input')]
    for base in bases:
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base):
            here = Path(current)
            if predicate(here, dirs, files):
                results.add(here.resolve())
            dirs[:] = [d for d in dirs if d not in {'Images', '.git', '.cache', '__pycache__',
                'node_modules', 'runs', 'outputs', '_support', '_archive'}]
            if len(here.relative_to(base).parts) >= 7:
                dirs[:] = []
    return sorted(results, key=str)

def resolve_one(override, candidates, kind, optional=False):
    if override is not None:
        path = Path(override).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'{kind}: {path}')
        return path
    if optional and not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(f'{kind}: found {len(candidates)} folders: {candidates}. Set the corresponding *_OVERRIDE.')
    return candidates[0]

PROJECT_ROOT = resolve_one(PROJECT_ROOT_OVERRIDE,
    [OFFLINE_ASSETS['root'] / 'source'] if globals().get('OFFLINE_ASSETS') else
    [Path.cwd()] if (Path.cwd() / 'clindistill').is_dir() else
    find_roots(lambda p, dirs, files: 'clindistill' in dirs and 'configs' in dirs), 'Project source')
sys.path.insert(0, str(PROJECT_ROOT))
PREPARED_ROOT = resolve_one(PREPARED_ROOT_OVERRIDE,
    find_roots(lambda p, dirs, files: 'release.json' in files and
               not json.loads((p / 'release.json').read_text(encoding='utf-8')).get('debug_subset')), 'Prepared release')
IMAGE_ROOT = resolve_one(IMAGE_ROOT_OVERRIDE,
    find_roots(lambda p, dirs, files: 'Images' in dirs and (p / 'Images' / 'Photographs').is_dir()
               and 'mini_dataset.json' not in files), 'COde images')
CLINICAL_CACHE = resolve_one(CLINICAL_CACHE_OVERRIDE,
    find_roots(lambda p, dirs, files: {'metadata.json', 'train.npz', 'validation.npz'} <= set(files) and
        json.loads((p / 'metadata.json').read_text(encoding='utf-8'))['request']['release_id'] ==
        json.loads((PREPARED_ROOT / 'release.json').read_text(encoding='utf-8'))['release_id']),
    'Clinical cache', optional=True)
OUTPUT_ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else PROJECT_ROOT
print('Source:', PROJECT_ROOT, '\nPrepared:', PREPARED_ROOT, '\nImages:', IMAGE_ROOT,
      '\nClinical cache:', CLINICAL_CACHE, '\nOutput:', OUTPUT_ROOT)
'''

train = [md('''
# COde 02 — Huấn luyện mô hình khóa luận

**Gắn 3 input rồi Run All:** `code-offline-v1`, `code-training-inputs`, output notebook 01 có clinical cache.
Chọn GPU và tắt Internet khi chạy bằng bundle. Không chạy lại notebook 00/01 trong notebook này.

Mặc định: **Teacher V + Teacher H → selective KD**, 3 seed, dữ liệu đầy đủ, ResNet-50 pretrained,
40 epochs tối đa, early stopping theo validation macro-F1. Tổng **9 lượt train**, không chạy baseline riêng.
Student sau huấn luyện chỉ dùng ảnh. Kết quả chọn mô hình ở đây là **validation**; notebook 03 đánh giá theo protocol đã khóa.

Cell tiếp theo là nơi duy nhất chỉnh đường dẫn, seed, nhánh nghiên cứu hoặc resume.
'''), code(r'''
from pathlib import Path
import os, sys, json, time

# Cấu hình duy nhất. Bình thường giữ nguyên và Run All.
SMOKE_TEST = os.environ.get('CLINDISTILL_SMOKE_TEST') == '1'  # chỉ phục vụ kiểm tra phần mềm
OFFLINE_MODE = Path('/kaggle').exists() and not SMOKE_TEST
OFFLINE_BUNDLE_OVERRIDE = None
PROJECT_ROOT_OVERRIDE = None
PREPARED_ROOT_OVERRIDE = None
IMAGE_ROOT_OVERRIDE = None
CLINICAL_CACHE_OVERRIDE = None
OUTPUT_ROOT_OVERRIDE = None
PREVIOUS_RUNS_OVERRIDE = None  # folder RUN_NAME từ output phiên cũ; None = tự tìm

RUN_NAME = 'code_clindistill_v2'
SEEDS = [2026, 2027, 2028] if not SMOKE_TEST else [2026]
EXPERIMENTS = ['kd_selective']  # Teacher V/H cần thiết được thêm tự động
RESUME = True
RUN_TRAIN = True
CHECK_IMAGE_FILES = True
SESSION_STARTED = time.monotonic()

if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {'.', '..'}:
    raise ValueError('RUN_NAME phải là tên một folder.')
if not SEEDS or len(SEEDS) != len(set(SEEDS)):
    raise ValueError('Danh sách seed rỗng hoặc trùng.')
print('Chế độ:', 'DEBUG — không dùng làm kết quả nghiên cứu' if SMOKE_TEST else 'HUẤN LUYỆN ĐẦY ĐỦ')
'''), md('''
## 1. Tự thiết lập offline và tìm đầu vào

Kiểm tra checksum, phiên bản môi trường, cài từ wheels và dùng weights trong bundle.
Báo `Offline setup OK` rồi notebook tự tìm prepared release, ảnh và cache đúng release.
Nếu có nhiều phiên bản khác nhau, điền đường dẫn override ở cell cấu hình đầu tiên.
'''), code(offline_setup.BOOTSTRAP_CELL.replace(
    "OFFLINE_MODE = Path('/kaggle').exists()", '# OFFLINE_MODE lấy từ cell cấu hình đầu tiên.').replace(
    'OFFLINE_BUNDLE_OVERRIDE = None  # folder chứa offline_manifest.json',
    '# OFFLINE_BUNDLE_OVERRIDE lấy từ cell cấu hình đầu tiên.')),
code(BOOTSTRAP.replace('PROJECT_ROOT_OVERRIDE = None', 'PROJECT_ROOT_OVERRIDE = globals().get("PROJECT_ROOT_OVERRIDE")').replace(
    'PREPARED_ROOT_OVERRIDE = None', 'PREPARED_ROOT_OVERRIDE = globals().get("PREPARED_ROOT_OVERRIDE")').replace(
    'IMAGE_ROOT_OVERRIDE = None', 'IMAGE_ROOT_OVERRIDE = globals().get("IMAGE_ROOT_OVERRIDE")').replace(
    'CLINICAL_CACHE_OVERRIDE = None', 'CLINICAL_CACHE_OVERRIDE = globals().get("CLINICAL_CACHE_OVERRIDE")').replace(
    "OUTPUT_ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else PROJECT_ROOT",
    "OUTPUT_ROOT = Path(OUTPUT_ROOT_OVERRIDE) if OUTPUT_ROOT_OVERRIDE else (Path('/kaggle/working') if Path('/kaggle/working').exists() else PROJECT_ROOT)")),
md('''
## 2. Kiểm tra trước huấn luyện và lập kế hoạch

Kiểm tra GPU, pretrained forward với AMP, manifest train/validation, giao bệnh nhân/ảnh,
clinical cache, các ảnh được dùng và cấu hình resume. Không đọc partition test.
Ghi lại kế hoạch, release, source hash, cache fingerprint và cấu hình; không tự đổi batch/loss khi gặp lỗi.
'''), code(r'''
import gc, shutil
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from clindistill.config import TrainConfig
from clindistill.artifacts import code_hash, file_hash, object_hash, read_json, verify_release, write_json
from clindistill.clinical import ClinicalCache
from clindistill.data import load_records, check_disjoint
from clindistill.models import VisualBagClassifier
from clindistill.suite import CORE, ABLATIONS, expand_plan, run_suite

PLAN = expand_plan(EXPERIMENTS)
release = verify_release(PREPARED_ROOT, splits=['train', 'validation'])
if release.get('debug_subset') and not SMOKE_TEST:
    raise ValueError('Đang gắn bộ mini. Hãy gắn prepared release đầy đủ.')
if not SMOKE_TEST and not torch.cuda.is_available():
    raise RuntimeError('Chưa nhận GPU. Chọn Accelerator rồi chạy lại từ đầu.')
device = torch.device('cuda' if torch.cuda.is_available() and not SMOKE_TEST else 'cpu')
base_config = TrainConfig.load(PROJECT_ROOT / 'configs/code' / ('smoke_cpu.json' if SMOKE_TEST else 'thesis.json'))
from dataclasses import replace
BACKBONE_WEIGHTS_PATH = OFFLINE_ASSETS['resnet'] if globals().get('OFFLINE_ASSETS') and not SMOKE_TEST else None
base_config = replace(base_config, backbone_weights_path=BACKBONE_WEIGHTS_PATH)
if not SMOKE_TEST:
    if base_config.debug_max_cases or base_config.backbone == 'tiny_debug' or not base_config.pretrained:
        raise ValueError('Full run cần pretrained backbone và debug_max_cases=0.')
    if base_config.freeze_backbone:
        raise ValueError('Cấu hình full run phải fine-tune backbone.')
    if OFFLINE_MODE and (not BACKBONE_WEIGHTS_PATH or not Path(BACKBONE_WEIGHTS_PATH).is_file()):
        raise FileNotFoundError('Thiếu pretrained weights trong bundle offline.')

records = {split: load_records(PREPARED_ROOT, split, 'photo_attention', base_config.debug_max_cases)
           for split in ('train', 'validation')}
check_disjoint(records['train'], records['validation'])
counts = np.bincount([r['single_label_index'] for r in records['train']], minlength=len(release['class_names']))
if (counts == 0).any():
    raise ValueError('Train thiếu lớp; quay lại prepared release.')
cache_fingerprint = None
if any(name.startswith(('teacher_', 'kd_')) for name in PLAN):
    if CLINICAL_CACHE is None:
        raise FileNotFoundError('Gắn output notebook 01 có metadata.json, train.npz và validation.npz.')
    clinical_check = ClinicalCache(CLINICAL_CACHE, release['release_id'], allow_debug=SMOKE_TEST)
    for split, rows in records.items():
        missing = [r['case_id'] for r in rows if r['case_id'] not in clinical_check.lookup]
        if missing:
            raise ValueError(f'Clinical cache thiếu {len(missing)} ca {split}.')
    cache_fingerprint = clinical_check.fingerprint
    del clinical_check

required_images = {p for rows in records.values() for r in rows for p in r['photo_paths']}
if 'photo_xray_attention' in PLAN:
    required_images.update(p for rows in records.values() for r in rows for p in r['radiograph_paths'])
if CHECK_IMAGE_FILES:
    print(f'Kiểm tra đường dẫn {len(required_images):,} ảnh...', flush=True)
    image_base = IMAGE_ROOT.resolve()
    for relative in sorted(required_images):
        path = (image_base / relative).resolve()
        if image_base not in path.parents or not path.is_file():
            raise FileNotFoundError(f'Ảnh thiếu hoặc đường dẫn không hợp lệ: {relative}')

# Xác nhận pretrained forward/AMP trước khi bắt đầu các lượt train dài.
probe_model = VisualBagClassifier(base_config, num_classes=len(release['class_names'])).to(device).eval()
with torch.no_grad(), torch.autocast(device_type=device.type, enabled=base_config.amp and device.type == 'cuda'):
    probe = probe_model(torch.zeros(1, 2, 3, base_config.image_size, base_config.image_size, device=device),
        torch.ones(1, 2, dtype=torch.bool, device=device), torch.zeros(1, 2, dtype=torch.long, device=device))
if not torch.isfinite(probe['logits']).all():
    raise FloatingPointError('Pretrained forward có giá trị không hữu hạn.')
parameter_count = sum(p.numel() for p in probe_model.parameters())
del probe_model, probe
gc.collect()
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print('GPU:', torch.cuda.get_device_name(0), '| số GPU hiện có:', torch.cuda.device_count())
    print('Mỗi lượt train hiện sử dụng GPU 0; các lượt chạy tuần tự.')

RUNS_ROOT = (OUTPUT_ROOT / '_support/smoke_runs' / RUN_NAME if SMOKE_TEST else OUTPUT_ROOT / 'runs' / RUN_NAME)
expected_protocol = {'base_config': base_config.to_dict(), 'code_sha256': code_hash(), 'release_id': release['release_id']}
# Tự phục hồi nguyên suite từ output notebook train đã gắn vào Input.
previous = Path(PREVIOUS_RUNS_OVERRIDE).resolve() if PREVIOUS_RUNS_OVERRIDE else None
if previous is None and RESUME and not RUNS_ROOT.exists():
    matches = []
    if Path('/kaggle/input').exists():
        for current, dirs, files in os.walk('/kaggle/input'):
            here = Path(current)
            if 'protocol.json' in files and here.name == RUN_NAME:
                if object_hash(read_json(here / 'protocol.json')) == object_hash(expected_protocol):
                    matches.append(here)
                dirs[:] = []
            else:
                dirs[:] = [d for d in dirs if d not in {'Images', '.cache', '__pycache__', 'models', 'wheels'}]
            if len(here.relative_to('/kaggle/input').parts) >= 9:
                dirs[:] = []
    if len(matches) > 1:
        raise ValueError('Có nhiều output resume phù hợp; đặt PREVIOUS_RUNS_OVERRIDE ở cell đầu.')
    previous = matches[0] if matches else None
if previous is not None and not RUNS_ROOT.exists():
    if object_hash(read_json(previous / 'protocol.json')) != object_hash(expected_protocol):
        raise ValueError('Output cũ khác code/config/release. Dùng đúng protocol hoặc RUN_NAME mới.')
    print('Khôi phục checkpoint từ:', previous, flush=True)
    RUNS_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = RUNS_ROOT.with_name(RUN_NAME + '.restoring')
    shutil.copytree(previous, staging, dirs_exist_ok=True)
    staging.rename(RUNS_ROOT)
if (RUNS_ROOT / 'protocol.json').exists():
    if object_hash(read_json(RUNS_ROOT / 'protocol.json')) != object_hash(expected_protocol):
        raise ValueError('RUN_NAME đã có protocol khác; giữ kết quả cũ và đặt tên run mới.')

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT = RUNS_ROOT / 'training_report'
REPORT_ROOT.mkdir(exist_ok=True)
execution_plan = [{'experiment': name, 'seed': seed} for seed in SEEDS for name in PLAN]
preflight = {'release_id': release['release_id'], 'release_stage': release['stage'],
    'pending_reviews': release.get('pending_reviews', []), 'code_sha256': code_hash(),
    'cache_fingerprint': cache_fingerprint, 'debug_run': SMOKE_TEST,
    'train_cases': len(records['train']), 'validation_cases': len(records['validation']),
    'checked_image_paths': len(required_images) if CHECK_IMAGE_FILES else 0,
    'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
    'torch': torch.__version__, 'device': str(device), 'base_config': base_config.to_dict(),
    'requested_experiments': EXPERIMENTS, 'expanded_plan': PLAN, 'seeds': SEEDS,
    'planned_runs': len(execution_plan), 'test_evaluated': False, 'publication_ready': False}
write_json(REPORT_ROOT / 'preflight.json', preflight)
write_json(REPORT_ROOT / 'execution_plan.json', execution_plan)
print(f"PREFLIGHT PASS | train={len(records['train']):,}, validation={len(records['validation']):,} | {len(execution_plan)} lượt")
print('Release:', release['stage'], '| pending reviews:', len(release.get('pending_reviews', [])))
print('Ngân sách:', base_config.epochs, 'epochs tối đa; patience:', base_config.patience,
      '| batch:', base_config.batch_size, '| accumulation:', base_config.accumulation_steps)
print('Kết quả:', RUNS_ROOT)
display(pd.DataFrame(execution_plan))
'''), md('''
## 3. Huấn luyện đề xuất trên dữ liệu đầy đủ

Teacher V/H học trước và được đóng băng khi distill Student. V:H=3:1, T=4, loss gồm CE + KD.
Ảnh Teacher và Student trong KD giống nhau; mô tả X-quang bị mask ở nhánh chỉ ảnh chụp.
V/H là giả thuyết phân nhóm trường cần kiểm chứng, không phải nhãn visualizability từ chuyên gia.

Mỗi epoch lưu `last.pt`; `best.pt` chọn bằng validation macro-F1. Chạy lại Run All sẽ resume ở epoch đã lưu.
Qua phiên khác: lưu **Output**, gắn output notebook train cũ vào Input; notebook tự copy suite phù hợp về working.
Không có cơ chế tự vượt giới hạn phiên Kaggle. Giữ Output ở chế độ Private vì predictions/config có ID hồ sơ.
'''), code(r'''
def collect_progress():
    rows = []
    for item in execution_plan:
        folder = RUNS_ROOT / item['experiment'] / f"seed_{item['seed']}"
        summary_file = folder / 'summary.json'
        row = {**item, 'complete': False, 'epochs_completed': 0, 'best_validation_macro_f1': None}
        if summary_file.exists():
            row.update(read_json(summary_file))
        elif (folder / 'history.json').exists():
            history = read_json(folder / 'history.json')
            row['epochs_completed'] = len(history)
            row['best_validation_macro_f1'] = max((h['validation_macro_f1'] for h in history), default=None)
        if (folder / 'best_validation_metrics.json').exists():
            metrics = read_json(folder / 'best_validation_metrics.json')
            row.update({k: metrics.get(k) for k in ['accuracy', 'macro_f1',
                'balanced_accuracy_observed_classes', 'macro_auroc_defined_classes', 'ece_10_bins', 'brier_multiclass']})
        rows.append(row)
    return rows

summaries = []
training_error = None
try:
    if RUN_TRAIN:
        for seed in SEEDS:
            free = shutil.disk_usage(RUNS_ROOT).free
            if not SMOKE_TEST and free < max(1024**3, int(parameter_count * 16 * 2)):
                raise OSError('Không đủ chỗ ghi checkpoint nguyên tử. Lưu Output và tiếp tục ở phiên có đủ dung lượng.')
            print(f'\n===== SEED {seed} | {len(PLAN)} phương pháp =====', flush=True)
            summaries.extend(run_suite(PREPARED_ROOT, IMAGE_ROOT, RUNS_ROOT, base_config,
                clinical_cache=CLINICAL_CACHE, selected=EXPERIMENTS, seeds=[seed], resume=RESUME,
                device=str(device)))
except BaseException as exc:
    training_error = f'{type(exc).__name__}: {exc}'
    raise
finally:
    progress_rows = collect_progress()
    progress = pd.DataFrame(progress_rows)
    progress.to_csv(REPORT_ROOT / 'run_progress.csv', index=False)
    completed = sum(bool(r['complete']) for r in progress_rows)
    write_json(REPORT_ROOT / 'suite_status.json', {'complete': completed == len(execution_plan),
        'completed_runs': completed, 'planned_runs': len(execution_plan), 'error': training_error,
        'session_seconds': time.monotonic() - SESSION_STARTED, 'test_evaluated': False,
        'debug_run': SMOKE_TEST, 'release_stage': release['stage'], 'publication_ready': False})
    print(f'Đã hoàn thành {completed}/{len(execution_plan)} lượt. Output:', RUNS_ROOT, flush=True)
'''), md('''
## 4. Bảng kết quả, learning curves và Student để bàn giao

Tổng hợp checkpoint tốt nhất trên validation cho từng seed; báo sample std, không thay std thiếu bằng 0.
Xuất Student chỉ dùng ảnh cho từng seed. Bảng này là kết quả chọn mô hình, không phải đánh giá test độc lập.
Thực hiện CI, so sánh và test trong notebook 03 sau khi chốt protocol và rà soát release.
'''), code(r'''
import matplotlib.pyplot as plt
from clindistill.reporting import export_student

progress = pd.DataFrame(collect_progress())
progress.to_csv(REPORT_ROOT / 'per_seed_validation.csv', index=False)
completed = progress[progress.complete.astype(bool)].copy()
if not completed.empty:
    metric_columns = ['macro_f1', 'accuracy', 'balanced_accuracy_observed_classes',
                      'macro_auroc_defined_classes', 'ece_10_bins', 'brier_multiclass']
    mean_std = completed.groupby('experiment')[metric_columns].agg(['mean', 'std', 'count'])
    mean_std.to_csv(REPORT_ROOT / 'validation_mean_std.csv')
    display(mean_std)

final_rows = completed[completed.experiment.isin(EXPERIMENTS)]
exports = []
for row in final_rows.to_dict('records'):
    if row['experiment'].startswith('teacher_'):
        continue
    folder = RUNS_ROOT / row['experiment'] / f"seed_{row['seed']}"
    destination = RUNS_ROOT / 'exports' / row['experiment'] / f"seed_{row['seed']}" / 'student.pt'
    artifact = export_student(folder, destination)
    exports.append({'experiment': row['experiment'], 'seed': row['seed'],
        'validation_macro_f1': row['best_validation_macro_f1'], 'path': str(artifact),
        'sha256': file_hash(artifact)})
write_json(REPORT_ROOT / 'student_exports.json', exports)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for row in final_rows.to_dict('records'):
    history_path = RUNS_ROOT / row['experiment'] / f"seed_{row['seed']}" / 'history.json'
    history = pd.DataFrame(read_json(history_path))
    label = f"{row['experiment']} / {row['seed']}"
    axes[0].plot(history.epoch, history.train_ce, label=label)
    axes[1].plot(history.epoch, history.train_kd, label=label)
    axes[2].plot(history.epoch, history.validation_macro_f1, label=label)
for ax, title in zip(axes, ['Train CE', 'Train KD (weighted)', 'Validation macro-F1']):
    ax.set(xlabel='Epoch', title=title)
if not final_rows.empty:
    axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(REPORT_ROOT / 'learning_curves.png', dpi=160)
plt.show()
status = read_json(REPORT_ROOT / 'suite_status.json')
print('TRAINING COMPLETE' if status['complete'] else 'TRAINING INCOMPLETE — xem run_progress.csv rồi resume')
print('Báo cáo:', REPORT_ROOT)
print('Student:', RUNS_ROOT / 'exports')
print('Notebook 03 dùng cùng RUN_NAME:', RUN_NAME)
'''), md('''
## Lưu và tiếp tục

Save Version kèm **Output**, giữ Private. Khi hết phiên, gắn output đó vào notebook 02 và Run All lại;
notebook chỉ tự khôi phục suite cùng RUN_NAME, code, config và release. Nếu có nhiều output phù hợp, đặt
`PREVIOUS_RUNS_OVERRIDE` ở cell cấu hình. Resume ở epoch đã lưu, không ở minibatch đang dang dở.

Mặc định 9 lượt cho mô hình đề xuất. Muốn mở rộng nghiên cứu có thể đổi `EXPERIMENTS` thành
`['kd_clinical', 'kd_full', 'kd_equal', 'kd_selective', 'kd_higher_h', 'kd_feature', 'kd_relation']`;
Teachers cần thiết được thêm tự động. Baselines ảnh chỉ chạy khi bạn chủ động thêm tên vào danh sách.
Release `experimental` và các pending reviews vẫn được ghi trong báo cáo; train xong không tự chuyển thành
release reviewed hoặc chứng nhận kết quả khóa luận. MeMoSA chưa nằm trong phạm vi notebook này.
''')]

evaluation = [md('''
# COde 03 — Đánh giá, so sánh và xuất Student

Chạy sau notebook 02. Mặc định đánh giá validation để kiểm tra; chuyển sang test sau khi khóa protocol.
Không huấn luyện hay chọn checkpoint trong notebook này.
Mỗi lần đánh giá kiểm tra release và source trùng phiên bản huấn luyện.

Nếu đánh giá Teacher trên test, trước đó cần thêm shard test của **cùng encoder cache** bằng mục 14 notebook 01.
Student KD không cần clinical cache để suy luận. Bảng so sánh toàn bộ suite gồm cả Teachers nên cần cache cho các hàng Teacher.
'''), code(BOOTSTRAP), code(r'''
from clindistill.reporting import evaluate_suite, export_student, predict_images
from clindistill.artifacts import read_json

SMOKE_TEST = os.environ.get('CLINDISTILL_SMOKE_TEST') == '1'
RUN_NAME = 'code_clindistill_v2'
RUNS_ROOT_OVERRIDE = None  # folder run đã COPY về /kaggle/working nếu từ phiên trước
RUNS_ROOT = Path(RUNS_ROOT_OVERRIDE) if RUNS_ROOT_OVERRIDE else (
    OUTPUT_ROOT / '_support' / 'smoke_runs' / RUN_NAME if SMOKE_TEST else OUTPUT_ROOT / 'runs' / RUN_NAME)
EVALUATION_SPLIT = 'validation'
ALLOW_EXPERIMENTAL_TEST = False
RUN_EVALUATION = True
print('Runs:', RUNS_ROOT, '| partition:', EVALUATION_SPLIT)
'''), code(r'''
per_seed = summary = comparisons = None
if RUN_EVALUATION:
    per_seed, summary, comparisons = evaluate_suite(PREPARED_ROOT, IMAGE_ROOT, RUNS_ROOT,
        split=EVALUATION_SPLIT, allow_experimental_test=ALLOW_EXPERIMENTAL_TEST,
        clinical_cache=CLINICAL_CACHE)
    display(per_seed)
    display(summary)
    print(json.dumps(comparisons, indent=2, ensure_ascii=False))
'''), md('''
## 1. So sánh nhiều seed

Error bar dưới đây là **sample std giữa seed**, không phải confidence interval.
CI trong từng run bootstrap theo split_group; paired CI dùng cùng case/group/seed giữa hai mô hình.
Các CI này có điều kiện trên mô hình đã fit, không thay thế biến thiên do train lại.
Nếu chỉ một seed, std không xác định; không trình bày là độ lệch chuẩn bằng 0.
'''), code(r'''
import numpy as np
import matplotlib.pyplot as plt
if per_seed is not None:
    report_dir = RUNS_ROOT / f'report_{EVALUATION_SPLIT}'
    agg = per_seed.groupby('experiment').macro_f1.agg(['mean', 'std', 'count'])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(agg.index, agg['mean'], color='#247ba0')
    valid = agg['count'] > 1
    ax.errorbar(np.flatnonzero(valid), agg.loc[valid, 'mean'], yerr=agg.loc[valid, 'std'],
                fmt='none', capsize=3, color='black')
    ax.set_ylabel('Macro-F1'); ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=65)
    ax.set_title(f'{EVALUATION_SPLIT}: mean and seed std' + (' [DEBUG]' if SMOKE_TEST else ''))
    plt.tight_layout(); plt.savefig(report_dir / 'comparison.png', dpi=160); plt.show()
'''), md('''
## 2. Confusion matrix và đường risk–coverage của một Student

Risk–coverage xếp ca theo max probability, báo tỷ lệ sai khi chỉ giữ ca tự tin nhất.
Nó không chứng minh mô hình biết khi nào cần khám lâm sàng. Attention theo ảnh không phải định vị tổn thương.
Các file dự đoán có case ID để đối chiếu lỗi với dữ liệu gốc; không nhúng hồ sơ bệnh nhân vào artifact inference.
'''), code(r'''
import pandas as pd
INSPECT_EXPERIMENT = 'kd_selective'
INSPECT_SEED = 2026
run_dir = RUNS_ROOT / INSPECT_EXPERIMENT / f'seed_{INSPECT_SEED}'
metrics_file = run_dir / f'evaluation_{EVALUATION_SPLIT}' / 'metrics.json'
if metrics_file.exists():
    result = read_json(metrics_file)
    classes = read_json(run_dir / 'run_config.json')['class_names']
    matrix = np.asarray(result['metrics']['confusion_matrix'])
    curve = pd.read_csv(run_dir / f'evaluation_{EVALUATION_SPLIT}' / 'risk_coverage.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(matrix, cmap='Blues')
    axes[0].set_xticks(range(len(classes)), classes, rotation=65, ha='right')
    axes[0].set_yticks(range(len(classes)), classes)
    axes[0].set_xlabel('Prediction'); axes[0].set_ylabel('Recorded target')
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[0].text(j, i, str(matrix[i, j]), ha='center', va='center')
    axes[1].plot(curve.coverage, curve.risk)
    axes[1].set(xlabel='Coverage', ylabel='Error risk', xlim=(0, 1), ylim=(0, 1))
    plt.tight_layout(); plt.savefig(run_dir / f'evaluation_{EVALUATION_SPLIT}' / 'diagnostics.png', dpi=160); plt.show()
else:
    print('Chưa có nhánh này. Chọn INSPECT_EXPERIMENT/SEED đã train.')
'''), md('''
## 3. Xuất Student dùng ảnh, không cần hồ sơ

Chọn checkpoint bằng validation trước khi mở test. `student.pt` chỉ có weights và cấu hình, không có optimizer hoặc clinical data.
Dùng cùng source clindistill với artifact; chỉ tải file .pt do bạn tạo/tin cậy.
Inference trả phân phối trên 6 nhãn nghiên cứu, không có lớp healthy và không dùng như kết luận khám bệnh.
'''), code(r'''
EXPORT_STUDENT = False  # bật sau khi chọn Student bằng validation
if EXPORT_STUDENT:
    artifact = export_student(run_dir, OUTPUT_ROOT / 'exports' / RUN_NAME / 'student.pt')
    print('Exported:', artifact)
    # Ví dụ; thay bằng đường dẫn ảnh thật, tương đối với IMAGE_ROOT:
    # result = predict_images(artifact, IMAGE_ROOT, ['Images/Photographs/example.jpg'])
    # print(result)
'''), md('''
Output tổng hợp: `report_<split>/per_seed.csv`, `mean_std.csv`, `paired_comparisons.json`, `comparison.png`, `status.json`.
Từng run có per-class recall/AUROC, ECE, Brier, confusion matrix và CI.

**Phần cần làm sau khi có kết quả:** kiểm tra ảnh gần trùng/clinical leakage, ablation trọng số và representation,
phân tích lỗi theo hỗ trợ từng lớp, chốt kết luận dựa trên kết quả thật. MeMoSA cần dữ liệu và mapping nhãn độc lập;
không áp classifier 6 lớp COde để tuyên bố external validation.
''')]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--only', choices=['train', 'evaluation'])
args = parser.parse_args()
for name, cells in [('COde_02_Train_Baselines.ipynb', train), ('COde_03_Evaluate.ipynb', evaluation)]:
    if args.only and (args.only == 'train') != name.startswith('COde_02'):
        continue
    if name.startswith('COde_03'):
        cells.insert(1, md('''## Thiết lập offline trước khi import thư viện

Trên Kaggle, gắn bundle tạo bởi notebook **00**. Cell sau cài bằng `--no-index`, kiểm tra phiên bản và dùng weights local.
Nếu đang chủ động chạy phiên online chưa có bundle, đặt `OFFLINE_MODE=False`; chế độ GPU cuộc thi dùng True.
'''))
        cells.insert(2, code(offline_setup.BOOTSTRAP_CELL))
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})
    nbf.validate(nb)
    for i, cell in enumerate(cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'{name}:{i}', 'exec')
    nbf.write(nb, ROOT / name)
    print(name, len(cells), 'cells')
