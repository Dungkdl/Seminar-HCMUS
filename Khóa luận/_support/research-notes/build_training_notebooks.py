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
# COde 02 — Baselines, Teachers và ClinDistill trên Kaggle

**Notebook này chỉ huấn luyện và chọn checkpoint qua validation.** Dữ liệu, split và clinical embeddings được tạo ở notebook 01.
Giữ tên file cũ để các đường dẫn trước đây vẫn dùng được; nội dung đã mở rộng từ baseline sang toàn bộ nhánh ClinDistill COde.

Trước Run All: gắn **source**, **COde có Images/**, **prepared release**, **clinical cache** vào Kaggle.
Nếu chưa có cache, dùng mục 14 notebook 01; có thể chạy riêng mục đó với prepared release đã có.
Bật GPU. Giữ torch/torchvision của môi trường Kaggle; không cài đè CUDA stack.
Đọc README nếu Internet bị tắt và cần gắn trọng số ResNet từ dataset riêng.

Student chỉ nhận ảnh khi suy luận. Teacher full/V/H dùng cùng ảnh với Student; không cấp X-quang riêng cho Teacher.
Nhánh photo+X-ray là baseline riêng. Nhãn hiện tại là **một nhóm bệnh được ghi nhận**, không phải bệnh duy nhất của bệnh nhân.
'''), code(BOOTSTRAP), md('''
## 1. Cấu hình thí nghiệm

`CORE` gồm 13 nhánh: single-photo, mean, Attention MIL, Set attention, photo+X-ray;
clinical-only/full/V/H Teachers; clinical KD, full KD, equal-group KD và selective-group KD.
`ABLATIONS` bổ sung trọng số ưu tiên H, feature KD và relation KD.

Mặc định chạy core trên 3 seed. Có thể chọn một phần, ví dụ `SELECTED=['kd_full']`: Teacher cần thiết được thêm tự động.
Mỗi nhánh/seed có thư mục riêng và resume riêng. Chạy cả bộ có thể cần nhiều phiên; không ước lượng thời gian chỉ từ tên RTX 6000.

**Batch mặc định 4 ca, tích lũy 4 bước**, tối đa 8 ảnh/ca, 224×224, AMP. Đây là cấu hình khởi đầu, không cam kết vừa mọi GPU.
Nếu OOM, giảm batch và tăng tích lũy, dùng RUN_NAME mới; giữ cùng cấu hình cho các nhánh so sánh.
Gradient accumulation không làm BatchNorm tương đương batch vật lý lớn.
'''), code(r'''
import torch
from dataclasses import replace
from clindistill.config import TrainConfig
from clindistill.artifacts import verify_release, read_json
from clindistill.suite import CORE, ABLATIONS, expand_plan, run_suite

SMOKE_TEST = os.environ.get('CLINDISTILL_SMOKE_TEST') == '1'
RUN_NAME = 'code_clindistill_v2'
SEEDS = [2026, 2027, 2028] if not SMOKE_TEST else [2026]
SELECTED = CORE  # CORE + ABLATIONS, hoặc ['photo_attention'], hoặc ['kd_selective']
RESUME = True
BACKBONE_WEIGHTS_PATH = OFFLINE_ASSETS['resnet'] if globals().get('OFFLINE_ASSETS') and not SMOKE_TEST else None
RUN_TRAIN = True

base_config = TrainConfig.load(PROJECT_ROOT / 'configs' / 'code' /
                               ('smoke_cpu.json' if SMOKE_TEST else 'thesis.json'))
base_config = replace(base_config, backbone_weights_path=BACKBONE_WEIGHTS_PATH)
RUNS_ROOT = (OUTPUT_ROOT / '_support' / 'smoke_runs' / RUN_NAME if SMOKE_TEST
             else OUTPUT_ROOT / 'runs' / RUN_NAME)
release = verify_release(PREPARED_ROOT, splits=['train', 'validation'])
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print('GPU:', gpu.name, '| VRAM GiB:', round(gpu.total_memory / 2**30, 2))
    print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
elif RUN_TRAIN and not SMOKE_TEST:
    raise RuntimeError('Bật GPU cho full run. SMOKE_TEST=True chỉ kiểm tra CPU, không tạo kết quả nghiên cứu.')
if any(n.startswith(('teacher_', 'kd_')) for n in expand_plan(SELECTED)) and CLINICAL_CACHE is None:
    raise RuntimeError('Thiếu clinical cache. Chạy mục 14 notebook 01 và gắn folder cache vào Kaggle.')
print('Release:', release['release_id'], '|', release['stage'])
print('Plan:', expand_plan(SELECTED), '| seeds:', SEEDS)
print('Run folder:', RUNS_ROOT)
print(json.dumps(base_config.to_dict(), indent=2))
'''), md('''
## 2. Huấn luyện

Teacher được train trước, đóng băng `eval()` + `requires_grad_(False)` trong KD. Chọn `best.pt` bằng macro-F1 validation.
Loss Student = CE + λ KD; KD dùng KL(Teacher || Student) với nhiệt độ T và hệ số T².
Selective KD chuẩn hóa trọng số V/H trên nhóm có dữ liệu của từng ca; không có clinical thì chỉ dùng CE.
V/H là cách phân nhóm trường để kiểm chứng giả thuyết, **chưa phải visualizability được chuyên gia xác nhận**.

Default V:H=3:1; `kd_equal` dùng 1:1; `kd_higher_h` dùng 1:3. Không chọn trọng số qua test.
Trong nhánh chỉ ảnh chụp, trường mô tả X-quang bị mask. Teacher và Student nhận cùng tensor ảnh của minibatch KD.

Kaggle hết phiên: Save Version để giữ output, gắn output phiên trước rồi copy **nguyên folder RUN_NAME** vào
`/kaggle/working/runs/` trước khi Run All. Không resume trực tiếp vào `/kaggle/input` chỉ đọc.
Config/code/data/cache/Teacher đổi sẽ bị từ chối resume; dùng RUN_NAME mới.
'''), code(r'''
summaries = []
if RUN_TRAIN:
    summaries = run_suite(PREPARED_ROOT, IMAGE_ROOT, RUNS_ROOT, base_config,
                         clinical_cache=CLINICAL_CACHE, selected=SELECTED, seeds=SEEDS, resume=RESUME)
'''), md('''
## 3. Kiểm tra validation và thời gian chạy

Đây là số liệu chọn mô hình. Kết quả CPU smoke không có ý nghĩa đánh giá phương pháp.
Ở từng run: `best.pt`, `last.pt`, `history.json`, `validation_predictions.csv`, `run_config.json`, `readiness.json`.
`train_kd` ghi phần loss distillation đã nhân các hệ số; feature/relation là ablation riêng, mặc định tắt.
'''), code(r'''
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
if summaries:
    frame = pd.DataFrame(summaries)
    display(frame[['experiment', 'seed', 'complete', 'epochs_completed', 'best_validation_macro_f1', 'debug_run']])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name in expand_plan(SELECTED):
        history = pd.DataFrame(read_json(RUNS_ROOT / name / f'seed_{SEEDS[0]}' / 'history.json'))
        axes[0].plot(history.epoch, history.train_loss, label=name)
        axes[1].plot(history.epoch, history.validation_macro_f1, label=name)
    axes[0].set_title('Training objective (different objectives across methods)')
    axes[1].set_title('Validation macro-F1 — first seed')
    for ax in axes:
        ax.set_xlabel('Epoch')
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()
'''), md('''
## 4. Tiếp theo

Mở notebook 03, đặt cùng RUN_NAME. Notebook 03 đọc các run đã hoàn thành để tạo bảng mean/std, CI và biểu đồ.
Đừng thay config sau khi xem test. Tình trạng `experimental` của dữ liệu vẫn được ghi trong mọi báo cáo.
MeMoSA, nhãn tooth/region và đa nhãn đầy đủ chưa có ground truth đã xác nhận trong bộ thực nghiệm này;
không tự sinh nhãn để lấp các phần đó của sơ đồ seminar.
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

for name, cells in [('COde_02_Train_Baselines.ipynb', train), ('COde_03_Evaluate.ipynb', evaluation)]:
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
