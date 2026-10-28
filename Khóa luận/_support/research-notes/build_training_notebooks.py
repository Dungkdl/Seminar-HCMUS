"""Generate the four existing notebook entry points for research protocol v4."""
import importlib.util
from pathlib import Path
import textwrap
import nbformat as nb

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('bank_generator', ROOT / '_support/research_bank.py')
bank = importlib.util.module_from_spec(spec); spec.loader.exec_module(bank)
def md(s): return nb.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s): return nb.v4.new_code_cell(textwrap.dedent(s).strip())
def save(name, cells):
    doc = nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    nb.validate(doc); nb.write(doc, ROOT / name)

CONFIG = '''from pathlib import Path
import os, sys, json
RESEARCH_BANK_OVERRIDE = None
PROJECT_ROOT_OVERRIDE = None
FEATURE_ROOT_OVERRIDE = None
OUTPUT_ROOT_OVERRIDE = None
PREVIOUS_STUDY_OVERRIDE = None
RUN_NAME = 'code_research_v1'
RUN_TRAIN = True
'''
FEATURES = '''from clindistill.artifacts import read_json
FEATURE_ROOT = research_one(FEATURE_ROOT_OVERRIDE,
    research_find(lambda p,d,f: 'feature_manifest.json' in f and
        not read_json(p / 'feature_manifest.json').get('debug') and
        {'train.npz','validation.npz'} <= set(f)), 'Research feature cache v4')
OUTPUT_ROOT = Path(OUTPUT_ROOT_OVERRIDE) if OUTPUT_ROOT_OVERRIDE else (
    Path('/kaggle/working') if Path('/kaggle/working').exists() else PROJECT_ROOT / '_support')
STUDY_ROOT = OUTPUT_ROOT / 'runs' / RUN_NAME
CONFIG = read_json(PROJECT_ROOT / 'configs/code/research.json')
CONFIG['run_name'] = RUN_NAME
print('Features:', FEATURE_ROOT, '\\nStudy:', STUDY_ROOT)
'''

save('COde_02_Train_Baselines.ipynb', [md('''
# COde 02 — DINOv3, Multimodal Teacher và Reliability KD

**Gắn research bank từ notebook00 và output notebook01 có feature_manifest.json, chọn GPU, tắt Internet, Run All.**
Notebook02 chỉ train: không audit CSV, chia lại split hay tạo embeddings.

Visual Teacher: DINOv3-L/16 + gated MIL. Multimodal Teacher: DINOv3-L + BGE-M3 + gated fusion.
Student: DINOv3-B/16 + gated MIL, chỉ nhận ảnh. **Encoder nền được cố định; MIL/projections/fusion/head được học.**
Protocol này chưa phải fine-tuning toàn bộ encoder. Mỗi lượt dùng một GPU, không tự dùng DDP.

9 hệ thống × 3 seeds và 2 Teacher × 5-fold × 3 seeds = **57 lượt fit**.
Đối chứng gồm ResNet-50 MIL, Swin-B MIL, DINO-B no-KD, KD Visual, KD Multimodal, Confidence KD, Reliability KD.
Đây là các so sánh cần thiết để kiểm chứng đóng góp trong plan.

Clinical chính chỉ age/gender/chief complaint tiếng Anh, trường có từ chẩn đoán bị chặn và ghi audit.
Chưa xác minh thời điểm clinical: vẫn ghi screened_exploratory. Chất lượng mô hình cần được đo, không mặc định đạt Q1.
'''), code(CONFIG), md('## 1. Cài offline và tìm cache\n\nBundle BiomedBERT cũ không đủ cho DINOv3/BGE-M3. Dùng bank v4 mới.'),
code(bank.BOOTSTRAP_CELL), code(FEATURES), md('''
## 2. Preflight và resume

Kiểm tra checksum, nhóm bệnh nhân/ảnh trùng và folds trước khi train.
Để nối phiên trước: gắn Output cũ, đặt PREVIOUS_STUDY_OVERRIDE tới study folder.
Lưu optimizer/scheduler/RNG ở cuối mỗi epoch; epoch bị ngắt giữa chừng sẽ được chạy lại.
'''), code('''
import shutil, torch
from clindistill.research.data import FeatureStore
from clindistill.research.training import validate_config, grouped_folds, study_plan, run_study
validate_config(CONFIG)
if not torch.cuda.is_available() and not CONFIG['debug']:
    raise RuntimeError('Chọn Settings > Accelerator > GPU rồi chạy lại từ đầu.')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), '| VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/2**30, 2))
else:
    print('CPU DEBUG: software verification only.')
print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
STORE = FeatureStore(FEATURE_ROOT, allow_debug=CONFIG['debug'])
for seed in CONFIG['seeds']:
    grouped_folds(STORE.data['train']['labels'], STORE.data['train']['groups'], CONFIG['folds'], seed)
print(json.dumps(study_plan(CONFIG), indent=2))
print('Clinical policy:', STORE.metadata['request']['policy'])
if PREVIOUS_STUDY_OVERRIDE:
    if STUDY_ROOT.exists():
        raise RuntimeError('Study đã tồn tại: để PREVIOUS_STUDY_OVERRIDE=None để resume tại chỗ.')
    shutil.copytree(Path(PREVIOUS_STUDY_OVERRIDE), STUDY_ROOT)
'''), md('''
## 3. Huấn luyện toàn bộ

Mỗi seed: Teacher Visual + Multimodal → group-OOF Teacher → Students/đối chứng.
Teacher OOF chỉ fit nhóm bệnh nhân ngoài fold giữ lại. Số epoch lấy từ validation độc lập;
không dùng nhãn fold giữ lại để chọn checkpoint. Student chỉ nhận ảnh, KD lấy logits OOF train.

Reliability = MM-correct × entropy confidence × [ε + (1−ε) × Visual probability of true class].
Gate temperature=1; KD temperature=4 và T². Trọng số chia cho **mean toàn OOF train**,
không chia cho tổng trọng số mỗi microbatch. Weight collapse sẽ báo lỗi, không âm thầm chuyển thành CE.
'''), code('''
if RUN_TRAIN:
    validation_summary = run_study(FEATURE_ROOT, STUDY_ROOT, CONFIG, require_gpu=not CONFIG['debug'])
    display(validation_summary)
else:
    print('RUN_TRAIN=False: chưa train.')
'''), md('## 4. Kết quả validation và learning curves\n\nChọn best checkpoint bằng macro-F1 validation. Test nằm riêng ở notebook03.'),
code('''
import pandas as pd
import matplotlib.pyplot as plt
if (STUDY_ROOT / 'validation_results.csv').exists():
    results = pd.read_csv(STUDY_ROOT / 'validation_results.csv'); display(results)
    methods = list(results.method.unique())
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), squeeze=False)
    for ax, method in zip(axes.flat, methods):
        for seed in CONFIG['seeds']:
            h = pd.read_csv(STUDY_ROOT / method / f'seed_{seed}' / 'history.csv')
            ax.plot(h.epoch, h.validation_macro_f1, label=str(seed))
        ax.set(title=method, xlabel='Epoch', ylabel='Validation macro-F1'); ax.legend()
    for ax in list(axes.flat)[len(methods):]: ax.set_visible(False)
    fig.tight_layout(); fig.savefig(STUDY_ROOT / 'learning_curves.png', dpi=160); plt.show()
'''), md('''
## 5. Xuất Student chỉ ảnh

Method reliability đã đăng ký; chọn seed bằng validation. Gói export gồm head + DINO-B encoder local,
không cần clinical khi inference. Giữ weights/output riêng tư theo điều khoản model/dataset.
'''), code('''
from clindistill.research.reporting import export_student
if RUN_TRAIN and (STUDY_ROOT / 'completed.json').exists():
    encoder = RESEARCH_ASSETS['models']['student'] if RESEARCH_ASSETS else Path(STORE.metadata['model_locations']['student'])
    info = export_student(FEATURE_ROOT, STUDY_ROOT, encoder, OUTPUT_ROOT / 'student_export' / RUN_NAME)
    print(json.dumps(info, ensure_ascii=False, indent=2))
print('Save Version giữ runs/ và student_export/. Notebook03 đọc Output này để đánh giá.')
''')])

save('COde_03_Evaluate.ipynb', [md('''
# COde 03 — Đánh giá nghiên cứu và khóa test

Gắn bank v4, cache notebook01 và output notebook02. Mặc định chỉ đánh giá validation.
Test chỉ mở sau khi hoàn thành thực nghiệm và ghi protocol đã khóa cùng các giới hạn review còn lại.
Không chỉnh phương pháp/hyperparameters theo test.
'''), code(CONFIG.replace('RUN_TRAIN = True', "SPLIT = 'validation'\nTEST_REVIEW_RECORD = None\nSTUDY_ROOT_OVERRIDE = None")),
code(bank.BOOTSTRAP_CELL), code(FEATURES), code('''
from clindistill.research.reporting import evaluate_study
if STUDY_ROOT_OVERRIDE:
    STUDY_ROOT = Path(STUDY_ROOT_OVERRIDE)
elif not (STUDY_ROOT / 'completed.json').exists():
    STUDY_ROOT = research_one(None, research_find(lambda p,d,f:
        {'study.json','completed.json'} <= set(f) and read_json(p / 'study.json')['config']['run_name'] == RUN_NAME,
        include_runs=True), 'Completed study')
table, summary = evaluate_study(FEATURE_ROOT, STUDY_ROOT, OUTPUT_ROOT / 'reports' / RUN_NAME / SPLIT,
                              split=SPLIT, test_review_record=TEST_REVIEW_RECORD)
display(table); display(summary)
'''), md('''
Đầu ra: mean/std qua seeds, per-class recall/AUROC, confusion matrices, ECE/Brier, risk–coverage,
case/subgroup analysis, group-bootstrap CI và 4 so sánh đã đăng ký.
CI điều kiện trên mô hình đã fit; không thay cho biến động retraining. CI của nhiều so sánh chưa hiệu chỉnh,
chỉ diễn giải mô tả. COde internal evaluation chưa thay thế external validation.

Mở test: notebook01 tạo thêm shard test với đúng bank/policy, không đổi cache train/validation;
đặt SPLIT='test' và TEST_REVIEW_RECORD mô tả protocol đã khóa cùng vấn đề annotation/clinical còn lại.
''')])

path = ROOT / 'COde_01_Data_Audit_and_Preparation.ipynb'
doc = nb.read(path, as_version=4)
doc.cells[1] = md('## Thiết lập research bank offline v4\n\nChạy setup trước import thư viện. Có prepared release rồi thì chạy setup và mục15, không cần audit lại.')
doc.cells[2] = code(bank.BOOTSTRAP_CELL)
for c in doc.cells:
    if c.cell_type == 'markdown' and c.source.startswith('## 8.'):
        c.source = '''## 8. Clinical context và audit

Manifest giữ các trường ứng viên để audit và tái lập phương pháp cũ; không đồng nghĩa đưa tất cả vào Teacher.
Nghiên cứu v4 ở mục15 chỉ chọn age/gender/chief_complaint theo research.json, không dùng examination hoặc
radiographic interpretation trong primary input. Không đưa diagnosis/anomalies/treatment/patient_record vào model.
Các flags dưới đây hỗ trợ review, không chứng minh thời điểm ghi clinical.'''
    if c.cell_type == 'markdown' and c.source.startswith('## 12.'):
        c.source = '''## 12. Luồng nghiên cứu hiện tại

00 chuẩn bị research bank → 01 khóa release và tạo frozen features ở mục15 →
02 huấn luyện DINOv3 Teachers, group-OOF và Student KD/đối chứng → 03 đánh giá.
Partial/full encoder fine-tuning và external validation là giai đoạn sau, không tự coi Stage1 là đã hoàn thành paper.'''
marker = next(i for i,c in enumerate(doc.cells) if c.cell_type == 'markdown' and c.source.startswith('## 14.'))
doc.cells = doc.cells[:marker] + [md('''
## 14. Cache cũ

BiomedBERT cache cũ thuộc phương pháp trước, giữ để tái lập nhưng không dùng thay BGE-M3 trong nghiên cứu mới.
Không sửa split/nhãn của prepared release hiện có.
'''), md('''
## 15. Frozen DINOv3 / BGE-M3 features

Có prepared release: chỉ chạy setup và cell này. Notebook02 chỉ train MIL/fusion/head.
Mặc định train/validation; chưa mở test. Clinical chỉ English age/gender/chief complaint.
Regex chặn từ chẩn đoán không chứng minh clinical có trước chẩn đoán; vẫn screened_exploratory.
'''), code('''
from clindistill.artifacts import read_json
from clindistill.research.data import prepare_features
PREPARED_ROOT_OVERRIDE = None
IMAGE_ROOT_OVERRIDE = None
RESEARCH_FEATURE_OUTPUT_OVERRIDE = None
MODEL_LOCATIONS_OVERRIDE = None
PREPARE_SPLITS = ['train', 'validation']  # thêm test sau khi khóa protocol ở notebook03
prepared = research_one(PREPARED_ROOT_OVERRIDE,
    [release_root] if 'release_root' in globals() else research_find(lambda p,d,f:
        'release.json' in f and not read_json(p / 'release.json').get('debug_subset')), 'Prepared release')
images = research_one(IMAGE_ROOT_OVERRIDE, research_find(lambda p,d,f:
    (p / 'Images/Photographs').is_dir() and 'mini_dataset.json' not in f), 'COde images')
models = MODEL_LOCATIONS_OVERRIDE or (RESEARCH_ASSETS['models'] if RESEARCH_ASSETS else None)
if models is None:
    raise RuntimeError('Gắn research bank v4 hoặc đặt MODEL_LOCATIONS_OVERRIDE tới pretrained models local.')
cfg = read_json(PROJECT_ROOT / 'configs/code/research.json')
destination = Path(RESEARCH_FEATURE_OUTPUT_OVERRIDE) if RESEARCH_FEATURE_OUTPUT_OVERRIDE else (
    Path('/kaggle/working/data/COde_processed/research_features_v1') if Path('/kaggle').exists()
    else PROJECT_ROOT.parent / 'Data/COde_processed/research_features_v1')
manifest = prepare_features(prepared, images, models, destination, cfg, splits=PREPARE_SPLITS)
print('Research cache:', destination, '\\nDimensions:', manifest['dimensions'], '\\nSplits:', manifest['splits'])
print('Save Version, gắn Output này vào notebook02. Không cần download/upload lại cache.')
''')]
for c in doc.cells:
    if c.cell_type == 'code':
        c.outputs, c.execution_count = [], None; c.metadata.pop('execution', None)
nb.validate(doc); nb.write(doc, path)

save('COde_00_Kaggle_Offline_Setup.ipynb', [md('''
# COde 00 — Research asset bank v4

Chạy **có Internet**, trên Linux/Kaggle cùng Python/scientific stack với GPU đích. Gắn source v4.
Bank cũ thiếu DINOv3/BGE-M3 nên không đủ. Nếu accelerator chặn Internet, dùng phiên free có Internet để chuẩn bị.

Trước khi tải: chấp nhận điều khoản ở Hugging Face cho facebook/dinov3-vitb16-pretrain-lvd1689m và
facebook/dinov3-vitl16-pretrain-lvd1689m. Thêm read token vào Kaggle Secrets tên HF_TOKEN, cấp quyền cho notebook.
Không dán token vào code/output/GitHub. BGE-M3: BAAI/bge-m3. Pin revision thực tế, không thay bằng random weights.
'''), code('''
from pathlib import Path
import os, sys, json, importlib.util
MODE = 'prepare'  # inspect để xem môi trường GPU đích; prepare để tải online
PROJECT_ROOT_OVERRIDE = None
BANK_ROOT = Path('/kaggle/working/code_research_bank_v1')
RUN_MINI_REHEARSAL = True
MINI_ROOT_OVERRIDE = None

def find(predicate):
    found = set()
    for base in (Path.cwd(), Path('/kaggle/input')):
        if not base.exists(): continue
        for current, dirs, files in os.walk(base):
            p = Path(current)
            if predicate(p, dirs, files): found.add(p.resolve())
            dirs[:] = [d for d in dirs if d not in {'Images','.git','.cache','_support','models','wheels','_archive'}]
            if len(p.relative_to(base).parts) >= 8: dirs[:] = []
    return sorted(found, key=str)
def one(override, found, name):
    if override: return Path(override).resolve()
    if len(found) != 1: raise RuntimeError(f'{name}: {found}. Set *_OVERRIDE.')
    return found[0]
source = one(PROJECT_ROOT_OVERRIDE, find(lambda p,d,f:
    (p / 'clindistill/research').is_dir() and 'configs' in d and p != (BANK_ROOT / 'source').resolve()), 'Source v4')
spec = importlib.util.spec_from_file_location('research_bank', source / '_support/research_bank.py')
bank = importlib.util.module_from_spec(spec); spec.loader.exec_module(bank)
print(json.dumps(bank.inventory(), indent=2))
if MODE == 'prepare' and RUN_MINI_REHEARSAL:
    if MINI_ROOT_OVERRIDE:
        mini = Path(MINI_ROOT_OVERRIDE)
    else:
        full_release = one(None, find(lambda p,d,f: 'release.json' in f and
            not bank.read_json(p / 'release.json').get('debug_subset')), 'Full prepared COde release')
        full_images = one(None, find(lambda p,d,f: (p / 'Images/Photographs').is_dir() and
            'mini_dataset.json' not in f), 'Full COde images')
        mini = BANK_ROOT.parent / 'research_mini_v1'
        mini_spec = importlib.util.spec_from_file_location('mini_builder', source / '_support/kaggle_offline.py')
        mini_builder = importlib.util.module_from_spec(mini_spec); mini_spec.loader.exec_module(mini_builder)
        if not (mini / 'mini_dataset.json').exists():
            import uuid
            staging = mini.with_name(mini.name + '_staging_' + uuid.uuid4().hex[:8])
            mini_builder.make_mini_code(full_release, full_images, staging, per_class=2, independent_groups=True)
            staging.rename(mini)
        elif bank.read_json(mini / 'prepared/release.json')['source_release_id'] != bank.read_json(full_release / 'release.json')['release_id']:
            raise RuntimeError('Mini thuộc release khác. Đặt MINI_ROOT_OVERRIDE tới mini đúng release.')
if MODE == 'prepare':
    if not os.environ.get('HF_TOKEN'):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:
            raise RuntimeError('Thêm Kaggle Secret HF_TOKEN và cấp quyền trước khi prepare.') from None
    info = bank.prepare(source, BANK_ROOT)
elif MODE != 'inspect': raise ValueError('MODE must be inspect or prepare.')
'''), md('''
## Thử pipeline thật trên mini COde

Gắn code-training-inputs đầy đủ: cell trên tự trích mini 2 nhóm độc lập/lớp từ prepared release,
không mở test. Mini cũ có thể có nhiều ca cùng bệnh nhân nên chưa đủ 2-fold OOF.
2-fold, 1 seed, 3 epochs kiểm tra phần mềm với pretrained thật; không dùng số đo này trong khóa luận.
Bank được gắn trực tiếp từ Output sang notebook01/02/03, không cần tải về máy.
'''), code('''
if MODE == 'prepare' and RUN_MINI_REHEARSAL:
    sys.path.insert(0, str(BANK_ROOT / 'source'))
    from clindistill.artifacts import read_json, write_json
    from clindistill.research.data import prepare_features
    from clindistill.research.training import run_study
    from clindistill.research.training import grouped_folds
    rows = [json.loads(line) for line in (mini / 'prepared/student_single_train.jsonl').read_text(encoding='utf-8').splitlines() if line]
    grouped_folds([r['single_label_index'] for r in rows], [r['split_group'] for r in rows], 2, 2026)
    models = {k: BANK_ROOT / v for k,v in info['models'].items()}
    cfg = read_json(BANK_ROOT / 'source/configs/code/research.json')
    cfg.update(debug=True, seeds=[2026], folds=2, epochs=3, patience=3, feature_dim=64,
        learning_rate=0.01, bootstrap_repetitions=10,
        methods=['student_no_kd','kd_visual','kd_multimodal','kd_confidence','kd_reliability'])
    cache = BANK_ROOT.parent / 'research_mini_features'
    with bank.block_network():
        prepare_features(mini / 'prepared', mini, models, cache, cfg)
        display(run_study(cache, BANK_ROOT.parent / 'research_mini_runs', cfg, require_gpu=False))
    write_json(BANK_ROOT / 'rehearsal.json', {'status':'passed','debug':True,'contains_test':False})
print('Save Version. 01/02/03 Add Input > Notebook Output > chọn output có research_bank_manifest.json.')
'''), md('''
Bank gồm source, Linux wheels, DINO-B/L, BGE-M3, ResNet/Swin weights; không gồm COde đầy đủ.
Giữ bank, mini, cache và checkpoints riêng tư. Nếu chưa có quyền DINO, dừng ở tải model và hoàn tất quyền truy cập;
không có cơ chế thay model hoặc sinh kết quả giả. Resume prepare dùng lại model downloads đã có.
''')])
print('Generated research notebooks 00/01/02/03.')
