"""Build a standalone, Vietnamese COde thesis data audit notebook."""
from pathlib import Path
import textwrap
import nbformat as nbf
import importlib.util

cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md(r'''
# COde — Kiểm định dữ liệu và chuẩn bị thực nghiệm khóa luận

**ClinDistill · Giai đoạn 1 · COde là dữ liệu chính**

Notebook độc lập với `explore-data.ipynb`, sử dụng dữ liệu đã có trên máy hoặc Kaggle.
Mục tiêu là tạo đơn vị học **một lần khám = một tập ảnh**, nhãn có định nghĩa rõ,
split không giao bệnh nhân/ảnh trùng chính xác, và clinical context dành riêng cho Teacher.
Notebook này chuẩn bị dữ liệu; chưa huấn luyện hoặc tuyên bố hiệu quả của KD.

**Bắt đầu ở mục 2:** tạo `complete_dataset_en.csv` chỉ giữ cột tiếng Anh và xem dashboard chất lượng.
Mục 7 trực quan dữ liệu train sau khi khóa split và cách chọn cohort.

**Cách chạy:** chọn kernel Python có pandas/numpy/Pillow/matplotlib; trên Kaggle gắn dataset đã giải nén,
sửa `DATA_ROOT_OVERRIDE` nếu tự tìm thấy nhiều bản, sau đó **Run All**. Không cần tải lại COde hoặc GPU.
Kiểm tra đầy đủ khoảng 58.000 ảnh có thể mất vài phút; cache cho phép chạy lại nhanh.
Dữ liệu gốc chỉ được đọc. Kết quả được ghi vào `../Data/COde_processed/` (dữ liệu) và `_support/audit/code_thesis_v1/` (kiểm định); Kaggle dùng `/kaggle/working/`.

**Đọc kết quả theo thứ tự:** kiểm tra schema → nhãn → ảnh → nhóm chống giao dữ liệu → split cố định →
clinical audit → manifest → điều kiện sẵn sàng. Các mục cần chuyên gia rà soát luôn được giữ là chưa xác nhận.
MeMoSA được bổ sung ở giai đoạn kiểm chứng riêng, không trộn nhãn trong notebook này.

Nguồn: [COde](https://www.nature.com/articles/s41597-026-07342-9),
`../Paper/ClinDistill_Literature_Review_CODe_MeMoSA_2026.pdf`, `_support/research-notes/KE_HOACH_KHOA_LUAN.md`.
''')

md('''
## 1. Cấu hình và khả năng tái lập

Các tham số dưới đây quyết định protocol. `ANNOTATION_ASSUMPTION_APPROVED=False` giữ các nhãn vắng
ở trạng thái chưa xác định trong manifest multi-label. Chỉ đổi sau khi rà soát annotation và thống nhất
với người hướng dẫn. Tập single-label là **một nhóm mục tiêu được ghi nhận**, không có nghĩa không mắc bệnh khác.
Split được khóa trên đĩa: chạy lại cùng cấu hình sẽ dùng đúng các ID cũ; đổi protocol thì dùng `RUN_NAME` mới.
Nếu cần cài thư viện trên môi trường mới: `%pip install pandas numpy pillow matplotlib`.
''')

code(r'''
from pathlib import Path
import os, sys, json, re, hashlib, random, platform, warnings, inspect
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, __version__ as PIL_VERSION
from IPython.display import display, Markdown

DATA_ROOT_OVERRIDE = None  # Ví dụ: Path('/kaggle/input/my-code/COde-Dataset')
RUN_NAME = 'code_thesis_v1'
SEED = 2026
SPLIT_RATIOS = {'train': 0.70, 'validation': 0.15, 'test': 0.15}
SPLIT_SEARCH_TRIALS = 128  # Cố định trước; chỉ cân bằng số ca/nhãn, không dùng kết quả model
IMAGE_WORKERS = min(8, os.cpu_count() or 2)
NEAR_DUPLICATE_MAX_HAMMING = 3
NEAR_DUPLICATE_BUCKET_CAP = 150  # bucket quá lớn được báo riêng, không âm thầm bỏ qua
REVIEW_SAMPLE_SIZE = 120
ANNOTATION_ASSUMPTION_APPROVED = False
CLINICAL_FIELDS_REVIEWED = False
LABEL_MAPPING_REVIEWED = False
REVIEWED_DUPLICATE_LINKS = None  # CSV path_a,path_b,decision; chỉ decision='duplicate' được gộp

BASE = Path.cwd()
# Local: support kernels started at the parent workspace or this project folder.
if not (BASE / 'clindistill').is_dir() and (BASE / 'Khóa luận' / 'clindistill').is_dir():
    BASE = BASE / 'Khóa luận'
WORK_ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else BASE
OUT = WORK_ROOT / '_support' / 'audit' / RUN_NAME
PROCESSED_ROOT_OVERRIDE = None  # Optional folder for the cleaned CSV and prepared releases.
PROCESSED_ROOT = (Path(PROCESSED_ROOT_OVERRIDE).resolve() if PROCESSED_ROOT_OVERRIDE is not None
    else (WORK_ROOT / 'data' / 'COde_processed' if Path('/kaggle/working').exists()
          else BASE.parent / 'Data' / 'COde_processed'))
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'figures').mkdir(exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
pd.set_option('display.max_columns', 16)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 180, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
    'axes.titlesize': 11, 'axes.labelsize': 10, 'figure.facecolor': 'white',
    'axes.facecolor': '#f8fafc', 'grid.color': '#e2e8f0', 'grid.alpha': 0.7})
BLUE, TEAL, ORANGE, RED = '#2563eb', '#0d9488', '#f59e0b', '#dc2626'

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def save_figure(name, *, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(OUT / 'figures' / name, bbox_inches='tight')
    plt.show()

print('Output:', OUT)
print('Python:', sys.version.split()[0], '| pandas:', pd.__version__, '| Pillow:', PIL_VERSION)
''')

code(r'''
# Tìm đúng thư mục chứa CSV; không quét từng ảnh để tìm dataset.
def discover_roots():
    candidates = []
    for root in [BASE, BASE / 'Data', BASE.parent / 'Data', BASE / 'COde', Path('/kaggle/input'), Path('/content')]:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root):
            depth = len(Path(current).relative_to(root).parts)
            dirs[:] = [d for d in dirs if d not in {'Images', '.git', '.cache', 'outputs', '_support', '_archive', 'node_modules'}]
            if depth >= 5:
                dirs[:] = []
            if 'complete_dataset.csv' in files or 'complete-dataset.csv' in files:
                candidates.append(Path(current).resolve())
    return sorted(set(candidates), key=str)

if DATA_ROOT_OVERRIDE is not None:
    DATA_ROOT = Path(DATA_ROOT_OVERRIDE).resolve()
else:
    candidates = discover_roots()
    if len(candidates) != 1:
        raise RuntimeError(f'Tìm thấy {len(candidates)} bản dữ liệu: {candidates}. Đặt DATA_ROOT_OVERRIDE rõ ràng; cần bản đã giải nén.')
    DATA_ROOT = candidates[0]
CSV_PATH = next((DATA_ROOT / f for f in ['complete_dataset.csv', 'complete-dataset.csv']
                 if (DATA_ROOT / f).is_file()), None)
if CSV_PATH is None or not (DATA_ROOT / 'Images').is_dir():
    raise FileNotFoundError('Cần CSV gốc và thư mục Images trong DATA_ROOT.')
SOURCE_SHA256 = sha256_file(CSV_PATH)
print('Dataset:', DATA_ROOT)
print('CSV SHA256:', SOURCE_SHA256)
existing_run = OUT / 'run_summary.json'
if existing_run.exists():
    previous_run = json.loads(existing_run.read_text(encoding='utf-8'))
    assert previous_run['source_csv_sha256'] == SOURCE_SHA256, 'CSV đã đổi; chọn RUN_NAME mới trước khi xuất kết quả.'
''')

md('''
## 2. Schema, ID và split được phát hành

Đọc ID dưới dạng chuỗi để giữ số 0 đầu. Kiểm tra bệnh nhân, lần khám, và tính nhất quán của filename.
Một bệnh nhân có nhiều lần khám nên việc chỉ tách `checkup_id` không đủ cho đánh giá người bệnh chưa thấy.
Các thống kê toàn bộ nhãn ở đây dùng kiểm định dữ liệu; không dùng test để tối ưu mô hình.
''')

code(r'''
raw = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
required = {'patient_id', 'checkup_id', 'photographs', 'radiographs', 'anomalies_en', 'split',
            'chief_complaint', 'present_illness', 'past_medical_record', 'examination', 'radiographs_examination'}
assert required <= set(raw), f'Thiếu cột: {required - set(raw)}'
assert not raw['checkup_id'].duplicated().any(), 'Trùng checkup_id: cần giải quyết trước.'
assert raw['patient_id'].str.fullmatch(r'\d+').all(), 'patient_id cần rà soát.'
assert raw['checkup_id'].str.fullmatch(r'\d+-\d+').all(), 'checkup_id không theo schema dự kiến.'
assert all(str(int(c.split('-')[0])) == str(int(p)) for c, p in zip(raw.checkup_id, raw.patient_id))
MISSING_MARKERS = {'', 'nan', 'none', 'null', 'n/a', 'na', 'not recorded', 'not available', 'unknown'}

def is_missing(value):
    return str(value).strip().casefold() in MISSING_MARKERS

print(f'Source CSV: {len(raw):,} visits, {raw.shape[1]} columns.')
''')

md('''
### 2.1. Làm sạch CSV chỉ giữ tiếng Anh

Xuất `complete_dataset_en.csv`: bỏ cột `_cn`, giữ ID dạng chuỗi, trim khoảng trắng đầu/cuối,
đưa missing placeholders về chuỗi rỗng. Giữ nguyên xuống dòng/nội dung y khoa, không tự dịch,
không xóa từ phủ định và không điền tuổi/giới/nhãn. Giữ mọi ca; chọn cohort train là bước riêng.

Kiểm tra duplicate IDs/rows, tuổi ngoài [0,120], timestamp không đọc được và ký tự Hán còn sót.
Giá trị cần review được báo, không tự sửa bằng suy đoán. Dashboard toàn CSV chỉ kiểm định chất lượng;
EDA nhãn/clinical chi tiết ở mục 7 chỉ dùng train sau khi khóa split.
''')

code(r'''
def clean_csv_value(value):
    # Preserve clinical wording, negation and paragraph breaks.
    value = str(value).strip()
    return '' if is_missing(value) else value

cn_columns = [c for c in raw.columns if c.casefold().endswith('_cn')]
english_columns = [c for c in raw.columns if c not in cn_columns]
clean_en = raw.loc[:, english_columns].copy()
cleaning_changes = []
for field in english_columns:
    before = clean_en[field].copy()
    clean_en[field] = before.map(clean_csv_value)
    cleaning_changes.append({'field': field, 'changed_cells': int((before != clean_en[field]).sum()),
        'missing_before': int(before.map(is_missing).sum()),
        'missing_after': int((clean_en[field] == '').sum())})

age_values = pd.to_numeric(clean_en.age, errors='coerce')
age_problem = (clean_en.age != '') & (age_values.isna() | ~age_values.between(0, 120))
parsed_times = pd.to_datetime(clean_en.checkup_time, format='mixed', dayfirst=True, errors='coerce')
date_problem = (clean_en.checkup_time != '') & parsed_times.isna()
han_pattern = '[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]'
han_mask = pd.DataFrame({c: clean_en[c].str.contains(han_pattern, regex=True) for c in english_columns})
quality_issues = []
for field, mask, issue in [('age', age_problem, 'invalid_age'),
                           ('checkup_time', date_problem, 'unparsed_timestamp')]:
    quality_issues.extend({'checkup_id': clean_en.at[i, 'checkup_id'], 'field': field,
                           'issue': issue} for i in clean_en.index[mask])
for field in english_columns:
    quality_issues.extend({'checkup_id': clean_en.at[i, 'checkup_id'], 'field': field,
                           'issue': 'Han_characters_in_English_column'} for i in clean_en.index[han_mask[field]])
pd.DataFrame(quality_issues, columns=['checkup_id', 'field', 'issue']).to_csv(OUT / 'csv_quality_issues.csv', index=False)
pd.DataFrame(cleaning_changes).to_csv(OUT / 'csv_cleaning_by_field.csv', index=False)
assert len(clean_en) == len(raw) and clean_en.checkup_id.tolist() == raw.checkup_id.tolist()
assert not clean_en.checkup_id.duplicated().any()
assert not any(c.casefold().endswith('_cn') for c in clean_en.columns)
assert not han_mask.to_numpy().any(), 'English columns contain Han characters; review csv_quality_issues.csv before export.'
CLEAN_CSV_PATH = PROCESSED_ROOT / 'complete_dataset_en.csv'
clean_en.to_csv(CLEAN_CSV_PATH, index=False, encoding='utf-8-sig')
roundtrip = pd.read_csv(CLEAN_CSV_PATH, dtype=str, keep_default_na=False)
pd.testing.assert_frame_equal(clean_en.reset_index(drop=True), roundtrip)
assert sha256_file(CSV_PATH) == SOURCE_SHA256, 'Source CSV unexpectedly changed.'
cleaning_summary = {'source_csv_sha256': SOURCE_SHA256, 'cleaned_csv_sha256': sha256_file(CLEAN_CSV_PATH),
    'rows_before': len(raw), 'rows_after': len(clean_en), 'columns_before': raw.shape[1],
    'columns_after': clean_en.shape[1], 'removed_columns': cn_columns,
    'changed_cells': int(sum(x['changed_cells'] for x in cleaning_changes)),
    'duplicate_checkup_ids': int(clean_en.checkup_id.duplicated().sum()),
    'duplicate_full_rows': int(clean_en.duplicated().sum()), 'quality_issues': len(quality_issues),
    'row_filtering': False, 'clinical_internal_whitespace_preserved': True}
write_json(OUT / 'csv_cleaning_summary.json', cleaning_summary)

# All following preparation operates on the cleaned English table.
df = clean_en.copy()
df['patient_id'] = df['patient_id'].map(lambda x: str(int(x)))
df['case_id'] = df['checkup_id']
df['original_split'] = df['split']
original_patient_sets = {s: set(g.patient_id) for s, g in df.groupby('original_split')}
original_overlap = sorted(original_patient_sets.get('train', set()) & original_patient_sets.get('test', set()))
overview = pd.DataFrame({'metric': ['checkups', 'patients', 'original train/test patient overlap'],
                         'value': [len(df), df.patient_id.nunique(), len(original_overlap)]})
overview.to_csv(OUT / 'dataset_overview.csv', index=False)
missing_table = pd.DataFrame([
    {'field': k, 'empty_count': int((clean_en[k] == '').sum()),
     'missing_or_placeholder_count': int(clean_en[k].map(is_missing).sum())} for k in english_columns])
missing_table['missing_percent'] = missing_table.missing_or_placeholder_count / len(clean_en) * 100
missing_table.to_csv(OUT / 'field_missingness.csv', index=False)
display(pd.DataFrame([{'rows': len(clean_en), 'patients': df.patient_id.nunique(),
    'English columns': len(english_columns), 'CN columns removed': len(cn_columns),
    'quality issues': len(quality_issues)}]))
display(pd.DataFrame(cleaning_changes).query('changed_cells > 0'))
display(missing_table.sort_values('missing_percent', ascending=False).head(12))
print('English CSV:', CLEAN_CSV_PATH)
''')

code(r'''
fig = plt.figure(figsize=(14, 8))
grid = fig.add_gridspec(2, 4, height_ratios=[1, 3.8], hspace=0.35, wspace=0.55)
cards = [('Visits kept', f'{len(clean_en):,}'), ('Patients', f'{df.patient_id.nunique():,}'),
         ('English columns', str(len(english_columns))), ('CN columns removed', str(len(cn_columns)))]
for column, (label, value) in enumerate(cards):
    ax = fig.add_subplot(grid[0, column]); ax.axis('off')
    ax.text(0.04, 0.70, value, fontsize=26, weight='bold', color=BLUE, transform=ax.transAxes)
    ax.text(0.04, 0.23, label, fontsize=11, color='#475569', transform=ax.transAxes)
left = fig.add_subplot(grid[1, :2]); right = fig.add_subplot(grid[1, 2:])
top_missing = missing_table.nlargest(10, 'missing_percent').sort_values('missing_percent')
bars = left.barh(top_missing.field, top_missing.missing_percent, color=ORANGE)
left.bar_label(bars, labels=[f'{p:.1f}% ({n:,})' for p, n in zip(top_missing.missing_percent,
    top_missing.empty_count)], padding=4, fontsize=9)
left.set_xlim(0, 130); left.set_xlabel('Missing visits (%)'); left.set_title('English fields: highest missingness')
left.grid(axis='x'); left.set_axisbelow(True)
bars = right.bar(['Source CSV', 'English CSV'], [raw.shape[1], clean_en.shape[1]], color=[BLUE, TEAL], width=0.5)
right.bar_label(bars, padding=5); right.set_ylim(0, raw.shape[1] * 1.2)
right.set_ylabel('Columns'); right.set_title('Columns before / after cleaning')
fig.suptitle('CSV cleaning overview — full source table', fontsize=16, weight='bold')
fig.text(0.5, 0.01, 'Missing values remain unknown; age, gender and disease labels are not imputed.', ha='center', color='#475569')
fig.subplots_adjust(top=0.88, bottom=0.12)
save_figure('00_csv_cleaning_overview.png', tight=False)
''')

code(r'''
# QA augmentation/ngôn ngữ: đếm theo lần khám thay vì theo dòng JSON.
qa_summary, qa_case_sets, qa_patient_sets = [], {}, {}
for filename in ['train-cls.json', 'test_cls.json', 'test-cls.json']:
    path = DATA_ROOT / filename
    if not path.exists():
        continue
    samples = json.loads(path.read_text(encoding='utf-8'))
    case_sets = [{Path(x.replace('\\', '/')).name.rsplit('-', 1)[0] for x in s.get('images', [])}
                 for s in samples]
    cases = set().union(*case_sets) if case_sets else set()
    patients = {str(int(c.split('-')[0])) for c in cases}
    qa_case_sets[filename], qa_patient_sets[filename] = cases, patients
    qa_summary.append({'file': filename, 'rows': len(samples), 'checkups': len(cases),
                       'patients': len(patients), 'empty_image_rows': sum(not c for c in case_sets),
                       'mixed_checkup_rows': sum(len(c) > 1 for c in case_sets)})
display(pd.DataFrame(qa_summary))
pd.DataFrame(qa_summary).to_csv(OUT / 'released_qa_audit.csv', index=False)
for test_name in ['test_cls.json', 'test-cls.json']:
    if test_name in qa_patient_sets and 'train-cls.json' in qa_patient_sets:
        print(test_name, 'patient overlap:', len(qa_patient_sets[test_name] & qa_patient_sets['train-cls.json']))
print('Pipeline phía dưới chỉ dùng CSV gốc; không nhân số ca theo QA.')
''')

md('''
## 3. Chính sách nhãn có phiên bản

Sáu nhóm mục tiêu: Caries, Gingivitis, Malocclusion, Pulpitis, Tooth Loss, Tooth Structure Loss.
Mapping chỉ gồm các tên tường minh; không tự suy diễn crowding, overbite hoặc bệnh gần nghĩa sang nhãn khác.
Một dòng có nhiều nhóm được giữ nguyên dạng multi-label. Không có nhãn mục tiêu **không đồng nghĩa normal**.

Xuất hai cohort để người hướng dẫn lựa chọn: `single_target_recorded_v1` (đúng một nhóm được ghi nhận),
và `multilabel_candidate_v1`. Multi-label có `label_mask=0` cho nhãn vắng theo mặc định; chỉ biết positive
thì chưa đủ train một BCE classifier sáu bệnh có meaningful negatives. Không được bỏ qua mask.
''')

code(r'''
TARGETS = ['Caries', 'Gingivitis', 'Malocclusion', 'Pulpitis', 'Tooth Loss', 'Tooth Structure Loss']
LABEL_MAPPING = {
    'Dental Caries': 'Caries', 'Gingivitis': 'Gingivitis', 'Malocclusion': 'Malocclusion',
    'Class I Malocclusion': 'Malocclusion', 'Class II Malocclusion': 'Malocclusion',
    'Class III Malocclusion': 'Malocclusion', 'Pulpitis': 'Pulpitis',
    'Tooth Loss': 'Tooth Loss', 'Tooth Structure Loss': 'Tooth Structure Loss',
}
mapping_folded = {k.casefold(): v for k, v in LABEL_MAPPING.items()}
def parse_labels(value):
    if is_missing(value):
        return []
    return sorted({x.strip() for x in re.split(r'[,;，；\n]+', value) if x.strip()})

df['source_labels'] = df.anomalies_en.map(parse_labels)
df['target_labels'] = df.source_labels.map(lambda xs: sorted({mapping_folded[x.casefold()] for x in xs
                                                              if x.casefold() in mapping_folded}))
df['unmapped_labels'] = df.source_labels.map(lambda xs: [x for x in xs if x.casefold() not in mapping_folded])
df['n_targets'] = df.target_labels.map(len)
df['label_status'] = ['missing_annotation' if not source else 'outside_target_mapping' if not target
                      else 'one_recorded_target' if len(target) == 1 else 'multiple_recorded_targets'
                      for source, target in zip(df.source_labels, df.target_labels)]
label_counts = Counter(x for xs in df.source_labels for x in xs)
label_inventory = pd.DataFrame([{'source_label': k, 'checkups': v, 'target': mapping_folded.get(k.casefold(), '')}
                                for k, v in label_counts.most_common()])
label_inventory.to_csv(OUT / 'label_inventory.csv', index=False)
write_json(OUT / 'label_policy.json', {'version': 'recorded_targets_v1', 'targets': TARGETS,
    'mapping': LABEL_MAPPING, 'missing_label_is_normal': False,
    'annotation_assumption_approved': ANNOTATION_ASSUMPTION_APPROVED,
    'single_label_definition': 'Exactly one mapped target recorded; other conditions may coexist.',
    'multilabel_mask': 'Observed positives only unless annotation completeness assumption is approved.'})
display(df.label_status.value_counts().rename_axis('status').to_frame('checkups'))
display(label_inventory.head(20))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pd.Series({k: sum(k in xs for xs in df.target_labels) for k in TARGETS}).plot.barh(ax=axes[0], title='Recorded target support')
df.n_targets.value_counts().sort_index().plot.bar(ax=axes[1], title='Mapped targets per checkup')
save_figure('01_label_support.png')
''')

md('''
## 4. Đường dẫn, giải mã toàn bộ ảnh và cache

File photo và radiograph có thể cùng basename; khóa ảnh luôn gồm **thư mục modality + filename**.
Quét đầy đủ: existence, JPEG decode, kích thước, SHA256 file, SHA256 pixel RGB sau EXIF transpose,
và dHash 64-bit. Pixel hash phát hiện ảnh pixel giống hệt dù container khác; không chứng minh đã tìm mọi ảnh gần trùng.
Cache được dùng lại khi đường dẫn, kích thước file và thời gian sửa không đổi; không phù hợp nếu file bị thay nhưng giữ cả hai metadata.
''')

code(r'''
def parse_image_names(value, folder, case_id):
    if is_missing(value):
        return []
    names = [x.strip() for x in value.split(',') if x.strip()]
    assert len(names) == len(set(names)), f'Ảnh bị lặp trong ca {case_id}'
    for name in names:
        assert '/' not in name and '\\' not in name and '..' not in name, f'Đường dẫn không hợp lệ: {name}'
        assert name.rsplit('-', 1)[0] == case_id, f'Ảnh không thuộc case: {case_id}, {name}'
    return [f'Images/{folder}/{name}' for name in names]

df['photo_paths'] = [parse_image_names(v, 'Photographs', c) for v, c in zip(df.photographs, df.case_id)]
df['radiograph_paths'] = [parse_image_names(v, 'Radiographs', c) for v, c in zip(df.radiographs, df.case_id)]
image_refs = pd.DataFrame([
    {'case_id': r.case_id, 'patient_id': r.patient_id, 'modality': modality, 'path': path}
    for r in df.itertuples() for modality, paths in [('photo', r.photo_paths), ('radiograph', r.radiograph_paths)]
    for path in paths
])
actual_paths = {p.relative_to(DATA_ROOT).as_posix() for folder in ['Photographs', 'Radiographs']
                for p in (DATA_ROOT / 'Images' / folder).glob('*') if p.is_file()}
referenced_paths = set(image_refs.path)
pd.DataFrame({'path': sorted(actual_paths - referenced_paths)}).to_csv(OUT / 'unreferenced_images.csv', index=False)
print('Image references:', len(image_refs), '| unreferenced files:', len(actual_paths - referenced_paths))

CACHE_VERSION = 'decode_rgb_dhash_v1'
cache_path = OUT / 'image_audit_cache.json'
cache_payload = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
cache = cache_payload.get('entries', {}) if cache_payload.get('version') == CACHE_VERSION else {}

def inspect_image(relative):
    path = DATA_ROOT / relative
    base = {'path': relative, 'ok': False, 'error': '', 'width': 0, 'height': 0,
            'file_sha256': '', 'pixel_sha256': '', 'dhash': '', 'size': 0, 'mtime_ns': 0}
    if not path.is_file():
        return {**base, 'error': 'missing_file'}
    stat = path.stat()
    base.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    old = cache.get(relative)
    if old and old['size'] == stat.st_size and old['mtime_ns'] == stat.st_mtime_ns:
        return old
    try:
        file_hash = sha256_file(path)
        with Image.open(path) as opened:
            opened.load()
            rgb = ImageOps.exif_transpose(opened).convert('RGB')
            width, height = rgb.size
            h = hashlib.sha256(f'RGB:{width}:{height}:'.encode())
            h.update(rgb.tobytes())
            tiny = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
            bits = (tiny[:, 1:] > tiny[:, :-1]).ravel()
            dhash = sum(int(bit) << i for i, bit in enumerate(bits))
        return {**base, 'ok': True, 'width': width, 'height': height,
                'file_sha256': file_hash, 'pixel_sha256': h.hexdigest(), 'dhash': f'{dhash:016x}'}
    except Exception as exc:
        return {**base, 'error': type(exc).__name__ + ': ' + str(exc)[:200]}

audit_records = []
with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
    for i, record in enumerate(pool.map(inspect_image, sorted(referenced_paths)), 1):
        audit_records.append(record)
        if i % 5000 == 0:
            print(f'Decoded/cache: {i:,}/{len(referenced_paths):,}', flush=True)
            write_json(cache_path, {'version': CACHE_VERSION, 'entries': {r['path']: r for r in audit_records}})
write_json(cache_path, {'version': CACHE_VERSION, 'entries': {r['path']: r for r in audit_records}})
image_audit = pd.DataFrame(audit_records)
image_audit.to_csv(OUT / 'image_audit.csv', index=False)
image_audit.loc[~image_audit.ok].to_csv(OUT / 'invalid_images.csv', index=False)
good_paths = set(image_audit.loc[image_audit.ok, 'path'])
df['valid_photo_paths'] = df.photo_paths.map(lambda xs: [x for x in xs if x in good_paths])
df['valid_radiograph_paths'] = df.radiograph_paths.map(lambda xs: [x for x in xs if x in good_paths])
df['n_photos'] = df.valid_photo_paths.map(len)
df['n_radiographs'] = df.valid_radiograph_paths.map(len)
df['has_valid_images'] = (df.n_photos + df.n_radiographs) > 0
display(image_audit.groupby('ok').agg(images=('path', 'size')))
display(image_audit.loc[image_audit.ok, ['width', 'height']].describe().round(1))
''')

md('''
## 5. Ảnh trùng và đơn vị chia tập

Gộp bệnh nhân thành connected components khi họ có ảnh trùng SHA256 file hoặc pixel RGB.
Toàn bộ component chỉ nằm trong một partition. Không xóa ca bệnh tự động chỉ vì có ảnh trùng.
Ảnh giống nhau nhưng nhãn khác được xuất để rà soát, không tự kết luận nhãn sai.

Near-duplicate là **ứng viên**, không tự gộp: dHash có thể giống nhau với hai ảnh răng khác nhau.
Chia hash thành bốn khối 16-bit để tìm cặp cách nhau tối đa 3 bit. Bỏ bucket quá lớn nhằm giới hạn chi phí,
và lưu số bucket bị bỏ. Vì vậy kiểm tra này không cấp chứng nhận “không còn ảnh gần trùng”.
''')

code(r'''
class UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

uf = UnionFind(sorted(set(df.patient_id)))
joined_images = image_refs.merge(image_audit, on='path', how='left', validate='many_to_one')
exact_rows = []
for hash_column in ['file_sha256', 'pixel_sha256']:
    good = joined_images.loc[joined_images.ok & (joined_images[hash_column] != '')]
    repeated = good.loc[good[hash_column].duplicated(keep=False)]
    for key, group in repeated.groupby(hash_column):
        people = sorted(set(group.patient_id))
        for patient in people[1:]:
            uf.union(people[0], patient)
        for r in group.itertuples():
            exact_rows.append({'hash_type': hash_column, 'hash': key, 'patient_id': r.patient_id,
                               'case_id': r.case_id, 'path': r.path, 'cross_patient': len(people) > 1})
exact_duplicates = pd.DataFrame(exact_rows, columns=['hash_type', 'hash', 'patient_id', 'case_id', 'path', 'cross_patient'])
exact_duplicates.to_csv(OUT / 'exact_duplicate_groups.csv', index=False)
case_labels = dict(zip(df.case_id, df.target_labels.map(lambda xs: '|'.join(xs))))
exact_duplicates['recorded_target_signature'] = exact_duplicates.case_id.map(case_labels)
if not exact_duplicates.empty:
    varying_labels = exact_duplicates.groupby(['hash_type', 'hash']).recorded_target_signature.transform('nunique') > 1
    exact_duplicates.loc[varying_labels].to_csv(OUT / 'exact_duplicate_label_review.csv', index=False)
else:
    exact_duplicates.to_csv(OUT / 'exact_duplicate_label_review.csv', index=False)
reviewed_links = []
if REVIEWED_DUPLICATE_LINKS is not None:
    links = pd.read_csv(REVIEWED_DUPLICATE_LINKS, dtype=str, keep_default_na=False)
    assert {'path_a', 'path_b', 'decision'} <= set(links)
    owners = image_refs.groupby('path').patient_id.agg(lambda xs: sorted(set(xs))).to_dict()
    for link in links.to_dict('records'):
        if link['decision'].strip().casefold() != 'duplicate':
            continue
        a, b = link['path_a'], link['path_b']
        assert a in owners and b in owners and a in good_paths and b in good_paths, 'Reviewed link không thuộc ảnh hợp lệ.'
        for pa in owners[a]:
            for pb in owners[b]:
                uf.union(pa, pb)
        reviewed_links.append([a, b])
df['split_group'] = df.patient_id.map(lambda p: 'group_' + uf.find(p))
display(df.groupby('split_group').agg(patients=('patient_id', 'nunique'), checkups=('case_id', 'size'))
        .sort_values('patients', ascending=False).head(10))
print('Exact duplicate rows (two hash methods may overlap):', len(exact_duplicates))

path_hash = {r.path: int(r.dhash, 16) for r in image_audit.itertuples() if r.ok}
hash_paths = defaultdict(list)
for path, value in path_hash.items():
    hash_paths[value].append(path)
buckets = defaultdict(list)
for value in sorted(hash_paths):
    for block in range(4):
        buckets[(block, (value >> (16 * block)) & 0xffff)].append(value)
hash_pairs = set()
skipped_buckets = []
for (block, key), values in buckets.items():
    if len(values) > NEAR_DUPLICATE_BUCKET_CAP:
        skipped_buckets.append({'block': block, 'key': key, 'distinct_hashes': len(values)})
        continue
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            if (a ^ b).bit_count() <= NEAR_DUPLICATE_MAX_HAMMING:
                hash_pairs.add((a, b))
near_rows = [{'hash_a': f'{a:016x}', 'hash_b': f'{b:016x}', 'hamming': (a ^ b).bit_count(),
              'images_a': len(hash_paths[a]), 'images_b': len(hash_paths[b])} for a, b in sorted(hash_pairs)]
near_rows += [{'hash_a': f'{a:016x}', 'hash_b': f'{a:016x}', 'hamming': 0,
               'images_a': len(paths), 'images_b': len(paths)} for a, paths in sorted(hash_paths.items()) if len(paths) > 1]
near_candidates = pd.DataFrame(near_rows, columns=['hash_a', 'hash_b', 'hamming', 'images_a', 'images_b'])
near_candidates.to_csv(OUT / 'near_duplicate_hash_candidates.csv', index=False)
pd.DataFrame(skipped_buckets, columns=['block', 'key', 'distinct_hashes']).to_csv(OUT / 'near_duplicate_skipped_buckets.csv', index=False)
write_json(OUT / 'dhash_image_index.json', {f'{h:016x}': paths for h, paths in sorted(hash_paths.items())})
print('Near-duplicate hash candidates:', len(near_candidates), '| skipped buckets:', len(skipped_buckets))
''')

md('''
## 6. Split bệnh nhân cố định, cân bằng gần đúng

Chia **toàn bộ ca**, bao gồm ca chưa có nhãn, trước khi tạo cohort. Điều này ngăn ca chưa có nhãn của bệnh nhân
test lọt vào self-supervised training sau này. Cân bằng số ca toàn bộ, số ca single-target, support từng nhãn
single-target và multi-target bằng tìm kiếm có seed. Đây là heuristic, không phải thuật toán iterative stratification.
Chỉ dùng metadata nhãn để thiết kế split, không dùng ảnh/điểm model để chọn test thuận lợi.

Các ảnh hỏng được loại khỏi bag; ca không còn ảnh bị loại khỏi cohort huấn luyện nhưng vẫn có split trong master.
''')

code(r'''
SPLITS = list(SPLIT_RATIOS)
ratios = np.array(list(SPLIT_RATIOS.values()), dtype=float)
assert len(SPLITS) == 3 and set(SPLITS) == {'train', 'validation', 'test'}
assert np.isclose(ratios.sum(), 1) and (ratios > 0).all()
df['eligible_single'] = (df.n_targets == 1) & df.has_valid_images
df['eligible_multi_candidate'] = (df.n_targets >= 1) & df.has_valid_images
features = pd.DataFrame({'all_cases': np.ones(len(df)), 'single_cases': df.eligible_single.astype(int)})
for target in TARGETS:
    features['single_' + target] = [int(ok and target in xs) for ok, xs in zip(df.eligible_single, df.target_labels)]
    features['multi_' + target] = [int(ok and target in xs) for ok, xs in zip(df.eligible_multi_candidate, df.target_labels)]
features['split_group'] = df.split_group.values
group_features = features.groupby('split_group', sort=True).sum()
groups = group_features.index.to_numpy()
assert len(groups) >= 3, 'Không đủ nhóm độc lập để chia tập.'
matrix = group_features.to_numpy(dtype=float)
fingerprint = digest({'source': SOURCE_SHA256, 'mapping': LABEL_MAPPING, 'seed': SEED,
    'ratios': SPLIT_RATIOS, 'trials': SPLIT_SEARCH_TRIALS,
    'case_groups': sorted(zip(df.case_id, df.split_group)), 'valid_paths': sorted(good_paths),
    'exact_pixel_content': sorted((r.path, r.pixel_sha256) for r in image_audit.itertuples() if r.ok)})
lock_path = OUT / 'split_lock.json'
if lock_path.exists():
    split_lock = json.loads(lock_path.read_text(encoding='utf-8'))
    assert split_lock['fingerprint'] == fingerprint, 'Protocol/dữ liệu đã đổi. Dùng RUN_NAME mới; không ghi đè split đã khóa.'
    group_assignment = split_lock['group_assignment']
    print('Loaded locked split:', lock_path)
else:
    rng = np.random.default_rng(SEED)
    cuts = np.round(np.cumsum(ratios)[:-1] * len(groups)).astype(int)
    goal_counts = ratios[:, None] * matrix.sum(axis=0)[None, :]
    best_score, best_assignment = float('inf'), None
    for trial in range(SPLIT_SEARCH_TRIALS):
        partitions = np.split(rng.permutation(len(groups)), cuts)
        observed = np.array([matrix[p].sum(axis=0) for p in partitions])
        score = float(np.mean(((observed - goal_counts) / np.maximum(goal_counts, 1)) ** 2))
        # Phạt lớp biến mất trong partition khi dữ liệu tổng thể có lớp đó.
        score += float(((observed[:, 2:] == 0) & (goal_counts[:, 2:] > 0)).sum()) * 100
        if score < best_score:
            best_score = score
            best_assignment = {str(groups[i]): split for split, indexes in zip(SPLITS, partitions) for i in indexes}
    group_assignment = best_assignment
    split_lock = {'version': 'group_random_search_v1', 'fingerprint': fingerprint, 'seed': SEED,
                  'score': best_score, 'group_assignment': group_assignment}
    write_json(lock_path, split_lock)
df['split'] = df.split_group.map(group_assignment)
assert df.split.notna().all()
assert (df.groupby('patient_id').split.nunique() == 1).all()
assert (df.groupby('split_group').split.nunique() == 1).all()
df[['case_id', 'patient_id', 'split_group', 'split', 'original_split']].to_csv(OUT / 'split_assignments.csv', index=False)

split_summary = df.groupby('split').agg(checkups=('case_id', 'size'), patients=('patient_id', 'nunique'),
    groups=('split_group', 'nunique'), single_target_cases=('eligible_single', 'sum'),
    multilabel_candidates=('eligible_multi_candidate', 'sum')).reindex(SPLITS)
display(split_summary)
split_summary.to_csv(OUT / 'split_summary.csv')
support = pd.DataFrame([
    {'split': split, 'target': t,
     'single_cases': sum(ok and t in xs for ok, xs in zip(g.eligible_single, g.target_labels)),
     'multi_positive_cases': sum(ok and t in xs for ok, xs in zip(g.eligible_multi_candidate, g.target_labels))}
    for split, g in df.groupby('split') for t in TARGETS
])
support.to_csv(OUT / 'label_support_by_split.csv', index=False)
display(support.pivot(index='target', columns='split', values='single_cases').reindex(TARGETS))
assert (support.single_cases > 0).all(), 'Có lớp mất khỏi một partition. Cần thiết kế protocol mới.'
''')

code(r'''
# Kiểm tra trực tiếp hash qua split sau khi khóa, và shortlist ứng viên gần trùng cần xem.
case_split = dict(zip(df.case_id, df.split))
joined_images['split'] = joined_images.case_id.map(case_split)
for column in ['file_sha256', 'pixel_sha256']:
    counts = joined_images.loc[joined_images.ok].groupby(column).split.nunique()
    assert (counts <= 1).all(), f'Exact duplicate leakage: {column}'
path_splits = joined_images.groupby('path').split.agg(set).to_dict()
hash_splits = {h: set().union(*(path_splits[p] for p in paths)) for h, paths in hash_paths.items()}
cross_near = []
for r in near_candidates.itertuples():
    a, b = int(r.hash_a, 16), int(r.hash_b, 16)
    partitions = hash_splits[a] | hash_splits[b]
    if len(partitions) > 1:
        cross_near.append({'hash_a': r.hash_a, 'hash_b': r.hash_b, 'hamming': r.hamming,
                           'partitions': ','.join(sorted(partitions)),
                           'example_a': hash_paths[a][0], 'example_b': hash_paths[b][0],
                           'review_decision': ''})
cross_review = pd.DataFrame(cross_near, columns=['hash_a', 'hash_b', 'hamming', 'partitions', 'example_a', 'example_b', 'review_decision'])
cross_review_path = OUT / 'cross_split_near_duplicate_review.csv'
if cross_review_path.exists():
    old_review = pd.read_csv(cross_review_path, dtype=str, keep_default_na=False)
    old_decisions = {(r.hash_a, r.hash_b): r.review_decision for r in old_review.itertuples()}
    cross_review['review_decision'] = [old_decisions.get((r.hash_a, r.hash_b), '') for r in cross_review.itertuples()]
cross_review.to_csv(cross_review_path, index=False)
print('PASS: patient/component/exact-file/exact-pixel disjoint.')
print('Cross-split near-duplicate hash candidates requiring review:', len(cross_near))
print('Nếu xác nhận ảnh gần trùng qua split: bổ sung liên kết nhóm và tạo protocol version mới trước training.')
print('Dùng REVIEWED_DUPLICATE_LINKS trỏ tới CSV path_a,path_b,decision; decision=duplicate sẽ gộp nhóm.')

def show_duplicate_candidate(hash_a, hash_b, max_images_per_hash=3):
    # Chỉ phục vụ xác minh trùng dữ liệu; không dùng để chọn model dựa trên test.
    paths = list(dict.fromkeys(hash_paths[int(hash_a, 16)][:max_images_per_hash] +
                              hash_paths[int(hash_b, 16)][:max_images_per_hash]))
    fig, axes = plt.subplots(1, len(paths), figsize=(4 * len(paths), 3), squeeze=False)
    for ax, path in zip(axes[0], paths):
        with Image.open(DATA_ROOT / path) as img:
            ax.imshow(ImageOps.exif_transpose(img).convert('RGB'))
        ax.set_title(path.split('/')[-1] + '\n' + ','.join(sorted(path_splits[path])), fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
# Ví dụ khi rà soát: show_duplicate_candidate(hash_a='...', hash_b='...')
''')

md('''
## 7. EDA trực quan trên train sau khi khóa split

Các biểu đồ dưới dùng **train theo split mới**, không dùng test để chọn model.
Phân biệt rõ: số ca ≠ số patient ≠ số ảnh. Nhãn là bệnh được ghi nhận trong hồ sơ,
không phải kết luận từng ảnh đều chứa bệnh. Thiếu nhãn/clinical không được diễn giải là âm tính.

Đọc theo thứ tự: ảnh và modality → nhân khẩu học → nhãn/cohort → đồng xuất hiện → chất lượng clinical
→ lịch sử khám → ảnh minh họa. Mỗi figure được lưu PNG trong `figures/` để dùng trong báo cáo.
''')

code(r'''
train_df = df.loc[df.split == 'train'].copy()
clinical_fields = ['chief_complaint', 'present_illness', 'past_medical_record', 'examination', 'radiographs_examination']
clinical_names = ['Chief complaint', 'Present illness', 'Medical history', 'Examination', 'X-ray description']
class_colors = ['#2563eb', '#0d9488', '#7c3aed', '#e11d48', '#d97706', '#0891b2']

def labeled_bars(ax, labels, values, *, horizontal=False, color=BLUE, percent_base=None):
    values = np.asarray(values)
    bars = ax.barh(labels, values, color=color) if horizontal else ax.bar(labels, values, color=color)
    annotations = [f'{int(v):,}' + (f' ({v / percent_base * 100:.1f}%)' if percent_base else '') for v in values]
    ax.bar_label(bars, labels=annotations, padding=4, fontsize=9)
    maximum = max(values.max() if len(values) else 0, 1)
    if horizontal:
        ax.set_xlim(0, maximum * 1.4); ax.grid(axis='x')
    else:
        ax.set_ylim(0, maximum * 1.3); ax.grid(axis='y')
    ax.set_axisbelow(True)
    return bars

eda_summary = pd.DataFrame([
    {'metric': 'train_visits', 'value': len(train_df)},
    {'metric': 'train_patients', 'value': train_df.patient_id.nunique()},
    {'metric': 'train_split_groups', 'value': train_df.split_group.nunique()},
    {'metric': 'train_photo_references_before_bag_dedup', 'value': int(train_df.n_photos.sum())},
    {'metric': 'train_xray_references_before_bag_dedup', 'value': int(train_df.n_radiographs.sum())},
    {'metric': 'train_single_target_with_photo', 'value': int((train_df.eligible_single & (train_df.n_photos > 0)).sum())},
])
eda_summary.to_csv(OUT / 'train_eda_summary.csv', index=False)
display(eda_summary)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, field, label in [(axes[0, 0], 'n_photos', 'Photographs'), (axes[0, 1], 'n_radiographs', 'Radiographs')]:
    counts = train_df[field].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=BLUE if field == 'n_photos' else TEAL, width=0.8)
    median = train_df[field].median()
    ax.axvline(median, color=RED, ls='--', label=f'Median: {median:g}')
    ax.set_xlabel(f'{label} per visit'); ax.set_ylabel('Visits')
    ax.set_title(f'Train: {label.lower()} count distribution'); ax.legend(); ax.grid(axis='y'); ax.set_axisbelow(True)
photo_present, xray_present = train_df.n_photos > 0, train_df.n_radiographs > 0
modality_counts = [int((photo_present & xray_present).sum()), int((photo_present & ~xray_present).sum()),
                   int((~photo_present & xray_present).sum()), int((~photo_present & ~xray_present).sum())]
labeled_bars(axes[1, 0], ['Photo + X-ray', 'Photo only', 'X-ray only', 'No image'], modality_counts,
             color=[TEAL, BLUE, ORANGE, RED], percent_base=len(train_df))
axes[1, 0].set_ylabel('Visits'); axes[1, 0].set_title('Train: available image modalities')
missing_counts = [int(train_df[k].map(is_missing).sum()) for k in clinical_fields]
labeled_bars(axes[1, 1], clinical_names, missing_counts, horizontal=True, color=ORANGE, percent_base=len(train_df))
axes[1, 1].set_xlabel('Visits with missing field'); axes[1, 1].set_title('Train: clinical missingness')
fig.suptitle('Image and clinical availability — locked training partition', fontsize=15, weight='bold')
save_figure('02_train_case_eda.png')
''')

md('''
### 7.1. Tuổi, giới và đơn vị thống kê

Tuổi/giới dưới đây đếm **theo lần khám**, không phải mỗi người một phiếu.
Tuổi thiếu được bỏ khỏi histogram nhưng số lượng được ghi; giới tính thiếu được giữ thành `Missing`.
Không loại tuổi chỉ vì khác khoảng tuổi mô tả trong paper; chỉ báo giá trị ngoài [0,120].
''')

code(r'''
age = pd.to_numeric(train_df.age, errors='coerce')
valid_age = age.between(0, 120)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].hist(age.loc[valid_age], bins=np.arange(0, 126, 5), color=BLUE, edgecolor='white')
axes[0].set_xlabel('Recorded age (years)'); axes[0].set_ylabel('Visits')
axes[0].set_title(f'Age per visit; missing/invalid: {int((~valid_age).sum()):,}')
if valid_age.any():
    axes[0].axvline(age.loc[valid_age].median(), color=RED, ls='--', label=f'Median: {age.loc[valid_age].median():g}')
    axes[0].legend()
gender_counts = train_df.gender.replace('', 'Missing').value_counts()
labeled_bars(axes[1], gender_counts.index, gender_counts.values, color=[TEAL, BLUE, ORANGE][:len(gender_counts)],
             percent_base=len(train_df))
axes[1].set_ylabel('Visits'); axes[1].set_title('Recorded gender per visit')
fig.suptitle('Train demographics — visit-weighted, without imputation', fontsize=15, weight='bold')
save_figure('03_train_demographics.png')
''')

md('''
### 7.2. Nhãn và các bước chọn cohort

Biểu đồ lớp dùng **cohort trainer thực nhận**: đúng một target và có photo.
Biểu đồ số target dùng mọi ca train. Một ca nhiều target góp positive cho nhiều lớp,
nhưng không vào CE classifier hiện tại. Các cột chọn cohort là tập lồng nhau, không phải các nhóm rời nhau.
''')

code(r'''
model_cohort = train_df.loc[train_df.eligible_single & (train_df.n_photos > 0)]
model_counts = [sum(t in xs for xs in model_cohort.target_labels) for t in TARGETS]
stages = ['All train visits', 'Target + valid image', 'Exactly one target', 'Single target + photo']
stage_counts = [len(train_df), int(train_df.eligible_multi_candidate.sum()), int(train_df.eligible_single.sum()), len(model_cohort)]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
labeled_bars(axes[0], TARGETS, model_counts, horizontal=True, color=class_colors, percent_base=len(model_cohort))
axes[0].set_title(f'Trainer class support (n={len(model_cohort):,})'); axes[0].set_xlabel('Visits')
labeled_bars(axes[1], stages[::-1], stage_counts[::-1], horizontal=True, color=TEAL)
axes[1].set_title('Nested cohort selection'); axes[1].set_xlabel('Visits')
target_count = train_df.n_targets.value_counts().sort_index()
labeled_bars(axes[2], [str(i) for i in target_count.index], target_count.values, color=BLUE)
axes[2].set_xlabel('Mapped targets recorded per visit'); axes[2].set_ylabel('Visits')
axes[2].set_title('0 targets does not mean healthy')
fig.suptitle('Training labels and cohort — recorded targets', fontsize=15, weight='bold')
save_figure('05_train_labels_and_cohort.png')
''')

md('''
### 7.3. Các bệnh được ghi nhận cùng nhau

Bên trái là số ca có cả hai target, diagonal là support mỗi target.
Bên phải chia mỗi hàng cho support của target ở hàng đó: tỷ lệ có thêm target ở cột,
**không phải xác suất bệnh thật**, vì annotation có thể chưa đầy đủ. Các nhãn không ghi nhận vẫn chưa xác định.
''')

code(r'''
cooccurrence = np.array([[sum(a in xs and b in xs for xs in train_df.target_labels) for b in TARGETS] for a in TARGETS])
conditional = np.divide(cooccurrence, cooccurrence.diagonal()[:, None],
    out=np.zeros_like(cooccurrence, dtype=float), where=cooccurrence.diagonal()[:, None] > 0) * 100
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, matrix, title, percentage in [(axes[0], cooccurrence, 'Recorded co-occurrence (visits)', False),
                                     (axes[1], conditional, 'Column target among row-positive visits (%)', True)]:
    image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100 if percentage else max(cooccurrence.max(), 1))
    ax.set_xticks(range(6), TARGETS, rotation=55, ha='right'); ax.set_yticks(range(6), TARGETS)
    ax.set_title(title)
    threshold = (100 if percentage else max(cooccurrence.max(), 1)) / 2
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{matrix[i,j]:.1f}' if percentage else f'{int(matrix[i,j]):,}',
                    ha='center', va='center', color='white' if matrix[i,j] > threshold else '#0f172a', fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
save_figure('06_train_target_cooccurrence.png')
''')

md('''
### 7.4. Độ đầy đủ và độ dài clinical

Heatmap là mẫu tối đa 120 ca train chọn cố định để nhìn pattern thiếu trường, không phải tỷ lệ toàn train.
Boxplot dùng số ký tự ở các giá trị không thiếu; giữ outlier trong thống kê, ẩn marker outlier để dễ đọc.
Chưa phải xác nhận clinical an toàn cho Teacher; bước screening/review ở mục 8 làm riêng.
''')

code(r'''
from matplotlib.colors import ListedColormap
sample_clinical = train_df.sample(n=min(120, len(train_df)), random_state=SEED).sort_values('case_id')
availability = np.array([[not is_missing(r[k]) for k in clinical_fields] for r in sample_clinical.to_dict('records')], dtype=int)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
image = axes[0].imshow(availability, cmap=ListedColormap(['#e2e8f0', TEAL]), vmin=0, vmax=1, aspect='auto', interpolation='nearest')
axes[0].set_xticks(range(5), clinical_names, rotation=40, ha='right'); axes[0].set_yticks([])
axes[0].set_ylabel(f'{len(sample_clinical)} sampled training visits'); axes[0].set_title('Field availability: gray=missing, teal=present')
lengths = [train_df.loc[~train_df[k].map(is_missing), k].str.len().to_numpy() for k in clinical_fields]
lengths = [x if len(x) else np.array([np.nan]) for x in lengths]
box_orientation = {'orientation': 'horizontal'} if 'orientation' in inspect.signature(axes[1].boxplot).parameters else {'vert': False}
boxes = axes[1].boxplot(lengths, **box_orientation, patch_artist=True, showfliers=False)
axes[1].set_yticks(range(1, len(clinical_names) + 1), clinical_names)
for patch in boxes['boxes']: patch.set_facecolor('#99f6e4')
axes[1].set_xlabel('Characters in non-missing text'); axes[1].set_title('Clinical text length — outlier markers hidden')
axes[1].grid(axis='x'); axes[1].set_axisbelow(True)
save_figure('07_train_clinical_quality.png')
''')

md('''
### 7.5. Lần khám theo thời gian và bệnh nhân

Histogram số lần khám đếm mỗi patient một lần trong train. Biểu đồ năm dùng thời gian đọc được;
nhóm `split_group` có thể nối nhiều patient IDs do ảnh trùng, không khẳng định họ là cùng một người.
''')

code(r'''
train_times = pd.to_datetime(train_df.checkup_time, format='mixed', dayfirst=True, errors='coerce')
years = train_times.dropna().dt.year.value_counts().sort_index()
visits_per_patient = train_df.groupby('patient_id').size().value_counts().sort_index()
patients_per_group = train_df.groupby('split_group').patient_id.nunique().value_counts().sort_index()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
labeled_bars(axes[0], [str(x) for x in years.index], years.values, color=BLUE)
axes[0].set_title(f'Train visits by year; unparsed: {int(train_times.isna().sum())}'); axes[0].set_ylabel('Visits')
axes[0].tick_params(axis='x', rotation=35)
axes[1].bar(visits_per_patient.index, visits_per_patient.values, color=TEAL)
axes[1].set_title('Follow-up structure'); axes[1].set_xlabel('Visits per patient'); axes[1].set_ylabel('Patients')
axes[2].bar(patients_per_group.index, patients_per_group.values, color=ORANGE)
axes[2].set_title('Duplicate-linked split groups'); axes[2].set_xlabel('Patient IDs per group'); axes[2].set_ylabel('Groups')
save_figure('08_train_time_and_followup.png')
''')

md('''
### 7.6. Ảnh mẫu theo từng lớp

Chọn một ca train có photo cho mỗi lớp; hiển thị tối đa ba photo và một X-quang.
Các cột trống được ghi rõ, không tạo ảnh thay thế. Tên lớp là nhãn ca, không phải nhãn đã xác nhận của từng ảnh.
''')

code(r'''
examples = model_cohort.groupby(model_cohort.target_labels.map(lambda xs: '|'.join(xs))).sample(n=1, random_state=SEED)
fig, axes = plt.subplots(len(examples), 4, figsize=(12, 2.6 * len(examples)), squeeze=False)
for row_index, r in enumerate(examples.itertuples()):
    paths = r.valid_photo_paths[:3] + [None] * max(0, 3 - len(r.valid_photo_paths[:3]))
    paths += r.valid_radiograph_paths[:1] or [None]
    for col, ax in enumerate(axes[row_index]):
        ax.axis('off'); path = paths[col]
        if path is not None:
            with Image.open(DATA_ROOT / path) as img:
                ax.imshow(ImageOps.exif_transpose(img).convert('RGB'))
            ax.set_title(('Photo' if col < 3 else 'Radiograph') + '\n' + r.target_labels[0], fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No additional photo' if col < 3 else 'No radiograph', ha='center', va='center', color='#64748b')
fig.suptitle('Training examples — visit labels, not image-level annotations', fontsize=14, weight='bold')
save_figure('04_train_examples.png')
''')

md('''
## 8. Clinical fields: allowlist, target mentions và bộ rà soát

Teacher chỉ nhận năm trường dự kiến cùng age/gender. Không đưa `patient_record`, `diagnosis`, `anomalies`,
treatment, management, instructions hoặc remarks vào input. EN/CN không được coi là mẫu độc lập.

Tạo hai phiên bản: raw allowlist và bản bỏ **câu** có tên target theo từ điển. Đây là sàng lọc heuristic,
không bảo đảm loại hết leakage, không thay thế đánh giá chuyên gia. Ví dụ “no caries” cũng bị bỏ để tránh giữ
từ phủ định gây đổi nghĩa. Cần kiểm tra các tên đồng nghĩa còn thiếu và các mô tả hậu điều trị.
Các nhóm V/H chỉ là prior; pocket depth, percussion, vitality không tự nhiên nhìn thấy được từ ảnh.

Bộ rà soát lấy từ train theo nhãn, kèm case không có nhãn và multi-target. Các cột quyết định để trống:
không tự sinh xác nhận lâm sàng. Tệp chứa nội dung hồ sơ phải được giữ trong môi trường dữ liệu được cấp quyền.
''')

code(r'''
FIELD_GROUPS = {'visual_candidate': ['examination', 'radiographs_examination'],
                'history_symptom': ['chief_complaint', 'present_illness', 'past_medical_record']}
ALLOWLIST = [field for fields in FIELD_GROUPS.values() for field in fields]
FORBIDDEN = {'patient_record', 'diagnosis', 'anomalies_en', 'anomalies_cn', 'treatment_plan',
             'management', 'medical_instructions', 'remarks', 'treatment_recommendations'}
assert not FORBIDDEN.intersection(ALLOWLIST)
TARGET_ALIASES = sorted(set(LABEL_MAPPING) | {'caries', 'carious', 'missing tooth', 'missing teeth',
    'tooth missing', 'teeth missing', 'loss of tooth structure', '牙龈炎', '龋齿', '牙髓炎', '牙齿缺失', '牙体缺损', '错颌'}, key=len, reverse=True)
target_pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in TARGET_ALIASES) + r')(?!\w)', re.I)
def screen_field(value):
    if is_missing(value):
        return '', [], False
    sentences = re.split(r'(?<=[.!?。！？;；])\s*|[\r\n]+', str(value).strip())
    removed = [s for s in sentences if target_pattern.search(s)]
    kept = ' '.join(s for s in sentences if s and not target_pattern.search(s))
    return kept, removed, bool(removed)

teacher_records, field_audit = [], []
for r in df.to_dict('records'):
    screened, original, available, screen_available, target_flags = {}, {}, {}, {}, {}
    for field in ALLOWLIST:
        original[field] = '' if is_missing(r[field]) else r[field].strip()
        clean, removed, flagged = screen_field(r[field])
        screened[field], available[field], screen_available[field], target_flags[field] = clean, bool(original[field]), bool(clean), flagged
        field_audit.append({'case_id': r['case_id'], 'split': r['split'], 'field': field,
                            'available': bool(original[field]), 'target_mention': flagged,
                            'removed_sentences': len(removed), 'raw_chars': len(original[field]), 'screened_chars': len(clean)})
    teacher_records.append({'case_id': r['case_id'], 'patient_id': r['patient_id'], 'split': r['split'],
        'clinical_raw_allowlist': original, 'clinical_screened': screened,
        'raw_available_mask': available, 'screened_available_mask': screen_available,
        'target_mention_flags': target_flags, 'age_raw': r.get('age', ''), 'gender_raw': r.get('gender', ''),
        'radiographic_text_without_valid_radiograph': bool(original['radiographs_examination']) and r['n_radiographs'] == 0})
field_audit = pd.DataFrame(field_audit)
field_audit.to_csv(OUT / 'clinical_field_audit.csv', index=False)
display(field_audit.loc[field_audit.split == 'train'].groupby('field').agg(
    available=('available', 'sum'), target_mentions=('target_mention', 'sum'),
    mean_raw_chars=('raw_chars', 'mean'), mean_screened_chars=('screened_chars', 'mean')).round(1))

strata = train_df.target_labels.map(lambda xs: '|'.join(xs) if xs else 'NO_MAPPED_TARGET')
review_ids = []
for _, group in train_df.groupby(strata):
    review_ids.extend(group.sample(n=1, random_state=SEED).case_id.tolist())
review_ids = sorted(set(review_ids))
if len(review_ids) > REVIEW_SAMPLE_SIZE:
    review_ids = sorted(random.Random(SEED).sample(review_ids, REVIEW_SAMPLE_SIZE))
remaining = train_df.loc[~train_df.case_id.isin(review_ids)]
review_ids += remaining.sample(n=min(max(REVIEW_SAMPLE_SIZE - len(review_ids), 0), len(remaining)), random_state=SEED).case_id.tolist()
review = train_df.loc[train_df.case_id.isin(review_ids), ['case_id', 'patient_id', 'anomalies_en', *ALLOWLIST]].copy()
for column in ['label_mapping_correct', 'annotation_complete_for_six_targets', 'post_diagnosis_leakage',
               'visual_grounding_review', 'image_time_alignment', 'reviewer', 'review_notes']:
    review[column] = ''
review_path = OUT / 'clinical_label_review_train.csv'
# Không ghi đè nhận xét mà người dùng đã điền khi Run All lần sau.
if review_path.exists():
    previous = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    assert set(previous.case_id) == set(review.case_id), 'Bộ review thay đổi; tạo RUN_NAME mới.'
else:
    review.to_csv(review_path, index=False, encoding='utf-8-sig')
print('Train review cases:', len(review), '| Clinical fields reviewed:', CLINICAL_FIELDS_REVIEWED)
''')

md('''
## 9. Xuất manifest Student/Teacher và cấu hình thực nghiệm

Đường dẫn ảnh tương đối với `DATA_ROOT`, có modality riêng. Manifest Student chỉ có ảnh, nhãn, ID, split;
không chứa tuổi/giới/hồ sơ. Teacher context nằm ở tệp riêng để tránh vô tình dùng clinical text khi inference.
ID dùng join/kiểm tra, không làm feature. Manifest không đưa ảnh hợp lệ trùng pixel nhiều lần trong cùng bag.
Các bảng loại trừ giải thích lý do mỗi ca không thuộc cohort chính.
''')

code(r'''
pixel_by_path = dict(zip(image_audit.path, image_audit.pixel_sha256))
def dedupe_bag(paths):
    seen, result = set(), []
    for path in paths:
        key = pixel_by_path[path]
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result

def make_visual_record(r):
    photos = dedupe_bag(r['valid_photo_paths'])
    xrays = dedupe_bag(r['valid_radiograph_paths'])
    y = [int(t in r['target_labels']) for t in TARGETS]
    return {'dataset': 'COde', 'case_id': r['case_id'], 'patient_id': r['patient_id'],
            'split_group': r['split_group'], 'split': r['split'],
            'photo_paths': photos, 'radiograph_paths': xrays,
            'bag_signature': digest(sorted(set(pixel_by_path[p] for p in photos + xrays))),
            'visual_modality_mask': [bool(photos), bool(xrays)],
            'target_labels': r['target_labels'], 'labels_multihot': y,
            'label_mask': ([1] * len(TARGETS) if ANNOTATION_ASSUMPTION_APPROVED and r['source_labels'] else y),
            'label_status': r['label_status'],
            'single_label_index': TARGETS.index(r['target_labels'][0]) if r['n_targets'] == 1 else None,
            'eligible_single': bool(r['eligible_single']),
            'eligible_multi_candidate': bool(r['eligible_multi_candidate'])}

visual_records = [make_visual_record(r) for r in df.to_dict('records')]
def write_jsonl(path, records):
    with Path(path).open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

write_jsonl(OUT / 'visual_master.jsonl', visual_records)
bag_audit = pd.DataFrame([{'case_id': r['case_id'], 'patient_id': r['patient_id'], 'split': r['split'],
    'bag_signature': r['bag_signature'], 'recorded_targets': '|'.join(r['target_labels'])}
    for r in visual_records if r['photo_paths'] or r['radiograph_paths']])
bag_audit.loc[bag_audit.bag_signature.duplicated(keep=False)].to_csv(OUT / 'identical_bag_review.csv', index=False)
assert (bag_audit.groupby('bag_signature').split.nunique() <= 1).all()
for split in SPLITS:
    write_jsonl(OUT / f'student_single_{split}.jsonl', [r for r in visual_records if r['split'] == split and r['eligible_single']])
    write_jsonl(OUT / f'student_multilabel_candidate_{split}.jsonl', [r for r in visual_records if r['split'] == split and r['eligible_multi_candidate']])
    write_jsonl(OUT / f'teacher_context_{split}.jsonl', [r for r in teacher_records if r['split'] == split])

exclusion = df[['case_id', 'patient_id', 'split', 'label_status', 'n_targets', 'has_valid_images']].copy()
exclusion['single_cohort_exclusion'] = [
    ';'.join(reason for condition, reason in [(not good, 'no_valid_images'), (n == 0, status),
                                            (n > 1, 'multiple_recorded_targets')] if condition)
    for good, n, status in zip(df.has_valid_images, df.n_targets, df.label_status)]
exclusion.to_csv(OUT / 'cohort_inclusion_exclusion.csv', index=False)

train_single = [r for r in visual_records if r['split'] == 'train' and r['eligible_single']]
train_counts = np.bincount([r['single_label_index'] for r in train_single], minlength=len(TARGETS))
class_weights = (len(train_single) / (len(TARGETS) * np.maximum(train_counts, 1))).tolist()
experiment_config = {
    'dataset': 'COde', 'task': 'single_target_recorded_v1', 'class_names': TARGETS,
    'seed': SEED, 'split_fingerprint': fingerprint,
    'image_root_resolve_at_runtime': True, 'training_class_counts': train_counts.tolist(),
    'optional_inverse_frequency_class_weights_train_only': class_weights,
    'primary_metric': 'macro_f1', 'bootstrap_unit': 'split_group',
    'experiments': ['single_photo', 'multi_photo_mean', 'multi_photo_attention_mil',
                    'photo_plus_radiograph_mil', 'clinical_only_teacher', 'multimodal_teacher',
                    'vanilla_kd_student', 'field_selective_kd_student'],
    'notes': ['Same patient split across all ablations.', 'Match cohort and image budget within each comparison.',
              'Teacher/student see identical visual inputs when isolating clinical KD.',
              'Bootstrap split_group because exact duplicate images connect different patient IDs.',
              'Report sensitivity to repeated identical bags; inspect identical_bag_review.csv.',
              'Do not use patient_id/case_id/split as model features.',
              'MeMoSA needs independent label head and verified encounter grouping.']}
write_json(OUT / 'experiment_config.json', experiment_config)
print('Exported all-case master, six cohort manifests, three separate Teacher contexts.')
''')

md('''
## 10. Kiểm tra đầu ra và minh họa đọc một bag

Kiểm tra bằng các file vừa xuất, không chỉ DataFrame trong RAM. Mẫu dưới đây chỉ lấy từ train và đọc ảnh RGB.
Nó chưa augmentation, normalization, resize hoặc padding; các bước đó thuộc notebook baseline sau.
''')

code(r'''
def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

loaded = {s: read_jsonl(OUT / f'student_single_{s}.jsonl') for s in SPLITS}
all_exported = sum(loaded.values(), [])
assert len(all_exported) == int(df.eligible_single.sum())
assert len({r['case_id'] for r in all_exported}) == len(all_exported)
for i, a in enumerate(SPLITS):
    for b in SPLITS[i + 1:]:
        assert not ({r['patient_id'] for r in loaded[a]} & {r['patient_id'] for r in loaded[b]})
for r in all_exported:
    assert 0 <= r['single_label_index'] < len(TARGETS)
    assert sum(r['labels_multihot']) == 1
    assert r['photo_paths'] or r['radiograph_paths']
    assert not (set(r) & (FORBIDDEN | set(ALLOWLIST) | {'age_raw', 'gender_raw', 'clinical_screened'}))
    assert all((DATA_ROOT / path).is_file() for path in r['photo_paths'] + r['radiograph_paths'])
teacher_by_case = {r['case_id']: r for r in teacher_records}
assert len(teacher_by_case) == len(df)
assert all(teacher_by_case[r['case_id']]['split'] == r['split'] for r in all_exported)
for r in read_jsonl(OUT / 'visual_master.jsonl'):
    assert len(r['label_mask']) == len(TARGETS) == len(r['labels_multihot'])
    if not ANNOTATION_ASSUMPTION_APPROVED:
        assert r['label_mask'] == r['labels_multihot']

example = loaded['train'][0]
bag = []
for path in example['photo_paths'] + example['radiograph_paths']:
    with Image.open(DATA_ROOT / path) as img:
        bag.append(ImageOps.exif_transpose(img).convert('RGB').copy())
print('PASS: exported manifests, labels, paths, patient disjointness, Teacher join and Student input isolation.')
print('Example training bag:', len(bag), 'images; recorded target:', TARGETS[example['single_label_index']])
''')

md('''
## 11. Báo cáo cuối và điều kiện chuyển sang baseline

**PASS kỹ thuật không đồng nghĩa nhãn được chuyên gia xác nhận.** Bảng dưới phân biệt rõ kiểm tra tự động
và việc còn cần rà soát. Chưa có kết quả model, chưa có chứng minh hiệu quả clinical KD.

Trước thí nghiệm chính: xử lý ứng viên ảnh gần trùng qua split; xác nhận mapping/cohort; rà soát leakage và
tính tương ứng thời điểm ảnh/hồ sơ. Khi đổi quyết định ảnh trùng hoặc mapping, tạo phiên bản protocol mới.
Khóa test cho tới giai đoạn đánh giá cuối, lưu config/checkpoint/prediction theo case ID và nhiều seed.
''')

code(r'''
checks = [
    {'check': 'unique checkup IDs and valid patient IDs', 'status': 'PASS'},
    {'check': 'all referenced images fully decoded or explicitly excluded', 'status': 'PASS', 'invalid_images': int((~image_audit.ok).sum())},
    {'check': 'patient / exact file / exact RGB pixel split disjointness', 'status': 'PASS'},
    {'check': 'exported manifests and isolated Student input', 'status': 'PASS'},
    {'check': 'near-duplicate visual review', 'status': 'REVIEW_REQUIRED' if cross_near or skipped_buckets else 'NO_CANDIDATES_FOUND_NOT_A_GUARANTEE'},
    {'check': 'recorded target mapping and single-target cohort definition', 'status': 'USER_DECLARED_REVIEWED' if LABEL_MAPPING_REVIEWED else 'REVIEW_REQUIRED'},
    {'check': 'label completeness for six-target multi-label training', 'status': 'USER_DECLARED_APPROVED' if ANNOTATION_ASSUMPTION_APPROVED else 'NOT_APPROVED'},
    {'check': 'clinical field content and temporal alignment', 'status': 'USER_DECLARED_REVIEWED' if CLINICAL_FIELDS_REVIEWED else 'REVIEW_REQUIRED'},
]
display(pd.DataFrame(checks).fillna(''))
run_summary = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'dataset': 'COde',
    'source_csv_sha256': SOURCE_SHA256, 'split_fingerprint': fingerprint,
    'english_csv_cleaning': cleaning_summary,
    'checkups': len(df), 'patients': int(df.patient_id.nunique()),
    'original_csv_patient_overlap': len(original_overlap), 'referenced_images': len(image_audit),
    'exact_duplicate_file_hash_groups': int(exact_duplicates.loc[exact_duplicates.hash_type == 'file_sha256', 'hash'].nunique()),
    'independent_split_groups': int(df.split_group.nunique()),
    'invalid_images': int((~image_audit.ok).sum()),
    'single_target_eligible': int(df.eligible_single.sum()), 'multilabel_candidates': int(df.eligible_multi_candidate.sum()),
    'cross_split_near_duplicate_hash_candidates': len(cross_near), 'near_duplicate_skipped_buckets': len(skipped_buckets),
    'checks': checks, 'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                                  'numpy': np.__version__, 'pillow': PIL_VERSION},
    'limitations': ['Labels are recorded conditions, not independently established image-level truth.',
                   'Near-duplicate candidates need review; dHash misses some transformed duplicates.',
                   'Clinical sentence screening is a heuristic, not verified leakage elimination.',
                   'Single-target cohort excludes co-occurring mapped targets and is selected.',
                   'No model training or held-out predictive evaluation in this notebook.']}
write_json(OUT / 'run_summary.json', run_summary)
report = '\n'.join([
    '# COde — kết quả kiểm định dữ liệu', '',
    '## CSV tiếng Anh và trực quan dữ liệu', '',
    f'- [complete_dataset_en.csv]({os.path.relpath(CLEAN_CSV_PATH, OUT).replace(chr(92), chr(47))}): {len(clean_en):,} ca, {len(english_columns)} cột tiếng Anh; bỏ {len(cn_columns)} cột `_cn`.',
    f'- Ô được trim/chuẩn hóa missing: {cleaning_summary["changed_cells"]:,}; vấn đề cần review: {len(quality_issues):,}.',
    '- Giữ mọi ca, không điền tuổi/giới/nhãn; CSV nguồn được kiểm tra giữ nguyên SHA256.',
    '- [Dashboard CSV](figures/00_csv_cleaning_overview.png); [nhãn/cohort train](figures/05_train_labels_and_cohort.png).',
    '- EDA train: modality, tuổi/giới, đồng xuất hiện target, clinical missingness/length, thời gian/lần khám, ảnh mẫu.', '',
    '## Kiểm định ảnh, nhóm và nhãn', '',
    f'- Số lần khám: {len(df):,}; số bệnh nhân: {df.patient_id.nunique():,}.',
    f'- Bệnh nhân giao train/test gốc: {len(original_overlap):,}.',
    f'- Ảnh đã kiểm tra: {len(image_audit):,}; ảnh thiếu/hỏng: {int((~image_audit.ok).sum()):,}.',
    f'- Nhóm file trùng SHA256: {exact_duplicates.loc[exact_duplicates.hash_type == "file_sha256", "hash"].nunique():,}.',
    f'- Nhóm chia tập sau liên kết ảnh trùng: {df.split_group.nunique():,}; bootstrap dùng split_group.',
    f'- Ca single-target đủ ảnh: {int(df.eligible_single.sum()):,}.',
    f'- Ca multi-label candidate đủ ảnh: {int(df.eligible_multi_candidate.sum()):,}.',
    f'- Ứng viên gần trùng qua split cần rà soát: {len(cross_near):,} nhóm hash.',
    f'- Bucket near-duplicate quá lớn chưa xét: {len(skipped_buckets):,}.', '',
    '## Việc còn lại', '',
    '- Rà soát clinical_label_review_train.csv và cross_split_near_duplicate_review.csv.',
    '- Chốt mapping và task với người hướng dẫn; nhãn vắng chưa được coi là âm tính đã xác nhận.',
    '- Xây baseline single-image → mean pooling → Attention MIL trên cùng split.',
    '- Sau baseline mới xây Teacher → vanilla KD → selective KD; MeMoSA bổ sung độc lập.', '',
    'Các manifest Student không có clinical input. Không có kết quả huấn luyện mới trong báo cáo này.',
])
(OUT / 'REPORT.md').write_text(report + '\n', encoding='utf-8')
artifacts = {p.name: sha256_file(p) for p in sorted(OUT.glob('*'))
             if p.is_file() and p.name not in {'artifact_checksums.json', 'image_audit_cache.json'}}
write_json(OUT / 'artifact_checksums.json', artifacts)
display(Markdown(report))
print('Artifacts saved in:', OUT)
''')

md('''
## 12. Lộ trình notebook tiếp theo

1. **COde baseline:** loader bag, padding masks, fixed patient split; single-photo, multi-photo mean/MIL,
   rồi photo+X-ray. Dùng macro-F1 cho task single-target, bootstrap theo `split_group` và tối thiểu ba seed ở kết quả chính.
   `split_group` có thể chứa nhiều patient ID liên kết bởi ảnh trùng; chỉ bootstrap patient ID sẽ bỏ qua tương quan này.
2. **COde clinical distillation:** clinical-only Teacher, multimodal Teacher, vanilla KD rồi field-selective KD.
   Dùng cùng ảnh và cùng Student khi muốn cô lập đóng góp clinical context.
3. **MeMoSA kiểm chứng:** adapter/head riêng; xác minh encounter/lesion trước khi gom ảnh, không trộn với nhãn COde.

Thư mục output gồm `REPORT.md`, `run_summary.json`, `split_lock.json`, `label_policy.json`,
`student_single_{train,validation,test}.jsonl`, `student_multilabel_candidate_*.jsonl`,
`teacher_context_*.jsonl`, CSV audit/review và hình EDA. Giữ `split_lock.json` cùng config khi đưa lên Kaggle.
''')

md('''
## 13. Xuất prepared release cho notebook train

Đây là ranh giới giữa xử lý dữ liệu và huấn luyện. Notebook 02 chỉ đọc thư mục `../Data/COde_processed/code_v1/`
được tạo ở đây; không chạy lại audit, tạo nhãn hoặc chia tập. Release giữ checksum và split fingerprint.
Nếu manifest thay đổi, dùng tên release mới thay vì sửa bản đã dùng để train.

Cell này cần project source có module `clindistill/`. Trên Kaggle gắn project source cùng dataset.
Các review còn mở được giữ nguyên trong release dưới trạng thái `experimental`; xuất file không biến thành dữ liệu đã xác nhận.
''')

code(r'''
PROJECT_ROOT_OVERRIDE = None  # thư mục có clindistill/
RELEASE_NAME = 'code_v1'
if PROJECT_ROOT_OVERRIDE is not None:
    project_root = Path(PROJECT_ROOT_OVERRIDE)
elif globals().get('OFFLINE_ASSETS'):
    project_root = OFFLINE_ASSETS['root'] / 'source'
elif (BASE / 'clindistill').is_dir():
    project_root = BASE
else:
    source_candidates = []
    if Path('/kaggle/input').exists():
        for current, dirs, files in os.walk('/kaggle/input'):
            root = Path(current)
            if 'clindistill' in dirs:
                source_candidates.append(root)
            dirs[:] = [d for d in dirs if d not in {'Images', '.git', '__pycache__', 'outputs', 'runs', '_support', '_archive'}]
            if len(root.relative_to('/kaggle/input').parts) >= 6:
                dirs[:] = []
    if len(source_candidates) != 1:
        raise RuntimeError('Gắn project source vào Kaggle và đặt PROJECT_ROOT_OVERRIDE tới thư mục có clindistill/.')
    project_root = source_candidates[0]
sys.path.insert(0, str(project_root))
from clindistill.artifacts import publish_release
release_root = PROCESSED_ROOT / RELEASE_NAME
release = publish_release(OUT, release_root)
print('Prepared release:', release_root)
print('Release ID:', release['release_id'])
print('Stage:', release['stage'])
print('Tiếp theo: mở COde_02_Train_Baselines.ipynb. Không đưa CSV gốc vào notebook train.')
''')

md('''
## 14. Chuẩn bị clinical embeddings cho Teacher/KD

**Có prepared release rồi: chạy cell thiết lập offline đầu notebook, rồi cell dưới đây**, không cần audit lại 58.000 ảnh.
Cell tự tìm source và prepared release; khi có nhiều bản, đặt các override. Dữ liệu cache được lưu riêng
trong `Data/COde_processed/clinical_biomedbert_v1/` ở local, hoặc `/kaggle/working/data/COde_processed/` ở Kaggle.

Mã hóa từng trường `clinical_screened` bằng BiomedBERT frozen, mean pooling theo attention mask;
max 192 token/trường, ghi số trường bị truncation. Không đọc diagnosis/treatment hoặc nhãn khi mã hóa.
Trường rỗng có vector 0 và mask riêng. Không huấn luyện tokenizer/encoder trên validation/test.
Encoder này là thiết kế triển khai của khóa luận, chưa phải tái tạo nguyên xi CPKD/MMPKD.

Kaggle GPU: mặc định bật bước này. Local CPU: mặc định tắt để không tự tải model lớn; có thể bật thủ công.
Nếu thiếu thư viện: `%pip install "transformers>=4.40,<6" "huggingface-hub>=0.23,<2" safetensors`.
Nếu tắt Internet: gắn folder BiomedBERT có config/tokenizer/weights và đặt `TEXT_ENCODER` tới folder đó.
Không có weights thì báo lỗi, không thay bằng random embeddings.

Mặc định **chỉ train/validation**. Khi protocol đã khóa và cần đánh giá Teacher trên test,
đặt `CACHE_SPLITS=('train', 'validation', 'test')`, dùng đúng encoder/cache đã train.
Student inference không cần shard test hoặc clinical cache.
''')
code(r'''
from pathlib import Path
import os, sys, json
import torch

PROJECT_ROOT_OVERRIDE = None
PREPARED_ROOT_OVERRIDE = None
CLINICAL_OUTPUT_OVERRIDE = None
TEXT_ENCODER = (OFFLINE_ASSETS['text_encoder'] if globals().get('OFFLINE_ASSETS')
                else 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
TEXT_REVISION = 'main'  # cache lưu checksum các assets; dùng commit ID nếu muốn pin ngay lần đầu
LOCAL_FILES_ONLY = bool(globals().get('OFFLINE_ASSETS'))
DEBUG_CLINICAL = os.environ.get('CLINDISTILL_SMOKE_TEST') == '1'
RUN_CLINICAL_ENCODING = torch.cuda.is_available() or DEBUG_CLINICAL
CACHE_SPLITS = ('train', 'validation')

def clinical_find(predicate):
    found = set()
    for base in [Path.cwd(), Path.cwd().parent / 'Data', Path('/kaggle/input')]:
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base):
            here = Path(current)
            if predicate(here, dirs, files):
                found.add(here.resolve())
            dirs[:] = [d for d in dirs if d not in {'Images', '.git', '__pycache__', 'runs', 'outputs', '_support', '_archive'}]
            if len(here.relative_to(base).parts) >= 7:
                dirs[:] = []
    return sorted(found, key=str)

def clinical_resolve(override, found, kind):
    if override is not None:
        return Path(override).resolve()
    if len(found) != 1:
        raise RuntimeError(f'{kind}: {found}. Set the corresponding *_OVERRIDE.')
    return found[0]

if RUN_CLINICAL_ENCODING:
    source_root = clinical_resolve(PROJECT_ROOT_OVERRIDE,
        [OFFLINE_ASSETS['root'] / 'source'] if globals().get('OFFLINE_ASSETS') else
        [Path.cwd()] if (Path.cwd() / 'clindistill').is_dir() else
        clinical_find(lambda p, dirs, files: 'clindistill' in dirs and 'configs' in dirs), 'Project source')
    # Reuse release just published when running the whole notebook.
    prepared_root = clinical_resolve(PREPARED_ROOT_OVERRIDE,
        [release_root] if 'release_root' in globals() else
        clinical_find(lambda p, dirs, files: 'release.json' in files and
            not json.loads((p / 'release.json').read_text(encoding='utf-8')).get('debug_subset')), 'Prepared release')
    sys.path.insert(0, str(source_root))
    from clindistill.clinical import prepare_clinical_cache
    cache_base = (Path('/kaggle/working/data/COde_processed') if Path('/kaggle/working').exists()
                  else source_root.parent / 'Data' / 'COde_processed')
    cache_destination = (Path(CLINICAL_OUTPUT_OVERRIDE) if CLINICAL_OUTPUT_OVERRIDE else
                         cache_base / ('clinical_debug_v1' if DEBUG_CLINICAL else 'clinical_biomedbert_v1'))
    metadata = prepare_clinical_cache(prepared_root, cache_destination, encoder=TEXT_ENCODER,
        revision=TEXT_REVISION, splits=CACHE_SPLITS, max_length=192, batch_size=32,
        local_files_only=LOCAL_FILES_ONLY, debug=DEBUG_CLINICAL)
    print('Clinical cache:', cache_destination)
    print('Encoder:', metadata)
    print('Save/upload this cache folder with the prepared release; then run notebook 02.')
else:
    print('Clinical encoding chưa chạy. Bật RUN_CLINICAL_ENCODING=True khi sẵn sàng hoặc chạy trên GPU Kaggle.')
''')

offline_spec = importlib.util.spec_from_file_location('offline_setup', Path(__file__).resolve().parents[1] / 'kaggle_offline.py')
offline_setup = importlib.util.module_from_spec(offline_spec)
offline_spec.loader.exec_module(offline_setup)
cells.insert(1, nbf.v4.new_markdown_cell('''## Thiết lập offline (chạy trước các cell khác)

Gắn bundle từ notebook 00 trong phiên Kaggle GPU tắt Internet. Cell sau cài wheels nội bộ, kiểm tra môi trường và tìm weights local.
Nếu dùng phiên online chưa có bundle, chủ động đặt OFFLINE_MODE=False. Local mặc định không cài gì.
'''))
cells.insert(2, nbf.v4.new_code_cell(offline_setup.BOOTSTRAP_CELL.strip()))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python', 'version': '3.13.0'}}
nbf.validate(nb)
target = Path(__file__).resolve().parents[2] / 'COde_01_Data_Audit_and_Preparation.ipynb'
if target.exists():
    previous = nbf.read(target, as_version=4)
    previous_cells = {c.source: c for c in previous.cells if c.cell_type == 'code'}
    for cell in nb.cells:
        if cell.cell_type == 'code' and cell.source in previous_cells:
            old = previous_cells[cell.source]
            cell.outputs, cell.execution_count = old.outputs, old.execution_count
            cell.metadata, cell.id = old.metadata, old.id
nbf.write(nb, target)
print(f'Created {target.name}: {len(cells)} cells')

if __name__ == '__main__':
    for i, cell in enumerate(cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'cell_{i}', 'exec')
    print('All code cells compile.')
