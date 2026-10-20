"""Generate the one-time online preparation notebook. Regular workflow remains 01 -> 02 -> 03."""
from pathlib import Path
import textwrap
import nbformat as nbf

root = Path(__file__).resolve().parents[2]
def md(s):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())

cells = [md('''
# COde 00 — Kiểm tra môi trường và chuẩn bị Kaggle OFFLINE

Thực hiện hướng dẫn của thầy: **thử COde nhỏ → chuẩn bị code/wheels/trọng số → gắn dữ liệu → bật GPU offline**.
Notebook này dùng một lần để chuẩn bị, không thay thế 01 xử lý dữ liệu / 02 train / 03 đánh giá.

**Phiên GPU đích:** có thể chạy `MODE='inspect'` dù Internet tắt để lấy Python/package versions và lỗi import.
**Phiên chuẩn bị:** mở Kaggle notebook có Internet (CPU được), cùng Python/scientific stack với phiên GPU;
gắn source ZIP và `code_mini_offline_check.zip`, chọn `MODE='prepare'`.
Ưu tiên Kaggle; Colab chỉ dùng nếu môi trường khớp. Không lấy wheels Windows cho Linux.

Ba loại khác nhau: wheel project chứa code; wheels thư viện chứa dependencies; file weights chứa ResNet/BiomedBERT.
Không có weights thì cài đủ wheels vẫn chưa chạy được pretrained model.
'''), code(r'''
from pathlib import Path
import os, sys, json, importlib.util
PROJECT_ROOT_OVERRIDE = None
MINI_ROOT_OVERRIDE = None  # folder chứa mini_dataset.json, prepared/, Images/
MODE = 'inspect'  # 'inspect' hoặc 'prepare'
BUNDLE_NAME = 'code_offline_v1'

def find_folders(predicate):
    found = set()
    for base in [Path.cwd(), Path.cwd().parent / 'Data', Path('/kaggle/input')]:
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base):
            here = Path(current)
            if predicate(here, files):
                found.add(here.resolve())
            dirs[:] = [d for d in dirs if d not in {'Images', '.cache', '__pycache__', '_support', 'runs', 'wheels', 'models'}]
            if len(here.relative_to(base).parts) >= 7:
                dirs[:] = []
    return sorted(found, key=str)

def choose(override, candidates, name):
    if override is not None:
        return Path(override).resolve()
    if len(candidates) != 1:
        raise RuntimeError(f'{name}: {candidates}; set its *_OVERRIDE.')
    return candidates[0]

PROJECT_ROOT = choose(PROJECT_ROOT_OVERRIDE,
    [Path.cwd()] if (Path.cwd() / '_support/kaggle_offline.py').exists() else
    find_folders(lambda p, f: (p / '_support/kaggle_offline.py').exists()), 'Project source')
spec = importlib.util.spec_from_file_location('kaggle_offline', PROJECT_ROOT / '_support/kaggle_offline.py')
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else PROJECT_ROOT / '_support'
PREP_ROOT = WORK / 'offline_preparation'
PREP_ROOT.mkdir(parents=True, exist_ok=True)
assert MODE in {'inspect', 'prepare'}
report = offline.probe(PREP_ROOT / 'environment_report.json')
print(json.dumps(report, ensure_ascii=False, indent=2))
'''), md('''
## 1. Đọc kết quả kiểm tra

`environment_report.json` ghi Python, OS/architecture, phiên bản thư viện, import scientific,
torchvision/NMS và Transformers/BERT. `ok=false` có thể là thiếu lib **hoặc** xung đột phiên bản; xem `details`.
Chỉ số `cuda=False` là bình thường khi chuẩn bị trên CPU. Ở phiên train RTX GPU phải nhận CUDA.

Đối chiếu report của phiên GPU và phiên online trước khi chuẩn bị. Runtime tự chặn khác Python/OS/architecture/base libraries.
Giữ torch/torchvision/scientific stack có sẵn. Bộ thêm pin Transformers 4.57.1, download cả dependencies và wheels binary.
Nếu import clinical còn lỗi sau cài, notebook dừng; giữ log để sửa đúng thư viện gây lỗi, không tự thay CUDA stack.
'''), code(r'''
BUNDLE = PREP_ROOT / BUNDLE_NAME
if MODE == 'prepare':
    if report['system'] != 'Linux':
        raise RuntimeError('Chạy prepare trên Kaggle ONLINE Linux. Windows chỉ inspect/build wheel project, không tạo wheels Linux mặc định.')
    if not report['imports']['scientific']['ok'] or not report['imports']['torchvision']['ok']:
        raise RuntimeError('Base scientific/torchvision bị lỗi. Dùng môi trường Kaggle hoạt động trước khi chuẩn bị extras.')
    MINI_ROOT = choose(MINI_ROOT_OVERRIDE,
        find_folders(lambda p, files: 'mini_dataset.json' in files and (p / 'prepared/release.json').exists()),
        'COde mini dataset')
    if (BUNDLE / 'offline_manifest.json').exists():
        offline.verify_bundle(BUNDLE)
        print('Dùng lại bundle đã chuẩn bị:', BUNDLE)
    else:
        offline.prepare_bundle(PROJECT_ROOT, BUNDLE)
    print('Assets đã chuẩn bị. Tiếp theo chạy smoke offline.')
else:
    print('Inspect only: không cài package hoặc tải model. Chuyển MODE=prepare trong phiên Kaggle có Internet.')
'''), md('''
## 2. Chạy thử mô hình thật với kết nối mạng bị chặn

Cell dùng **ResNet-50 pretrained và BiomedBERT thật từ disk**, mã hóa clinical và chạy baseline + full/selective KD.
Ngân sách smoke: 12 ca train + 12 validation, ảnh 64px, backbone frozen, 1 epoch; **không dùng để báo cáo hiệu quả nghiên cứu**.
Sockets bị chặn trong đoạn mã hóa/model/training. Các lệnh pip dùng `--no-index --find-links`.

Lần này mới xác nhận bộ assets đầy đủ trong môi trường online-preparation. Sau khi gắn sang GPU, chạy lại bootstrap/import
và smoke ở đó nếu muốn kiểm tra CUDA thực. CPU smoke không chứng minh driver GPU đích hoạt động.

Không mở test, không dùng hồ sơ/nhãn ngoài bộ mini. Bộ mini chứa dữ liệu thật: giữ quyền truy cập như COde gốc.
'''), code(r'''
if MODE == 'prepare':
    smoke_report = offline.smoke_bundle(BUNDLE, MINI_ROOT, PREP_ROOT / (BUNDLE_NAME + '_smoke'))
    print(json.dumps(smoke_report, indent=2))
'''), md('''
## 3. Gói mang sang phiên GPU

Chỉ zip sau smoke thành công. ZIP chứa `source/`, `wheels/`, `models/`, checksums, base constraints và biên bản.
Không có COde đầy đủ trong ZIP này: gắn dataset COde riêng, prepared release riêng.
Giữ cả folder output/Save Version; ở phiên GPU Add Input output chuẩn bị là đủ, không bắt buộc download rồi upload lại.

Khi attach ZIP chưa được giải nén, giải nén vào `/kaggle/working` và đặt OFFLINE_BUNDLE_OVERRIDE.
Mở notebook 01/02/03 từ source mới: cell đầu tự cài wheel offline, dùng weights nội bộ, không download.
Nếu đã import torch/transformers trước khi đổi package, restart kernel rồi chạy từ đầu.
'''), code(r'''
if MODE == 'prepare':
    info = offline.verify_bundle(BUNDLE)
    if info['status'] != 'offline_smoke_passed':
        raise RuntimeError('Chưa có smoke offline thành công.')
    archive = offline.zip_directory(BUNDLE, PREP_ROOT / (BUNDLE_NAME + '.zip'))
    print('UPLOAD/ATTACH:', archive)
    print('Bundle folder:', BUNDLE)
    print('Tiếp theo: gắn COde gốc + prepared; notebook 01 tạo clinical cache đầy đủ; notebook 02 train; notebook 03 evaluate.')
''')]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}})
nbf.validate(nb)
for i, cell in enumerate(cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'offline:{i}', 'exec')
nbf.write(nb, root / 'COde_00_Kaggle_Offline_Setup.ipynb')
print('COde_00_Kaggle_Offline_Setup.ipynb', len(cells), 'cells')
