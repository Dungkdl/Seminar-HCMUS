"""Refresh entry points and reviewed bootstrap, retaining the data-audit implementation."""
import importlib.util
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('research_bootstrap', ROOT / '_support/research_bank.py')
bank = importlib.util.module_from_spec(spec); spec.loader.exec_module(bank)


def load(prefix):
    path = next(ROOT.glob(prefix + '*.ipynb'))
    return path, json.loads(path.read_text(encoding='utf-8'))


def put(nb, index, source):
    nb['cells'][index]['source'] = textwrap.dedent(source).strip() + '\n'


def save(path, nb):
    for cell in nb['cells']:
        if isinstance(cell['source'], str):
            cell['source'] = cell['source'].splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')


PATHS_CELL = '''
from clindistill.artifacts import read_json
FEATURE_ROOT = research_one(FEATURE_ROOT_OVERRIDE,
    research_find(lambda p,d,f: 'feature_manifest.json' in f and
        not read_json(p / 'feature_manifest.json').get('debug') and
        {'train.npz','validation.npz'} <= set(f)), 'Research feature cache')
OUTPUT_ROOT = Path(OUTPUT_ROOT_OVERRIDE) if OUTPUT_ROOT_OVERRIDE else (
    Path('/kaggle/working') if Path('/kaggle/working').exists() else PROJECT_ROOT / '_support')
CONFIG = read_json(PROJECT_ROOT / 'configs/code' / CONFIG_FILE)
RUN_NAME = RUN_NAME or CONFIG['run_name']
CONFIG['run_name'] = RUN_NAME
STUDY_ROOT = OUTPUT_ROOT / 'runs' / RUN_NAME
print('Features:', FEATURE_ROOT, '\\nStudy:', STUDY_ROOT)
'''


def refresh():
    path, nb = load('COde_00')
    put(nb, 0, '''
        # COde 00 — Research wheel bank v5

        Chạy ONLINE trên Linux/Kaggle có cùng Python/scientific stack với phiên GPU đích.
        Gắn source 0.5.0 đã giải nén. Chưa có prepared release vẫn chạy: RUN_MINI_REHEARSAL=False.
        Bank chứa project wheel, dependency wheelhouse + requirements.lock, source/configs và pretrained weights có checksum.
        Chấp nhận điều khoản DINOv3-B/L, thêm Kaggle Secret HF_TOKEN và cấp quyền cho notebook.
        Hướng dẫn: _support/KAGGLE_RUNBOOK.md. Bank v4 không dùng cho source mới.
    ''')
    source = ''.join(nb['cells'][1]['source']).replace('code_research_bank_v1', 'code_research_bank_v2')
    source = source.replace('RUN_MINI_REHEARSAL = True', 'RUN_MINI_REHEARSAL = False').replace('Source v4', 'Source v5')
    put(nb, 1, source)
    source = ''.join(nb['cells'][3]['source']).replace("sys.path.insert(0, str(BANK_ROOT / 'source'))", "bank.installed_project_matches(info['project_files'])")
    put(nb, 3, source)
    put(nb, 2, '''
        ## Rehearsal tùy chọn sau khi có prepared release

        Lần đầu chỉ tạo bank. Sau notebook01, có thể đặt RUN_MINI_REHEARSAL=True,
        gắn prepared release và ảnh để thử pretrained thật trên mini độc lập theo nhóm bệnh nhân.
        Đây là kiểm tra phần mềm, không phải kết quả nghiên cứu; không mở test.
    ''')
    save(path, nb)

    path, nb = load('COde_01')
    put(nb, 2, bank.BOOTSTRAP_CELL)
    put(nb, 1, '''
        ## Thiết lập wheel bank v5

        Gắn output notebook00, chạy bootstrap trước import. Lần đầu audit raw CSV/Images rồi xuất code_v1.
        Nếu đã có prepared release, chạy bootstrap rồi mục15; không chia lại split.
        Trên Kaggle, code import từ wheel đã cài; source_root chỉ cung cấp configs/docs.
    ''')
    source = ''.join(nb['cells'][45]['source'])
    if "if not globals().get('RESEARCH_ASSETS')" not in source:
        source = source.replace('sys.path.insert(0, str(project_root))', "if not globals().get('RESEARCH_ASSETS'):\n    sys.path.insert(0, str(project_root))")
    put(nb, 45, source)
    source = ''.join(nb['cells'][48]['source']).replace('research_features_v1', 'research_features_v2').replace('bank v4', 'bank v5')
    if 'TEST_PROTOCOL_PATH = ' not in source:
        source = source.replace("PREPARE_SPLITS = ['train', 'validation']", "TEST_PROTOCOL_PATH = None  # test_protocol.json đã khóa ở notebook03\nPREPARE_SPLITS = ['train', 'validation']")
    source = source.replace('manifest = prepare_features(prepared, images, models, destination, cfg, splits=PREPARE_SPLITS)',
        "test_protocol = read_json(TEST_PROTOCOL_PATH) if TEST_PROTOCOL_PATH else None\nmanifest = prepare_features(prepared, images, models, destination, cfg, splits=PREPARE_SPLITS, test_protocol=test_protocol)")
    put(nb, 48, source)
    put(nb, 47, '''
        ## 15. Frozen DINOv3 / BGE-M3 features v2

        Có prepared release thì chạy bootstrap và cell này. Mặc định chỉ train/validation.
        Clinical: English age/gender/chief complaint. Review chưa xác minh vẫn screened_exploratory.
        Source builder mới cần cache mới research_features_v2; giữ nguyên split code_v1.
        Để thêm test: notebook03 khóa protocol trước, gắn test_protocol.json rồi đặt PREPARE_SPLITS=['test'].
        Copy cache train/validation từ input sang working trước khi thêm shard test.
    ''')
    save(path, nb)

    path, nb = load('COde_02')
    put(nb, 0, '''
        # COde 02 — ClinDistill: KD thành phần và đối chứng

        Gắn bank v5 + cache v2, chọn GPU, tắt Internet, Run All.
        Không đọc raw CSV, chia lại split hoặc mở test. DINO/BGE cố định, chỉ học MIL/fusion/head.
        Mặc định 12 hệ thống × 3 seeds + 30 OOF fits = 66 fits.
        CONFIG_FILE='research_controls.json' thêm single/mean MIL và clinical controls: 81 fits.
        Các gate là giả thuyết cần kiểm nghiệm, không phải chứng nhận visual sufficiency.
    ''')
    put(nb, 1, '''
        from pathlib import Path
        import os, sys, json
        RESEARCH_BANK_OVERRIDE = None
        PROJECT_ROOT_OVERRIDE = None
        FEATURE_ROOT_OVERRIDE = None
        OUTPUT_ROOT_OVERRIDE = None
        PREVIOUS_STUDY_OVERRIDE = None
        CONFIG_FILE = 'research.json'
        RUN_NAME = None  # dùng run_name của config; đổi khi thay protocol
        RUN_TRAIN = True
    ''')
    put(nb, 2, '## 1. Cài wheel offline và tìm cache\n\nDùng bank v5. Không trộn bank/cache/checkpoint cũ với source 0.5.0.')
    put(nb, 3, bank.BOOTSTRAP_CELL)
    put(nb, 4, PATHS_CELL)
    put(nb, 7, '''
        ## 3. Huấn luyện

        Visual/MM Teachers → group-OOF → Students. Có label smoothing, correctness-only,
        confidence-only, visual-support-only và reliability KD. OOF không fit nhóm held-out.
        Weight chia mean toàn OOF train; CE vẫn học trên ca có gate=0.
        _oof/seed_*/weight_diagnostics.json báo effective cases, zero fraction theo lớp/nhóm Teacher.
        Clinical shuffle chỉ lấy context từ train của lần fit; validation/test giữ clinical tự nhiên.
    ''')
    source = ''.join(nb['cells'][10]['source']).replace('fig, axes = plt.subplots(3, 3, figsize=(16, 11), squeeze=False)',
        'rows = (len(methods) + 2) // 3\n    fig, axes = plt.subplots(rows, 3, figsize=(16, 3.7 * rows), squeeze=False)')
    put(nb, 10, source)
    save(path, nb)

    path, nb = load('COde_03')
    put(nb, 1, '''
        from pathlib import Path
        import os, sys, json
        RESEARCH_BANK_OVERRIDE = None
        PROJECT_ROOT_OVERRIDE = None
        FEATURE_ROOT_OVERRIDE = None
        OUTPUT_ROOT_OVERRIDE = None
        CONFIG_FILE = 'research.json'
        RUN_NAME = None
        SPLIT = 'validation'
        TEST_REVIEW_RECORD = None
        STUDY_ROOT_OVERRIDE = None
        LOCK_TEST_PROTOCOL = False
    ''')
    put(nb, 2, bank.BOOTSTRAP_CELL)
    put(nb, 3, PATHS_CELL)
    put(nb, 4, '''
        from clindistill.research.reporting import evaluate_study, freeze_test_protocol
        if STUDY_ROOT_OVERRIDE:
            STUDY_ROOT = Path(STUDY_ROOT_OVERRIDE)
        elif not (STUDY_ROOT / 'completed.json').exists():
            STUDY_ROOT = research_one(None, research_find(lambda p,d,f:
                {'study.json','completed.json'} <= set(f) and read_json(p / 'study.json')['config']['run_name'] == RUN_NAME,
                include_runs=True), 'Completed study')
        if LOCK_TEST_PROTOCOL and Path('/kaggle/input') in STUDY_ROOT.resolve().parents:
            import shutil
            writable = OUTPUT_ROOT / 'runs' / RUN_NAME
            if writable.exists():
                raise RuntimeError('Set STUDY_ROOT_OVERRIDE to the existing working study before locking.')
            shutil.copytree(STUDY_ROOT, writable)
            STUDY_ROOT = writable
        if LOCK_TEST_PROTOCOL:
            protocol = freeze_test_protocol(FEATURE_ROOT, STUDY_ROOT, TEST_REVIEW_RECORD)
            print('Locked:', STUDY_ROOT / 'test_protocol.json', protocol['identity'])
        table, summary = evaluate_study(FEATURE_ROOT, STUDY_ROOT, OUTPUT_ROOT / 'reports' / RUN_NAME / SPLIT,
                                       split=SPLIT, test_review_record=TEST_REVIEW_RECORD)
        display(table); display(summary)
    ''')
    put(nb, 5, '''
        Kết quả có per-class precision/recall/F1, calibration, paired contrasts (có MM KD−Visual KD),
        teacher_strata.json cho bốn nhóm Teacher đúng/sai và pending review status.
        Đây là phân tích bất đồng mô hình, không chứng minh ca bệnh cần clinical.
        CI chưa hiệu chỉnh nhiều so sánh và có điều kiện trên mô hình đã fit.

        Lần đầu chỉ SPLIT='validation'. Để khóa test: LOCK_TEST_PROTOCOL=True và TEST_REVIEW_RECORD mô tả thực tế.
        Save Version giữ test_protocol.json. Notebook01 dùng file này để tạo test shard trên bản copy cache.
        Sau đó notebook03 dùng cache đã thêm test, đúng study đã khóa, SPLIT='test', LOCK_TEST_PROTOCOL=False
        và cùng TEST_REVIEW_RECORD. Không tự xóa pending_reviews khi khóa protocol.
    ''')
    save(path, nb)


if __name__ == '__main__':
    refresh()
