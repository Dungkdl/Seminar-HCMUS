# Chạy ClinDistill trên Kaggle bằng wheel offline (0.5.0)

Quy trình: **đóng gói source trên máy → 00 tải wheels/weights có Internet → 01 xử lý dữ liệu và tạo features → 02 train → 03 đánh giá**.

Code trong phiên Kaggle chạy từ project wheel đã cài, có kiểm tra khớp source. Thư mục source trong bank cung cấp notebook/config/tài liệu. DINO/BGE được cố định; GPU trích features, sau đó chỉ huấn luyện MIL/fusion/head. Đây chưa phải full encoder fine-tuning.

## 1. Tạo gói upload trên Windows

Từ thư mục repo, dùng Python đã có setuptools và wheel. Không cần tải torch/weights để đóng gói:

```powershell
py -3.14 -m pip install "setuptools>=68" "wheel>=0.43"
py -3.14 "Khóa luận/_support/research-notes/package_kaggle.py"
```

Các file được tạo tại `Khóa luận/_support/dist/`:

- `clindistill-code-source.zip`: source, configs, 4 notebook và project wheel mới.
- `project_wheel/clindistill_code-0.5.0-py3-none-any.whl`: wheel Python thuần, dùng được trên Linux.
- `package_manifest.json`: SHA256 của các artifact.

Chỉ project wheel là portable. Wheels của dependencies như tokenizers/safetensors phải chuẩn bị trên Linux cùng Python với Kaggle đích; không lấy Windows wheels đưa sang Linux.

Upload source ZIP thành dataset source. Trong Kaggle phải thấy `pyproject.toml`, `clindistill/`, `configs/` và `_support/` sau giải nén. Nếu dịch vụ giữ ZIP nguyên dạng, giải nén vào working trước:

```python
from pathlib import Path
import zipfile
archive = Path('/kaggle/input/TEN-DATASET/clindistill-code-source.zip')
destination = Path('/kaggle/working/clindistill-source')
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        target = (destination / member.filename).resolve()
        if destination.resolve() not in target.parents or (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Unsafe ZIP member')
    z.extractall(destination)
```

Đặt `PROJECT_ROOT_OVERRIDE` của notebook00 tới thư mục đã giải nén nếu tự tìm thấy nhiều source.

## 2. Notebook00 — chuẩn bị bank khi có Internet

1. Import `COde_00_Kaggle_Offline_Setup.ipynb` từ source mới vào Kaggle.
2. Add Input: source dataset. Lần đầu chưa cần prepared release hoặc raw data.
3. Chấp nhận điều khoản ở Hugging Face cho [DINOv3-B](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) và [DINOv3-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
4. Tạo Kaggle Secret `HF_TOKEN`, cấp quyền cho notebook. Không ghi token vào source.
5. Bật Internet. Giữ `MODE='prepare'`, `RUN_MINI_REHEARSAL=False`, `BANK_ROOT='/kaggle/working/code_research_bank_v2'`. Run All.
6. Save Version để giữ output. Phiên sau gắn **Notebook Output**, không cần tải bank về máy rồi upload lại.

Bank chứa `research_bank_manifest.json` schema5, `requirements.lock`, `base-constraints.txt`, `wheels/`, `models/`, `source/`. Model revisions và mọi artifact được ghi checksum. Không có fallback sang random weights khi thiếu quyền model.

Nếu GPU đích chặn Internet, dùng một phiên Linux có Internet để chuẩn bị. Trước đó có thể mở notebook00 trong image GPU đích với `MODE='inspect'` để đối chiếu Python/base packages. Chuẩn bị và chạy phải cùng scientific stack; bootstrap dừng nếu khác, không tự thay torch/torchvision/CUDA.

Dung lượng bank gồm DINO-L/B, BGE-M3 và hai CNN weights. Kiểm tra quota output/disk trong phiên của bạn; chưa có đo thời gian/VRAM Kaggle thực cho bản này.

## 3. Notebook01 — raw data → prepared release → frozen features

Gắn **bank00 + CODe gốc đã giải nén**. Thư mục dữ liệu phải chứa `complete_dataset.csv` và `Images/Photographs`, `Images/Radiographs`.

Trên GPU offline, chạy setup trước mọi import. Nếu tự tìm thấy nhiều dữ liệu, sửa cell cấu hình audit:

```python
DATA_ROOT_OVERRIDE = Path('/kaggle/input/TEN-DATASET/COde-Dataset')
```

Lần đầu Run All: kiểm tra ảnh/nhãn → nhóm bệnh nhân và ảnh trùng → khóa split → xuất release → trích features. Quy trình này đọc raw data, không thay đổi dữ liệu gốc.

Đầu ra mặc định:

- `/kaggle/working/_support/audit/code_thesis_v1/`: báo cáo, review candidates và figures.
- `/kaggle/working/data/COde_processed/code_v1/`: prepared release, manifests, split lock, review status.
- `/kaggle/working/data/COde_processed/research_features_v2/`: feature manifest, train.npz và validation.npz.

Nếu đã có `code_v1`, gắn prepared release và ảnh; chỉ chạy setup đầu notebook rồi mục15. Đặt `PREPARED_ROOT_OVERRIDE` và `IMAGE_ROOT_OVERRIDE` nếu cần. Không chạy lại chia tập để tìm split thuận lợi.

Cache v1 không dùng với builder mới. Tạo cache v2 trên cùng split. Save Version giữ cả release, cache và audit. Feature vectors/IDs/clinical contexts vẫn là dữ liệu riêng tư.

**Các review cần người thực hiện:** near-duplicate, mapping nhãn, tính đầy đủ annotation và thời điểm clinical. Không bật cờ reviewed chỉ để bỏ cảnh báo. Không có healthy class; single-target là một target được ghi nhận, không khẳng định chỉ mắc một bệnh.

## 4. Notebook02 — train từ cache, không cần gắn raw images

Gắn **bank00 + output cache01**, chọn GPU, tắt Internet, restart kernel rồi Run All từ đầu.

Cell đầu:

```python
CONFIG_FILE = 'research.json'
RUN_NAME = None  # tự dùng code_research_v2 từ config
```

Profile mặc định có 12 hệ thống ×3 seeds và 30 Teacher OOF fits = **66 fits**. Bao gồm Student no-KD, label smoothing, Visual/MM KD, confidence/correctness/visual-support/reliability KD, hai Teachers và ResNet/Swin controls.

Để thêm kiểm chứng multi-image/clinical, đổi **trước khi chạy**:

```python
CONFIG_FILE = 'research_controls.json'
RUN_NAME = None  # code_research_controls_v2, thư mục study riêng
```

Profile này có **81 fits**, thêm Student một ảnh, mean pooling, Teacher demographics-only, Teacher clinical bị hoán vị trong TRAIN, Teacher cùng kiến trúc multimodal nhưng clinical bị mask. Nó dùng cùng cache/split; validation/test không bị hoán vị. Không gộp study khác config vào cùng run_name.

Đầu ra: `/kaggle/working/runs/<run_name>/` và `/kaggle/working/student_export/<run_name>/`. Save Version sau train. Không diễn giải số fit heads thành số lần fine-tune toàn bộ DINO.

### Tiếp tục khi Kaggle hết phiên

Gắn output lần trước, giữ nguyên bank/cache/config. Đặt `PREVIOUS_STUDY_OVERRIDE` tới folder có study.json; notebook copy sang working và resume cuối epoch. Nếu working đã có study, để override=None để resume tại chỗ. Source/config/cache thay đổi phải dùng run_name mới; không hứa resume giữa minibatch.

## 5. Notebook03 — đánh giá validation và phân tích lỗi

Gắn **bank00 + cache01 + output train02**. Chọn đúng CONFIG_FILE/run_name như 02. Giữ `SPLIT='validation'`, `LOCK_TEST_PROTOCOL=False`, Run All.

Đọc các file trong `/kaggle/working/reports/<run_name>/validation/`:

- `mean_std.csv`, `per_seed.csv`: metric theo seed và trung bình/độ lệch chuẩn.
- `paired_comparisons.json`: có MM Teacher−Visual Teacher và MM KD−Visual KD.
- `teacher_strata.json`: bốn nhóm cả hai đúng/chỉ MM đúng/chỉ Visual đúng/cả hai sai, kết quả từng phương pháp/từng lớp.
- `status.json`: review còn mở và giới hạn nghiên cứu.
- Mỗi model/seed có predictions, per-class precision/recall/F1, calibration, confusion matrix và risk–coverage.

Trong output train02: `_oof/seed_*/weight_diagnostics.json` và `oof_diagnostics.csv` cho biết trọng số, tỷ lệ gate=0 và effective cases theo lớp/nhóm Teacher. Dùng để kiểm tra gate có giảm hướng dẫn ở lớp hiếm hoặc nhóm chỉ MM đúng không. Đây là bất đồng giữa mô hình, không phải ground truth visual sufficiency.

## 6. Chỉ mở test sau khi khóa protocol

1. Hoàn thành các thí nghiệm đã đăng ký và lựa chọn bằng train/validation.
2. Trong 03 vẫn để SPLIT='validation', đặt `LOCK_TEST_PROTOCOL=True`, viết `TEST_REVIEW_RECORD` mô tả protocol đã cố định và review còn mở. Run All. Study từ input sẽ được copy sang working trước khi tạo test_protocol.json. Save Version.
3. Mở 01, gắn bank + raw images + prepared release + cache train/validation + output 03. Copy cache từ input sang thư mục working; **không sửa cache gốc**. Chạy bootstrap và mục15 với destination là bản copy, PREPARE_SPLITS=['test'], TEST_PROTOCOL_PATH trỏ tới test_protocol.json.
4. Builder kiểm tra source/feature policy khớp protocol trước khi tạo shard test. Train/validation shards giữ checksum và identity cũ. Save Version.
5. Mở 03 với cache có đủ ba shards và study đã khóa: SPLIT='test', LOCK_TEST_PROTOCOL=False, cùng TEST_REVIEW_RECORD. Đánh giá một lần theo protocol đã khóa; không điều chỉnh phương pháp theo điểm test.

Khóa lưu hash checkpoints/config/source. Thay các mục này sau khi khóa sẽ bị từ chối. Record là xác nhận của người thực hiện, không tự chứng nhận annotation/clinical hay xóa pending reviews.

## 7. Kiểm tra trước khi chạy dài

- Bank v5/project0.5.0, cache v2, đúng config/run_name.
- Bootstrap in đường dẫn project đã cài trong site-packages; không import từ source trong bank.
- Dùng weights thật có quyền truy cập; debug/tiny-model tests không thay cho pretrained experiments.
- Chạy mini rehearsal nếu cần sau khi có prepared release. Không báo metric mini như kết quả khóa luận.
- Giữ output phiên trước, raw predictions và checksums. Code/tests chạy local không chứng minh Kaggle đã train full hoặc phương pháp tốt hơn.

Nguồn kỹ thuật: [pip wheel](https://pip.pypa.io/en/stable/cli/pip_wheel/), [pip download](https://pip.pypa.io/en/stable/cli/pip_download/), [Transformers DINOv3](https://huggingface.co/docs/transformers/v4.57.1/model_doc/dinov3), [BGE-M3](https://huggingface.co/BAAI/bge-m3).
