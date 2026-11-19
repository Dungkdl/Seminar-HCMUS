# Chạy ClinDistill 0.6.0 trên Kaggle

Bank6/source 0.6.0; giữ prepared `code_v1` và cache `research_features_v3`0.5.1 nếu checksums/policy/models khớp. **Tạo bank và study mới; không patch bank cũ hoặc resume heads/OOF cũ.** DINO/BGE frozen; GPU trích features, sau đó học heads/MIL/fusion.

## Nếu đã có output feature4.225 train/917 validation

Đây là đường chạy ưu tiên cho các output đã tạo ở bản trước:

```text
Source0.6.0 + bank5 cũ → notebook00 → bank v6 mới, cùng pretrained weights/revisions
Cache train/validation cũ + bank v6 → notebook02 Visual → notebook03 Visual
                              → notebook02 Clinical → notebook03 Clinical
                              → notebook02 Transfer → notebook03 Transfer
```

**Bỏ qua notebook01** nếu cache đã lưu có `feature_manifest.json`, `train.npz`, `validation.npz` và preflight khớp. Không cần upload lại ảnh, prepared metadata hoặc trích features lần nữa chỉ vì nâng source. Notebook02 kiểm tra hash models của bank so với cache trước fit; không khớp phải dừng để chọn đúng bank đã tạo cache.

| Phiên | Add Input | Môi trường | Output cần giữ |
|---|---|---|---|
| 00 inspect | Source0.6.0 | GPU đích, `MODE='inspect'` | Copy inventory JSON |
| 00 prepare | Source0.6.0 + output bank cũ | CPU/Internet ON | Bank6 gồm source/wheel/dependencies/weights |
| 02 visual | Bank6 + output features cũ | GPU/Internet OFF | `runs/code_visual_v6`, Student export |
| 03 visual | Bank6 + features + output02 visual | GPU/Internet OFF | Validation report |
| 02/03 clinical | Bank6 + cùng features;03 thêm output02 clinical | GPU/Internet OFF | `runs/code_clinical_v6` và report, không Student export |
| 02/03 transfer | Bank6 + cùng features;03 thêm output02 transfer | GPU/Internet OFF | `runs/code_transfer_v6`, report và Student export |

Các đường dẫn mẫu dưới đây cần thay bằng **Copy Path** của input thực tế. Root bank chứa `research_bank_manifest.json`; root features chứa `feature_manifest.json`; root study chứa `study.json` và sau khi hoàn thành có `completed.json`.

## 1. Đóng gói local

Từ root repo, dùng Python có setuptools/wheel:

```powershell
python -m pip install "setuptools>=68,<82" "wheel>=0.43"
python "Khóa luận/_support/research-notes/package_kaggle.py"
```

Output ở `Khóa luận/_support/dist/`: `clindistill-code-source.zip`, `project_wheel/clindistill_code-0.6.0-py3-none-any.whl`, năm notebooks trong `notebooks/`, `package_manifest.json`. Source ZIP mặc định không có dữ liệu/weights. Third-party binary wheels phải tạo trên Linux cùng Python/base image GPU đích, không dùng Windows wheels.

Nếu cần prepared ZIP/addon:

```powershell
python "Khóa luận/_support/research-notes/package_kaggle.py" --prepared-root "Data/COde_processed/code_v1" --image-root "E:/Hoc Tap/Data/COde-Dataset"
```

Addon chứa source/wheel và patient/clinical metadata + original image hashes, không chứa ảnh/weights. Giữ addon/dataset/output có dữ liệu trong quyền truy cập phù hợp dữ liệu gốc. Không commit ZIP này.

Upload source ZIP đã giải nén thành input; đặt `PROJECT_ROOT_OVERRIDE` nếu có nhiều source. Không gắn source0.5.1 rồi chạy notebook00 mới; schema guard yêu cầu helper bank v6.

## 2. Notebook00 — inspect GPU target, chuẩn bị CPU online

1. Trên GPU đích chạy `MODE='inspect'` và giữ inventory JSON. Không tải model/đọc Secret trong mode này.
2. Nếu GPU chặn Internet: dừng phiên, chọn Accelerator=None (CPU), bật Internet, khởi động lại. Copy inventory vào `TARGET_ENVIRONMENT`.
3. Gắn source 0.6.0, đặt `MODE='prepare'`, `BANK_ROOT=Path('/kaggle/working/clindistill_research_v6')`, `RUN_MINI_REHEARSAL=False`.
4. Chấp nhận điều khoản [DINO-B](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)/[DINO-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), tạo Kaggle Secret `HF_TOKEN` và cấp notebook quyền đọc. Không ghi token vào source.
5. Run All rồi Save Version; phiên sau gắn Notebook Output, không cần download/upload bank lớn lại.

Để dùng lại weights của bank5 đã tạo features, gắn output00 cũ và sửa cell đầu notebook00:

```python
PROJECT_ROOT_OVERRIDE = Path('/kaggle/input/TEN-SOURCE-MOI')  # root có pyproject.toml và clindistill/
MODE = 'prepare'
TARGET_ENVIRONMENT = {...}  # nguyên JSON inventory từ phiên GPU inspect mới
BANK_ROOT = Path('/kaggle/working/clindistill_research_v6')
REUSE_MODELS_FROM = Path('/kaggle/input/TEN-OUTPUT-00-CU/clindistill_research_v3')
RUN_MINI_REHEARSAL = False
```

`clindistill_research_v3` là tên folder lịch sử của bank5; trỏ theo folder thực tế, không đoán từ version. Đặt source override vì output bank cũ cũng chứa source0.5.1. Chức năng reuse chỉ copy pretrained files đã kiểm tra toàn bộ hashes và giữ nguyên pinned revisions; không mang source/wheels cũ sang bank mới. Không cần tải lại các weights đã xác minh; dependencies/source/wheel được dựng mới. Nếu chưa có bank cũ, giữ `REUSE_MODELS_FROM=None` để tải theo quy trình thông thường.

CPU/GPU phải cùng Python/platform/base scientific packages, **bao gồm fsspec**. Chỉ cho Torch/vision cùng release khác `+cpu`→cùng `+cuNNN`; GPU activation vẫn kiểm tra đúng CUDA build. Không cài wheelhouse baseline vào kernel ClinDistill và không thay Torch/CUDA để ép bank chạy.

Bank gồm:

```text
clindistill_research_v6/
  research_bank_manifest.json     # schema6, source/wheel/model checksums
  base-constraints.txt            # target scientific versions incl. fsspec
  requirements.lock
  dependency_report.json         # CPU preparation metadata, not GPU results
  wheels/                        # project0.6.0 + addons, no Torch/CUDA wheels
  models/                        # DINO-B/L, BGE-M3, ResNet/Swin
  source/                        # code, configs, helpers, notebooks, docs
```

Transformers4.57.1/hub0.36.2 pinned, setuptools<82 bảo vệ requirement Torch2.11. Không tự nâng fsspec. Sau cài đặt, conflict của transitive closure thực dùng gây lỗi trước khi đánh dấu bank ready. Conflict image ngoài closure báo riêng; `RESEARCH_ASSETS['dependency_report']` cho biết trạng thái sau GPU activation. Không gọi toàn bộ image sạch chỉ vì closure của ClinDistill sạch.

CPU model loading không xác nhận GPU execution. Fresh bank v6/activation/training thực vẫn cần chạy trên tài khoản Kaggle của người thực hiện; software tests không thay thế bước này.

## 3. Notebook01 — prepared release và features

Lần đầu gắn bank00 + CODe gốc có `complete_dataset.csv`, `Images/Photographs`, `Images/Radiographs`; Run All audit/export. Nếu có prepared release, dùng **Features Only** để không audit/chia lại split. Gắn bank00 + prepared metadata + ảnh, GPU ON/Internet OFF.

Feature preparation dùng `configs/code/features.json` cho tất cả profiles. Không đổi cohort, max8 policy hoặc preprocessing để xử lý thiếu ảnh. Cache cũ0.5.1 dùng được vì builder không đổi; cache cũ trước0.5.1 cần tạo lại. Không bỏ verification/checksum.

Output:

```text
/kaggle/working/data/COde_processed/code_v1/
/kaggle/working/data/COde_processed/research_features_v3/
  feature_manifest.json           # schema4; tên thư mục v3 không phải schema
  train.npz
  validation.npz
  clinical_audit_summary.json
```

Save Version giữ release/cache/audit; clinical contexts, feature vectors và IDs là dữ liệu riêng tư. Review nhãn/clinical/near-duplicate còn mở; xem [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md).

### Tái dùng ảnh đã làm phẳng

Nếu ảnh đang ở `data/images` của assets cũ, giữ nguyên ảnh và baseline manifests. Gắn addon/prepared metadata với `image_index.json`; cls JSON của baseline không thay patient/clinical metadata ClinDistill.

```python
IMAGE_ROOT_OVERRIDE = Path('/kaggle/input/TEN-DATASET/code_full_offline_assets/data')
IMAGE_BATCH_SIZE = 'auto'
IMAGE_WORKERS = 'auto'
CLINICAL_BATCH_SIZE = 'auto'
```

Adapter nhận original tree, basename hoặc `sha1(canonical_path)[:10] + '_' + basename`; bắt buộc SHA256 khớp ảnh gốc, phân biệt photo/X-ray cùng basename. Nó tạo canonical filesystem aliases dưới ancestor chung Input/Working, không đổi bytes/installed wheel/manifest. Không ghi đè Images directory có sẵn của người dùng.

`image_layout_report.json` trong working báo coverage/missing_required. Thiếu/sai bytes phải dừng trước encoders; gắn thêm ảnh gốc/supplement đúng release, không lọc bỏ ca để chạy tiếp. Supplement local đã tạo trước có1.417 photos/22,5MB theo max8 policy; đây là reference comparison, không chứng nhận coverage của mọi dataset upload. Test labels không được adapter mở.

## 4. Notebook02 — chạy từng giai đoạn

Gắn bank00 + cache01, GPU ON/Internet OFF, restart kernel/Run All:

```python
CONFIG_FILE = 'research_visual.json'
RUN_NAME = None  # code_visual_v6, mặc định12 fits
```

Nếu auto-detect có nhiều input, đặt trong cell đầu02:

```python
RESEARCH_BANK_OVERRIDE = Path('/kaggle/input/TEN-OUTPUT-00-MOI/clindistill_research_v6')
FEATURE_ROOT_OVERRIDE = Path('/kaggle/input/TEN-OUTPUT-FEATURES/data/COde_processed/research_features_v3')
PREVIOUS_STUDY_OVERRIDE = None  # chỉ đặt khi resume đúng profile
RUN_TRAIN = True
```

Notebook03 dùng cùng bank/features, cùng `CONFIG_FILE` và:

```python
STUDY_ROOT_OVERRIDE = Path('/kaggle/input/TEN-OUTPUT-02/runs/code_visual_v6')
SPLIT = 'validation'
LOCK_TEST_PROTOCOL = False
TEST_REVIEW_RECORD = None
```

Lặp lại02/03 với `research_clinical.json`/`code_clinical_v6`, rồi `research_transfer.json`/`code_transfer_v6`. Chuyển profile không phải resume stage trước. Giữ cùng feature cache/split để so sánh; clinical có/không có lợi đều cần báo trung thực.

Đọc validation bằng notebook03. Tiếp theo `research_clinical.json` 18 fits và `research_transfer.json` 45 fits nếu phù hợp. Core gộp `research.json` 63 fits; extended `research_controls.json` 87 fits. Mỗi profile có run_name riêng; giữ cùng config ở03. Không đổi config giữa chừng/resume study cũ.

Visual stage không tạo OOF; clinical stage không xuất Student. KD/weighted CE tạo group-OOF, cosine horizon giữ cùng full Teacher. Đừng diễn giải số fits heads là số lần fine-tune toàn DINO.

Output `/kaggle/working/runs/<run_name>/` chứa study/protocol/checkpoints/results/readiness. Student stages thêm `/kaggle/working/student_export/<run_name>/`. Save Version giữ các thư mục này.

Trong output, kiểm tra có `completed.json`, `validation_summary.csv`, `cohort_summary.json`, `research_readiness.json`; transfer có thêm `_oof/seed_*/targets.json` và `weight_diagnostics.json`. Reports03 có `per_seed.csv`, `mean_std.csv`, `paired_comparisons.json`, `teacher_strata.json`, `status.json`. Visual stage có thể không có Teacher-disagreement strata vì không fit MM Teacher.

Files trong `/kaggle/working` cần được lưu thành Notebook Version có output rồi gắn bằng Add Input ở phiên sau; xem [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks). Sau Save Version hãy xác nhận những file nêu trên xuất hiện trong Output trước khi dừng phiên. Không mặc định các file chỉ tồn tại ở phiên tương tác đã được lưu.

### Resume

Gắn output02 trước đó; đặt `PREVIOUS_STUDY_OVERRIDE` tới `.../runs/<run_name>` của input. Notebook copy sang working rồi resume epoch-boundary. Giữ đúng source/config/cache. Checkpoints đã complete được kiểm tra/reuse; không trộn stage hoặc study khác. Giới hạn session không phải lý do bỏ provenance.

## 5. Notebook03 — validation trước, test sau protocol lock

Gắn bank00 + cache01 + study02. Đặt cùng `CONFIG_FILE`, `SPLIT='validation'`, `LOCK_TEST_PROTOCOL=False`. Báo metric/CI ở validation là exploratory vì checkpoint được chọn trên nó.

Sau khi chốt phương pháp bằng train/validation:

1. Notebook03 đặt `LOCK_TEST_PROTOCOL=True`, `TEST_REVIEW_RECORD` mô tả protocol/giới hạn thực, `SPLIT='validation'`. Attached study chỉ đọc được copy vào working trước lock. Save Version giữ `test_protocol.json` và study.
2. Notebook01 gắn protocol vừa lưu. Copy nguyên cache train/validation từ input sang working, đặt `RESEARCH_FEATURE_OUTPUT_OVERRIDE` tới bản copy, `TEST_PROTOCOL_PATH` tới protocol và `PREPARE_SPLITS=['test']`. Giữ `features.json`/prepared release/model policy. Save Version cache có test.
3. Notebook03 gắn đúng study đã khóa và cache có test; đặt `SPLIT='test'`, `LOCK_TEST_PROTOCOL=False`, cùng record. Không thay config/checkpoints/source. Không lựa chọn hyperparameter/method/seed sau khi xem test.

Protocol lock không xóa pending reviews hoặc chứng nhận clinical timing. `confirmatory` chỉ được phép khi reviewed release và evidence thực đạt điều kiện. Exploratory test phải ghi giới hạn tương ứng; không biến nó thành kết quả xác nhận.

## 6. Hiệu năng và việc cần xác nhận

Auto profile khi còn≥70GiB: DINO-B/ResNet256, DINO-L/Swin128, text128, tối đa8 workers; encoder OOM chia microbatch giữ thứ tự. Head batch32/accumulation1 FP32; evaluation batch256. Đây là starting profile, chưa phải benchmark tối ưu.

Đọc `feature_runtime.json`, images/s và peak VRAM; đặt `IMAGE_WORKERS=0` nếu lỗi multiprocessing. Train heads nhỏ không cần chiếm100% GPU. Không đổi LR/optimizer batch/image policy chỉ để tăng VRAM. Cần đo pretrained full training, resume/export và latency/VRAM cấp lượt khám; không suy ra hiệu năng từ synthetic tests.
