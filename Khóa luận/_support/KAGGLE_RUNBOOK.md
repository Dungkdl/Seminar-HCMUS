# Chạy ClinDistill trên Kaggle bằng wheel offline (0.5.1)

## Cấu trúc assets Qwen đang có và GPU Pro 6000

Với bộ hiện tại (`data/images`, `cls_train.json`, `cls_test.csv`, `packaging_manifest`), giữ nguyên các file đó. Hai file cls là input của baseline Qwen, không chứa đầy đủ patient/clinical metadata để thay thế prepared release ClinDistill. Gói bổ sung mới dùng metadata từ CODe gốc đã audit, không chia theo split của Qwen.

```text
code_full_offline_assets/
├── Qwen2.5-VL-3B-Instruct/          # baseline gốc
├── official_code/                 # baseline gốc
├── wheelhouse/                    # dependencies baseline gốc
├── manifest_sha256.json           # manifest baseline gốc
├── requirements-runtime...        # baseline gốc
├── data/
│   ├── images/                    # ảnh đang có
│   ├── cls_train.json             # baseline gốc
│   ├── cls_test.csv               # baseline gốc
│   ├── packaging_manifest...
│   └── COde_processed/code_v1/     # từ clindistill-assets-addon.zip
│       ├── release.json
│       ├── image_index.json       # SHA256 ảnh gốc, kiểm tra ánh xạ ảnh bị làm phẳng
│       ├── student_single_*.jsonl
│       └── teacher_context_*.jsonl
├── clindistill_source/             # từ clindistill-assets-addon.zip
│   ├── pyproject.toml
│   ├── clindistill/
│   ├── configs/
│   ├── _support/
│   └── COde_0*.ipynb
└── clindistill_research_v3/         # output notebook00, tạo trên Linux đích
    ├── research_bank_manifest.json
    ├── base-constraints.txt
    ├── requirements.lock
    ├── wheels/                     # wheels ClinDistill, tách wheelhouse Qwen
    ├── models/                     # DINO-B/L, BGE-M3, ResNet/Swin
    └── source/
```

**Không bắt buộc gom và upload lại ảnh.** Gắn đồng thời dataset assets cũ, dataset addon mới và Notebook Output của bank00; notebook tìm trong toàn bộ `/kaggle/input`. Nếu gom vào một dataset, giải nén addon vào root assets; giữ manifest Qwen nguyên trạng và dùng manifest ClinDistill riêng. Không để nhiều bản prepared release/bank cùng lúc nếu chưa đặt override.

Trên Windows, từ root repo, build addon gồm source/wheel + prepared metadata + chỉ mục hash ảnh:

```powershell
py -3.14 "Khóa luận/_support/research-notes/package_kaggle.py" --prepared-root "Data/COde_processed/code_v1" --image-root "E:/Hoc Tap/Data/COde-Dataset"
```

`Khóa luận/_support/dist/clindistill-assets-addon.zip` là file cần upload thêm. Nó không chứa ảnh hay weights; có patient/clinical metadata, giữ dataset private. Import các notebook mới từ `_support/dist/notebooks/`, không dùng bản 0.5.0 cũ.

### Chạy theo đúng thứ tự

1. **00:** gắn addon/source đã giải nén, **Accelerator=None (CPU), Internet ON**, `HF_TOKEN` có quyền DINO, `RUN_MINI_REHEARSAL=False`; `BANK_ROOT=Path('/kaggle/working/clindistill_research_v3')`. So inventory với phiên GPU trước khi tải; Save & Run All, giữ output. Không cài wheelhouse Qwen vào kernel này.
2. **01 Features Only:** import `COde_01_Features_Only.ipynb`, gắn **assets cũ + addon + output00**, chọn GPU Pro 6000, Internet OFF. Run All chỉ bootstrap và trích features; không cần `complete_dataset.csv`, không chạy lại audit/split. Mặc định chỉ train/validation.
3. **02:** gắn **output00 + output01**, GPU ON, Internet OFF. Giữ `CONFIG_FILE='research.json'`, `RUN_NAME=None`. Study mới là `runs/code_research_v3`, 66 fits. Profile controls dùng `research_controls.json`, 81 fits; giữ đúng config ở notebook03.
4. **03:** gắn **output00 + output01 + output02**, `SPLIT='validation'`, `LOCK_TEST_PROTOCOL=False`; Run All. Chưa mở test khi chưa khóa protocol.

Nếu auto-detect ảnh không đúng, trong cell features đặt root chứa `images`:

```python
IMAGE_ROOT_OVERRIDE = Path('/kaggle/input/TEN-DATASET/code_full_offline_assets/data')
# Không trỏ tới cls_train.json. Nếu trỏ trực tiếp thư mục images cũng được.
IMAGE_BATCH_SIZE = 'auto'
IMAGE_WORKERS = 'auto'
CLINICAL_BATCH_SIZE = 'auto'
```

Notebook giữ nguyên canonical image paths của CODe, tìm ảnh theo basename trong bundle và so SHA256 với ảnh gốc trước khi chạy encoder. Thiếu ảnh, trùng basename hoặc ảnh đã bị đổi/resize sẽ dừng. Không bỏ ca thiếu ảnh rồi train trên cohort khác; khi đó gắn thêm CODe-Dataset gốc và đặt IMAGE_ROOT_OVERRIDE tới root có `Images/Photographs`.

### Bundle có tên `images/<10 ký tự hash>_<tên ảnh gốc>`

Notebook01 mới nhận dạng quy tắc `sha1(canonical_path)[:10] + '_' + basename`. Ví dụ photo `Images/Photographs/0127-001-01.jpg` thành `b42b6cfb7a_0127-001-01.jpg`, còn radiograph cùng basename thành `f3f5a4e7c9_0127-001-01.jpg`. Mỗi file vẫn phải có SHA256 trùng index gốc; không chỉ bỏ prefix rồi lấy file đầu tiên.

Adapter tạo filesystem aliases canonical tại ancestor chung của Input/Working (trên Kaggle là `/kaggle/Images/...`), không đổi bytes hoặc chỉnh installed wheel/bank manifest. Chỉ tạo trong thư mục có marker của adapter; không ghi đè thư mục Images có sẵn của người dùng. `image_layout_report.json` nằm trong `/kaggle/working/data/COde_processed/image_layout/`, ghi ánh xạ, hash identity và coverage. Input và workspace phải có ancestor chung dưới filesystem root. Các alias được tạo lại ở mỗi session.

JSON cls của paper không tham chiếu toàn bộ ảnh của cohort ClinDistill. Theo union `train-cls.json` + `test_cls.json`, thiếu 1.020 ảnh train, 211 validation và 186 test so với photo policy max8 của prepared release hiện tại. Gói `clindistill-cohort-image-supplement.zip` gồm 1.417 ảnh gốc (~22,5 MB) và manifest, không có clinical/labels/weights. Đây là so sánh references local, chưa phải xác nhận toàn bộ file thực có trên Kaggle.

Để chạy với bundle Qwen đã upload:

1. Import notebook01 Features Only mới, gắn **bank00 đã hoàn thành + addon đã có + Qwen assets**.
2. Giải nén supplement, upload `images/` và `code_image_supplement.json` thành một dataset **private**, gắn thêm vào notebook01. Notebook tự tìm manifest và đối chiếu release_id; không cần thay dataset ảnh lớn hoặc chạy lại notebook00.
3. Chọn Pro6000, Internet OFF, giữ auto batch/paths và Save & Run All. Chờ `IMAGE LAYOUT READY` rồi các encoder chạy.

Nếu cohort đang yêu cầu còn ảnh thiếu/sai bytes, adapter dừng trước khi trích features, liệt kê `missing_required` trong coverage report. Không giảm cohort hoặc gộp photo với X-ray để chạy tiếp. Test feature extraction vẫn do wheel kiểm tra frozen protocol; adapter không đọc test labels hoặc chạy model trên test.

Tạo lại supplement trên máy từ root repo:

```powershell
py -3.14 "Khóa luận/_support/research-notes/package_image_supplement.py" --prepared-root "Data/COde_processed/code_v1" --original-root "E:/Hoc Tap/Data/COde-Dataset" --image-index-archive "Khóa luận/_support/dist/code_v1_prepared.zip" --output "Khóa luận/_support/dist/clindistill-cohort-image-supplement.zip"
```

### Cấu hình hiệu năng

**Khi Kaggle báo `Internet cannot be enabled for this competition with the current accelerator`:** dừng session Pro6000 của notebook00, chọn Accelerator=None, bật Internet và khởi động session mới. Chỉ notebook00 cần mạng để đọc Secrets/tải wheels/weights. Notebook01/02 dùng Pro6000 offline sau khi bank đã lưu.

Notebook00 mới thử kết nối công khai tới Hugging Face trước khi đọc Secrets; phân biệt lỗi mạng, quyền đọc Kaggle Secret và token rỗng, không in giá trị secret hoặc exception có thể chứa nó. Chưa đọc được Secret thì chưa xác thực token với Hugging Face. [Kaggle Secrets client](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py), [Kaggle web client](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_web_client.py).

CPU và GPU có thể khác Python/Torch/CUDA build. Trên GPU, chạy `MODE='inspect'`, sao chép inventory JSON vào `TARGET_ENVIRONMENT` của notebook00 online. Bản mới cho phép **chỉ Torch và torchvision khác hậu tố `+cpu` → cùng hậu tố `+cuNNN`**, giữ đúng release gốc, Python/platform và tất cả scientific packages. Trường hợp đã ghi nhận ngày 03/11/2026: CPU là torch2.11.0+cpu/torchvision0.26.0+cpu; Pro6000 là torch2.11.0+cu130/torchvision0.26.0+cu130, các mục còn lại khớp. Đây là các bản phân phối CPU/CUDA của cùng release theo [PyTorch](https://pytorch.org/get-started/previous-versions/).

Manifest ghi `preparation_environment` là CPU thật, `base_environment` là GPU đích và chỉ chứng nhận local CPU model loading. Khi activate trên GPU, kiểm tra target vẫn **khớp hoàn toàn**, không bỏ hậu tố CUDA hoặc sửa manifest bằng tay. Wheelhouse addons bị chặn nếu chứa torch/torchvision/torchaudio/triton hoặc NVIDIA CUDA wheels; không cài lại Torch trong CPU/GPU session. Compiled Torch extensions chưa được hỗ trợ bởi cơ chế này.

Notebook00 mới chứa inline helpers nên có thể thay riêng file notebook, dùng tiếp addon0.5.1 đã upload; không cần upload lại ảnh/prepared metadata/source chỉ để sửa thông báo lỗi này.

Notebook in tên GPU, total/free VRAM, compute capability, Torch/CUDA và chạy một kernel kiểm tra. RTX PRO 6000 Blackwell là sm_120; PyTorch 2.7 đã giới thiệu build CUDA 12.8 hỗ trợ Blackwell. Dùng image/build hỗ trợ GPU thực tế; không dùng wheels Windows hoặc tự ghi đè torch của bank trong kernel offline. [NVIDIA GPU list](https://developer.nvidia.com/cuda/gpus), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).

| Công việc | Tự chọn khi còn ≥70 GiB VRAM | Ghi chú |
|---|---:|---|
| DINOv3-B | 256 ảnh/batch | Image autocast BF16 nếu GPU hỗ trợ, FP16 nếu không |
| DINOv3-L | 128 ảnh/batch | OOM tự chia microbatch, giữ thứ tự |
| ResNet50 / Swin-B | 256 / 128 ảnh/batch | Cùng preprocessing/split |
| BGE-M3 | 128 văn bản/batch | FP32 như protocol cũ, gom độ dài để giảm padding |
| Đọc/resize ảnh | tối đa 8 workers, prefetch 2 | Pinned CPU buffers, nonblocking transfer |
| MIL/head train | batch 32, accumulation 1 | Features resident trên GPU nếu đủ bộ nhớ; FP32 |
| Validation/predict | batch 256 | Không đổi optimizer steps |

Đây là **điểm khởi đầu theo phần cứng**, chưa được benchmark trên Pro 6000. Nếu GPU ít VRAM trống, batch tự giảm. Không tự tăng learning rate, số ảnh/ca, epochs hay batch tối ưu hóa để chiếm VRAM. Pipeline frozen-feature huấn luyện head nhỏ nên train02 có thể không đạt 100% GPU; đánh giá bằng thời gian/epoch và images/s. `torch.compile` chưa bật mặc định vì có nhiều fit ngắn; FP8 chưa dùng vì cần kiểm chứng sai số riêng.

Đọc `feature_runtime.json` trong cache; log từng chunk có images/s và peak GPU memory (theo kernel session). Nếu cần thử batch lớn hơn sau khi đo, có thể đặt `IMAGE_BATCH_SIZE=256`; chỉ chỉnh feature batching, CUDA OOM của encoder sẽ chia nhỏ. Đặt `IMAGE_WORKERS=0` để xử lý lỗi multiprocessing, hoặc 4 nếu CPU/RAM hạn chế. Giữ cùng source/config/cache khi resume. Nguyên tắc tải dữ liệu: [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

Source 0.5.1 cần bank mới và cache `research_features_v3`; prepared split `code_v1` giữ nguyên. Không nối checkpoint/cache 0.5.0 vào study mới.

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
- `project_wheel/clindistill_code-0.5.1-py3-none-any.whl`: wheel Python thuần, dùng được trên Linux.
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
5. Chọn **Accelerator=None (CPU)** nếu GPU đích chặn mạng; bật Internet. So `TARGET_ENVIRONMENT` với inventory GPU. Giữ `MODE='prepare'`, `RUN_MINI_REHEARSAL=False`, `BANK_ROOT='/kaggle/working/clindistill_research_v3'`. Run All.
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
- `/kaggle/working/data/COde_processed/research_features_v3/`: feature manifest, train.npz và validation.npz.

Nếu đã có `code_v1`, gắn prepared release và ảnh; chỉ chạy setup đầu notebook rồi mục15. Đặt `PREPARED_ROOT_OVERRIDE` và `IMAGE_ROOT_OVERRIDE` nếu cần. Không chạy lại chia tập để tìm split thuận lợi.

Cache v1 không dùng với builder mới. Tạo cache v3 trên cùng split. Save Version giữ cả release, cache và audit. Feature vectors/IDs/clinical contexts vẫn là dữ liệu riêng tư.

**Các review cần người thực hiện:** near-duplicate, mapping nhãn, tính đầy đủ annotation và thời điểm clinical. Không bật cờ reviewed chỉ để bỏ cảnh báo. Không có healthy class; single-target là một target được ghi nhận, không khẳng định chỉ mắc một bệnh.

## 4. Notebook02 — train từ cache, không cần gắn raw images

Gắn **bank00 + output cache01**, chọn GPU, tắt Internet, restart kernel rồi Run All từ đầu.

Cell đầu:

```python
CONFIG_FILE = 'research.json'
RUN_NAME = None  # tự dùng code_research_v3 từ config
```

Profile mặc định có 12 hệ thống ×3 seeds và 30 Teacher OOF fits = **66 fits**. Bao gồm Student no-KD, label smoothing, Visual/MM KD, confidence/correctness/visual-support/reliability KD, hai Teachers và ResNet/Swin controls.

Để thêm kiểm chứng multi-image/clinical, đổi **trước khi chạy**:

```python
CONFIG_FILE = 'research_controls.json'
RUN_NAME = None  # code_research_controls_v3, thư mục study riêng
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

- Bank v5/project0.5.1, cache v3, đúng config/run_name.
- Bootstrap in đường dẫn project đã cài trong site-packages; không import từ source trong bank.
- Dùng weights thật có quyền truy cập; debug/tiny-model tests không thay cho pretrained experiments.
- Chạy mini rehearsal nếu cần sau khi có prepared release. Không báo metric mini như kết quả khóa luận.
- Giữ output phiên trước, raw predictions và checksums. Code/tests chạy local không chứng minh Kaggle đã train full hoặc phương pháp tốt hơn.

Nguồn kỹ thuật: [pip wheel](https://pip.pypa.io/en/stable/cli/pip_wheel/), [pip download](https://pip.pypa.io/en/stable/cli/pip_download/), [Transformers DINOv3](https://huggingface.co/docs/transformers/v4.57.1/model_doc/dinov3), [BGE-M3](https://huggingface.co/BAAI/bge-m3).
