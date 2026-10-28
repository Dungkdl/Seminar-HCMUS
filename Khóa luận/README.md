# ClinDistill — bắt đầu từ đây

Nếu bundle cũ lỗi `Index put requires the source and destination dtypes match` khi chạy
AMP tại `models.py`, dùng `_support/recover_amp_bundle.py` từ source mới:
`recover_and_smoke(BUNDLE, MINI_ROOT)`. Helper sửa source, build lại project wheel,
cập nhật checksum và chạy smoke ở thư mục mới trước khi tạo ZIP; giữ weights đã tải.
Chạy trong phiên chuẩn bị còn hoạt động và có quyền ghi bundle dưới `/kaggle/working`.

Bộ code hiện hỗ trợ **COde: chuẩn bị → baselines → Teachers → KD → so sánh → xuất Student dùng ảnh**.
Luồng nghiên cứu giữ **01 → 02 → 03**; thêm **00 dùng một lần để chuẩn bị offline theo hướng dẫn của thầy**.
Dữ liệu dùng chung nằm trong `../Data/`; source và hướng dẫn nằm trong `Khóa luận/`.

> Đây là pipeline để thực hiện khóa luận, chưa phải khóa luận đã có kết quả. Chưa train đầy đủ trên RTX 6000/Kaggle trong phiên sửa code này.
> Phạm vi lần này chỉ COde theo yêu cầu của bạn; MeMoSA để sau. Chưa có cơ sở tuyên bố đạt Q1 hoặc selective KD tốt hơn baseline.

## Cần mở file nào?

| File | Làm gì |
|---|---|
| [00 — Kaggle offline setup](COde_00_Kaggle_Offline_Setup.ipynb) | Kiểm tra môi trường, tải wheels Linux + weights khi còn Internet, thử chạy khi chặn mạng, đóng gói |
| [01 — Data preparation](COde_01_Data_Audit_and_Preparation.ipynb) | Làm sạch tiếng Anh, EDA, kiểm định ảnh/nhãn, khóa split, xuất manifest; **mục 14 tạo clinical embeddings** |
| [02 — Train](COde_02_Train_Baselines.ipynb) | Run All: Teacher V/H → selective KD, 3 seed; kiểm tra đầu vào, resume, báo cáo và xuất Student |
| [03 — Evaluate](COde_03_Evaluate.ipynb) | Metrics, bootstrap, mean/std, so sánh cặp, risk–coverage, phân tích lỗi, xuất Student |
| [Giải thích kỹ thuật](_support/TECHNICAL_GUIDE.md) | Định nghĩa task, model, field masks, công thức loss, thực nghiệm và các giới hạn |
| [Cấu hình full run](configs/code/thesis.json) | ResNet-50, số ảnh, batch, tích lũy gradient, epochs, KD weights |

Tên notebook 02 vẫn giữ `Train_Baselines` để không làm hỏng đường dẫn cũ, nhưng nội dung đã bao gồm Teacher/KD.
02 và 03 được xóa output của bản smoke cũ để tránh nhầm là kết quả của code mới.

## Dữ liệu ở một chỗ

```text
Data/
├── COde-Dataset (1)/          dữ liệu gốc; Images/ và complete_dataset.csv
└── COde_processed/
    ├── complete_dataset_en.csv   8.775 hồ sơ, 22 cột tiếng Anh
    ├── code_v1/                 release.json + manifest + teacher_context
    └── clinical_biomedbert_v1/   tạo bằng mục 14 notebook 01, trước khi train Teacher/KD
Khóa luận/
├── README.md
├── COde_00_...ipynb              chuẩn bị offline một lần
├── COde_01_...ipynb
├── COde_02_...ipynb
├── COde_03_...ipynb
├── clindistill/                  code notebook gọi
├── configs/code/
├── runs/code_clindistill_v2/     chỉ xuất hiện khi train full
└── _support/                    tài liệu sâu, tests, audit, smoke và ZIP upload
Paper/                           các paper của khóa luận
```

CSV sạch dùng để đọc/EDA. **Train đọc manifest và ảnh gốc**, không đọc lại CSV, không nối các JSON QA thành mẫu mới.
Prepared release đang có 6 lớp: Caries, Gingivitis, Malocclusion, Pulpitis, ToothLoss, ToothStructureLoss.
Cohort thực dùng có ảnh chụp: **4.225 train / 917 validation / 910 test**.
Đơn vị dự đoán là lần khám; split_group giữ các bệnh nhân/ảnh trùng đã phát hiện cùng tập.
Chỉ có một nhãn mục tiêu được ghi nhận không đồng nghĩa chỉ mắc một bệnh. Không có lớp healthy.

## Chạy trên Kaggle — làm theo thứ tự này

### Quy trình GPU bị chặn Internet — dùng phần này trước

Thầy yêu cầu ba việc: trích ít dữ liệu để thử; xác định/cất sẵn dependencies; mang code + data + wheels sang phiên GPU.
Wheel là gói cài Python, **không tự chứa ResNet/BiomedBERT weights hoặc dữ liệu COde**.
`pip download` thu thập wheel và dependencies; `pip install --no-index --find-links ...` cài từ disk mà không hỏi package index.
Tham khảo [pip download chính thức](https://pip.pypa.io/en/stable/cli/pip_download/) và
[offline Transformers](https://huggingface.co/docs/transformers/installation#offline-mode).

**Đã tạo sẵn trên máy:**

| Artifact | Nội dung / trạng thái |
|---|---|
| [code_mini_offline_check.zip](_support/dist/code_mini_offline_check.zip) | 12 ca train + 12 validation, đủ 6 lớp, 128 ảnh, khoảng 2 MB; không lấy test |
| [project wheel](_support/dist/project_wheel/clindistill_code-0.2.0-py3-none-any.whl) | Code ClinDistill Python thuần; đã build và cài thử offline; không chứa thư viện bên thứ ba |
| [source ZIP](_support/dist/clindistill-code-source.zip) | 00–03, source, helper offline, cấu hình và wheel project |
| `code_offline_v1.zip` | **Chưa có tại local**; notebook 00 tạo trên Kaggle online sau khi thu thập wheels Linux và weights, rồi chạy smoke thành công |

Bộ mini thực nằm ở `../Data/COde_processed/mini_offline_check/`, không trộn vào full release.
Notebook chính bỏ qua mini khi tự tìm dữ liệu; nếu cố ý dùng mini hãy đặt đường dẫn override.
Mini ZIP chứa ảnh/hồ sơ thật, giữ cùng phạm vi truy cập như COde gốc; không publish công khai mặc định.

**Làm lần lượt:**

1. Gắn source ZIP vào notebook ở môi trường GPU đích; import notebook **00**, giữ `MODE='inspect'`.
   Nó chạy được khi không có Internet và xuất `environment_report.json`: Python, package versions, lỗi import cụ thể.
2. Mở **một notebook Kaggle có Internet**, có thể CPU, cùng image/Python/base packages. Gắn source và mini ZIP.
   Colab chỉ thay được nếu môi trường khớp; không giả định wheel từ Colab/Windows sẽ tương thích GPU đích.
3. Trong 00 chọn `MODE='prepare'`, kiểm tra source/mini paths rồi Run All.
   Notebook tải wheel dependencies (pin Transformers 4.57.1), dùng project wheel đã build,
   tải ResNet-50/BiomedBERT từ nguồn chính thức, giữ nguyên torch/torchvision/scientific stack.
4. Notebook tự cài bằng `--no-index` rồi thử pretrained models trên mini COde với sockets bị chặn.
   Đây là kiểm tra pipeline ở ngân sách nhỏ, không phải kết quả khóa luận.
5. Chỉ khi smoke qua mới tạo `offline_preparation/code_offline_v1.zip`.
   Save Version để giữ output. Gắn folder output vào notebook GPU, hoặc download/upload ZIP làm input riêng.
6. Phiên GPU: gắn **offline bundle + COde đầy đủ + prepared release**. Mở 01, chạy cell setup đầu rồi mục 14 tạo clinical cache đầy đủ.
   Sau đó 02 train, 03 evaluate. Các notebook dùng ResNet/BiomedBERT local từ bundle, không tải Internet.
   Nếu clinical cache đã được chuẩn bị đầy đủ, gắn cache và đi thẳng 02.

Nếu bundle chỉ hiện dưới dạng ZIP chưa giải nén, giải nén vào `/kaggle/working` rồi đặt `OFFLINE_BUNDLE_OVERRIDE`
tới folder có `offline_manifest.json`. Không nhập đường dẫn tới file ZIP vào biến này.
Bundle tự kiểm tra checksum, Python minor, hệ điều hành, architecture và base package versions.
Nếu lệch, chuẩn bị lại trên môi trường khớp; **không tự cài đè torch/torchvision CUDA** để cố vượt qua kiểm tra.
Nếu thiếu import dù package có version, xem `details` trong report: có thể là dependency bị xung đột.
Restart kernel nếu đã import thư viện trước khi đổi phiên bản.

Đã kiểm tra local: cài project wheel `--no-index --no-deps` vào thư mục riêng, import từ chính wheel đã cài,
chạy 6 nhánh baseline/Teacher/KD trên mini COde trong khi chặn sockets, dùng **tiny_debug**.
[Biên bản](_support/smoke_runs/offline_wheel_check/verification.json).
Chưa kiểm chứng wheelhouse Linux hoặc pretrained GPU trong phiên Kaggle của bạn; notebook 00 thực hiện bước đó.

### Trường hợp online thông thường

Các mục A–E dưới đây mô tả dữ liệu và cách train. Nếu đã dùng offline bundle, source/weights được cung cấp từ bundle,
không cần tải lại. Nếu chủ động chạy online không có bundle, đặt `OFFLINE_MODE=False` trong cell setup đầu notebook.

### A. Gắn đầu vào

1. Upload [source ZIP](_support/dist/clindistill-code-source.zip) thành dataset source; kiểm tra thấy `clindistill/` và `configs/` sau giải nén.
2. Gắn COde gốc đã có `Images/Photographs` và `Images/Radiographs`.
3. Upload [prepared ZIP](_support/dist/code_v1_prepared.zip) hoặc output notebook 01; kiểm tra thấy `release.json`.
4. Import notebook 01 nếu cần tạo cache; notebook 02 khi bắt đầu train. Bật accelerator GPU.

Notebook tìm folder dưới `/kaggle/input` và `/kaggle/working`. Có nhiều bản thì đặt `*_OVERRIDE` ở cell đầu.
`IMAGE_ROOT_OVERRIDE` là folder **chứa Images/**, không phải folder Photographs.
`PREPARED_ROOT_OVERRIDE` là folder **chứa release.json**.
Nếu upload ZIP vẫn còn nguyên file nén trong dataset, giải nén vào `/kaggle/working` rồi đặt override tới folder giải nén.

### B. Tạo clinical embeddings một lần

Bạn đã có prepared release nên **chạy cell setup đầu rồi mục 14 notebook 01**, không cần audit lại toàn bộ ảnh.

- Bật Internet lần đầu để tải BiomedBERT; hoặc gắn folder model offline rồi đặt `TEXT_ENCODER` tới folder đó.
- Nếu thiếu thư viện, chạy một cell `%pip install "transformers>=4.40,<6" "huggingface-hub>=0.23,<2" safetensors`.
- Giữ cặp torch/torchvision của Kaggle; không cài lại CUDA stack chỉ để chạy project.
- Trên GPU, mục 14 mặc định chạy. Giữ `DEBUG_CLINICAL=False` cho nghiên cứu.
- Mặc định chỉ mã hóa train/validation. Kết quả tại `/kaggle/working/data/COde_processed/clinical_biomedbert_v1/`.
- Save Version để giữ output; phiên train khác gắn output cache vào, hoặc upload folder cache thành dataset riêng.

Cache chứa vector hồ sơ bệnh nhân, không phải model công khai; giữ cùng phạm vi truy cập như dữ liệu gốc.
Text encoder frozen, không huấn luyện trên validation. Token/trường bị cắt được đếm trong `train.json`, `validation.json` của cache.
Không có encoder thì báo lỗi, không tự thay bằng random vectors.

### C. Train

Mở notebook 02 mới, gắn **3 input**: `code-offline-v1`, `code-training-inputs` (ảnh + prepared),
output notebook 01 có clinical cache. Chọn GPU, tắt Internet, **Run All**. Cấu hình mặc định:

```python
RUN_NAME = 'code_clindistill_v2'
SEEDS = [2026, 2027, 2028]
EXPERIMENTS = ['kd_selective']  # tự thêm Teacher V/H: 9 lượt train trên 3 seed
RESUME = True
```

Notebook 02 có một cell cấu hình duy nhất, kiểm tra GPU/AMP, release/cache/ảnh và các nhóm split trước train.
Tự lưu bảng validation từng seed, mean/std, learning curves và xuất Student mỗi seed vào `exports/`.
Muốn mở rộng đối chứng/ablation, đổi `EXPERIMENTS`; Teachers phụ thuộc tự thêm. Mặc định không chạy baseline riêng.
Giữ RUN_NAME và cấu hình nền để tiếp tục, không tạo lại split. Test và đánh giá độc lập ở notebook 03.

Default: **ResNet-50 pretrained ImageNet, 224×224, tối đa 8 photo, batch 4 ca × tích lũy 4, 40 epochs, patience 8, AMP**.
Notebook in tên GPU/VRAM thật. Không giả định mọi RTX 6000 cùng VRAM.
Nếu OOM, giảm batch/increase accumulation, dùng RUN_NAME mới và áp dụng đồng nhất cho các nhánh so sánh.
Tích lũy gradient không làm BatchNorm giống batch vật lý lớn; relation KD dùng các ca trong microbatch.

ResNet offline: gắn state_dict chính thức tương ứng `torchvision.models.ResNet50_Weights.DEFAULT` (bao gồm `fc`)
và đặt `BACKBONE_WEIGHTS_PATH`. Không tự chuyển sang random-init khi download lỗi.

### D. Giữ kết quả và resume

Kết quả ghi vào `/kaggle/working/runs/code_clindistill_v2/<experiment>/seed_<seed>/`.
`best.pt` chỉ giữ phần cần đánh giá; `last.pt` giữ cả optimizer/scheduler/scaler/RNG để resume cuối epoch.
**Save Version/giữ output sau phiên**. Checkpoint trong ổ làm việc không tự trở thành bản lưu lâu dài.

Phiên mới: gắn Output notebook train cũ, giữ nguyên code/config/data/cache và `RESUME=True`.
Notebook 02 tự tìm suite cùng RUN_NAME/protocol trong Input và copy về working trước khi resume.
Nếu có nhiều bản phù hợp, đặt `PREVIOUS_RUNS_OVERRIDE` tới folder RUN_NAME muốn dùng ở cell cấu hình đầu.
Notebook 03 vẫn cần folder runs có quyền ghi theo hướng dẫn của notebook đó.

Không train vào `/kaggle/input` chỉ đọc. Chạy cả 3 seed và mọi ablation tạo nhiều GB checkpoint; có thể chia phiên theo nhóm
thí nghiệm rồi giữ output riêng. Không xóa `last.pt` nếu còn muốn resume. Đường dẫn cache có thể đổi khi chuyển phiên,
nhưng nội dung/checksum phải giữ nguyên. Với weights file offline, config chứa path nên giữ đường dẫn thống nhất giữa phiên.
Checkpoint smoke cũ thuộc phiên bản code cũ **không dùng để resume/evaluate bằng v2**; chúng vẫn được giữ trong `_support` để đối chiếu.

### E. Đánh giá

Notebook 03 dùng cùng RUN_NAME, mặc định `EVALUATION_SPLIT='validation'`.
Chọn phương pháp/hyperparameter bằng validation, khóa protocol rồi mới mở test.
Release hiện vẫn `experimental`; test cần đặt `ALLOW_EXPERIMENTAL_TEST=True` nếu muốn đánh giá thăm dò và vẫn ghi rõ trạng thái này.
Đây không phải thao tác xác nhận nhãn hay ảnh gần trùng đã được kiểm định.

Nếu báo cáo cả Teacher trên test: chạy mục 14 notebook 01 với `CACHE_SPLITS=('train','validation','test')` trên cùng encoder/cache.
Có cache gắn read-only thì copy nguyên cache về working và đặt `CLINICAL_OUTPUT_OVERRIDE` tới bản copy trước khi thêm shard test.
Notebook 03 phải trỏ `CLINICAL_CACHE_OVERRIDE` vào cache đã có đủ shard. Student riêng không cần clinical data khi inference.

Output: CSV từng seed, mean/std, paired CI, hình so sánh, confusion matrix, risk–coverage, case_analysis, subgroup metrics.
`EXPORT_STUDENT=True` xuất artifact chỉ chứa Student; inference API `predict_images` không yêu cầu nhãn hoặc hồ sơ.

## Đã hiện thực đến đâu so với seminar?

| Hạng mục | Trạng thái thực tế |
|---|---|
| COde English cleaning, EDA, audit, patient/duplicate grouping | Có; một số manual reviews vẫn mở |
| Single, mean, MIL, Set attention, photo+X-ray | Có |
| Clinical-only và multimodal full/V/H Teachers | Có; frozen BiomedBERT + trainable field projections/late fusion |
| Vanilla và selective/weighted KD | Có; Teacher frozen; Student inference chỉ ảnh |
| Feature/relation distillation | Có dưới dạng ablation `kd_feature` / `kd_relation` |
| Multi-seed, calibration, CI, error/subgroup analysis | Có |
| Clinical dependency | Có phân tích bất đồng Teacher/visual và risk–coverage; chưa có nhãn “cần lâm sàng” thật |
| Learned visualizability, cross-attention image–text | Chưa triển khai; selective hiện dùng nhóm trường và trọng số cố định |
| Đa nhãn đầy đủ, tooth/region heads, CBCT 3D | Chưa có annotation/protocol phù hợp trong nhánh COde này |
| MeMoSA external validation | Chưa có dữ liệu; cần task/head và split riêng, không ánh xạ tùy tiện sáu nhãn COde |
| Kết quả khóa luận / bài báo | Phải chạy thực nghiệm thật, phân tích và viết kết luận; không cam kết Q1 |

Không đưa treatment/diagnosis trực tiếp vào Teacher. Mask tên bệnh trong text chưa chứng minh loại bỏ hết leakage.
Phân nhóm V/H là giả thuyết thực nghiệm; không gọi V là thông tin chắc chắn suy ra được từ ảnh.

## Kiểm chứng bản code này

- 20 kiểm thử CPU đã qua, gồm backend Transformers offline bằng BERT nhỏ, AMP,
  phục hồi wheel/source/manifest và Run All notebook 02 trên dữ liệu tổng hợp.
  Kiểm thử notebook xác nhận xuất Student, kiểm tra cache lỗi, resume cùng phiên và phục hồi từ output phiên trước.
- Đã thực thi cell clinical preparation và toàn bộ cell notebook 02/03 trên COde thật ở chế độ debug:
  16 nhánh, 1 seed, 2 epochs, 12 ca mỗi partition; checkpoint, resume và báo cáo tạo thành công.
- [Biên bản smoke](_support/smoke_runs/verification_6036cbf08e93/verification.json) là bằng chứng chạy code,
  **không phải đánh giá chất lượng BiomedBERT/ResNet-50 hoặc hiệu quả KD**.
- CUDA/RTX 6000, tải pretrained weights online và full 3-seed training chưa được chạy trong môi trường local này.

Chi tiết công thức, dữ liệu và nguồn tham khảo ở [TECHNICAL_GUIDE.md](_support/TECHNICAL_GUIDE.md).
Lệnh kiểm thử từ thư mục `Khóa luận`: `python -m unittest discover -s _support/tests -v`
(cài dependencies project và extra `clinical` để chạy cả kiểm thử Transformers).
