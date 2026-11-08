# Kiểm chứng implementation v5 — 02/11/2026

## Bổ sung 08/11/2026: regression clinical và output Kaggle notebook01

- **65 tests qua** trong môi trường pinned Python3.14.2/torch2.11.0+cu128/torchvision0.26.0+cu128, Transformers4.57.1/hub0.36.2 trên Windows. Năm notebook qua nbformat/AST với outputs sạch; đây là bộ kiểm thử phần mềm local.
- Bổ sung ba regression tests dùng synthetic context có text, tuổi/giới khác 0 và các trường hợp thiếu dữ liệu. Full phải giữ nguyên context/cờ hiện diện; demographics-only chỉ xóa text; masked phải xóa cả hai và tắt cờ. Các chế độ không được sửa cache nguồn.
- Kiểm tra đường `dataset_for → collate_visits → forward_batch`: thay riêng text hoặc demographics có thể đổi logits của MM Teacher ở cả gated/concat fusion; masked control không phản ứng với các giá trị đã bị che. Student không nhận context, không đổi logits khi context thay đổi và từ chối clinical input trực tiếp. Đây là kiểm chứng phần mềm với head ngẫu nhiên, không phải bằng chứng clinical cải thiện chất lượng pretrained model.
- Mutation check trong bản sao ignored: đảo `clinical_mode == 'masked'` thành `!= 'masked'`. Bộ 62 tests cũ vẫn qua; hai regression tests mới thất bại với 14 subtest failures. Source/model/data builder thực không đổi; project module hash vẫn `d4e1c107c9bbaf3219ed72652115426ef9ff88a79659761f444fd8005b843ae0`. Không yêu cầu tạo lại bank00 hoặc features01.
- Đã đọc notebook có output `code-01-features-only-v051.ipynb` do người dùng tải về (artifact giữ ngoài Git). Papermill ghi phiên chạy 04/11/2026 không có exception, khoảng 63,2 phút; code cells khớp Features Only hiện tại. Torch runtime ghi CUDA13.0/torch2.11.0+cu130 và RTX PRO6000 94,97 GiB, dù metadata Kaggle có trường `isGpuEnabled=false` và TensorFlow có cảnh báo driver.
- Log xác nhận layout ánh xạ 35.385 ảnh byte-verified và các encoder xử lý đủ 28.667 ảnh của train/validation. Cache được báo có 4.225 train/917 validation; dimensions student1536/teacher2048/ResNet2048/Swin1024/clinical1024, không có text bị cắt. Đây là bằng chứng từ log thực; chưa tải các `.npz`/manifest để đối chiếu checksum độc lập, chưa có full training hoặc test evaluation.
- Runtime ghi batch DINO-B/ResNet256, DINO-L/Swin128, clinical128, workers8. Các images/s và peak VRAM trong log là telemetry của phiên này, chưa phải benchmark có kiểm soát hoặc chứng minh GPU đã tối ưu.
- Pip báo dependency conflicts, trong đó Torch yêu cầu `setuptools<82` nhưng bank có84.2.1; các xung đột fsspec/HF với thư viện Kaggle khác cũng được ghi nhận. Notebook01 vẫn hoàn thành; regression tests lần này không giải quyết dependency compatibility. Review nhãn/clinical timing/near-duplicates vẫn chưa xác nhận.

## Bổ sung 04/11/2026: SHA1-prefixed Qwen photos và coverage

- **62 tests qua** trên môi trường pinned Transformers4.57.1/hub0.36.2 đã nêu. Sáu tests mới kiểm tra SHA1 filename mapping, photo/X-ray trùng basename, original SHA256, coverage/supplement, idempotence, path/user-directory protection và encoder features giống hệt qua canonical filesystem view.
- Kiểm tra filesystem trên Windows dùng hardlinks khi thiếu quyền symlink. Kaggle dùng symlinks; chưa trực tiếp xác nhận phiên Kaggle chạy adapter này. Các tiny encoder outputs giống hệt không thay thế pretrained experiments.
- Đối chiếu được cả 10 filename mẫu và 2 references cùng basename do người dùng cung cấp với SHA1 canonical path. Project wheel/modules và feature builder không đổi; adapter thao tác layout filesystem, không monkeypatch model/data functions hoặc sửa bank manifest. Không cần tải lại weights.
- Tạo supplement 1.417 ảnh (~22,5 MB): số references của photo policy max8 không nằm trong union cls JSON paper là 1.020 train/211 validation/186 test. Mọi ảnh trong ZIP đã so SHA256 với image index của prepared release. Đây không phải số ảnh thiếu đã kiểm kê trực tiếp trên Kaggle.
- User đã báo notebook00 hoàn thành. Chưa xác nhận trích features đầy đủ hoặc train full ClinDistill trên Kaggle. Pending clinical/annotation/near-duplicate reviews vẫn giữ nguyên.

## Bổ sung 03/11/2026: online CPU → Pro6000 offline

- **56 tests qua** với Transformers4.57.1/huggingface-hub0.36.2 trên môi trường local đã nêu. Tám tests mới phân biệt network/Secret access/empty token, không lộ token trong traceback, kiểm tra môi trường và chuẩn bị bank CPU cho CUDA target.
- CPU→CUDA chỉ cho phép Torch/torchvision cùng release, khác hậu tố CPU/CUDA; Python/platform và mọi scientific package còn lại phải khớp. GPU activation vẫn kiểm tra đúng CUDA build, không nới fingerprint.
- Test mô phỏng chuẩn bị đầy đủ bank ghi targetCUDA và preparationCPU riêng, kiểm tra resume, từ chối CPU activation và Torch/CUDA wheels trong addon wheelhouse. HF API/model loading và subprocess trong test này được giả lập, không phải tải pretrained hoặc chạy GPU thật.
- Notebook00 replacement chứa prepare/helper tương thích với addon0.5.1 đã upload. Kiểm tra compatibility patch nạp được vào chính helper cũ trong addon local. Project wheel/modules và feature builder không đổi, không cần thay prepared release/cache chỉ vì lỗi setup này.
- User output xác nhận RTX PRO6000 nhận đúng CUDA13.0/Torch2.11.0+cu130, CPU có cùng scientific stack ngoài Torch/vision +cpu. Chưa xác nhận bank mới đã tải hoàn tất hoặc inference/training trên Kaggle GPU.

## Bổ sung package 0.5.1: assets Qwen và cấu hình GPU

- **48 tests qua** với Python 3.14.2, torch 2.11.0+cu128, Transformers 4.57.1, huggingface-hub 0.36.2. Năm notebook qua nbformat/AST, không có execution outputs.
- Kiểm tra ánh xạ `data/images` bằng SHA256 ảnh gốc, từ chối thiếu hash/khác bytes/trùng basename/path traversal. Tiny encoders tạo cùng features từ ảnh gốc và ảnh làm phẳng.
- Kiểm tra multiprocessing loader qua nhiều chunk/resume, chia microbatch khi CUDA OOM giả lập, giữ thứ tự và nhớ batch nhỏ cho lần sau; không nuốt lỗi runtime khác.
- Kiểm tra resident batches (thực thi trên CPU) khớp ordinary DataLoader cả visit order, padding, clinical shuffle và target indices. Test này chưa đo CUDA throughput.
- Kiểm tra đóng gói addon: source/wheel + prepared metadata/hash index; không chứa ảnh/weights/Qwen assets. Bank vẫn kiểm tra installed wheel khớp source.
- Profile ≥70 GiB VRAM trống khởi đầu DINO-B/ResNet 256, DINO-L/Swin 128, clinical 128; tối đa 8 image workers. Optimizer batch32/accumulation1, head FP32 và protocol/split giữ nguyên.

Chưa benchmark Pro 6000/Kaggle hoặc xác nhận full pretrained CODe experiments. Batch profile là điểm khởi đầu; log images/s và peak GPU memory dùng để đo trên máy đích. Cache cũ không dùng với builder0.5.1; dùng research_features_v3, giữ code_v1 split. Các review chuyên môn bên dưới vẫn mở.

## Phần mềm bản 0.5.0

- 42 tests qua trong môi trường Python 3.14.2 / torch 2.11.0+cu128 / torchvision 0.26.0+cu128.
- Chạy lại toàn bộ 42 tests trong virtualenv dùng đúng Transformers **4.57.1**, huggingface-hub **0.36.2**; đều qua.
- Tests bao gồm notebook02 Run All trên synthetic cache, group-OOF, gradient KD, batch-one weighting, controls clinical/MIL, Teacher strata, khóa test, resume, export/inference và kiểm tra wheel trong virtualenv riêng.
- Test wheel cố ý chạy tại thư mục có source để xác nhận import vẫn đến từ wheel đã cài; source/wheel lệch module bị từ chối.
- Bốn notebook qua nbformat validation, các code cell parse được, không có execution output. Generator cập nhật notebook chạy lại không đổi nội dung.
- Script đóng gói đã tạo project wheel 0.5.0, source ZIP và prepared ZIP; kiểm tra tính toàn vẹn archive và module wheel/source.

Các tests dùng synthetic visits/features và tiny encoders khởi tạo ngẫu nhiên. Đây là kiểm chứng phần mềm, không phải chất lượng pretrained DINO/BGE trên CODe. Chưa chạy tải gated weights hoặc train full trong Kaggle GPU.

## Audit dữ liệu gốc trên máy

Thực thi các cell audit/preparation của notebook01 trên CODe gốc, dừng trước trích features:

| Mục | Kết quả |
|---|---:|
| Lượt khám / bệnh nhân | 8.775 / 4.800 |
| Ảnh tham chiếu đã giải mã, kiểm tra hash | 57.994 |
| Ảnh thiếu/hỏng | 0 |
| Cohort photo, train / validation / test | 4.225 / 917 / 910 |
| Giao bệnh nhân / exact-file / exact-pixel qua split mới | 0 |
| Cặp hash gần trùng qua split cần review | 5.257 |
| Near-duplicate buckets lớn chưa xét hết | 35 |

Prepared release được tạo tại local; không commit patient manifests/clinical contexts hoặc raw predictions. Các cặp gần trùng là ứng viên, chưa phải duplicate được xác nhận. Mapping nhãn, tính đầy đủ annotation và clinical temporal alignment vẫn chưa được chuyên gia xác nhận; release giữ trạng thái experimental.

Không suy ra hiệu quả KD, clinical dependency, khả năng tổng quát ngoài CODe hoặc độ an toàn sử dụng từ các kiểm tra trên.
