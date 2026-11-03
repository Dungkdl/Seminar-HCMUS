# Kiểm chứng implementation v5 — 02/11/2026

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
