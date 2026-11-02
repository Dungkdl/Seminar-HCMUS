# Kiểm chứng implementation v5 — 02/11/2026

## Phần mềm

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
