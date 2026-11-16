# ClinDistill — phân loại target nha khoa được ghi nhận trên CODe

Nghiên cứu kiểm tra liệu clinical context dùng khi huấn luyện có giúp Student chỉ nhận ảnh phân loại tốt hơn trên cùng cohort/split hay không. Hiện triển khai **CODe photo-only, DINOv3/BGE-M3 frozen, học MIL/projection/fusion/head**. Không có healthy class, không bao phủ mọi bệnh nha khoa và chưa phải mô hình chẩn đoán lâm sàng đã thẩm định.

**Bắt đầu tại [Khóa luận/README.md](Khóa%20luận/README.md)** và [runbook Kaggle](Khóa%20luận/_support/KAGGLE_RUNBOOK.md). Package **0.6.0**, bank **v6**, giữ prepared `code_v1` và builder/cache `research_features_v3` của0.5.1. Tạo study mới; không resume checkpoint từ phương pháp cũ.

| Giai đoạn | Profile | Câu hỏi | Fits, 3 seeds |
|---|---|---|---:|
| Visual | `research_visual.json` | Một ảnh, mean, attention MIL và Visual Teacher | 12 |
| Clinical | `research_clinical.json` | Clinical-only, demographics, shuffle và masked control | 18 |
| Transfer | `research_transfer.json` | Student no-KD, Visual KD và MM KD | 45 |
| Core gộp | `research.json` | Các hệ thống cốt lõi trên cùng protocol | 63 |
| Mở rộng | `research_controls.json` | Reliability/gate components, weighted CE, CNN/Swin | 87 |

Notebook02 mặc định bắt đầu **visual**. Các stage có run_name riêng và lặp một số fits; không cộng chúng thành số fits của core gộp. Chỉ chuyển giai đoạn sau khi đọc validation; kết quả bất lợi cho clinical/KD vẫn là kết quả hợp lệ.

| Notebook | Vai trò |
|---|---|
| `00_Kaggle_Offline_Setup` | Inspect target, chuẩn bị dependency bank/pretrained assets trên Linux online |
| `01_Data_Audit_and_Preparation` | Audit, review templates, prepared split và features |
| `01_Features_Only` | Dùng prepared release có sẵn, không chia lại split |
| `02_Train_Baselines` | Chạy từng profile, OOF khi cần, resume và export Student |
| `03_Evaluate` | Validation; khóa protocol trước khi chuẩn bị/mở test |

## Điều kiện bằng chứng

- Software checks dùng synthetic data/tiny random encoders, không chứng minh hiệu quả pretrained.
- Log Kaggle cung cấp trước đây xác nhận bước trích feature; repo chưa có full ClinDistill training/test results được kiểm chứng.
- Nhãn là target ghi nhận trong hồ sơ. Review annotation/clinical timing/near-duplicates còn mở; absent labels không tự là âm tính hoặc khỏe mạnh.
- Reliability là trọng số theo mẫu từ dự đoán OOF, chưa phải chọn lọc field hay thước đo visual sufficiency.
- `exploratory` giữ pending reviews; `confirmatory` bị chặn khi release/review chưa đủ điều kiện. Hoàn thành software run không tự nâng trạng thái nghiên cứu.

Phạm vi/phương pháp: [plan hiện hành](CODe_Research_Implementation_Plan.md). Sơ đồ: [Seminar.md](Seminar.md). Kế hoạch/poster ban đầu được giữ ở `docs/archive/` như tài liệu lịch sử.

## Source và dữ liệu

`Khóa luận/clindistill/research/` là pipeline hiện hành; CLI `python -m clindistill` dùng pipeline này. Trainer cũ chỉ truy cập bằng `python -m clindistill legacy ...`.

`Baseline/`, `run-all-2.ipynb`, `explore-data.ipynb` được giữ độc lập; Qwen không thuộc đợt củng cố này. [Paper/README.md](Paper/README.md) ghi nguồn và sửa nhầm file DentVLM.

Không commit ảnh CODe, patient records/IDs, clinical vectors, raw predictions, weights/checkpoints. Bundles và kiểm tra local ở ignored `_support/dist/`. Dữ liệu có quyền sử dụng ở `Data/` hoặc đường dẫn override; source bundle mặc định không chứa dataset.

## Kiểm tra local

Trong môi trường khoa học/PyTorch tương thích:

```bash
cd "Khóa luận"
python -m pip install "setuptools>=68,<82" "wheel>=0.43"
python -m pip install -e ".[clinical,dev]"
python -m unittest discover -s _support/tests -p "test_*.py"
python -m clindistill plan --config configs/code/research_visual.json
```

[#5](https://github.com/Dungkdl/Seminar-HCMUS/issues/5) theo dõi đợt sửa này; [#1](https://github.com/Dungkdl/Seminar-HCMUS/issues/1) vẫn mở cho nghiên cứu/review; [#4](https://github.com/Dungkdl/Seminar-HCMUS/issues/4) theo dõi xác nhận dependency/runtime trên Kaggle thật.
