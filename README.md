# ClinDistill — khóa luận trên COde

Pipeline nghiên cứu phân loại ảnh nha khoa, sử dụng thông tin lâm sàng khi huấn luyện Teacher và knowledge distillation sang Student chỉ dùng ảnh. Phạm vi hiện tại là **COde**; MeMoSA để giai đoạn sau.

**Bắt đầu tại [Khóa luận/README.md](Khóa%20luận/README.md)**: chuẩn bị Kaggle offline, dữ liệu, train và đánh giá.

| Notebook | Vai trò |
|---|---|
| [00 — Offline setup](Khóa%20luận/COde_00_Kaggle_Offline_Setup.ipynb) | Chuẩn bị wheels và pretrained weights khi còn Internet |
| [01 — Data preparation](Khóa%20luận/COde_01_Data_Audit_and_Preparation.ipynb) | Làm sạch, EDA, kiểm định, chia tập và clinical embeddings |
| [02 — Train](Khóa%20luận/COde_02_Train_Baselines.ipynb) | Baselines, Teachers, knowledge distillation và ablation |
| [03 — Evaluate](Khóa%20luận/COde_03_Evaluate.ipynb) | Metrics, so sánh thực nghiệm và xuất Student |

## Các tài liệu và code bổ sung

- `Paper/`: các paper tham khảo, literature review và báo cáo seminar đã có tại máy làm việc. Quyền tác giả thuộc các tác giả/nhà xuất bản tương ứng.
- `Baseline/Seminar/`: notebook và script baseline ban đầu; `COde_HF/` chỉ gồm README, thông tin Git và Data Usage Agreement, không có dataset.
- `Seminar.png`: hình seminar.
- `explore-data.ipynb`, `run-all-2.ipynb`: các notebook ban đầu, giữ để tham khảo. Pipeline hiện tại bắt đầu từ notebook 00–03 trong `Khóa luận/`.

Các notebook được xóa output trước khi đưa lên Git. Notebook baseline đọc token Hugging Face từ biến môi trường `HF_TOKEN` thay cho token ghi trực tiếp; cung cấp biến này khi thực sự cần xác thực.

## Dữ liệu và artifact

Repo lưu mã nguồn, notebook không có output, cấu hình, tests và tài liệu kỹ thuật. COde, bản mini, clinical embeddings, weights, checkpoints và các ZIP upload Kaggle không nằm trong Git. Các đường dẫn artifact trong hướng dẫn chi tiết trỏ tới file được tạo tại máy làm việc, không phải file có sẵn sau khi clone.

Đặt dữ liệu có quyền sử dụng tại `Data/COde-Dataset (1)/` cạnh thư mục `Khóa luận/`, gồm `complete_dataset.csv` và `Images/`, hoặc dùng biến đường dẫn override trong notebook. Notebook 01 tạo prepared release. Giữ dữ liệu trên Kaggle trong phạm vi truy cập phù hợp dữ liệu gốc.

Sau khi có prepared release `Data/COde_processed/code_v1/`, chạy `Khóa luận/_support/research-notes/package_kaggle.py` để tạo source ZIP và prepared ZIP tại `_support/dist/`. Notebook 00 có thể build project wheel từ source nếu chưa có wheel; môi trường chuẩn bị cần setuptools và wheel.

## Kiểm tra mã nguồn

Trong môi trường có dependencies khoa học/PyTorch tương thích:

```bash
cd "Khóa luận"
python -m pip install -e ".[clinical]"
python -m unittest discover -s _support/tests -p "test_*.py"
```

Lệnh cài trên dùng cho môi trường có Internet. Với GPU chặn Internet, thực hiện notebook 00 theo hướng dẫn chi tiết; giữ cặp torch/torchvision của môi trường đích.

Tests dùng dữ liệu tổng hợp. Smoke tại local chỉ xác nhận pipeline; chưa phải kết quả train đầy đủ trên RTX 6000/Kaggle và chưa chứng minh hiệu quả nghiên cứu.
