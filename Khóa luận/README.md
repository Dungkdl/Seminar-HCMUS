# ClinDistill trên CODe — research v5 / package 0.5.0

**Bắt đầu bằng [hướng dẫn Kaggle từng bước với wheel offline](_support/KAGGLE_RUNBOOK.md).**

Luồng chạy: tạo source ZIP/project wheel trên máy → notebook00 online tải Linux wheels và pretrained weights → notebook01 audit dữ liệu/tạo cache → notebook02 train → notebook03 đánh giá.

| Notebook | Input chính | Công việc |
|---|---|---|
| [00 — Offline setup](COde_00_Kaggle_Offline_Setup.ipynb) | Source mới, Internet, HF_TOKEN có quyền DINO | Bank v5 với dependency lock, project wheel, DINO-B/L, BGE-M3, ResNet/Swin và checksum. Không cần prepared release trong lần đầu. |
| [01 — Data audit/preparation](COde_01_Data_Audit_and_Preparation.ipynb) | Bank00 + CODe CSV/Images gốc | Patient/duplicate-group split, review artifacts, code_v1 và research_features_v2. Có prepared release thì chỉ chạy setup + mục15. |
| [02 — Train](COde_02_Train_Baselines.ipynb) | Bank00 + cache01 | Teachers, group-OOF targets, Students/đối chứng, resume và export image-only Student. Không mở test. |
| [03 — Evaluate](COde_03_Evaluate.ipynb) | Bank/cache/study hoàn thành | Mean/std, per-class, calibration, paired comparisons, Teacher-disagreement strata và khóa protocol trước test. |

## Chuẩn bị gói upload

Từ thư mục gốc repo:

```powershell
py -3.14 -m pip install "setuptools>=68" "wheel>=0.43"
py -3.14 "Khóa luận/_support/research-notes/package_kaggle.py"
```

Output ở `_support/dist/`: `clindistill-code-source.zip`, `project_wheel/clindistill_code-0.5.0-py3-none-any.whl`, `package_manifest.json`. Không gồm raw images hay pretrained weights. Third-party binary wheels được notebook00 chuẩn bị trên Linux có cùng Python/base environment với phiên đích.

Nếu cần đóng gói prepared release riêng, thêm `--prepared-root "đường-dẫn-code_v1"`. Không thay dữ liệu gốc. Save Version và gắn Notebook Output giữa các phiên Kaggle; không bắt buộc download/upload lại bank hoặc cache lớn.

## Các profile thí nghiệm

- [research.json](configs/code/research.json): **66 fits**, 3 seeds, 5 group folds. Student no-KD, label smoothing, Visual/MM KD, confidence/correctness/visual-support/reliability KD và CNN/Transformer controls.
- [research_controls.json](configs/code/research_controls.json): **81 fits**, thêm một ảnh/mean pooling và clinical controls (demographics, shuffled-train, masked same-architecture Teacher). Cùng cache/split, run_name riêng.

Đổi `CONFIG_FILE` ở cell đầu notebook02/03 để chọn profile. Không sửa config giữa chừng rồi resume vào study cũ.

## Phạm vi và mức hoàn thành

Photo-only, một lượt khám là một mẫu có nhiều ảnh; sáu target được ghi nhận, không có healthy class. DINOv3-B/L và BGE-M3 đều frozen. Huấn luyện MIL/projection/fusion/head, chưa phải full encoder fine-tuning. MeMoSA và X-ray/CBCT không nằm trong protocol v5 này.

Clinical mặc định chỉ English age/gender/chief complaint đã sàng lọc. Chưa review thời điểm thì giữ `screened_exploratory`; không tự chứng nhận clinical có trước chẩn đoán. Pending reviews về nhãn/near-duplicate/clinical được giữ trong artifacts và report.

Bank v5/project0.5.0 cần được chuẩn bị mới. Giữ `code_v1` split nếu đã có; tạo cache v2 vì builder thay đổi. Không trộn weights/cache/checkpoints v2/v3/v4 với nghiên cứu mới.

Kiểm thử dùng synthetic features và tiny encoders để kiểm tra phần mềm. Chưa có kết quả full pretrained CODe/Kaggle được chứng minh bởi lần sửa này. Việc thêm đối chứng và báo cáo không bảo đảm reliability KD sẽ thắng, không xác nhận visual sufficiency hay độ an toàn lâm sàng.

Chi tiết phương pháp: [TECHNICAL_GUIDE.md](_support/TECHNICAL_GUIDE.md). Kết quả kiểm tra phần mềm và audit dữ liệu: [VERIFICATION.md](_support/VERIFICATION.md). Công việc theo [Issue #2](https://github.com/Dungkdl/Seminar-HCMUS/issues/2), tách phần baseline Qwen còn ở [Issue #1](https://github.com/Dungkdl/Seminar-HCMUS/issues/1).

## Kiểm tra local

Trong môi trường có dependencies tương thích:

```powershell
cd "Khóa luận"
python -m pip install -e ".[clinical,dev]"
python -m unittest discover -s _support/tests -p "test_*.py"
```

Không đưa dataset, clinical vectors/IDs, raw predictions, weights hoặc checkpoints lên Git. `_support/dist`, audit và training outputs được ignore.
