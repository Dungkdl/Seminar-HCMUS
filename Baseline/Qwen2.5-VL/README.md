# Tái lập baseline CODe với Qwen2.5-VL 3B và 7B

Thư mục lưu 7 notebook đã thực thi và bảng so sánh do người thực hiện cung cấp ngày 30/10/2026. Giữ nguyên nội dung và output của các file nguồn để đối chiếu log; việc thêm vào Git không chạy lại mô hình hoặc sửa phương pháp đánh giá.

Đây là nhánh tái lập baseline Qwen của paper CODe. Pipeline ClinDistill DINOv3/MIL/KD hiện tại nằm ở [Khóa luận/](../../Khóa%20luận/README.md).

## Notebook và thứ tự sử dụng

| File | Vai trò | Trạng thái thể hiện trong output |
|---|---|---|
| [run-all-2.ipynb](run-all-2.ipynb) | Pipeline 3B ban đầu: zero-shot, train và eval | Zero-shot hoàn thành; bước train dừng vì phiên bản `torchao` không tương thích. Dùng các notebook tách stage bên dưới cho lần train đã hoàn thành. |
| [code-finetune-3b-only.ipynb](code-finetune-3b-only.ipynb) | Fine-tune LoRA 3B | Hoàn thành 3 epochs, best adapter ở step 2619. |
| [21-3b-eval-finetuned-only.ipynb](21-3b-eval-finetuned-only.ipynb) | Đánh giá adapter 3B | Hoàn thành 1.793 dòng test. |
| [31-7b-zero-shot-only.ipynb](31-7b-zero-shot-only.ipynb) | Zero-shot 7B | Hoàn thành 1.793 dòng; có cảnh báo right-padding khi generation. |
| [40-7b-lora-finetune-only.ipynb](40-7b-lora-finetune-only.ipynb) | Fine-tune LoRA 7B | Có checkpoint đến step 1600; lần chạy được tiếp tục bằng notebook resume. |
| [resume-only-after-40-7b-finetune.ipynb](resume-only-after-40-7b-finetune.ipynb) | Tiếp tục lần train 7B bị timeout | Resume từ step 1600 và hoàn thành step 1746, đủ 3 epochs. Chỉ dùng khi có checkpoint đầy đủ của lần chạy trước. |
| [41-7b-eval-finetuned-only.ipynb](41-7b-eval-finetuned-only.ipynb) | Đánh giá adapter 7B | Hoàn thành 1.793 dòng test. |

[CODe_3B_7B_experiment_comparison.xlsx](CODe_3B_7B_experiment_comparison.xlsx) tổng hợp metric trong log, thông tin training và notebook nguồn. Các số liệu này chưa được chấm lại bằng parser đã sửa.

Bản `run-all-2.ipynb` ở đây có cùng nội dung các cell với [bản ở thư mục gốc](../../run-all-2.ipynb), đồng thời giữ output của lần chạy được cung cấp. Không cần chạy cả hai bản.

## Dữ liệu và artifact cần gắn trên Kaggle

- Các notebook tìm bộ assets offline trong `/kaggle/input`, gồm model, dependencies và dữ liệu đã đóng gói như `cls_train.json`, `cls_test.csv`, ảnh. Các tên này thuộc bundle dùng khi chạy; dữ liệu CODe gốc có thể dùng tên khác.
- Notebook eval fine-tuned cần output training tương ứng: `stage_ft3b` hoặc `stage_ft7b`. Notebook resume cần checkpoint có optimizer, scheduler và trainer state.
- Predictions CSV, adapters, pretrained weights, ảnh và dữ liệu gốc không nằm trong thư mục Git này. Log hoàn thành không xác nhận các artifact Kaggle vẫn còn được lưu.
- Notebook là bản ghi của môi trường Kaggle lúc chạy; cần kiểm tra lại assets và phiên bản thư viện trước lần thực thi mới.

## Giới hạn đã xác nhận trong review

Theo [Issue #1 — Research review](https://github.com/Dungkdl/Seminar-HCMUS/issues/1):

1. **Bộ chấm chỉ nhận tên bệnh tiếng Anh.** Test gốc có 896 đáp án tiếng Anh và 897 đáp án tiếng Trung; đáp án không khớp bị chuyển thành tập rỗng. Vì vậy exact-set accuracy và F1 trong log chưa phải metric đã được xác minh. Cần chấm lại từ raw predictions với mapping Anh–Trung và xử lý câu trả lời không hợp lệ rõ ràng.
2. **Validation chia theo dòng JSON có giao dữ liệu.** Tái tạo split seed 42 cho thấy 650/735 dòng validation có ít nhất một ảnh cũng xuất hiện trong train. Nghiên cứu chính cần chia theo bệnh nhân trước khi tạo các biến thể.
3. **1.793 dòng test là biến thể của 600 lượt khám**, mỗi lớp có 100 lượt. Không xem các biến thể là những ca độc lập khi báo metric hoặc khoảng tin cậy. Split phát hành cũng có bệnh nhân chung giữa train và test.
4. **Zero-shot 7B có cảnh báo right-padding**, trong khi eval fine-tuned đặt left-padding. Cần chạy lại suy luận với cấu hình đánh giá phù hợp trước khi kết luận chênh lệch do fine-tuning.

Giữ các kết quả này làm bằng chứng tái lập ban đầu với giới hạn được ghi rõ. Không dùng chúng để khẳng định patient-disjoint generalization, 3B vượt 7B, vượt paper hoặc chứng minh hiệu quả của ClinDistill.
