# Kaggle — test Run 4 và thăm dò X-quang Run 5

Dùng L4, Internet Off và Notebook Output bank `clindistill_research_v6` từ notebook `notebook308b8c1b62`. Gắn cache của `code-01-features-only-v051`. Hai notebook mới nằm trong `_support/dist/run4-run5/` sau khi chạy `python "Khóa luận/_support/research-notes/package_run4_run5.py"` từ root repo.

## Run 4 — test đã khóa

Run 4 đã train 57 lượt chính và 30 OOF fits. Test dưới đây đo các checkpoint đã chọn trên validation. Xem `student_mean` là baseline chỉ ảnh và giữ `kd_reliability`, `student_no_kd` và `kd_multimodal` để diễn giải KD. Notebook đánh giá tất cả 19 cấu hình × 3 seed đã đăng ký. Đánh giá này là exploratory vì review nhãn, dữ liệu clinical và ảnh gần trùng vẫn mở.

Import `Run04_Locked_Test.ipynb`. Dùng **Copy Path** trong Kaggle Input để lấy từng đường dẫn. Làm ba phiên riêng, mỗi phiên chọn Save Version → Save & Run All. Giữ nguyên `REVIEW_RECORD` trong cả ba phiên.

| Phiên | Input ngoài bank | Biến cấu hình | Output cần giữ |
|---|---|---|---|
| 1 | Cache train/validation và Notebook Output Run 4 | `MODE='lock'`. `FEATURE_ROOT` là cache cũ. `STUDY_ROOT` là thư mục `runs/code_controls_v6` chứa `completed.json` | `runs/code_controls_v6/` gồm checkpoint, metadata và `test_protocol.json` |
| 2 | Cache train/validation, output phiên 1, prepared release `code_v1`, ảnh CODe gốc | `MODE='features'`. `FEATURE_ROOT` vẫn là cache cũ. `PREPARED_ROOT` là thư mục chứa `release.json`. `IMAGE_ROOT` là thư mục cha của `Images/Photographs`. `TEST_PROTOCOL_PATH` là file khóa từ phiên 1 | `run4_features_with_test/` gồm `train.npz`, `validation.npz`, `test.npz` và manifest |
| 3 | Output phiên 1 và 2 | `MODE='evaluate'`. `STUDY_ROOT` trỏ thư mục study đã khóa từ phiên 1. `FEATURE_ROOT` trỏ `run4_features_with_test` từ phiên 2 | `reports/code_controls_v6/test/` |

Ví dụ gán một đường dẫn trong cell cấu hình: `STUDY_ROOT = Path('/kaggle/input/notebooks/OWNER/NOTEBOOK/runs/code_controls_v6')`. Phiên 2 chỉ trích shard test và giữ bytes train/validation. Dùng bank gốc để source/checkpoint hashes khớp. Nếu chỉ có ảnh làm phẳng, dùng notebook Features Only của bank với `PREPARE_SPLITS=['test']`, bản copy cache và protocol từ phiên 1. Adapter cũ cần index hash kèm ảnh đúng release. Notebook test mới nhận cây ảnh gốc.

Đọc `mean_std.csv`, `paired_comparisons.json`, `status.json` và `<method>/seed_<seed>/metrics.json`. Dự kiến 910 lượt khám từ prepared release hiện tại. Khi đã xem kết quả, test này được coi là đã mở. Nếu dùng điểm đó để đổi thiết kế Run 5, cần một holdout độc lập để xác nhận kết quả mới.

## Run 5 — ảnh chụp + X-quang, validation

Import `Run05_Xray_Validation.ipynb`. Gắn bank, cache ảnh chụp train/validation, prepared release `code_v1` và X-quang gốc theo đường dẫn `Images/Radiographs/...`. `PREPARED_ROOT` trỏ tới thư mục chứa `release.json`. `IMAGE_ROOT` trỏ tới thư mục cha của `Images/Radiographs`. Gắn các input chứa dữ liệu theo quyền truy cập của CODe. `FEATURE_ROOT` giữ cache cũ từ Run 1–4. Notebook chỉ đọc train và validation.

- Cohort gồm các lượt khám có cả photo và X-quang. Dự kiến 2.098 train và 429 validation. Notebook kiểm tra ID, nhãn, group và thứ tự ảnh chụp với cache, rồi kiểm tra trùng byte X-quang qua hai tập.
- Dùng lại photo features và DINO-B của bank. Với X-quang, chọn tối đa hai file theo vị trí cách đều trong danh sách tên đã sắp xếp. Đây là quy tắc xác định, không phân biệt loại radiograph. Mỗi lượt lấy trung bình descriptor X-quang rồi nối vào từng descriptor ảnh chụp.
- Hai mô hình cùng kiến trúc Mean MIL. Nhánh photo baseline có phần X-quang đặt bằng 0, nhánh còn lại nhận đặc trưng X-quang thật. Cùng seed, batch, 30 epoch tối đa, patience 8 và checkpoint theo Macro-F1 validation. Không clinical, KD hay test.
- Chọn **Save Version → Save & Run All**. Output là `runs/code_run5_xray_v1/`, gồm 6 fits, `paired_cohort.json`, `mean_std.csv`, `metrics.json`, `paired_comparisons.json`, `completed.json` và cache chunks. Xem `photo_xray_minus_photo` để đọc mức chênh và khoảng tin cậy bootstrap theo split group.

Đây là pilot frozen DINO-B trên X-quang với tối đa hai file. Phạm vi kết quả là các lượt khám có hai modality và cách chọn ảnh đã đăng ký. Nếu pilot có tín hiệu, các lượt sau có thể kiểm tra thêm loại ảnh hoặc ngân sách ảnh trước khi thử Clinical/KD.

### Gói X-quang riêng nếu Kaggle chưa có ảnh gốc

Từ thư mục `Khóa luận`, dùng Python đã cài dependencies ClinDistill và chạy:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python "_support/research-notes/package_run5_radiographs.py" --prepared-root "../Data/COde_processed/code_v1" --image-root "E:/Hoc Tap/Data/COde-Dataset"
```

Gói local ở ignored `_support/dist/run4-run5/` chỉ lấy các file X-quang đã chọn cho train/validation. Giải nén ZIP thành private Kaggle input và giữ cấu trúc `Images/Radiographs`. Gắn thêm prepared release hiện có để join với cache.
