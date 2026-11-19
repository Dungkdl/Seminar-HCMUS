# Review nhãn, clinical và ảnh gần trùng

Đây là việc đánh giá dữ liệu thật bởi người có năng lực, không phải unit test. Chưa có review mới được hoàn tất trong đợt sửa phần mềm0.6.0.

## Trạng thái hiện có

Artifact local đã kiểm tra trong review: 120 phiếu clinical/label chưa điền;5.257 candidate pairs gần trùng chưa có decision; 35 buckets lớn chưa xét hết. Candidate không tự là duplicate xác nhận; không được tự gộp toàn bộ hoặc coi chúng an toàn vì exact-hash checks đã qua.

Từ thư mục `Khóa luận`:

```bash
python -m clindistill review-status --prepared ../Data/COde_processed/code_v1 --audit _support/audit/code_review_v5
```

Lệnh chỉ báo aggregate progress và giữ nguyên pending flags. Số dòng điền đủ không đồng nghĩa quyết định đúng/đã adjudicate; report không tự xác nhận chuyên môn. Không commit các CSV review có patient IDs/clinical text.

## Annotation và cohort

Người review cần xem schema/mapping nguồn và ca có/không nhiều bệnh, ghi tên/ngày/protocol và quyết định có lý do. Tách `label_mapping_correct`, `annotation_complete_for_six_targets` và `visual_grounding_review`. Một mapped target được ghi nhận không khẳng định các target còn lại âm tính hoặc có đủ bằng chứng trên photo.

Tập120 ca train là điểm bắt đầu, không chứng nhận mọi ca toàn dataset. Bổ sung phân tầng lớp hiếm/missingness/ca bất đồng nếu hợp lý; đăng ký cách lấy mẫu và phạm vi kết luận. Không lấy test prediction để chọn ca/sửa mapping theo hướng tăng điểm.

Nếu có label corrections hoặc thay cohort: adjudicate bất đồng, giữ record gốc và tạo mapping/prepared release version mới. Không sửa `release.json`/checksum hay đánh dấu PASS bằng tay để vượt guard.

## Clinical content và thời điểm

Review ai ghi field, thời điểm so với chẩn đoán/điều trị, nguồn EN và tính trung thực ngữ nghĩa. Chief complaint vẫn có thể chứa diagnosis/được bổ sung sau khám; regex không xác minh được điều đó. Diagnosis/treatment/examination/full record không là input chính.

Ghi `post_diagnosis_leakage`, reviewer/date/provenance và cách xử lý. Nếu timing không kiểm chứng được, giữ `screened_exploratory` và không tuyên bố lợi ích pre-diagnostic. Việc thông tin hậu chẩn đoán có thể thuộc một thiết lập LUPI khác không cho phép gọi nó là pre-diagnostic.

## Near-duplicate

Review candidate bằng ảnh ở nơi có quyền truy cập; bổ sung tìm kiếm các35 buckets bị bỏ qua và thừa nhận giới hạn perceptual hash. Quyết định duplicate/distinct/uncertain cần provenance; uncertain chưa giải quyết không tự PASS.

Các liên kết duplicate xác nhận đưa vào CSV `path_a,path_b,decision` với `decision='duplicate'`, trỏ `REVIEWED_DUPLICATE_LINKS` trong notebook01 và tạo `RUN_NAME`/prepared release/split version mới. Không reuse checkpoints/features của split cũ như bằng chứng cho split sửa. Nếu không thay group/mapping, vẫn lưu review/adjudication evidence bên ngoài Git và bảo vệ tính bất biến release gốc.

## Điều kiện nghiên cứu chính

Người phụ trách phải duyệt evidence/protocol và xuất reviewed release có provenance, checksums và trạng thái nhất quán. Software không biến CSV được điền hoặc một chuỗi `review_record` thành phê duyệt. `confirmatory` bị chặn khi metadata còn pending in-scope/debug/experimental; `exploratory` vẫn chạy và báo đầy đủ giới hạn. Approval cho complete multi-label negatives không phải điều kiện của task single-target; trạng thái chưa duyệt đó vẫn được giữ trong report.

Khóa test protocol là bước kỹ thuật riêng sau selection trên validation. Nó không hoàn thành expert review. Không báo clinical safety/visual sufficiency chỉ từ accuracy, Teacher disagreement, attention hoặc calibration.
