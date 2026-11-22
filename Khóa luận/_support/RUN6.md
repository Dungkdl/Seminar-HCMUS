# Run 6 — Ghép ảnh chụp và X-quang trên validation

Run 6 so ba cấu hình trên các ca có cả hai loại ảnh. Giữ cùng prepared release, split, photo cache, DINO-B, tối đa hai phim mỗi ca, CE và ba seed của Run 5. Dùng cùng lịch AdamW/cosine, tối đa 30 epoch, patience 8. Đây là phép thử kiến trúc exploratory trên train/validation.

| Cấu hình | Kiến trúc |
|---|---|
| `run5_early_fusion` | Lấy trung bình vector X-quang trước, nối vào từng vector ảnh chụp, rồi chạy Mean MIL gốc |
| `two_branch_xray_masked` | Projection và mean riêng theo loại ảnh, vector nhánh X-quang bằng 0 trước classifier |
| `two_branch_photo_xray` | Projection từng ảnh và từng phim riêng, mean theo loại ảnh rồi ghép hai vector cấp lượt khám |

Hai nhánh mới có cùng kiến trúc và khởi tạo theo seed. So với early fusion, kiến trúc hai nhánh thay cả vị trí gộp và cách phân bổ chiều latent. Chênh lệch giữa chúng đo thiết kế tổng thể. Cặp có X-quang và che X-quang đo đóng góp của modality trong kiến trúc mới.

Từ root repo tạo notebook bằng:

```powershell
python "Khóa luận/_support/research-notes/package_run6.py"
```

Import `Khóa luận/_support/dist/run6/Run06_Xray_Fusion_Validation.ipynb`. Chọn **L4, Internet Off**, rồi gắn đúng bốn input của Run 5:

| Input | Loại | Biến trong cell đầu |
|---|---|---|
| `notebook308b8c1b62` | Notebook Output có bank v6 | `RESEARCH_BANK_OVERRIDE` |
| `code-01-features-only-v051` | Notebook Output có cache `research_features_v3` | `FEATURE_ROOT` |
| `code-v1` | Dataset private có `release.json` | `PREPARED_ROOT` |
| `radiograph-images` | Dataset private có `Images/Radiographs` | `IMAGE_ROOT` |

Các đường dẫn mặc định lấy từ notebook Run 5 đã chạy. Dùng **Copy Path** nếu mount khác. Chạy **Save Version → Save & Run All**. Notebook dùng lại photo cache và trích lại các phim đã chọn với cùng encoder và preprocessing. Có 9 fits, không OOF hoặc KD. Kết quả vào `runs/code_run6_xray_fusion_v1/`. Nếu chạy lại trên cùng thư mục, checkpoints đã hoàn thành được kiểm tra hash và dùng lại. Fit hai nhánh đang dở sẽ chạy lại từ đầu.

Đọc `mean_std.csv`, `paired_comparisons.json` và `metrics.json`. Kiểm tra hai contrast `two_branch_photo_xray_minus_run5_early_fusion` và `two_branch_photo_xray_minus_two_branch_xray_masked`. Dấu dương có lợi cho nhánh mới dùng X-quang. CI bootstrap nhóm ca được tính có điều kiện trên các model đã fit. Đối chiếu F1/recall và số ca đúng cho Tooth Loss, Tooth Structure Loss và Caries trước khi chốt kiến trúc để thử loss KD.
