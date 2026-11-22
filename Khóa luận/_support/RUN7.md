# Run 7 — So sánh loss trên Student ảnh chụp + X-quang

Student giữ hai nhánh Mean của Run 6, DINO-B frozen và cùng tối đa hai phim mỗi lượt. Cả ba Student nhận đúng cùng ảnh, train cùng cohort và dùng cùng ba seed. Multimodal Teacher giữ hai nhánh ảnh đó và thêm clinical context đã mã hóa trong cache. Teacher này cũng dùng DINO-B để kiểm tra cấu hình hiện tại.

| Student | Loss |
|---|---|
| `ce` | Cross-entropy với nhãn thật |
| `kd_vanilla` | CE + KL với dự đoán OOF của Multimodal Teacher |
| `kd_correctness` | CE + KL từ cùng Teacher, chỉ áp KD cho ca Teacher OOF đoán đúng |

Correctness gate chia trọng số cho tỷ lệ OOF đúng của toàn train. CE vẫn dùng mọi ca. Hai nhánh KD giữ T=4 và hệ số KD=1 như profile trong bank. Teacher full-fit chọn epoch trên validation. Mỗi Teacher OOF học bốn trong năm fold theo nhóm bệnh nhân/ảnh trùng, chạy đúng số epoch đã chọn với nguyên cosine horizon. Sau đó dự đoán fold còn lại. Mỗi ca train có đúng một dự đoán OOF theo seed.

Mỗi seed có 3 Student + 1 Teacher đầy đủ + 5 Teacher OOF, tổng **27 fits**. Test vẫn đóng. Clinical context là exploratory vì thời điểm ghi nội dung chưa xác minh. Phép so chính kiểm tra KD từ Teacher đa phương thức và tác động của correctness gate. So riêng với image-only Teacher OOF sẽ cần thêm một nhánh KD khác.

## Chạy trên Kaggle

Import `Khóa luận/_support/dist/run7/Run07_Xray_KD_Validation.ipynb`. Có thể tạo lại bằng:

```powershell
python "Khóa luận/_support/research-notes/package_run7.py"
```

Chọn **L4, Internet Off** và gắn đúng bốn input của Run 6.

| Input | Biến trong cell đầu |
|---|---|
| Notebook Output `notebook308b8c1b62`, bank v6 | `RESEARCH_BANK_OVERRIDE` |
| Notebook Output `code-01-features-only-v051`, cache ảnh chụp và clinical | `FEATURE_ROOT` |
| Dataset private `code-v1`, có `release.json` | `PREPARED_ROOT` |
| Dataset private `radiograph-images`, có `Images/Radiographs` | `IMAGE_ROOT` |

Đường dẫn đã điền theo Run 6. Sửa bằng Copy Path nếu Kaggle mount khác rồi chọn **Save Version → Save & Run All**. Run 7 trích lại cùng các phim đã chọn và sử dụng cache ảnh chụp/clinical cũ. Không cần gắn output Run 6. Luôn so các điểm trong cùng Run 7.

Kết quả vào `runs/code_run7_xray_kd_v1/`. Thư mục phải có `completed.json`, `mean_std.csv`, `metrics.json`, `paired_comparisons.json` và `_oof/seed_*/targets.json`. Artifacts OOF chứa thông tin cấp lượt khám và phải giữ ngoài Git.

Đọc `kd_vanilla_minus_ce` để kiểm tra KD thường, rồi `kd_correctness_minus_kd_vanilla` để đo lợi ích của gate. Dấu dương nghĩa là phương pháp đứng trước `minus` có Macro-F1 cao hơn. Kiểm tra đồng thời F1/recall của Tooth Loss, Tooth Structure Loss và Caries, cùng tỷ lệ ca mỗi lớp nhận KD trong file `targets.json`. Paired CI bootstrap theo nhóm ca có điều kiện trên các model đã fit.
