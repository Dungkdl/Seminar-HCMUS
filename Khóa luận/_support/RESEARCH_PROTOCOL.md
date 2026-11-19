# ClinDistill — protocol nghiên cứu hiện hành

Cập nhật 16/11/2026. Package 0.6.0, bank v6; CODe photo-only, frozen DINOv3/BGE-M3. Các kế hoạch/poster trước được archive trong repository; văn bản này mô tả protocol hiện hành.

## Tác vụ và phạm vi

Một mẫu là một lượt khám có ít nhất một photo hợp lệ và đúng một target mapping từ hồ sơ: Caries, Gingivitis, Malocclusion, Pulpitis, Tooth Loss, Tooth Structure Loss. Bệnh khác vẫn có thể cùng tồn tại. Nhãn vắng không tự là âm tính; không có healthy/out-of-scope class.

Prepared photo cohort hiện có4.225/917/910 train/validation/test; kết quả chỉ áp dụng cohort chọn lọc này. Nhãn lượt khám không phải annotation tổn thương từng ảnh. Không suy rộng thành chẩn đoán mọi bệnh nhân.

Split theo connected components patient + exact duplicates; near-duplicate review còn mở. Clinical chính là English age/gender/chief complaint đã screening, chưa được xác nhận pre-diagnostic. X-ray/CBCT, multi-label đầy đủ, encoder fine-tuning và MeMoSA không thuộc implementation hiện hành.

## Câu hỏi và thứ tự

1. **Visual:** single → mean → attention MIL trên cùng Student encoder/split/budget. DINO-L là Visual Teacher tham chiếu; DINO-B là Student.
2. **Clinical:** MM so với Visual Teacher, same-architecture masked control, demographics-only, shuffled-train và clinical-only. Clinical-only không nhận tensor ảnh trong API model.
3. **Transfer:** no-KD so với Visual KD và MM KD trên cùng cached features/initialization policy. OOF dự đoán từng ca train một lần; giữ toàn bộ patient/duplicate group ngoài fold train.
4. **Exploratory weighting:** confidence/correctness/visual-support/reliability, label smoothing và matched weighted-CE control. Báo zero fraction/effective cases theo lớp và bốn nhóm Teacher đúng/sai; không coi đó là ground truth visual sufficiency.

Notebook02 bắt đầu visual 12 fits, tiếp clinical 18 rồi transfer 45. Core gộp63 fits tránh một số fits lặp. Controls 87 fits là mở rộng. Nếu clinical/KD không có lợi, phân tích và báo kết quả đó; không mở test để chọn hướng hoặc ép giả thuyết đúng.

## Phương pháp hiện có

- Frozen DINOv3-B/L: CLS concat mean patch, bỏ register tokens, L2 normalize; image-only MIL heads. BGE-M3 dense normalized CLS cho text.
- Teachers độc lập; Student không nhận clinical. Clinical-only chỉ dùng text/demographics/missingness.
- KD: `CE + lambda * T² * KL(p_teacher || p_student)`; targets OOF train, không fit held-out groups.
- OOF dùng số epoch tốt nhất của full Teacher và **giữ cosine horizon config.epochs**, tái tạo prefix schedule. Validation chọn schedule chung; không phải nested CV hoàn toàn độc lập.
- Reliability: `MM-correct * entropy-confidence * [epsilon+(1-epsilon)*p_visual(y)]`, chuẩn hóa bằng mean OOF train. Đây là sample weighting, chưa phải field selection hay chứng nhận kiến thức nhìn thấy từ ảnh.
- Weighted-CE: `CE + lambda * (w/mean_train(w)) * CE`, không dùng teacher soft logits trong auxiliary loss. Nó kiểm tra giải thích cạnh tranh do weighting, không cô lập hết mọi yếu tố KD.

## Đánh giá

Macro-F1 cấp lượt khám là primary metric; báo support/per-class recall/F1, balanced accuracy, AUROC khi xác định, ECE/Brier và risk–coverage. Bootstrap theo split_group; mean/std qua seeds đo biến động tối ưu hóa, không thay thế CI lấy mẫu. Paired CI hiện là mô tả, có điều kiện trên models đã fit, chưa hiệu chỉnh multiplicity.

Validation chọn checkpoint nên metric validation là exploratory. Khóa model/config/source bằng `lock-test` sau khi chốt phương pháp rồi mới tạo shard test. Lock không phê duyệt review. Không chọn method/seed/hyperparameter theo test. Export method đăng ký trong config, seed chọn bằng validation.

Review nhãn, clinical content/timing và near-duplicates cần người có năng lực đánh giá/adjudicate. Software chỉ kiểm tra consistency/provenance và báo pending; xem [review protocol](REVIEW_PROTOCOL.md). Sửa mapping/duplicate groups phải tạo release/split mới, không sửa manifests/checksums tại chỗ.

## Bằng chứng còn thiếu

Chưa có full ClinDistill pretrained training/test experiment được kiểm chứng trong repo. Log features trước đây và synthetic tests là hai loại bằng chứng riêng. Cần fresh Kaggle bank v6/CPU→GPU activation, training/resume/export, kiểm chứng shards và đo latency/VRAM cấp lượt khám. Review chuyên môn/external validation vẫn là việc thật còn lại.

Novelty cần đối chiếu từng đóng góp với LUPI/generalized distillation, MMPKD, CPKD, TriDeNT/MIL; ghép encoders/MIL/gate chưa đủ xác lập phương pháp mới. MeMoSA nếu train head khác là kiểm chứng phương pháp trên bài toán thứ hai, không mặc định là external validation classifier CODe.

Nguồn: [CODe](https://www.nature.com/articles/s41597-026-07342-9), [MMPKD](https://arxiv.org/abs/2508.06558), [MIL](https://proceedings.mlr.press/v80/ilse18a.html). [#5](https://github.com/Dungkdl/Seminar-HCMUS/issues/5) theo dõi implementation; giữ umbrella [#1](https://github.com/Dungkdl/Seminar-HCMUS/issues/1) mở.
