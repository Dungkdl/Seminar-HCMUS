# Kỹ thuật COde research v4

Tài liệu giải thích code hiện có trong `clindistill/research`, cấu hình `configs/code/research.json`.
Đây là implementation **Stage 1** của plan; không gọi frozen-feature training là full encoder fine-tuning.
Luồng thao tác ngắn ở [README](../README.md).

## 1. Bài toán và split

Một sample = một lượt khám có ≥1 photo và đúng một target được ghi nhận trong sáu target.
Nhãn này chưa chứng minh bệnh nhân chỉ có một bệnh. Không có healthy class; multi-label chưa được duyệt.
Cohort hiện tại: 4.225 train, 917 validation, 910 test. Các hệ thống dùng đúng cùng cohort/split.

Giữ `code_v1/release.json` và manifests; không chạy lại chia tập trong notebook02.
`split_group` là connected component bệnh nhân và ảnh trùng byte/pixel, mạnh hơn chỉ group theo patient_id.
Train chỉ mở cache train/validation. Test có entry point riêng và cần ghi protocol đã khóa.
Metadata audit của toàn dataset không được dùng để tối ưu model theo test.

## 2. Feature preparation nằm ở notebook01

`research/data.py::prepare_features` xác minh release và local weights, rồi trích xuất frozen features.
Không đưa labels vào foundation encoders, không fit scaler/PCA trên validation/test.

- Ảnh: EXIF → RGB → resize/pad giữ tỷ lệ, bilinear, 224×224; normalize theo processor DINO hoặc ImageNet cho CNN.
- Chọn tối đa8 photo/ca bằng vị trí filename cách đều, cố định cho full fit/OOF/Student/Teacher.
  Đây không phải bộ canonical dental views được chuyên gia đánh dấu.
- DINO-B hidden768, DINO-L hidden1024. Descriptor = concat(CLS, mean patch tokens),
  **bỏ register tokens**, L2 normalize → Student1536 chiều, Teacher2048 chiều.
- ResNet50 trước `fc`:2048 chiều; Swin-B trước `head`:1024 chiều; L2 normalize.
- BGE-M3 dense = **normalized CLS1024**, không dùng BiomedBERT mean pooling.
  Tokenize tối đa256 tokens; đếm trường bị truncation trong manifest mỗi split.
- GPU encoder ảnh dùng BF16 khi hỗ trợ, nếu không FP16; vectors cache float16, heads tính float32.
  Text encoder frozen/eval, lưu dense float32. Không có augmentation ảnh ngẫu nhiên trong protocol frozen cache.

Ảnh được encode theo microbatch8 và checkpoint chunk256 ảnh có checksum. Đổi bytes ảnh/model/policy/source builder
sẽ từ chối trộn cache. Cùng protocol có thể bổ sung shard test sau này mà không đổi identity train/validation.

Nguồn: [DINOv3 models](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m),
[Transformers DINOv3 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/model_doc/dinov3),
[BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).
L2 pooling/resize-pad là protocol thực nghiệm của dự án, không tuyên bố là preprocessing tối ưu được các paper bảo đảm.

## 3. Clinical whitelist và giới hạn leakage

Mặc định chỉ age, gender, chief_complaint tiếng Anh, lấy theo case_id/patient_id khớp manifest.
Không lấy diagnosis, anomalies, treatment, management, full patient_record, examination hay radiographic interpretation.
Age: hợp lệ0–120, chia120, thêm cờ missing. Gender: one-hot male/female/unknown. Tổng5 demographic dimensions.
Chief complaint thiếu → vector text0; cờ clinical_present tính từ text hoặc demographics có sẵn.

Trường có pattern chẩn đoán trực tiếp bị loại **toàn trường**, ghi clinical_leakage_audit.csv;
không sửa một câu y khoa rồi giả định câu đã an toàn. Không dùng y của sample để quyết định whitelist hay rewrite text.
`clinical_whitelist.yaml` là JSON hợp lệ theo YAML1.2, ghi fields/language/screening/review status.
Tỷ lệ thiếu/chặn có thể tính từ audit; không xuất raw clinical text trong báo cáo public.

Regex chỉ là sàng lọc. Code không xác nhận ai ghi clinical vào thời điểm nào; default
clinical_temporal_verified=false, status=screened_exploratory. Chỉ thêm present_illness/past_medical_record
khi có clinical_temporal_verified=true **và clinical_review_record**; việc khai báo phải dựa trên review thực.
Không diễn giải teacher benefit là ích lợi pre-diagnostic đã chứng minh khi chưa hoàn tất review này.

## 4. Kiến trúc

Gated MIL có projection Linear → LayerNorm → GELU cho mỗi ảnh h_i. Attention:

`a_i = softmax_i(wᵀ[tanh(Vh_i) ⊙ sigmoid(Uh_i)])`, `z = Σ a_i h_i`.

Mask padding trước softmax; mỗi bag phải có ≥1 ảnh. MIL bất biến theo thứ tự ảnh.
Width256, dropout0.2; classification head trả sáu logits. Không coi attention là giải thích tổn thương đã được xác nhận.

- Visual Teacher: frozen DINO-L descriptors → MIL → head.
- MM Teacher: cùng DINO-L/MIL, `t = TextProjection([BGE, demographics])`,
  `g = sigmoid(W[z,t]) × clinical_present`, `fused = LayerNorm(z + g×t)` → head.
  Text được mask khi clinical thiếu. Có lựa chọn concat cho thí nghiệm riêng, mặc định gated.
- Student: frozen DINO-B descriptors → MIL → head, API từ chối clinical/demographics trong forward.
  Targets/labels chỉ xuất hiện trong loss, không được nối vào visual features.
- ResNet/Swin controls: frozen encoder tương ứng → cùng gated MIL/head.

Các encoder có dữ liệu/recipe pretraining khác nhau; so sánh ResNet/Swin/DINO không tách riêng tác động kiến trúc.

BGE/DINO không được optimizer cập nhật trong v4. Biểu diễn cached không phụ thuộc nhãn huấn luyện.
Nếu chuyển sang fine-tune foundation encoder trong tương lai, phải tạo encoder riêng theo fold;
không dùng features từ encoder đã fine-tune trên toàn train để giả làm OOF.

## 5. OOF targets

Mỗi seed: fit Visual/MM Teachers trên train, chọn best epoch theo macro-F1 validation độc lập.
Sau đó StratifiedGroupKFold5 trên **train split_group**. Mỗi fold fit lại head/MIL/fusion từ đầu,
chỉ dùng các nhóm training của fold. Số epoch cố định bằng best_epoch của full-fit Teacher tương ứng.
Không tính metric held-out fold trong fit và không chọn checkpoint bằng nhãn held-out fold.

Dự đoán fold giữ lại tạo logits cho từng ca train đúng một lần. Lưu IDs, fold, train/heldout groups,
checksum checkpoint, source/cache fingerprint trong `_oof/seed_*/targets.json` và `targets.npz`.
Validation được dùng chọn training schedule chung; đây không phải nested CV độc lập hoàn toàn với validation.
Test chưa được mở. Inference Student không gọi bất kỳ Teacher nào.

## 6. Loss và weighting

Teacher và no-KD: CE trên nhãn single-target. Students KD:

`L = CE(y, student) + λ × T² × KL(softmax(teacher/T) || softmax(student/T))`.

Mặc định λ=1, T=4, hướng KL Teacher→Student. Teacher logits detach.
KD Visual lấy Visual OOF; KD Multimodal lấy MM OOF, weights=1.
Confidence KD lấy MM OOF với `c=1−H(p_m)/log(K)`.

Reliability KD:

`w = 1[argmax(p_m)=y] × c × [ε+(1−ε)×p_v(y)]`, ε=0.1.

Gate probabilities dùng temperature1, riêng với temperature4 của KL. y chỉ là nhãn train của ca có logits OOF.
Loss weighted = `mean_batch[(w / mean_OOF_train(w)) × KL_T]`.
**Không** dùng mean_batch(w) làm mẫu số: batch1 sẽ làm trọng số bị triệt tiêu.
Microbatches được nhân số mẫu/thực số mẫu trong accumulation window, kể cả remainder cuối epoch.
Nếu mean weights=0 thì báo lỗi để kiểm tra Teacher; không tự fallback thành CE hay bỏ thí nghiệm.

Correctness gate có thể bỏ mẫu khó/rare classes, là giả thuyết cần kiểm chứng. Lưu mean/zero fraction theo class.
Các đối chứng vanilla/confidence là bắt buộc để tách lợi ích weighting khỏi việc chỉ đổi backbone.
Chưa có cơ sở nói proposed weighting sẽ thắng.

Optional `kd_feature` có head alignment sang **mean frozen DINO-L image descriptor** và cosine loss,
weight theo reliability. Không align fused latent của nhiều Teacher folds vì các latent có hệ tọa độ học khác nhau.
Method này không chạy mặc định; phải đăng ký thành study khác nếu thêm vào so sánh.

## 7. Huấn luyện, resume và selection

AdamW LR1e-3, weight decay1e-4, cosine decay; batch32, accumulation1; clip grad norm5.
30 epochs tối đa, patience8 cho full fits. CE không dùng class weighting/sampling trong protocol mặc định,
để so sánh các loss trên cùng objective phân phối train. Macro-F1 và per-class metrics làm rõ imbalance.
Seeds2026/2027/2028, dataloader shuffle có seed theo epoch, deterministic mode cảnh báo nếu op chưa deterministic.

`best.pt`: head model/spec + best epoch/validation metrics. `last.pt`: model+optimizer+scheduler+RNG/history.
Resume từ epoch boundary. Check config/source/cache/OOF identity; không resume vào study khác.
Kết quả saved seeds đã hoàn thành được dùng lại; không train lại chỉ vì Run All lần nữa.
Không cam kết bitwise giống nhau khi đổi GPU/PyTorch; CPU epoch-resume đã được kiểm thử chính xác.

## 8. Đánh giá và xuất Student

9 hệ thống ×3seeds =27 full fits;30 OOF fits bổ sung. Report mean/std của các seed đã đăng ký,
không chọn seed có test tốt nhất. Per-class recall/AUROC, macro-F1, accuracy, Brier, ECE10bins,
confusion matrix, risk–coverage, case/error/subgroup analysis.

Group bootstrap resample split_group, giữ các visit liên quan đi cùng nhau.
Paired bootstrap so sánh đúng cùng cases/seeds; CI điều kiện trên mô hình đã fit, không phải CI cho retraining.
Bốn contrasts: MM−Visual Teacher, Reliability−NoKD, Reliability−VanillaMM, Reliability−Confidence.
CI hiện chưa hiệu chỉnh nhiều phép so sánh: diễn giải mô tả, không tự gọi significance xác nhận.

Test mặc định khóa; notebook03 cần completed study và test_review_record mô tả protocol đã cố định/giới hạn còn lại.
Không tự đánh dấu các pending_reviews thành PASS khi thêm record này.
Khi thêm test features, cache train/validation phải giữ nguyên checksum.

Export chọn method đã đăng ký, seed tốt nhất theo validation. Gồm `student_head.pt`, `image_encoder/`, manifest.
`predict_student(export_root, photo_paths, image_root, work_dir)` nhận raw photos; không có clinical/label argument.
Gói inference này cần package source cùng version; weights tiếp tục chịu điều khoản DINOv3.

## 9. Kiểm chứng và phần chưa làm

Tests dùng synthetic visits/features và tiny randomly initialized DINOv3/XLM-R models để kiểm tra backend,
local-only I/O, dimensions/pooling, mask, KL/gradients, group folds, checksums, resume, report/export/inference.
Notebook02 được thực thi cả các code cells với fixture CPU. Tests không chứng minh pretrained performance.
Bank00 có thêm rehearsal mini COde thật bằng pretrained models thật; cần chạy trong môi trường Kaggle của bạn.
Chưa có full COde DINO results hay kiểm chứng tốc độ/VRAM của phiên L4 thực trong lần sửa code local này.

Chưa làm Stage2/3 partial/full encoder fine-tuning, hyperparameter sweeps, exhaustive clinical/MIL/robustness ablations,
MeMoSA external validation hoặc chứng nhận clinical timing/annotation. Việc này phải được ghi là giới hạn,
không đổi tên các checks phần mềm thành “đã hoàn thành khóa luận/bài Q1”.
