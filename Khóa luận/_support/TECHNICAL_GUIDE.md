# Kỹ thuật ClinDistill COde v2

Tài liệu mô tả code hiện có, không phải kết quả đạt được. Luồng thao tác Kaggle nằm ở [README](../README.md).
Bản v2 thay trainer visual-only v1 bằng baselines + Teachers + KD, giữ ranh giới 01 xử lý dữ liệu / 02 train / 03 đánh giá.

Notebook 02 hiện mặc định `EXPERIMENTS=['kd_selective']`, seed 2026/2027/2028:
Teacher V/H và Student selective KD, tổng 9 lượt. Baselines và các đối chứng vẫn có trong source,
chỉ chạy khi chủ động chọn. Một cell cấu hình điều khiển toàn notebook; Run All thực hiện preflight,
phục hồi suite phù hợp từ output phiên trước, train, tổng hợp validation và xuất Student từng seed.
`training_report/` có preflight, kế hoạch, tiến độ, bảng mean/std và learning curves; `exports/` có Student.
Logic trainer/model vẫn trùng bộ offline đã sửa AMP, nên thay notebook không bắt tải lại bundle.

## Bổ sung: phiên GPU không có Internet

Notebook `COde_00_Kaggle_Offline_Setup.ipynb` chuẩn bị một lần trên Linux Kaggle online trước luồng 01–03.
Helper `_support/kaggle_offline.py` chỉ dùng standard library khi import, nên chạy inventory/install trước khi import thư viện ML.
Mỗi notebook nghiên cứu có cell bootstrap đầu tiên để tìm bundle, kiểm tra checksum/môi trường và cài wheels với `--no-index`.
ResNet dùng state_dict local; BiomedBERT dùng local folder và `local_files_only=True`; bật HF offline flags.

Wheel `clindistill_code-0.2.0-py3-none-any.whl` chỉ đóng gói module của project; thư viện native cần wheel đúng nền tảng.
Notebook 00 giữ scientific/torch/torchvision stack, tải Transformers 4.57.1 + dependencies bằng pip download `--only-binary=:all:`.
Source bundle có wheel project portable đã build nên không cần compiler khi cài project.
Gói offline cuối gồm wheels, weights và source; manifest kiểm tra SHA256, Python minor, OS/architecture và base package versions.

Bộ mini có 12 train + 12 validation (2/lớp/tập), sao chép đủ bag của ca được chọn, tổng 128 ảnh; không truy cập test.
Release riêng có `debug_subset=true`, không đổi release nghiên cứu. Auto-discovery notebook 01–03 bỏ qua mini.
Smoke local từ wheel đã cài dùng tiny_debug; smoke trong notebook 00 mới dùng pretrained thật với sockets bị chặn.
Full offline Linux/GPU verification cần chạy notebook 00 trên môi trường Kaggle thực; không suy ra từ Windows CPU smoke.

## 1. Bài toán, đơn vị dữ liệu và protocol

Một sample là **một lần khám** (`case_id`), có một bag ảnh và một nhãn mục tiêu được ghi nhận.
Một bệnh nhân có thể có nhiều lần khám. Ảnh trùng có thể xuất hiện dưới nhiều patient IDs nên chỉ split theo patient chưa đủ.
Notebook 01 tạo `split_group` bằng liên kết patient/ảnh trùng chính xác, rồi khóa train/validation/test.

- Bảng nguồn 8.775 lần khám / 4.800 patient IDs, 57.994 ảnh đã audit ở lần chuẩn bị trước.
- CSV sạch 22 cột tiếng Anh, giữ đủ 8.775 dòng. Không tạo bệnh nhân mới từ QA JSON.
- Cohort single-target có photo: 4.225 train, 917 validation, 910 test.
- Cả baseline photo và photo+X-ray dùng **cùng cohort photo-available**, không lọc X-quang để làm baseline dễ hơn.
- Sáu lớp theo `release.json`: Caries, Gingivitis, Malocclusion, Pulpitis, ToothLoss, ToothStructureLoss.
- Nhãn thiếu không tự coi là âm tính. Multi-label candidate chưa được trainer này dùng; không có lớp healthy.
- “Single-target recorded” không có nghĩa bệnh nhân chỉ mắc một bệnh.

Train đọc `student_single_train.jsonl`, `student_single_validation.jsonl`. Teacher/KD thêm cache từ `teacher_context_*`.
Không đọc JSON QA gốc hoặc tự chia lại tập trong train. `verify_release(splits=...)` bỏ qua nội dung manifest test trong training;
metadata chuẩn bị có thống kê toàn bộ tập, không dùng kết quả model test để quyết định cấu hình.

Release `code_v1` còn experimental. Audit đã xử lý trùng byte/pixel; near-duplicate, label mapping và clinical leakage
còn review thủ công. Chất lượng split/nhãn là giới hạn thật của nghiên cứu, không biến mất khi đổi sang GPU tốt.

## 2. Từ ảnh thành tensor

`clindistill/data.py`:

1. Load theo đường dẫn tương đối với folder chứa `Images/`; chặn path đi ra ngoài image root.
2. Áp EXIF orientation, chuyển RGB, resize/pad giữ tỉ lệ về 224×224. Không random crop vùng có thể chứa bệnh.
3. Train chỉ đổi brightness/contrast photo trong [0.9, 1.1]; X-quang không áp hai augmentation này.
4. Normalize theo ImageNet. Baseline một ảnh chọn filename đầu, không được gọi là góc chụp chuẩn lâm sàng.
5. Train multi-image lấy tối đa 8 photo, tối đa 2 X-quang cho dual modality; sample ảnh theo seed/epoch/case.
6. Validation/test lấy ảnh trải đều theo thứ tự filename cố định.
7. Collate thành images `[B,N,3,H,W]`, mask `[B,N]`, modalities `[B,N]`, labels `[B]`.

Padding không đi vào encoder hoặc BatchNorm. KD dùng chính tensor ảnh Student đã nhận để chạy frozen Teachers,
không lấy một phiên bản augmentation khác và không cấp thêm ảnh cho Teacher.
Ảnh ngoài số lượng cap bị bỏ theo quy tắc có ghi lại trong config, không tuyên bố sử dụng mọi ảnh của bag ở một lượt forward.

## 3. Visual models

`clindistill/models.py` dùng ResNet-50 pretrained ImageNet, bỏ fc, projection 2048→256 + ReLU.
Photo và radiograph dùng chung encoder; modality embedding hai loại được cộng vào feature, khởi tạo 0.
Đây không phải hai backbone tách biệt, và không có CBCT 3D encoder.

| experiment | Pooling |
|---|---|
| single_photo | Một ảnh, mean |
| photo_mean | Mean các ảnh hợp lệ |
| photo_attention | MIL: Linear(256,128) → tanh → Linear(128,1), masked softmax, weighted sum |
| photo_set_attention | Hai self-attention blocks, 4 heads, không positional encoding; một learned query attention pooling |
| photo_xray_attention | MIL trên photo + X-quang có sẵn |

Set pooling là triển khai gọn theo tinh thần SAB/PMA, không tuyên bố khớp mọi chi tiết của code paper.
Hai block dùng residual/layer norm/feed-forward, mask padding ở keys. Output không đổi khi hoán vị ảnh lúc eval.
Classifier Student: dropout 0.2 → linear sáu lớp. Attention được xuất khi inference theo thứ tự ảnh được loader chọn;
đây là trọng số bag, không phải heatmap vị trí tổn thương.

## 4. Clinical encoder nằm ở bước chuẩn bị

`clindistill/clinical.py`, mục 14 notebook 01:

- Mặc định `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (trước đây tên PubMedBERT).
- Encoder frozen/eval, không fine-tune; mean last hidden state theo attention mask, có bao gồm special tokens.
- Mỗi trường riêng, tối đa 192 token, batch mã hóa 32. Lưu số trường nonempty/truncated theo split.
- Năm trường theo thứ tự cố định: examination, radiographs_examination, chief_complaint, present_illness, past_medical_record.
- Chỉ dùng `clinical_screened`; không dùng raw diagnosis, treatment, target label, age/gender làm input Teacher.
- Rỗng → vector 0 + available=False. Một case không có trong cache → lỗi, không giả là hồ sơ rỗng.
- Cache `[cases,5,D]` với D lấy từ encoder config (BiomedBERT base thường 768).

Cache có metadata model/field/pooling/max_length/release ID, SHA256 assets encoder, checksum shard và context nguồn.
Train/validation tạo trước; test chỉ thêm khi đánh giá. Thêm test không đổi fingerprint train/validation.
Khi thêm test, phải dùng đúng encoder assets cũ; không chấp nhận model revision đã thay đổi.
Đường dẫn có thể khác giữa local/Kaggle nhưng bytes của cache phải khớp.

`debug=True` dùng vector từ hash văn bản chỉ để kiểm tra pipeline không cần tải model. Runtime chặn debug cache ở backbone nghiên cứu.
Debug vectors không có ý nghĩa ngữ nghĩa và không phải kết quả BiomedBERT. Không phát hành debug cache như prepared data nghiên cứu.

## 5. Teachers: cấu trúc và missingness

`clindistill/distillation.py`:

| Method | Input |
|---|---|
| teacher_clinical | Clinical fields, không có visual encoder |
| teacher_full | Visual bag + năm clinical fields được phép |
| teacher_v | Visual bag + examination / radiographs_examination |
| teacher_h | Visual bag + chief_complaint / present_illness / past_medical_record |

Mỗi field có Linear(D,256)+ReLU riêng. Sau mask, lấy mean các field available. Concatenate pooled text,
five-bit availability mask và pooled visual nếu có; fusion Linear→ReLU→dropout→classification.
Đây là **late fusion**, không phải cross-attention token ảnh–text.

Train Teacher dùng field dropout 0.2; mask sau dropout được truyền vào head. Eval/KD không áp dropout.
Không có field vẫn được train CE: multimodal Teacher dùng ảnh; clinical-only học prior/missingness.
Khi KD, một Teacher không có clinical input hợp lệ của nhóm đó bị bỏ khỏi distillation cho ca tương ứng.

Quy tắc tránh gán tính “nhìn thấy” quá mức:

- Nhánh photo-only mask radiographs_examination, kể cả cache có text đó.
- Nhánh dual chỉ dùng mô tả X-quang nếu ca có ảnh X-quang thật trong batch.
- V là tên nhóm trường mang giả thuyết gần nội dung khám/ảnh, không phải nhãn visualizability đã xác nhận.
- Examination vẫn có thể chứa xét nghiệm, cảm giác đau hoặc kết luận không suy ra từ ảnh. Screen bằng regex không đủ chứng nhận an toàn leakage.

Teacher full/V/H có ảnh nên là upper reference khi có clinical context, không mặc định là upper bound toán học.
Teacher không đủ tốt có thể làm Student kém hơn; phải báo cả kết quả Teacher trước khi diễn giải KD.

## 6. Loss Student

Ký hiệu Student logits s_i, Teacher logits t_ig, nhiệt độ T=4, a_ig∈{0,1} cho nhóm g có clinical hợp lệ.
Teacher được train riêng, chọn bằng validation, đóng băng `eval()` và `requires_grad_(False)` khi train Student.

```text
q_ig = softmax(t_ig / T)
p_i  = softmax(s_i / T)
w_ig = a_ig * weight_g / sum_h(a_ih * weight_h)   (tổng 0 thì toàn bộ w=0)
L_logit = mean_i [sum_g w_ig * T² * KL(q_ig || p_i)]
L = CE(y, s) + kd_weight * L_logit
```

PyTorch nhận log_softmax Student làm input, softmax Teacher làm target; tính KL float32 trong AMP.
Không đảo hướng KL. Không tính mean trên cả số lớp rồi vô tình làm nhỏ loss C lần.
Ca không có clinical có KD=0, vẫn được CE và vẫn nằm trong mẫu số toàn batch.

- `kd_clinical`: một Teacher text-only; baseline privileged clinical-only.
- `kd_full`: một Teacher multimodal full; vanilla logit KD.
- `kd_equal`: V/H Teachers, trọng số 1:1.
- `kd_selective`: V/H Teachers, mặc định 3:1.
- `kd_higher_h`: V/H Teachers, 1:3 làm đối chứng hướng ưu tiên.

Đóng góp selective là **giả thuyết trọng số theo nhóm trường**. Có thể bị bác bỏ bởi equal/full hoặc reverse control.
Không gọi đây là reproduction MMPKD/CPKD hoặc thuật toán tự học visualizability.

Ablation representation (`kd_full` Teacher):

```text
L_feature = mean_i [a_i * mean_d (normalize(zS_i)-normalize(zT_i))²]
L_relation = mean_{i != j, a_i=a_j=1} (cos(zS_i,zS_j)-cos(zT_i,zT_j))²
L = CE + kd_weight*L_logit + feature_kd_weight*L_feature + relation_kd_weight*L_relation
```

zT là fused hidden của Teacher; zS là pooled Student feature, cùng dimension.
Hai feature spaces không có đảm bảo cùng tọa độ: feature loss là ablation cần kiểm chứng, không mặc định phải tốt.
Relation tính trong microbatch; gradient accumulation không tăng số cặp relation. Dưới hai ca available thì relation=0.
Hai hệ số representation mặc định 0; suite bật riêng từng cái bằng 1 ở `kd_feature` / `kd_relation`.
`train_kd` ghi tổng phần distillation đã nhân hệ số; `train_ce` ghi CE.

## 7. Training loop, GPU và tái lập

- AdamW lr=1e-4, weight_decay=1e-4; CosineAnnealingLR theo epoch, T_max=40.
- Batch=4 ca, gradient accumulation=4. Cuối epoch xử lý cả window thiếu; scale theo số ca thực trong window.
- AMP CUDA fp16 + GradScaler, unscale trước clip global grad norm=1; CPU không bật AMP.
- 40 epochs tối đa, early stop sau 8 epoch không cải thiện validation macro-F1.
- CE mặc định không weighted. Có option class_weighted_loss dùng counts từ train; không dùng tần suất test.
- All-six-class macro-F1, lớp vắng có F1=0; chọn best khi lớn hơn strict, không chọn lại bằng test.
- Seed 2026/2027/2028; Python/NumPy/torch/CUDA seeds; deterministic algorithms warn_only.
- Seed của loader theo epoch; ảnh lấy mẫu/augment theo seed+epoch+case. Worker Windows=0, Kaggle dùng config workers=2.
- Nếu freeze_backbone=True thì cả weights và BatchNorm stats của encoder frozen.
- `run_config.json` lưu config, code hash, release, cache/Teacher hashes, train/val IDs, package versions và GPU/VRAM.
- History có thời gian epoch; summary có peak allocated GPU memory. VRAM báo từ runtime, không đoán theo ảnh chụp màn hình.

Grad accumulation không làm BN tương đương batch lớn. CUDA/driver khác có thể không tái lập bitwise dù cùng seed;
kiểm thử bitwise resume ở CPU chỉ chứng minh logic checkpoint trên môi trường đó.

`best.pt`: model/config/class_names/release/fingerprint, không có optimizer để tiết kiệm dung lượng.
`last.pt`: thêm optimizer/scheduler/scaler/RNG/history để resume cuối epoch. Mất giữa epoch sẽ chạy lại epoch đó.
Không dùng checkpoint chưa tin cậy (training checkpoint dùng pickle để chứa RNG/optimizer).

`run_suite` tự thêm Teachers phụ thuộc; khóa base config + code + release trong protocol.json.
Có thể đổi SELECTED để chạy thêm thí nghiệm, nhưng thay protocol cần RUN_NAME mới.
Mỗi Student seed s dùng Teacher seed s. Nếu Teacher config/cohort/ảnh/seed/cache khác, runtime từ chối.

## 8. Đánh giá và cách đọc kết quả

Notebook 03 không train. Student KD được load bằng VisualBagClassifier, không load Teacher/text encoder/cache.
Đánh giá Teacher cần đúng clinical cache; nếu test chưa có shard thì phải chuẩn bị trước.

| Output | Ý nghĩa |
|---|---|
| accuracy, macro-F1 | Kết quả phân loại sáu lớp |
| balanced_accuracy_observed_classes | Mean recall chỉ các lớp có support trong split được đánh giá |
| per_class_recall / AUROC / support | AUROC null khi không có positive hoặc negative |
| macro_auroc_defined_classes | Mean AUROC của lớp tính được, luôn đọc cùng support |
| Brier | Mean theo case của tổng bình phương sai khác C lớp, không chia thêm cho C |
| ECE | 10 bin confidence bằng nhau; không dùng temperature calibration fitted trên test |
| bootstrap CI | Percentile 95%, resample split_group giữ nguyên các case trong nhóm |
| mean/std | Mean và sample std qua seed; std một seed là undefined |
| paired CI | So sánh trên cùng case/group/seed; giữ cùng resample cho hai mô hình |
| risk–coverage / AURC | Sắp max probability giảm dần; lỗi trong prefix được giữ, AURC là mean discrete risk |
| case_analysis.csv | Case IDs, nhãn, dự đoán, confidence/entropy, số photo, có X-quang; lỗi tự tin cao ở đầu |
| subgroup_metrics.json | Một/nhiều photo, có/không X-quang; luôn kèm support |

Paired bootstrap tính delta macro-F1 trung bình qua các fitted seeds, CI chỉ phản ánh resampling cohort có điều kiện trên
các model đó. Không gọi đó là tổng bất định retraining; không suy ra significance bằng nhiều phép thử không hiệu chỉnh.
Nếu hai phương pháp chưa có cùng bộ seed thì không tạo paired comparison cho cặp đó.

`clinical_dependency_descriptive.json` đếm Teacher-full/visual cùng đúng, Teacher-only đúng, visual-only đúng, cả hai sai.
Bất đồng mô hình không phải nhãn thật “ca này cần clinical”. Chưa có head dự đoán need-clinical-info đã được xác nhận.
Risk–coverage cũng không thay thế xác minh lâm sàng.

Release còn experimental, test bị chặn mặc định. Override phục vụ exploratory evaluation vẫn ghi trạng thái chưa kiểm định,
không tự nâng thành reviewed/publication-ready. Không chọn hyperparameter sau mỗi lần xem test.

## 9. Student artifact

`export_student` xuất model state_dict + config + class names + hashes; không có patient IDs, optimizer, Teacher hoặc clinical vectors.
`predict_images(artifact, image_root, photo_paths, radiograph_paths=())` nhận path ảnh tương đối; không nhận nhãn/clinical text.
Dùng cùng preprocessing/cap/rule chọn ảnh lúc eval. Trả xác suất sáu lớp và bag attention, không chẩn đoán ngoài label space.
Dùng cùng source version khi inference; kiểm tra code hash để tránh loader/model bị đổi âm thầm.

## 10. Ranh giới của khóa luận và việc còn cần dữ liệu thật

Nhánh này thực hiện COde, 2D multi-image, frozen text features, late fusion, supervised/privileged KD, nhóm field fixed weights.
Seminar còn mô tả 3D/CBCT, multi-task tooth/region, learned selection, cross-attention và MeMoSA; không đủ annotation/dataset
để tự bật tất cả mà vẫn có ý nghĩa khoa học. Không tạo nhãn giả để làm đủ ô trong sơ đồ.

MeMoSA cần file thực, kiểm tra image–patient–lesion, định nghĩa target và split độc lập; classifier sáu bệnh COde không đồng nhất
với nhãn MeMoSA. Tách hai câu hỏi: transfer representation với head mới trên MeMoSA, hoặc kiểm chứng cùng phương pháp trên
mỗi dataset. Không gọi một trong hai là zero-shot external validation nếu đã fine-tune dữ liệu đích.

Trước kết luận: hoàn tất review dữ liệu, chạy full seeds và các đối chứng, báo Teacher quality, hiệu quả/chi phí Student,
đọc lỗi và missingness, chốt kết quả thật kể cả không cải thiện. Không thể xác nhận Q1 chỉ từ notebook hay GPU.

## 11. Nguồn và kiểm chứng code

- [Attention MIL — Ilse et al.](https://proceedings.mlr.press/v80/ilse18a.html): tham khảo aggregation.
- [Set Transformer — Lee et al.](https://proceedings.mlr.press/v97/lee19d.html): permutation-invariant attention pooling.
- [BiomedBERT model card của Microsoft](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext): encoder biomedical frozen dùng ở bước preparation.
- [PyTorch KL API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.kl_div.html): log-input, target distribution và reduction.
- [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html): accumulation/scaler/unscale/clip.
- Hướng phạm vi đối chiếu với `Paper/ClinDistill_Literature_Review_CODe_MeMoSA_2026.pdf` và `Seminar.png` trong workspace.

Tests nằm trong `_support/tests`: padding/permutation, nhóm split, checksum, KD direction/T², missing groups,
Teacher detach, CPU resume tương đương, training không cần test manifests, suite/report, inference không nhãn/clinical.
Kiểm thử tiny_debug chỉ xác nhận tính đúng của pipeline, không xác nhận hiệu quả ResNet-50/BiomedBERT.
