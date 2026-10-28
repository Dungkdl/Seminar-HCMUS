# CODe — Kế hoạch triển khai nghiên cứu Teacher–Student hướng tới bài báo khoa học

> **Phiên bản:** 1.0 · **Ngày:** 28/10/2026  
> **Trạng thái:** Kế hoạch nghiên cứu và thử nghiệm, **chưa phải** phương pháp đã được xác nhận là vượt SOTA.  
> **Dataset trọng tâm:** COde / Casangels Oro-dental Dataset (CODe).  
> **Đích sử dụng:** Khóa luận tốt nghiệp + cơ sở xây dựng bài báo học thuật. Không dùng mô hình chưa thẩm định cho chẩn đoán thực tế.

---

## 0. Tóm tắt: cần làm gì trước?

**Đề xuất nghiên cứu:** *Reliability-Aware Privileged Clinical Knowledge Distillation for Visual-Only Multi-Image Dental Diagnosis*.

**Bài toán:** Khi huấn luyện, Teacher được sử dụng ảnh nha khoa **và thông tin lâm sàng hợp lệ tại thời điểm trước chẩn đoán**. Student chỉ nhận **ảnh** cả khi huấn luyện lẫn suy luận. Mục tiêu kiểm tra liệu Student học được lợi ích từ Teacher đa phương thức mà không bị ép sao chép các kết luận không có đủ bằng chứng trên ảnh.

**Cấu hình chính để thử:**

- **Visual Teacher \(T_v\):** DINOv3 ViT-L/16 + bộ tổng hợp đa ảnh (MIL) + classification head.
- **Multimodal Teacher \(T_m\):** DINOv3 ViT-L/16 + MIL + BGE-M3 (hoặc text encoder đa ngôn ngữ được kiểm chứng) + gated fusion + classification head.
- **Visual Student \(S\):** DINOv3 ViT-B/16 + MIL + classification head. **Không nhận clinical text** khi train hoặc suy luận; nhãn/soft target từ Teacher được sử dụng trong hàm loss.
- **Baseline:** ResNet-50 + MIL; Swin-B + MIL; Student DINOv3-B không có KD.
- **Phần đề xuất cần kiểm định:** KD có xét độ tin cậy, chất lượng Teacher và mức hỗ trợ từ thị giác (reliability / visual-support aware KD).

> **Ưu tiên số 1:** Đừng thuê GPU lớn trước khi hoàn thành data audit, kiểm tra nhãn, rò rỉ lâm sàng và chia tập theo bệnh nhân. Lỗi nghiên cứu ở bước này không thể sửa bằng mô hình lớn hơn.

### Checklist chạy thử nhanh — theo thứ tự

- [ ] Tải bản CODe chính thức; đọc paper, data card, giấy phép và sơ đồ các trường dữ liệu.
- [ ] Hoàn tất `00_data_audit.ipynb`; xác định chính xác `patient_id`, `checkup_id`, danh sách ảnh, nhãn, trường clinical có sẵn.
- [ ] Hoàn tất `01_patient_split.ipynb`; lưu manifest train/validation/test cố định và khóa test.
- [ ] Train **ResNet-50 + MIL image-only** để xác nhận pipeline hoạt động end-to-end.
- [ ] Train **DINOv3-B image-only** và **DINOv3-L visual-only** trên **cùng split**.
- [ ] Huấn luyện Teacher đa phương thức DINOv3-L + clinical text **không chứa đáp án chẩn đoán**.
- [ ] Distill DINOv3-B với **vanilla KD**, so sánh với không KD và KD từ Visual Teacher.
- [ ] Triển khai reliability-aware KD; thực hiện ablation kiểm tra phần đóng góp thật.
- [ ] Chạy nhiều seed, đánh giá trên test theo bệnh nhân, báo cáo số liệu với khoảng tin cậy.

---

## 1. Mục tiêu nghiên cứu và giả thuyết

### RQ1 — Clinical benefit

Thông tin lâm sàng **có trước chẩn đoán** có giúp Teacher đa phương thức tốt hơn Visual Teacher chỉ dùng ảnh không?

- **H1:** `T_m(image + clinical)` có Macro-F1 cao hơn `T_v(image)` trên **cùng lượt khám / cùng split / cùng định nghĩa nhãn**.
- **Điều kiện:** clinical phải không chứa kết luận chẩn đoán, mã nhãn, treatment plan hay ghi chú viết sau khi xác định bệnh.

### RQ2 — Privileged knowledge transfer

Student chỉ nhìn ảnh có tốt hơn nếu học từ Teacher đa phương thức không?

- **H2:** `S(image) + KD(T_m)` tốt hơn `S(image)` và (lý tưởng) tốt hơn `S(image) + KD(T_v)`.
- Nếu H2 không đúng, phải phân tích giới hạn **thông tin chỉ nhìn ảnh không thể suy ra**, không ép kết quả theo giả thuyết.

### RQ3 — Reliability / visual support

Khi Teacher đúng vì triệu chứng nhưng Student không có đủ bằng chứng thị giác, liệu cơ chế **chọn lọc hoặc điều tiết KD** giúp tăng chất lượng, hiệu chỉnh xác suất và khả năng từ chối dự đoán không?

- **H3 (cần kiểm nghiệm):** reliability-aware KD tốt hơn vanilla KD hoặc confidence-only KD trên metric đã đăng ký trước.
- **Lưu ý novelty:** Teacher đa phương thức + Student chỉ ảnh và confidence-weighted KD đều đã có tiền lệ. Điểm mới phải được xác lập bằng literature review + thiết kế phương pháp cụ thể + kiểm chứng chặt, không tuyên bố mới chỉ vì dùng một công thức trọng số.

---

## 2. Dataset CODe: dữ kiện đã xác nhận và các điểm cần audit

### 2.1. Quy mô và bản gốc

CODe được mô tả trong bài báo *A benchmark multimodal oro-dental dataset for large vision-language models* (*Scientific Data*, 2026):

| Thành phần | Quy mô báo cáo |
|---|---:|
| Bệnh nhân | 4.800 |
| Lượt khám (checkups) | 8.775 |
| Ảnh trong miệng | 50.000 |
| Ảnh X-ray | 8.056 |
| Benchmark phân loại gốc | 6 loại bất thường, test 600 bản ghi |

Paper cho biết ảnh được tổ chức trong `Photographs` và `Radiographs`; mỗi dòng dữ liệu văn bản tương ứng một lượt khám và có tham chiếu tên file ảnh. **Không giả định tên cột thực tế trước khi mở CSV.** Test 600 bản ghi trong paper là một protocol riêng, chưa được mặc định xem là patient-disjoint.

**Nguồn chính:**

- Paper: <https://www.nature.com/articles/s41597-026-07342-9>
- Dataset: <https://huggingface.co/datasets/zirak-ai/COde>
- Dataset DOI: <https://doi.org/10.57967/hf/6421>
- Repository / preprocessing: <https://github.com/zirak-ai/COde>

### 2.2. Hai giao thức đánh giá (KHÔNG trộn số liệu)

**Protocol P1 — benchmark gốc (reproduction):**

- Tái tạo chính xác six-class task, các trường input và split/nhãn được cung cấp bởi nhóm tác giả (nếu có đầy đủ).
- Dùng để **đối chiếu** với bài CODe gốc và VLM benchmark.
- Kiểm tra và công khai liệu có bệnh nhân xuất hiện ở cả train và test hay không.

**Protocol P2 — patient-disjoint nghiên cứu chính (đề xuất):**

- Chia bệnh nhân thành train/validation/test, gợi ý **70/15/15** theo số bệnh nhân, dùng group-stratification nếu khả thi.
- Tất cả lượt khám, ảnh, bản dịch, text variants và augmentations của cùng bệnh nhân nằm trong **một** split.
- Ghi rõ số lượng bệnh nhân, lượt khám, ảnh và từng lớp của từng split.
- **Chỉ dùng P2 làm cơ sở chính** để kết luận khả năng tổng quát trên bệnh nhân chưa gặp; P1 có thể là báo cáo bổ sung.

### 2.3. Quyết định loại nhãn

Bài gốc có six-class task: **Caries, Gingivitis, Malocclusion, Pulpitis, Tooth Loss, Tooth Structure Loss**.

Trước khi train:

1. Tạo bảng phân bố **nhãn theo lượt khám** cho sáu lớp.
2. Xác minh mỗi ví dụ benchmark thực tế có **một** nhãn hợp lệ hay nhiều bệnh cùng tồn tại.
3. **Nếu single-label**: `CrossEntropyLoss`, `softmax` và Macro-F1 đa lớp.
4. **Nếu multi-label**: thiết kế **protocol riêng**, `BCEWithLogitsLoss` và KD phù hợp đầu ra sigmoid; không dùng chung metric/công thức với six-class benchmark.
5. Không tự suy luận rằng một bệnh **không được đề cập** nghĩa là bệnh **âm tính** nếu quy trình annotation chưa xác nhận.

**Quyết định ưu tiên:** Thử tái tạo *six-class benchmark* trước để có đối chiếu rõ ràng; mở rộng multi-label sau khi đã kiểm chứng chất lượng nhãn.

### 2.4. Data audit: các phép kiểm tra bắt buộc

Tạo `reports/data_audit.md` và `reports/data_audit.json` gồm:

- [ ] Các cột CSV, loại dữ liệu, giá trị thiếu, ID bất thường.
- [ ] Số `patient_id` / `checkup_id` duy nhất; số lượt khám mỗi bệnh nhân.
- [ ] Số ảnh Photo/X-ray mỗi lượt khám; modality thiếu; ảnh không đọc được.
- [ ] Loại ảnh: intraoral, PaX, PeX, CBCT 2D (chỉ gắn nhãn khi nguồn xác định được).
- [ ] Ảnh trùng theo hash / near-duplicate; lượt khám trùng / bản ghi lặp.
- [ ] Nhãn sáu lớp; mất cân bằng lớp; bệnh xuất hiện đồng thời.
- [ ] Phân bố tuổi, giới tính; trường text gốc tiếng Trung và bản dịch.
- [ ] Chất lượng liên kết `checkup_id` ↔ image filenames.
- [ ] Phân bố độ phân giải và aspect ratio (không tự upscale rồi coi là tăng thông tin).
- [ ] Khả năng có ảnh sau điều trị, ảnh theo dõi hoặc thông tin tương lai trong cùng record.
- [ ] Giấy phép dataset, điều kiện sử dụng, cách công bố code/model mà không phát tán dữ liệu nhạy cảm.

### 2.5. Chống clinical leakage (đặc biệt quan trọng)

| Nhóm trường | Quy tắc ban đầu |
|---|---|
| Age, Gender | Có thể dùng nếu có ở thời điểm dự đoán |
| Chief Complaint | Dùng **sau audit** nội dung / thời điểm |
| Present Illness, Past Medical History | Chỉ dùng nếu xuất hiện **trước chẩn đoán** và không sao chép kết luận |
| Clinical Examination | **Protocol bổ sung** nếu định nghĩa rõ thời điểm thông tin xuất hiện |
| Diagnosis, Diseases | **Cấm dùng làm input Teacher** trong protocol chính |
| Treatment Plan, Management, Follow-up diagnosis | **Cấm dùng** vì khả năng chứa trực tiếp đáp án |
| Radiographs Examination / kết luận từ bác sĩ | Không dùng trong protocol chính, trừ khi thiết kế một task khác rõ ràng |

Đặc biệt **không chỉ lọc theo tên cột**: cần kiểm tra mẫu văn bản vì Chief Complaint cũng có thể chứa cụm diễn đạt gần như đáp án. Việc loại bỏ các từ bệnh cần có quy tắc audit nhất quán, không được tùy tiện xóa những triệu chứng có ý nghĩa.

**Đầu ra bắt buộc:** `clinical_whitelist.yaml`, `clinical_leakage_audit.csv`, báo cáo tỷ lệ record bị loại / bị masking và ví dụ được ẩn danh.

---

## 3. Đơn vị dự đoán, pipeline đa ảnh và kiến trúc

### 3.1. Đơn vị học và đánh giá

- **Input 1 mẫu:** một **lượt khám** gồm `photos[]`, `radiographs[]` và (Teacher only) `clinical_fields`.
- **Output 1 mẫu:** nhãn/điểm số ở **mức lượt khám**, không tự gán cùng nhãn bệnh cho tất cả ảnh trong lượt khám.
- **Mức phân chia dữ liệu:** bệnh nhân; **mức dự đoán chính:** lượt khám; **mức bootstrap CI:** bệnh nhân.

### 3.2. Hierarchical Multi-Image Learning (MIL)

```text
Photos (0..N) ──> Shared / modality-aware image encoder ──> Photo tokens ──> Masked attention pool ─┐
                                                                                                     ├──> Visit representation
X-rays (0..M) ──> Shared / modality-aware image encoder ──> X-ray tokens ──> Masked attention pool ───┘
                                                                                                               │
                                                                    ┌──────────────────────────────────────────┤
                                                                    │                                          │
                                                            Visual head                                  Clinical fusion
                                                                    │                                          │
                                                               Visual logits                            Multimodal logits
```

**Mức độ triển khai:**

1. **V0:** Mean pooling các embedding của tất cả ảnh để có pipeline tối thiểu.
2. **V1:** Gated Attention MIL với mask số ảnh khác nhau.
3. **V2:** Hierarchical MIL: pool từng modality trước, fusion hai modality sau; token `missing modality` rõ ràng.
4. **V3 (ablation):** Modality-aware cross-attention / aggregator transformer nếu V2 đã chứng minh hữu ích.

**Quy tắc xử lý ảnh:**

- Bảo toàn aspect ratio; resize + padding. Không tự ý horizontal flip hoặc crop khiến mất/đảo dấu hiệu lâm sàng.
- Train có thể sampling một số ảnh mỗi lượt khám; validation/test **phải dùng quy tắc chọn ảnh cố định** hoặc sử dụng toàn bộ ảnh theo batch/chunk deterministically.
- Với image-level feature caching, lưu cả image ID, modality, split, phiên bản preprocessing, model hash.
- Một ảnh thuộc nhiều lượt khám phải được rà soát vì có nguy cơ rò rỉ.

### 3.3. Bộ mô hình

| Thành phần | Mô hình | Vai trò | Ưu tiên |
|---|---|---|---|
| CNN baseline | ResNet-50 pretrained + MIL | Mốc đối chiếu | Bắt buộc |
| Transformer baseline | Swin-B pretrained + MIL | Mốc đối chiếu thứ hai | Nên có |
| Student baseline | DINOv3 ViT-B/16 + MIL | Không KD | **Bắt buộc** |
| Visual Teacher | DINOv3 ViT-L/16 + MIL | Chỉ ảnh | **Bắt buộc** |
| Multimodal Teacher | DINOv3 ViT-L/16 + MIL + BGE-M3 + Fusion | Ảnh + clinical | **Bắt buộc** |
| Student nhẹ | DINOv3 ConvNeXt-Tiny + MIL | Kiểm tra nén | Mở rộng |
| Alternative Teacher | SigLIP2-L + MIL / fusion | Kiểm tra độ phụ thuộc backbone | Mở rộng |
| VLM benchmark | Qwen3-VL / DentVLM theo điều kiện giấy phép | Đối chứng lớn trên cùng protocol | Mở rộng |

**Checkpoint tham khảo:**

- DINOv3 ViT-B: <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
- DINOv3 ViT-L: <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>
- DINOv3 ConvNeXt-Tiny: <https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m>
- BGE-M3: <https://huggingface.co/BAAI/bge-m3>
- SigLIP2-L: <https://huggingface.co/google/siglip2-large-patch16-384>

**Chú ý:** Meta DINOv3 checkpoints yêu cầu chấp nhận điều kiện truy cập / giấy phép ở Hugging Face. Kiểm tra quyền tải và điều kiện công bố weights; không mặc định mọi checkpoint có thể chia sẻ lại.

### 3.4. Text Encoder và Fusion

**Khởi đầu:** BGE-M3 frozen, encode text pre-diagnostic ở ngôn ngữ gốc, kết hợp biểu diễn ảnh với tuổi/giới tính đã xử lý missing values.

So sánh ba mức:

1. `clinical = age + gender` (metadata only).
2. `clinical = age + gender + chief_complaint` (primary clinical input, sau audit).
3. `clinical = (2) + present_illness/past_history` **nếu** được xác minh có sẵn trước chẩn đoán.

Fusion baselines:

- **F0:** concat + MLP.
- **F1:** gated fusion (khởi đầu đề xuất).
- **F2:** cross-attention / token-level fusion (chỉ nếu có lợi).

Ablation ngôn ngữ: tiếng Trung gốc / bản dịch tiếng Anh / metadata-only; cùng tập bản ghi hợp lệ.

---

## 4. Knowledge Distillation: triển khai từ đơn giản đến phương pháp đề xuất

### 4.1. Ba nguồn supervision

- `CE`: hard label đã được xác minh.
- `KD(T_v)`: phân phối dự đoán của Visual Teacher.
- `KD(T_m)`: phân phối dự đoán của Multimodal Teacher.

**Single-label six-class loss:**

\[
\mathcal L_{\mathrm{KD}}=T^2\,D_{\mathrm{KL}}\left(\operatorname{softmax}(z_T/T)\;\|\;\operatorname{softmax}(z_S/T)\right)
\]

\[
\mathcal L_{\mathrm{student}}=(1-\alpha)\mathcal L_{\mathrm{CE}}+\alpha\mathcal L_{\mathrm{KD}}
\]

Lưu ý thứ tự KL (Teacher || Student) và reduction phù hợp. Ví dụ hyperparameter để **tuning trên validation**: `T ∈ {2, 4}`, `alpha ∈ {0.1, 0.3, 0.5, 0.7}`. Không chọn dựa trên test.

### 4.2. Proposed: reliability-aware KD (ý tưởng cần kiểm định)

Đặt:

- \(p_v(y\mid I)\): xác suất Visual Teacher.
- \(p_m(y\mid I,C)\): xác suất Multimodal Teacher.
- \(p_s(y\mid I)\): xác suất Student.

**Một trọng số ứng viên** cho mẫu huấn luyện \(i\) trong thiết lập single-label:

\[
\tilde w_i = \mathbf 1[\arg\max p_{m,i}=y_i]\cdot
\left(1-\frac{H(p_{m,i})}{\log K}\right)\cdot
\left[\epsilon+(1-\epsilon)p_{v,i}(y_i)\right]
\]

Sau đó chuẩn hóa / chặn \(w_i\) và thử:

\[
\mathcal L = \mathcal L_{\mathrm{CE}}+
\lambda\frac{\sum_i w_i\,\mathcal L_{\mathrm{KD},i}(T_m,S)}{\sum_i w_i+\delta}
\]

**Lưu ý nghiên cứu quan trọng:** công thức trên **không mặc định tốt hơn** vanilla KD. Nếu Visual Teacher còn yếu, weighting theo \(p_v(y_i)\) có thể loại bỏ mẫu khó — chính nơi thông tin clinical hữu ích. Phải kiểm tra ablation và đối chứng; có thể cân nhắc trọng số dựa trên disagreement, ranking hoặc learned gate được regularize.

**Nguyên tắc chống sai lệch:**

- Teacher logits / reliability weights tốt nhất nên lấy từ **out-of-fold predictions trên training patients**, không lấy ngay dự đoán quá tự tin của Teacher trên những mẫu đã học.
- Không sử dụng validation/test để train gate hoặc tính feature statistics phục vụ train.
- Teacher đóng băng khi distill; kiểm tra `requires_grad=False` cho toàn Teacher.
- Giữ soft-target artifacts kèm `checkup_id` và hash; không được ghép nhầm prediction của các lượt khám.
- Nếu phát triển multi-label, **định nghĩa lại loss** cho sigmoid/BCE; không tái sử dụng công thức softmax six-class.

### 4.3. Ablation bắt buộc cho KD

| Mã | Student | KD target | Mục tiêu |
|---|---|---|---|
| KD0 | DINOv3-B | Không KD | Student baseline |
| KD1 | DINOv3-B | Visual Teacher, vanilla KL | Lợi ích KD thông thường |
| KD2 | DINOv3-B | Multimodal Teacher, vanilla KL | Lợi ích privileged clinical KD |
| KD3 | DINOv3-B | Multimodal Teacher, confidence-only weight | Đối chứng chọn lọc đơn giản |
| KD4 | DINOv3-B | Multimodal Teacher, proposed reliability weight | **Phương pháp chính** |
| KD5 | DINOv3-B | KD4 + feature alignment | Kiểm tra thêm feature KD |

**So sánh cốt lõi:** KD4 vs KD2, KD2 vs KD1, KD1 vs KD0; làm paired comparison trên cùng test.

---

## 5. Bảng thí nghiệm tổng thể (research matrix)

| ID | Mô hình | Input | Output / câu hỏi |
|---|---|---|---|
| E01 | ResNet-50 + MIL | Ảnh | CNN baseline |
| E02 | Swin-B + MIL | Ảnh | Transformer baseline |
| E03 | DINOv3-B + MIL | Ảnh | Student không KD |
| E04 | DINOv3-L + MIL | Ảnh | Visual Teacher \(T_v\) |
| E05 | DINOv3-L + BGE-M3 + gated fusion | Ảnh + clinical hợp lệ | Multimodal Teacher \(T_m\) |
| E06 | E03 + KD(E04) | Ảnh | Visual KD |
| E07 | E03 + vanilla KD(E05) | Ảnh | Clinical privileged KD |
| E08 | E03 + confidence-weighted KD(E05) | Ảnh | Reliability baseline |
| **E09** | **E03 + proposed reliability-aware KD(E05)** | **Ảnh** | **Ứng viên phương pháp chính** |
| E10 | E09 + feature KD | Ảnh | Feature-alignment ablation |
| E11 | DINOv3 ConvNeXt-Tiny + E09 KD | Ảnh | Student nhỏ |
| E12 | Teacher SigLIP2-L + corresponding KD | Ảnh / clinical khi train Teacher | Backbone generalization |

**Không chạy tất cả E01–E12 cùng lúc.** Thứ tự đúng: E01 → E03 → E04 → E05 → E06/E07 → E08/E09; mở rộng E02/E10–E12 khi protocol chính ổn định.

### Ablation bổ sung (ưu tiên theo sức thuyết phục)

1. **Clinical**: image-only; + demographics; + complaint; + valid prediagnostic history.
2. **Multi-image**: mean pooling / attention MIL / hierarchical MIL; photos only / X-rays only / both (cùng tập bệnh nhân đủ modality).
3. **KD**: no KD / visual KD / multimodal KD / confidence-only / proposed reliability-aware.
4. **Resolution**: 224 / 384 / 448; kiểm soát tiền xử lý và tổng số token ảnh.
5. **Robustness**: thiếu X-ray, thiếu ảnh, ảnh nhiễu, clinical text thiếu/sai; test tương ứng cho Teacher và Student.
6. **Ngôn ngữ**: văn bản tiếng Trung / bản dịch tiếng Anh khi bản gốc cho phép.

---

## 6. Cấu hình GPU và chiến lược tiết kiệm tài nguyên

| Hạ tầng | Công việc thích hợp | Hạn chế cần nhớ |
|---|---|---|
| **1× NVIDIA L4 24 GB** | audit, baseline, frozen embeddings, MIL head, Student, LoRA/adapters | Multi-image full fine-tune ViT-L có thể OOM |
| **2–4× L4 (24 GB/GPU)** | parallel experiments / DDP, thử nhiều seed | VRAM **không tự gộp**; phân tán cần xử lý riêng |
| **1× A100 80 GB** | Teacher ViT-L, BF16, nhiều ảnh/lượt khám, ablation | Ưu tiên nếu thuê được với chi phí hợp lý |
| **1× H100 80 GB hoặc RTX PRO 6000 96 GB** | VLM đối chứng lớn / nhiều thử nghiệm nặng | Giá và hỗ trợ phần mềm cần so sánh trước |

**Chiến lược train khuyến nghị:**

1. Stage 1: frozen DINOv3 image embeddings + train MIL/head; frozen text embeddings + fusion.
2. Stage 2: partial fine-tune visual backbone, layer-wise LR decay; frozen/partial text encoder.
3. Stage 3: khi có lợi rõ ràng mới thử full fine-tuning Teacher hoặc resolution lớn.
4. Stage 4: đóng băng Teacher, cache logits (nếu protocol cho phép) rồi huấn luyện Student.

**Khởi đầu trên A100:** BF16; AdamW; 384 px để dò, 448 px khi kiểm chứng; LR backbone `1e-5..3e-5`, heads `1e-4..1e-3`; warmup 5%; 15–30 epochs + early stopping; accumulation tùy số ảnh/lượt khám. **Đây là khoảng để tuning, không phải cấu hình tối ưu đã được chứng minh.**

**Khởi đầu trên L4:** batch rất nhỏ theo lượt khám (1–2); chia encode ảnh thành image microbatch; activation checkpointing nếu cần; frozen features trước; mixed precision `bf16` hoặc `fp16` theo mức hỗ trợ phần cứng/phần mềm được kiểm tra bằng chạy thực tế.

**Không sử dụng GPU được cấp riêng cho một cuộc thi Kaggle vào công việc khóa luận nếu vi phạm quy định cuộc thi.** Dùng quota Kaggle thông thường hoặc máy thuê hợp lệ.

---

## 7. Metrics, tính hợp lệ và thống kê

### Primary metric (đăng ký trước trên validation)

- **Six-class single-label:** Macro-F1 tại **mức lượt khám**.
- Báo cáo thêm Accuracy, Balanced Accuracy, per-class Precision/Recall/F1, Confusion Matrix, AUROC/AUPRC khi định nghĩa phù hợp.
- **Multi-label extension:** macro/micro F1, macro mAP/AUPRC và per-label recall; threshold chọn trên validation.

### Reliability / selective prediction

- Expected Calibration Error (ECE), Brier Score, negative log-likelihood nếu phù hợp.
- Risk–coverage curve / selective prediction: mô hình được phép **abstain** khi không đủ bằng chứng.
- Phân tích nhóm có Teacher đa phương thức đúng nhưng Teacher chỉ ảnh sai.
- Tách rõ điểm đánh giá **hiệu năng dự đoán** khỏi kết luận **an toàn sử dụng lâm sàng**.

### Statistical protocol

- **Ít nhất 3 seeds** cho so sánh chính; giữ nguyên split và evaluation protocol.
- Báo cáo trung bình ± độ lệch chuẩn theo seed.
- Dùng **paired patient-cluster bootstrap** để tạo CI 95% cho chênh lệch Macro-F1 trên test khi so sánh các hệ thống.
- Khóa hyperparameter trước khi mở test; không chọn model / temperature / threshold sau khi xem điểm test.
- Nếu nhiều phép kiểm định, nêu trước primary comparison và điều chỉnh multiple comparisons phù hợp.

### Model efficiency

- Model parameter count, size checkpoint, peak VRAM khi inference, latency / visit với cùng số ảnh, throughput.
- Student inference **không được truy cập** clinical text, Teacher output hay các embedding chứa clinical từ lúc train.
- Báo cáo số ảnh xử lý ở inference; không chỉ tính latency của một ảnh trong khi bài toán dùng cả lượt khám.

---

## 8. Chuẩn đánh giá lâm sàng bổ sung (nếu có hợp tác nha sĩ)

**Rất đáng làm cho bài báo, nhưng phải coi đây là một study riêng:**

- Tạo tập con 200–500 lượt khám (quy mô đề xuất, tùy khả năng) được chuyên gia đánh giá **chỉ nhìn ảnh**.
- Ghi nhận: nhìn ảnh có đủ bằng chứng để phân loại không, tổn thương nhìn thấy ở ảnh nào, mức chắc chắn, cần thêm dữ liệu gì.
- Nếu có ít nhất hai người chấm: báo cáo agreement và cách xử lý bất đồng; tránh cho người chấm thấy nhãn gốc trong lượt đánh giá image-only.
- Dùng tập này để kiểm nghiệm giả thuyết **visual support**, không dùng làm training labels hay gate training nếu muốn giữ như independent evaluation.
- Không vẽ bounding box / heatmap như ground truth nếu không có chú thích vị trí tổn thương do bác sĩ xác nhận.

Tham khảo checklist báo cáo: TRIPOD+AI; STARD-AI tùy loại nghiên cứu. CODe đến từ một trung tâm; giới hạn khả năng khái quát phải được thừa nhận.

---

## 9. Cấu trúc dự án gợi ý

```text
code-dental-kd/
├── README.md
├── requirements.txt                # hoặc pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── resnet50_baseline.yaml
│   ├── dinov3b_student.yaml
│   ├── dinov3l_visual_teacher.yaml
│   ├── dinov3l_multimodal_teacher.yaml
│   └── kd_reliability.yaml
├── data/
│   ├── raw/                        # KHÔNG commit bộ ảnh bệnh nhân
│   ├── manifests/
│   ├── splits/
│   └── cache/                      # embeddings/logits — theo quyền truy cập dữ liệu
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_patient_split.ipynb
│   ├── 02_resnet50_mil.ipynb
│   ├── 03_dinov3_visual.ipynb
│   ├── 04_multimodal_teacher.ipynb
│   ├── 05_vanilla_kd.ipynb
│   ├── 06_reliability_kd.ipynb
│   └── 07_evaluation.ipynb
├── src/
│   ├── data/{audit.py,manifest.py,dataset.py,splits.py,transforms.py}
│   ├── models/{encoders.py,mil.py,fusion.py,teacher.py,student.py}
│   ├── losses/{kd.py,reliability.py}
│   ├── training/{train.py,evaluate.py,seed.py}
│   └── utils/{logging.py,checkpoint.py}
├── tests/
│   ├── test_patient_disjoint.py
│   ├── test_no_label_leakage.py
│   ├── test_mil_masks.py
│   └── test_kd_loss.py
├── reports/
│   ├── data_audit.md
│   ├── label_protocol.md
│   ├── split_protocol.md
│   └── results.md
└── runs/                           # logs, checkpoints, metrics; thường gitignore
```

**Lệnh khởi tạo thư mục** (chạy trong terminal Linux/macOS):

```bash
mkdir -p code-dental-kd/{configs,data/{raw,manifests,splits,cache},notebooks,reports,runs,tests}
mkdir -p code-dental-kd/src/{data,models,losses,training,utils}
cd code-dental-kd
git init
printf 'data/raw/\ndata/cache/\nruns/\n*.ckpt\n__pycache__/\n.env\n' > .gitignore
```

### Mẫu `configs/data.yaml` (sửa sau audit)

```yaml
dataset_name: COde
raw_root: data/raw
manifest_path: data/manifests/checkups.parquet
split_path: data/splits/patient_split_v1.csv
label_protocol: six_class_single_label   # chỉ dùng sau khi audit xác nhận
classes:
  - Caries
  - Gingivitis
  - Malocclusion
  - Pulpitis
  - Tooth Loss
  - Tooth Structure Loss
patient_group_key: patient_id            # đổi thành tên cột thực tế sau audit
visit_key: checkup_id                    # đổi thành tên cột thực tế sau audit
clinical_whitelist:
  - age
  - gender
  - chief_complaint                  # chỉ sau leakage audit
# KHÔNG đưa diagnosis/diseases/treatment_plan vào clinical input
seed: 42
train_fraction: 0.70
val_fraction: 0.15
test_fraction: 0.15
```

### Hợp đồng dữ liệu nội bộ (`manifest`)

Mỗi dòng manifest nên biểu diễn **một lượt khám**:

```json
{
  "patient_id": "REDACTED_PATIENT_ID",
  "checkup_id": "REDACTED_VISIT_ID",
  "photos": ["relative/path/photo_1.jpg"],
  "radiographs": ["relative/path/xray_1.jpg"],
  "clinical_pre_diagnostic": {"age": null, "gender": null, "chief_complaint": null},
  "label": null,
  "split": null
}
```

Đây chỉ là **schema mục tiêu**, không phải khẳng định CODe gốc có chính xác những key này.

---

## 10. Các notebook và tiêu chí hoàn thành (Definition of Done)

### `00_data_audit.ipynb` — bước đầu tiên

**Input:** CSV chính thức, đường dẫn ảnh, metadata, hướng dẫn paper.

**Phải xuất:**

- `reports/data_audit.md` có thống kê số mẫu và các bất thường.
- `data/manifests/checkups.parquet` với 1 dòng / lượt khám.
- `reports/label_protocol.md` giải thích task six-class và mapping nhãn.
- Danh sách các trường clinical hợp lệ và nguy cơ rò rỉ.

**Pass khi:** mọi `checkup_id` có thể liên kết ảnh hợp lệ; số lượng/missing được báo cáo; nhãn và nguồn nhãn rõ ràng.

### `01_patient_split.ipynb`

**Phải xuất:** `data/splits/patient_split_v1.csv`, `reports/split_protocol.md`, hash manifest và random seed.

**Pass khi:** số bệnh nhân train/val/test giao nhau = **0**; không có ảnh trùng qua split; stratification được giải thích và lớp hiếm được kiểm tra.

### `02_resnet50_mil.ipynb`

**Mục tiêu:** xác nhận pipeline baseline đúng trước khi dùng backbone lớn.

**Pass khi:** loss giảm trên một small subset, có checkpoint, metric per-class và **không** dùng clinical trong forward.

### `03_dinov3_visual.ipynb`

**Chạy:** DINOv3-B Student baseline và DINOv3-L Visual Teacher.

**Pass khi:** cùng label protocol, augmentation và split; kiểm tra frozen feature vs partial fine-tune; lưu logits và run configs.

### `04_multimodal_teacher.ipynb`

**Chạy:** Visual Teacher + text encoder + clinical fusion.

**Pass khi:** đánh giá cùng validation visits với Visual Teacher; report ảnh hưởng metadata, complaint, source language; kiểm chứng không leakage.

### `05_vanilla_kd.ipynb`

**Chạy:** KD0, KD1, KD2.

**Pass khi:** teacher frozen, KD loss tính đúng, Student inference chỉ có ảnh; chạy thử so sánh nhiệt độ và alpha trên validation.

### `06_reliability_kd.ipynb`

**Chạy:** KD3, KD4, (KD5 nếu hợp lý).

**Pass khi:** có confidence-only baseline, weighting không collapse, teacher logits gắn đúng checkup IDs, phương pháp vượt/không vượt vanilla KD đều được ghi nhận trung thực.

### `07_evaluation.ipynb`

**Phải xuất:** `reports/results.md`, bảng per-class metrics, confusion matrix, CI, robustness, calibration, latency, một bảng ablation chính.

**Pass khi:** mô hình/threshold đã khóa trước test; test không được tái dùng để chọn siêu tham số; code tạo lại figures và tables có thể chạy từ checkpoints.

---

## 11. Các unit tests tối thiểu phải có

- `test_patient_disjoint.py`: không giao `patient_id` giữa train/val/test.
- `test_no_label_leakage.py`: denylist cột chẩn đoán không xuất hiện trong clinical feature builder; kiểm tra các từ khóa nhạy theo quy tắc audit.
- `test_mil_masks.py`: thêm padded images không đổi predictions nếu mask đúng; missing modality không tạo NaN.
- `test_kd_loss.py`: KL gần 0 khi logits Student = logits Teacher; gradient chỉ chảy vào Student; trọng số không âm; không chia 0.
- `test_manifest.py` (khuyến nghị): mỗi ảnh có file tồn tại, nhãn hợp lệ, ID duy nhất; không ghép logits sai visit.
- `test_inference_visual_only.py`: inference chỉ truyền image tensors/masks vẫn thành công, không có text encoder trong Student artifact.

---

## 12. Kế hoạch nguồn lực và mốc triển khai (không phải cam kết thời gian)

| Giai đoạn | Điều kiện hoàn thành | GPU ưu tiên |
|---|---|---|
| G0 — Data protocol | Audit + labels + patient-disjoint split | CPU / L4 |
| G1 — Visual baseline | ResNet-50 + DINOv3-B + DINOv3-L | L4 / A100 |
| G2 — Multimodal Teacher | Fusion hoạt động; không clinical leakage | A100 80 GB |
| G3 — Vanilla KD | No-KD vs Visual-KD vs Multimodal-KD | L4 / A100 |
| G4 — Proposed KD | Reliability ablations hoàn chỉnh | L4 / A100 |
| G5 — Paper evaluation | Nhiều seed, CI, robustness, efficiency | A100 nếu nhiều run |
| G6 — Paper draft | Methods, Results, Limitations, Reproducibility | CPU |

**Dừng/đổi hướng nếu:**

- Nhãn không đáng tin cậy hoặc không thể tạo protocol kiểm chứng → ưu tiên audit/annotation thay vì tăng backbone.
- Multimodal Teacher không tốt hơn Visual Teacher trên cùng validation → kiểm tra text, fusion, leakage, quality; không vội làm privileged KD.
- Vanilla privileged KD liên tục tệ hơn KD từ Visual Teacher → nghiên cứu độ không khớp giữa nhãn và khả năng quan sát từ ảnh; không tăng alpha vô hạn.
- Reliability weight chỉ cải thiện một seed/nhóm nhỏ → chưa đủ cơ sở kết luận có phương pháp mới tốt hơn.

---

## 13. Hướng viết bài báo và các bảng/đồ thị cần lưu

**Dàn ý draft:** Abstract → Introduction → Related Work → Data/Task & Leakage Protocol → Proposed Dual-Teacher Reliability-Aware KD → Experimental Setup → Baselines → Main Results → Ablation → Reliability/Clinical Analysis → Limitations/Ethics → Conclusion.

**Bảng & hình tối thiểu:**

1. Table 1 — Dataset và patient-disjoint split statistics.
2. Figure 1 — Sơ đồ hai Teacher + Visual-only Student.
3. Table 2 — Main result: image baseline / Teacher / vanilla KD / proposed KD (mean±SD và CI).
4. Table 3 — Ablation: clinical input, MIL, KD weighting.
5. Figure 2 — Per-class confusion matrix / F1.
6. Figure 3 — Reliability diagram và risk–coverage.
7. Table 4 — Model size, latency/visit, inference VRAM.
8. Failure-case analysis — mẫu không đủ thông tin ảnh / clinical-image conflict (sau khi bảo đảm quyền sử dụng và ẩn danh).

**Kết luận cần bảo vệ được:** không chỉ “đạt accuracy X”, mà là liệu đặc trưng học từ privileged clinical information có tạo lợi ích ổn định cho visual-only Student, và trong điều kiện nào transfer không hợp lệ.

---

## 14. Tài liệu đọc và đối chứng

1. **CODe paper** — <https://www.nature.com/articles/s41597-026-07342-9>
2. **CODe dataset** — <https://huggingface.co/datasets/zirak-ai/COde>
3. **CODe preprocessing/evaluation code** — <https://github.com/zirak-ai/COde>
4. **DINOv3 paper / model family** — <https://arxiv.org/abs/2508.10104> và <https://huggingface.co/collections/facebook/dinov3>
5. **Attention-based Multiple Instance Learning (Ilse et al.)** — <https://proceedings.mlr.press/v80/ilse18a.html>
6. **BGE-M3** — <https://huggingface.co/BAAI/bge-m3>
7. **SigLIP2** — <https://huggingface.co/google/siglip2-large-patch16-384>
8. **DentVLM source** — <https://github.com/ZJUI-AI4H/DentVLM> (kiểm tra giấy phép, cấu hình thực tế và tính tương thích với task CODe).
9. **TRIPOD+AI** — kiểm tra checklist phù hợp loại nghiên cứu dự đoán trước khi viết bài.
10. **STARD-AI** — kiểm tra checklist phù hợp nếu báo cáo nghiên cứu độ chính xác chẩn đoán.

> **Yêu cầu literature review:** cập nhật các bài privileged information learning (LUPI), multimodal-to-unimodal distillation, confidence/reliability weighted KD, uncertainty-aware KD, missing-modality robustness. Lập bảng đối chiếu *đóng góp của từng paper* trước khi tự tuyên bố novelty.

---

## 15. Việc cần làm NGAY trong buổi triển khai đầu tiên

1. [ ] Clone GitHub chính thức, đọc `data-preprocessing.ipynb`, `SFT-Config.yaml`, `evaluate.py`.
2. [ ] Tải CODe và xem `README`/dataset tree; ghi lại dung lượng và license.
3. [ ] Mở CSV: xuất `columns`, `shape`, 5 bản ghi đã ẩn thông tin cá nhân và thống kê `patient_id/checkup_id`.
4. [ ] Kiểm tra 20 lượt khám ngẫu nhiên: tên file ảnh, modality, văn bản lâm sàng, nhãn và nguy cơ leakage.
5. [ ] Chốt nhãn chính: **six-class single-label** hay task khác có quy tắc rõ ràng.
6. [ ] Xây manifest và group split; viết unit test patient overlap.
7. [ ] Chạy ResNet-50 + mean pooling cho 1–2 epoch **chỉ để smoke test**, không dùng kết quả này làm kết luận nghiên cứu.
8. [ ] Lưu environment versions, seed, split hash, log configs trước khi thử DINOv3.

**Sản phẩm đầu tiên cần có:** `data_audit.md`, `label_protocol.md`, `patient_split_v1.csv`, `checkups.parquet`. Khi bốn file này tin cậy, phần modeling mới đáng để đầu tư GPU thuê.

---

### Các nguyên tắc bất biến của dự án

- Mọi baseline phải so sánh **cùng protocol, cùng visit set**, chỉ khác thành phần đang kiểm nghiệm.
- Student **không nhận clinical input trong forward**, kể cả vô tình qua cached features/metadata.
- Patient-level split + chống duplicate/leakage là điều kiện bắt buộc.
- Mọi lựa chọn kiến trúc/hyperparameter thực hiện từ train/validation, **không tuning trên test**.
- Công bố kết quả không thuận lợi và giới hạn tính tổng quát. Đây là nghiên cứu hỗ trợ quyết định, **không thay thế chẩn đoán của bác sĩ**.

## Ghi nhận triển khai v4 (2026-10-28)

Đã triển khai Stage 1 trong bốn notebook hiện có ở `Khóa luận/`, không tạo bộ notebook thứ hai.
Tham số chính nằm ở `Khóa luận/configs/code/research.json`; giải thích code ở `Khóa luận/_support/TECHNICAL_GUIDE.md`.

- **Encoder frozen:** DINOv3-B/L và BGE-M3 cố định; huấn luyện gated MIL/projection/fusion/head. Chưa làm Stage 2/3 partial/full encoder fine-tuning.
- **Ngôn ngữ chính:** chỉ tiếng Anh theo yêu cầu của người làm khóa luận. Chinese/original-language ablation để sau.
- **Clinical:** mặc định age/gender/chief_complaint, chặn trường có từ chẩn đoán và ghi audit. Chưa xác minh thời điểm thì vẫn `screened_exploratory`; không coi regex là chứng nhận pre-diagnostic.
- **OOF:** 5 folds theo split_group của train (liên kết patient và ảnh trùng), hai Teacher riêng theo fold. Không huấn luyện trên nhóm bệnh nhân đang dự đoán; số epoch cố định được chọn từ validation độc lập.
- **Sửa chuẩn hóa loss:** weighted KL dùng mean weight cố định từ toàn bộ OOF train. Chia cho tổng weight của mỗi microbatch sẽ làm gate mất tác dụng ở batch1 nên không áp dụng cách đó.
- **Tách temperatures:** reliability gate dùng temperature1, distillation KL dùng temperature4 và T²; Teacher logits detach.
- **Feature KD optional:** align sang frozen DINO-L image descriptor, tránh ghép latent fused của các Teacher folds có hệ tọa độ khác nhau.
- **Matrix mặc định:** E01–E09, 3 seeds, tổng27 full fits +30 Teacher OOF fits. E10 được hỗ trợ dưới method kd_feature nhưng chưa bật mặc định; E11/E12 và ablations mở rộng để sau.
- **Không thay split:** dùng lại code_v1, tạo research_features_v1 riêng; cache BiomedBERT/checkpoints v2/v3 không dùng làm kết quả của phương pháp mới.
- **Test:** notebook02 không mở test. Notebook03 mặc định validation, cần ghi protocol đã khóa trước khi tạo/đánh giá shard test. Pending review của release vẫn được giữ.

Kiểm thử tổng hợp kiểm chứng đường đi phần mềm (OOF/KD/resume/Student chỉ ảnh), không chứng minh chất lượng pretrained trên COde.
Cần quyền truy cập DINOv3 trên Hugging Face để notebook00 tải weights thật; chưa có full COde DINO results trong phiên sửa local này.
Các yêu cầu về review nhãn/near-duplicate/clinical timing, so sánh partial fine-tuning, ablations mở rộng và external validation vẫn là công việc nghiên cứu tiếp theo.
