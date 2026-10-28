# COde — nghiên cứu DINOv3 và clinical knowledge distillation

Bản v4 thực hiện **Stage 1 frozen features** trong [plan](../CODe_Research_Implementation_Plan.md):
Visual Teacher DINOv3-L, Multimodal Teacher DINOv3-L + BGE-M3, Student DINOv3-B chỉ dùng ảnh.
Notebook02 chạy các hệ thống chính và đối chứng bằng một lần **Run All**.
Chỉ làm COde; MeMoSA để giai đoạn kiểm chứng sau.

| Mở file | Công việc |
|---|---|
| [00 — Offline setup](COde_00_Kaggle_Offline_Setup.ipynb) | Tải đúng weights/wheels khi có Internet, pin revisions/checksums, thử mini COde |
| [01 — Data preparation](COde_01_Data_Audit_and_Preparation.ipynb) | Audit/EDA/release giữ nguyên; **mục15** tạo cache DINO/BGE mới |
| [02 — Train](COde_02_Train_Baselines.ipynb) | Preflight → Teachers → group-OOF → các Student/đối chứng → export Student |
| [03 — Evaluate](COde_03_Evaluate.ipynb) | Mean/std, per-class, calibration, bootstrap/paired comparisons, lỗi và subgroup |
| [Cấu hình nghiên cứu](configs/code/research.json) | Nguồn cấu hình duy nhất cho phương pháp mới |
| [Kỹ thuật chi tiết](_support/TECHNICAL_GUIDE.md) | Biểu diễn, loss, OOF, checkpoints, giới hạn của nghiên cứu |

Tên `COde_02_Train_Baselines.ipynb` được giữ để các đường dẫn đang dùng không bị hỏng.
Code v2/v3 còn trong package để tái lập thí nghiệm cũ; notebook mới gọi `clindistill.research`.

## Thay đổi so với bản cũ

- Teacher Visual và Multimodal là hai mô hình độc lập, cùng encoder DINO-L; Student dùng DINO-B.
- Clinical mặc định chỉ **age, gender, chief_complaint tiếng Anh**; không dùng examination/radiographic interpretation/diagnosis/treatment làm đầu vào chính.
- Teacher targets của train lấy bằng **5-fold out-of-fold theo split_group**, thay cho dự đoán trên chính mẫu Teacher đã fit.
- So sánh no-KD, vanilla KD từ Visual/MM, Confidence KD và Reliability KD trên cùng cohort.
- Weight normalization cố định từ toàn OOF train: trọng số vẫn có tác dụng khi batch=1 và tích lũy gradient.
- Không tự đổi weights thiếu thành random; không dùng cache BiomedBERT cũ thay BGE-M3.

**Encoder ảnh và text được cố định.** MIL, projection, fusion và classification heads được huấn luyện.
Partial/full fine-tuning encoder ở Stage 2/3 của plan chưa được triển khai trong protocol này.
L4 phù hợp để trích xuất đặc trưng lần lượt rồi chạy các heads/OOF; code không tự huấn luyện DDP trên nhiều GPU.

## Chạy trên Kaggle theo thứ tự

1. Upload [source ZIP](_support/dist/clindistill-code-source.zip) mới thành dataset source riêng tư.
   Phải thấy `clindistill/research/`, `configs/code/research.json` và `_support/research_bank.py` sau giải nén.
2. Chuẩn bị quyền Hugging Face cho [DINOv3-B](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
   và [DINOv3-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
   Chấp nhận điều khoản, tạo read token, thêm **Kaggle Secret `HF_TOKEN`**, cấp quyền cho notebook00.
   Token không lưu trong source, manifest hay output bank.
3. Mở **00 mới**, môi trường Linux/Kaggle **có Internet**, cùng Python/scientific stack với phiên GPU đích.
   Gắn source mới và `code-training-inputs` có ảnh/prepared release. Giữ `MODE='prepare'`, Run All.
   Notebook tự trích mini có **2 nhóm bệnh nhân độc lập/lớp**; mini cũ có ca cùng nhóm nên chưa đủ OOF.
   Bank tải DINO-B/L, [BGE-M3](https://huggingface.co/BAAI/bge-m3), ResNet/Swin pretrained và Linux wheels.
   Mặc định thử mini thật trước khi chuyển sang full study. **Save Version** để giữ output.
4. Mở **01 mới** trong phiên GPU offline. Gắn **output bank00 + code-training-inputs** có ảnh và prepared release.
   Vì đã có `code_v1`, chỉ chạy **setup đầu notebook rồi mục15**, không cần audit lại 58 nghìn ảnh.
   Mục15 mặc định tạo train/validation ở `/kaggle/working/data/COde_processed/research_features_v1`.
   **Save Version** để giữ cache.
5. Mở **02 mới**, gắn **output bank00 + output cache01**, chọn GPU, tắt Internet, **Run All**.
   Không cần gắn raw data khi chỉ train từ cache. Output ở `runs/code_research_v1/` và `student_export/code_research_v1/`.
6. **Save Version** sau train. Mở **03**, gắn bank/cache/output train; Run All để báo cáo validation.
   Chỉ mở test sau khi khóa protocol và ghi rõ các vấn đề review còn lại.

Bank mới lớn hơn gói cũ vì chứa DINO-L/B và BGE-M3. Gắn **Notebook Output trực tiếp** sang phiên sau,
không cần download ZIP về máy rồi upload lại. `code_offline_v1.zip` vẫn phục vụ pipeline cũ nhưng **không đủ cho v4**.
Nếu đổi packages sau khi đã import, restart session rồi chạy lại từ đầu.

Nếu môi trường GPU đích khác phiên chuẩn bị: notebook00 `MODE='inspect'` in Python/base packages để đối chiếu.
Bootstrap kiểm tra chính xác phiên bản và SHA256 trước khi cài `--no-index`; không tự cài lại CUDA stack.
Dependencies bổ sung pin Transformers **4.57.1**, có hỗ trợ DINOv3; binary wheels phải đúng Linux/Python.
Xem [Transformers DINOv3](https://huggingface.co/docs/transformers/v4.57.1/model_doc/dinov3)
và [pip download](https://pip.pypa.io/en/stable/cli/pip_download/).

## Notebook02 chạy những gì?

| Hệ thống | Encoder ảnh | Clinical trong forward | Mục đích |
|---|---|---|---|
| ResNet-50 MIL | ResNet-50 ImageNet | Không | Đối chứng CNN |
| Swin-B MIL | Swin-B ImageNet | Không | Đối chứng Transformer phân cấp |
| Student no-KD | DINOv3-B | Không | Mốc kiểm tra lợi ích KD |
| Visual Teacher | DINOv3-L | Không | Teacher ảnh mạnh hơn |
| Multimodal Teacher | DINOv3-L | BGE-M3 + age/gender | Kiểm tra lợi ích privileged information |
| KD Visual | DINOv3-B | Không | Distill từ Visual Teacher OOF |
| KD Multimodal | DINOv3-B | Không | Vanilla KD từ MM Teacher OOF |
| Confidence KD | DINOv3-B | Không | Weight chỉ dùng entropy confidence |
| Reliability KD | DINOv3-B | Không | MM correctness + confidence + Visual support |

Mặc định **3 seeds (2026/2027/2028), 5 folds**, tối đa 30 epochs/head, patience8, batch32,
AdamW LR=1e-3, weight decay=1e-4, gated MIL width256, dropout0.2, ảnh224, tối đa8 photo/ca.
9 hệ thống ×3 =27 full fits, cộng30 OOF fits = **57 fits**. Encoder không được fine-tune trong các fits này.
Teacher OOF chạy số epoch cố định lấy từ best validation epoch của Teacher tương ứng;
không chọn checkpoint dựa trên nhãn fold đang giữ lại.

Mỗi hệ thống lưu `best.pt`, `last.pt`, `protocol.json`, `history.csv`, `result.json`.
Source/config/cache/targets thay đổi sẽ chặn resume vào thư mục cũ. Đổi RUN_NAME khi thay phương pháp.
Nếu ngắt Kaggle: Save Version giữ output, gắn vào phiên sau và đặt `PREVIOUS_STUDY_OVERRIDE` tới study folder.
Resume ở cuối epoch, không hứa tiếp tục đúng giữa minibatch.

## Data và cache ở đâu?

```text
Data/
  COde-Dataset (1)/              CSV/JSON/Images gốc
  COde_processed/
    complete_dataset_en.csv     CSV tiếng Anh để đọc/EDA
    code_v1/                    release.json + manifests, không đổi split
    clinical_biomedbert_v1/      cache cũ, chỉ dùng tái lập v2/v3
    research_features_v1/        cache mới từ mục15 notebook01
Khóa luận/
  COde_00...03...ipynb           bốn notebook chính
  configs/code/research.json
  clindistill/research/          data, models, losses, training, reporting
  _support/                     helpers, tests, tài liệu kỹ thuật, dist
Paper/                          paper tham khảo
```

Cache mới có `feature_manifest.json`, `train.npz`, `validation.npz`, clinical whitelist/audit và `_chunks/`
để tiếp tục trích xuất ảnh bị ngắt. Chunk không chứa một lần train khác hay split mới.
Cache chứa cả ảnh/vector lâm sàng, ID và labels: giữ riêng tư như dữ liệu gốc, **không push GitHub**.
Gói source/wheel không chứa dataset hay pretrained weights. Dữ liệu full ở một chỗ; outputs trên Kaggle ở `/kaggle/working`.

Cohort single-recorded-target có photo: **4.225 train / 917 validation / 910 test**, sáu lớp theo release.
Đơn vị dự đoán là lượt khám; split_group nối bệnh nhân có ảnh trùng. Không coi nhãn không ghi nhận là âm tính,
không tạo lớp healthy, không dùng JSON QA như những bệnh nhân mới.

## Mức độ hoàn thành và giới hạn

Code có kiểm thử synthetic cho DINO/XLM-R backends local, OOF group separation, KD gradient,
Student chỉ ảnh, checksum, full notebook02 Run All, resume và export/inference.
Đó là kiểm chứng phần mềm, **chưa phải kết quả huấn luyện DINO pretrained trên COde đầy đủ**.
Chưa tải gated DINO weights ở máy local này; notebook00 thực hiện việc đó bằng quyền Hugging Face của bạn.

Release hiện vẫn **experimental**: cần review near-duplicates, mapping/single-target labels và thời điểm clinical.
Chặn từ chẩn đoán không chứng minh chief complaint là pre-diagnostic; default policy ghi **screened_exploratory**.
Chỉ thêm history sau khi có review record. Kết quả chưa thể mặc định gọi là bài Q1, SOTA hay đủ nộp mà không chạy/đánh giá.
Các bước partial fine-tuning, clinical/MIL ablations mở rộng và external validation là phần phát triển tiếp của plan.

Chạy kiểm thử: `py -3.13 -m unittest discover -s _support/tests -p "test_*.py"` từ `Khóa luận/`.
