# Kỹ thuật ClinDistill 0.6.0 — frozen-feature, selected-cohort classification

Code hiện hành ở `clindistill/research/`. [README](../README.md), [Kaggle runbook](KAGGLE_RUNBOOK.md), [review protocol](REVIEW_PROTOCOL.md), [protocol nghiên cứu](RESEARCH_PROTOCOL.md).

## Hợp đồng dữ liệu và tác vụ

Một mẫu = một lượt khám có ≥1 photo và đúng một recorded mapped target trong sáu lớp. Các nhãn khác có thể cùng tồn tại; nhãn vắng không được xác nhận âm tính. Không có healthy class. Cohort photo hiện tại4.225/917/910 train/validation/test, chưa được review chuyên môn đầy đủ.

`split_group` là connected component patient IDs và ảnh trùng byte/pixel. Chia theo nhóm và bootstrap nhóm; pending near-duplicate review được truyền từ prepared release sang cache/study/report. Không thay split trong training. `FeatureStore` mở train/validation; test cần entry point/protocol riêng.

`research/data.py` giữ nguyên so với0.5.1 để cache v3 đã tạo có thể tiếp tục dùng. Schema feature manifest vẫn4; tên thư mục cache v3 và schema4 là hai khái niệm khác nhau. Checksum và builder hash vẫn bắt buộc. `features.json` cố định pretrained assets/preprocessing cho toàn bộ profile, kể cả CNN controls.

## Frozen features

- EXIF → RGB → resize/pad giữ tỷ lệ, bilinear224×224; normalize theo DINO processor hoặc ImageNet cho CNN. Tối đa8 photo, chọn theo vị trí filename cách đều; không phải canonical clinical views.
- DINO-B hidden768, DINO-L hidden1024; concat CLS + mean patch, bỏ register tokens, L2 normalize →1536/2048 dimensions. BF16/FP16 encoder trên GPU, cache image vectors float16, head float32.
- ResNet50/Swin-B trước head →2048/1024 dimensions, L2 normalize. Pretraining recipes khác nhau; so sánh backbone không cô lập riêng kiến trúc.
- BGE-M3 dense normalized CLS1024, text tối đa256 tokens, báo truncation. Text frozen/eval, cache float32. Clinical thiếu → vector0; demographics5 chiều: age/120, age-present, male/female/unknown.
- Không fit scalers/PCA trên test; không đưa labels vào foundation encoders. Frozen protocol không có augmentation ảnh ngẫu nhiên.
- Chunk checksums bảo vệ resume; batching/worker có thể đổi cho throughput, không đổi optimizer batch/split/policy.

## Clinical và các model

Mặc định English age/gender/chief complaint; diagnosis/treatment/examination/full patient record không làm input. Regex loại toàn field có pattern chẩn đoán, không rewrite câu rồi tuyên bố an toàn. Regex không xác minh thời điểm; policy mặc định `screened_exploratory`. Thêm history fields cần review record và temporal verification thực.

Gated MIL: projection Linear→LayerNorm→GELU, attention `softmax(wᵀ[tanh(Vh)⊙sigmoid(Uh)])`, weighted sum, padding mask. Single chọn ảnh hợp lệ đầu theo policy; mean tôn trọng mask. Attention không phải annotation tổn thương.

Visual Teacher dùng DINO-L/MIL; MM thêm projected BGE/demographics và gated residual fusion. Masked control giữ cùng MM architecture, clinical bị che toàn bộ. Demographics-only che text; shuffled control hoán vị toàn context giữa các group trong TRAIN, validation/test tự nhiên.

Clinical-only control có text projection/head riêng, không có MIL và API từ chối visual tensors. `forward_batch` chỉ truyền clinical arguments cho control này. Student DINO-B/MIL/head từ chối mọi clinical argument; inference không gọi Teacher.

## Profiles, OOF và loss

Visual12 fits → clinical 18 → transfer 45, hoặc core gộp63. Controls 87 chứa Reliability/gate components, label smoothing, weighted CE và ResNet/Swin. Counts dùng3 seeds/5 folds; stages độc lập lặp một số full fits. CLI `plan` là nguồn count hiện hành. Clinical-only stage không xuất Student.

OOF chỉ tạo khi có KD hoặc weighted CE. Mỗi seed fit full Visual/MM Teachers, chọn epoch theo validation; StratifiedGroupKFold5 trên train groups, fit mỗi Teacher riêng từ đầu. Held-out fold không tham gia fit/checkpoint selection. **Cosine horizon luôn config.epochs**, kể cả khi OOF dừng tại selected epoch. Vì vậy OOF dùng prefix cùng lịch full Teacher. Protocol lưu scheduler horizon/source/config/feature/target identities.

Validation chọn schedule chung nên OOF không phải nested CV hoàn toàn độc lập với validation. Lưu train/heldout IDs/groups, fold assignment, checkpoint hashes và aligned logits; mỗi ca train nhận một prediction held-out. Không dùng encoder đã fine-tune toàn train để giả OOF.

`L = CE + lambda*T²*KL(softmax(teacher/T) || softmax(student/T))`, lambda1/T4 mặc định; logits Teacher detach. Gate dùng temperature1:

`w = 1[MM argmax=y] * (1-H(p_MM)/log(K)) * [epsilon+(1-epsilon)*p_Visual(y)]`, epsilon0.1.

Weighted loss là batch mean của `(w/mean_OOF_train(w))*KL_T`. Không chuẩn hóa lại theo mean microbatch; batch1 không triệt tiêu gate. Mean weight0 gây lỗi, không silent fallback. CE vẫn học mọi mẫu của KD, kể cả gate0.

Weighted-CE control dùng `CE + lambda*(w/mean_train(w))*CE`, thay auxiliary KL bằng CE và không đọc teacher soft logits trong loss. Nó kiểm tra lời giải thích do weighting; không phải bằng chứng cô lập toàn bộ cơ chế KD. `history.csv` ghi `auxiliary_loss_kind` để phân biệt.

Correctness/visual-support gates là heuristic. Có thể giảm KD ở ca khó/lớp hiếm hoặc MM-only-correct, chưa chứng minh visual sufficiency/clinical dependency. `weight_diagnostics.json` và `teacher_strata.json` giữ cách diễn giải mô tả. Optional `kd_feature` align sang frozen DINO-L image descriptor, không fused latent khác hệ tọa độ giữa folds.

## Training, evaluation và export

AdamW LR1e-3/WD1e-4; cosine30 epochs, patience8 full fits; batch32/accumulation1, gradient clip5. Không class weighting/sampling mặc định. Seeds2026/2027/2028; validation chọn checkpoint macro-F1. Resume epoch-boundary phục hồi model/optimizer/scheduler/RNG, kiểm tra fingerprint; đổi source cần study mới.

Report mean/std qua seeds, per-class support/precision/recall/F1, accuracy/balanced accuracy, AUROC khi xác định, ECE/Brier, risk–coverage. Bootstrap split_group và paired same-cases/seeds. CI điều kiện trên models đã fit, không CI retraining; chưa hiệu chỉnh multiplicity. Chỉ xuất contrast có đủ systems; visual-only stage không yêu cầu MM strata.

`freeze_test_protocol` khóa completed study/config/source/checkpoints trước khi tạo test shard. `review_record` ghi protocol/giới hạn, không phê duyệt chuyên môn. Giữ shard train/validation checksums khi thêm test. Không lựa chọn theo test.

Export dùng `export_method` đăng ký trong config; seed theo validation, không test. Artifact gồm head, DINO-B encoder, manifest/hash/source và photo-only API. Clinical stage không có Student nên notebook bỏ export. `python -m clindistill` dùng research CLI; `legacy` truy cập trainer cũ rõ ràng.

## Readiness và runtime

Mỗi study ghi `research_readiness.json`, task contract, cohort/per-class clinical availability và pending reviews. `confirmatory` bị chặn nếu debug, release chưa `reviewed`, còn pending, hoặc clinical chưa có review provenance khi sử dụng. Dù đủ điều kiện, software không tự chứng nhận novelty/clinical safety/publication readiness.

Bank6 cài project wheel0.6.0 đúng source; addons Transformers4.57.1/hub0.36.2, setuptools68–<82, fsspec được giữ theo target. Sau install, kiểm tra transitive dependency closure thực dùng. Conflict trong closure gây lỗi; conflict thư viện Kaggle ngoài closure được báo riêng. CPU preparation chỉ xác nhận metadata/local CPU backend loading; activation GPU vẫn kiểm tra target chính xác. Cần fresh Kaggle run để xác nhận môi trường thật.
