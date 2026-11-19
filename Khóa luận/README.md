# ClinDistill — package 0.6.0, bank v6, frozen-feature research

Phân loại một trong sáu target ghi nhận trên cohort có photo và đúng một mapped target. DINOv3-B/L và BGE-M3 frozen; chỉ MIL/projection/fusion/head được train. Clinical dùng cho Teachers; Student chỉ nhận ảnh. Không có healthy class, chưa chứng minh mọi nhãn đều quan sát được trên photo.

**Thao tác Kaggle: [_support/KAGGLE_RUNBOOK.md](_support/KAGGLE_RUNBOOK.md).**

## Các giai đoạn

| Config | Hệ thống | Fits / 3 seeds | Export |
|---|---|---:|---|
| `research_visual.json` | Visual Teacher; Student single, mean, attention | 12 | `student_no_kd` |
| `research_clinical.json` | Visual/MM; clinical-only, demographics, masked, shuffled | 18 | Không có Student |
| `research_transfer.json` | Visual/MM; no-KD, Visual KD, MM KD | 45 | `kd_multimodal` |
| `research.json` | Core gộp, gồm các đối chứng nền tảng | 63 | `kd_multimodal` |
| `research_controls.json` | CNN/Swin, Reliability/components, label smoothing, weighted CE, clinical/MIL controls | 87 | `kd_reliability` |

Notebook02/03 mặc định `CONFIG_FILE='research_visual.json'`. Sau validation visual, chuyển sang clinical rồi transfer nếu phù hợp. Controls dùng để kiểm nghiệm giả thuyết; không mặc định phải thắng. Không sửa config rồi resume vào study cũ. Stage độc lập lặp một số fits; core gộp tránh các fits lặp đó.

`features.json` cố định encoder/preprocessing cho mọi profile, vẫn tạo DINO-B/L, ResNet, Swin và BGE để cache dùng được cho controls. Không dùng profile visual để dựng cache thiếu backbone rồi kỳ vọng chạy controls đầy đủ.

## Luồng thực hiện

1. Local đóng gói bằng `_support/research-notes/package_kaggle.py`.
2. Notebook00 **bank v6/source 0.6.0** online trên Linux: inspect target, tải wheels/weights. CPU→CUDA chỉ cùng Torch/vision release; scientific stack bao gồm fsspec phải khớp.
3. Notebook01 audit tạo `code_v1`, hoặc Features Only nếu đã có prepared release. Không chia lại split khi chưa đổi protocol. Features dùng `features.json`.
4. Notebook02 chạy từng profile; giữ output để resume và đánh giá.
5. Notebook03 validation; khóa source/config/checkpoints trước khi tạo/mở test shard. Protocol lock không xóa pending reviews.

## CLI hiện hành

Chạy từ `Khóa luận` sau khi cài project:

```bash
python -m clindistill plan --config configs/code/research_visual.json
python -m clindistill review-status --prepared ../Data/COde_processed/code_v1 --audit _support/audit/code_review_v5
python -m clindistill train --config configs/code/research_visual.json --features /path/research_features_v3 --study /path/runs/code_visual_v6
python -m clindistill evaluate --features /path/research_features_v3 --study /path/runs/code_visual_v6 --output /path/reports/visual
```

`features`, `lock-test`, `export-student` có `--help` riêng. `--allow-cpu-debug` chỉ cho config `debug=true`. Trainer cũ raw-image/BiomedBERT dùng `python -m clindistill legacy ...`; không dùng `research*.json` cho legacy trainer.

## Migration 0.5.1 → 0.6.0

- Giữ prepared release/split và ảnh gốc. `research/data.py` không đổi: cache v3 từ0.5.1 giữ được nếu checksums/policy/models khớp.
- Tạo source ZIP/project wheel/bank v6 mới. Không patch hàm trong memory hoặc sửa manifest để vượt kiểm tra bank cũ.
- Tạo study mới với run_name `_v6`; scheduler/controls/CLI đổi source hash. Không resume heads/OOF targets cũ.
- `features.json` giữ image policy/model fingerprints như chuẩn0.5.1. Không thay policy để cố đọc cache không tương thích.

## Review và giới hạn

`review-status` chỉ đếm tiến độ, không xuất clinical text/IDs hoặc tự phê duyệt. Đọc [REVIEW_PROTOCOL.md](_support/REVIEW_PROTOCOL.md) để thực hiện/adjudicate review và tạo release mới khi cần sửa dữ liệu.

Mặc định `study_mode='exploratory'`. Mỗi study có `research_readiness.json`, cohort summary và source/feature/config identity. `confirmatory` cần release `reviewed`, không pending reviews/debug, clinical review có provenance khi dùng clinical. Đạt các điều kiện này vẫn không tự chứng minh clinical safety hoặc novelty.

Phương pháp: [TECHNICAL_GUIDE.md](_support/TECHNICAL_GUIDE.md). Bằng chứng: [VERIFICATION.md](_support/VERIFICATION.md). [Protocol hiện hành](_support/RESEARCH_PROTOCOL.md). Không đưa dataset/vectors/predictions/weights/checkpoints lên Git.
