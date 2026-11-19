# Chỉ mục tài liệu nghiên cứu

Các PDF ở đây thuộc tác giả/nhà xuất bản tương ứng. Tên file không tự xác nhận nội dung; đối chiếu title/DOI trước khi trích dẫn.

| Tài liệu | Nguồn chính | Cách dùng trong ClinDistill |
|---|---|---|
| CODe, Scientific Data 2026 | [Paper](https://www.nature.com/articles/s41597-026-07342-9), [dataset](https://huggingface.co/datasets/zirak-ai/COde) | Schema/labels/protocol dữ liệu; không mặc định split phát hành patient-disjoint |
| MeMoSA, Scientific Data 2026 | [Paper](https://www.nature.com/articles/s41597-026-06998-7) | Hướng nghiên cứu thứ hai; chưa có implementation/validation hiện hành |
| MMPKD | [Paper](https://arxiv.org/abs/2508.06558) | Privileged KD; teacher clinical-only trong bài không đồng nhất với MM Teacher của dự án |
| Attention-based Deep MIL | [Paper](https://proceedings.mlr.press/v80/ilse18a.html) | Baseline pooling, không tự là đóng góp mới |
| DINOv3 | [Paper](https://arxiv.org/abs/2508.10104), [DINO-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | Frozen visual features hiện hành |
| BGE-M3 | [Model card](https://huggingface.co/BAAI/bge-m3) | Frozen dense CLS text features |
| DentVLM | [Official repository](https://github.com/ZJUI-AI4H/DentVLM), [publisher reference](https://www.nature.com/articles/s41467-026-75718-x) | Related work; chưa tái lập trong pipeline này |

## Sửa nhầm PDF DentVLM

File trước đây mang tên `A multimodal vision-language model for comprehensive dental diagnosis and enhanced clinical practice.pdf` thực chất trùng byte với `s41598-026-52410-0.pdf`, bài oral-cancer multimodal fusion. SHA256 cả hai là `74a6a473d5ff7b8264c662b0dc45df9228c8d116b942030b0b2c1c76be68c230`.

Đã bỏ bản sao mang tên sai, giữ bài oral-cancer đúng nguồn. Repo không giả thay nó bằng PDF DentVLM chưa được xác minh. Official DentVLM repository xác nhận title tương ứng; dùng nguồn chính khi tải/đọc bài. Tài liệu literature review PDF cũ cần đối chiếu với ghi chú này, không suy ra đã đọc đúng bài chỉ từ danh sách tên file.

Trước khi tuyên bố novelty, lập bảng riêng về teacher inputs, train/inference modalities, selection mechanism, OOF/label protocol và bằng chứng của LUPI/generalized distillation, MMPKD, CPKD, TriDeNT/MIL. Chỉ mục nguồn không thay thế literature comparison đã hoàn thành.
