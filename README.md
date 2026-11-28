# MONKEY_V2 · Low-resolution scene text detection

CS431: MonkeyOCRv2-AS → FPNC → DBNet, ảnh TextVQA với word-level ground truth từ TextOCR. Mục tiêu là đo ảnh hưởng native LR2, tách thích nghi miền khỏi thích nghi độ phân giải, và kiểm tra cải tiến vượt baseline đã calibration. Đây là nhánh **text detection**, chưa phải reproduce toàn bộ document parsing/recognition của MonkeyOCRv2.

**Mới nhất 28/11/2026:** [ICDAR2015 gốc, job192199 đã COMPLETED](docs/handoff/results/icdar2015_transfer_192199.md), đủ12 setting ×500 ảnh, không train/retune. B0 reproduce88,78%; B-Clean82,20%; B1 75,80%; B1+ASF+EDSR+Q78,09%. Tổ hợp tăng2,293pp so B1 nhưng chưa vượt B0/B-Clean và không tăng Recall chữ nhỏ trong phép đo này. [Audit dataset và B0 từng ảnh](docs/handoff/artifacts/icdar2015_192199/), [index500 ảnh](docs/handoff/data/icdar2015_reproduce/). Kết quả TextOCR development mới nhất nằm trong [bảng28/11](docs/handoff/results/comparison_20261128.md); không gộp hai benchmark.

**Cập nhật 22/11/2026:** B-Clean/B1, PR/quad diagnosis, CRAFT screen và latent 10.000 updates đã có báo cáo. Chưa thấy lợi ích của residual/gated adapter sau FPNC. [Fusion111683 đã COMPLETED](docs/handoff/artifacts/fusion_completed_111683/README.md): ASF/DySample gần ngang B1 về Hmean; ASF có tăng recall chữ nhỏ kèm tăng FP. [EDSR recheck](docs/edsr-recheck-v1.md) kiểm lại trên B1 đã fine-tune, cùng calibration và thêm Clean/transfer.

## Đọc theo thứ tự

1. [Handoff: dữ liệu, metric, kết quả, giới hạn](handoff.md).
2. [Workflow hiện tại và vấn đề cần giải quyết](docs/research/current-workflow.md).
3. [Mục lục toàn bộ kết quả](docs/handoff/results/README.md).
4. [Checkpoint B0, B-Clean, B1, PRC và latent](docs/handoff/checkpoints.md).
5. [Artifacts mới: CI, per-source counts, review và provenance](docs/handoff/artifacts/research_20261122/README.md).
6. [TRAIN IDs + SHA256 ảnh](docs/handoff/data/README.md), [protocol đã khóa](docs/handoff/protocol/).

## Baseline hiện tại

2.262 remaining-validation sources, đã dùng nhiều lần để chọn checkpoint/ngưỡng và phân tích. Các số này là kết quả development, không phải fresh holdout.

| Model | Clean Hmean | Native LR2 Hmean | LR2 FP | LR2 recall chữ <4 px |
|---|---:|---:|---:|---:|
| B0 — detector gốc | 74,09% | 62,38% | 4.069 | 1,01% |
| B-Clean — fine-tune 100% Clean | 80,86% | 75,51% | 6.578 | 15,96% |
| B1 — fine-tune 50% Clean / 50% LR2 | 79,43% | 76,90% | 11.997 | 45,79% |

Đây là grid hậu xử lý mở rộng, chọn riêng từng model bằng cùng quy tắc. B-Clean/B1 dùng mask `.40`, score `.60`; B0 `.30/.45`, epsilon `.005`. Các kết quả lịch sử `.30/.40` được giữ nguyên trong báo cáo gốc. Không trộn hai operating points.

## Tải và kiểm tra

```bash
git lfs install
git clone https://github.com/ncat218/MONKEY_V2.git
cd MONKEY_V2
git lfs pull
python docs/handoff/artifacts/research_20261122/verify_bundle.py
```

Nếu đã clone: `git pull --ff-only && git lfs pull`. Weights và ZIP review dùng Git LFS; file vài dòng bắt đầu `version https://git-lfs` là pointer, chưa phải dữ liệu tải đủ. Xem [cách nạp detector](docs/handoff/checkpoints.md); verifier mặc định kiểm hash/artifacts, không chạy inference hay submit job.

Code submit nằm trong repo để tái lập từng giai đoạn; không chạy tất cả launcher. [Thiết kế fusion đã kiểm nghiệm](docs/fusion_research_and_generalization.md) gồm pilot/full trong cùng một job, dừng khi probe kỹ thuật FAIL. Không sửa code worker hay submit trùng khi job còn chạy.

Ảnh dataset, optimizer states và cache P-map toàn bộ không được đưa vào Git. Repo có IDs, hash, config, checkpoint inference, số đếm theo nguồn và review có chọn lọc. [Hướng dẫn cũ và các đề xuất trước đây](docs/archive/README.md) được lưu để truy lịch sử; không phải workflow mặc định hiện tại.

[EDSR × ASF](docs/edsr-asf-v1.md): nhánh kết hợp dùng checkpoint ASF đã train, chạy sau EDSR163012 để tái dùng đối chứng; inference-only, cùng ngưỡng và cohort.
