# MONKEY_V2 · Low-resolution scene text detection

CS431: MonkeyOCRv2-AS → FPNC → DBNet, ảnh TextVQA với word-level ground truth từ TextOCR. Mục tiêu là đo ảnh hưởng native LR2, tách thích nghi miền khỏi thích nghi độ phân giải, và kiểm tra cải tiến vượt baseline đã calibration. Đây là nhánh **text detection**, chưa phải reproduce toàn bộ document parsing/recognition của MonkeyOCRv2.

**Cập nhật 22/11/2026:** B-Clean/B1, PR/quad diagnosis, CRAFT screen và latent 10.000 updates đã có báo cáo. Chưa thấy lợi ích của residual/gated adapter sau FPNC. Job fusion **106090 FAILED ở cleanup khi đánh giá ASF**, sau khi Nearest/ASF đã train đủ10k. [Recovery v2](docs/fusion-recovery-v2.md) giữ checkpoint/cache và chạy tiếp phần thiếu; chưa có kết quả cuối đầy đủ.

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

Code submit nằm trong repo để tái lập từng giai đoạn; không chạy tất cả launcher. [Thiết kế fusion đang chạy](docs/fusion_research_and_generalization.md) gồm pilot/full trong cùng một job, dừng khi probe kỹ thuật FAIL. Không sửa code worker hay submit trùng khi job còn chạy.

Ảnh dataset, optimizer states và cache P-map toàn bộ không được đưa vào Git. Repo có IDs, hash, config, checkpoint inference, số đếm theo nguồn và review có chọn lọc. [Hướng dẫn cũ và các đề xuất trước đây](docs/archive/README.md) được lưu để truy lịch sử; không phải workflow mặc định hiện tại.
