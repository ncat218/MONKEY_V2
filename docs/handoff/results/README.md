# Mục lục kết quả · 22/11/2026

Xem [handoff](../../../handoff.md) để hiểu B0/B-Clean/B1, dữ liệu và các giới hạn. Tất cả số main-flow dưới đây là development; không có TEST evaluation mới. Giữ raw reports nguyên trạng, kể cả tên/path HPC lịch sử. Các link tại đây trỏ đến artifacts tải được từ GitHub.

| Giai đoạn | Báo cáo | Trạng thái / điểm cần nhớ |
|---|---|---|
| S1 LowRes | [Report](../artifacts/research_20261122/s1/results/pilot_report.md) | Job11492, 500 nguồn; không so trực tiếp với remaining-validation |
| S2-A | [Report](s2a_report.md) | Job11907, lựa chọn LR2 theo protocol |
| S2-B blur/JPEG | [Report](s2b_report.md) | 500 nguồn; descriptive, không tune final method |
| B1 | [Training report](b1_report.md), [calibration](calibration_report.md) | Native và common `.30/.40`; đã có checkpoint |
| B-Clean | [Report](bclean_report.md) | Clean-only control cùng budget |
| EDA B0/B-Clean/B1 | [Paired EDA](paired_adaptation_eda_report.md), [B1 diagnostics](b1_cal_diagnostics.md), [G4](g4_report.md) | Lịch sử common point; proxy chưa phải visual cause |
| PRC M | [Report](prc_report.md) | Không tăng đáng kể; mask active3,04% |
| Restoration RGB | [Bicubic/EDSR report](restoration_report.md) | +0,60 pp EDSR trên ngưỡng cũ, FP/latency tăng; chưa đo Clean |
| Feature KD,48499 | [Report](../artifacts/research_20261122/feature_kd/results/pilot_report.md) | Thêm1k, không vượt continued control |
| PR/quad calibration,63985 | [Report và toàn grid](../artifacts/pr_quad_audit_v1/README.md) | Chọn threshold trên development đã xem; không ghi đè frozen reports |
| Fixed diagnosis,73827 | [Report](../artifacts/research_20261122/fixed_diagnostics/results/fixed_diagnostics_report.md) | `.40/.60`,127 paired cards; review PENDING |
| CRAFT screen,74220 | [Report](../artifacts/research_20261122/craft/results/craft_report.md) | Standalone defaults; chưa train auxiliary head |
| Latent pilot,81918 | [Report](../artifacts/research_20261122/latent_pilot/results/latent_report.md) | Recovery1k; không vượt B1 |
| Latent10k,88935 | [Report/CI](../artifacts/research_20261122/latent_10k/results/latent_report.md) | A0/A1/A2 đủ10k, không có lợi ích adapter |
| Fusion recovery111683 | [Final report](../artifacts/fusion_completed_111683/fusion_report.md), [multi-metric analysis](../artifacts/fusion_completed_111683/README.md) | COMPLETED;4 nhánh10k,7.524 tasks/model; có tiny-recall gain/FP tradeoff |
| EDSR recheck163012 | [Protocol](../../edsr-recheck-v1.md), [GPU smoke PASS](../artifacts/edsr_recheck_163012/) | RUNNING; cùng B1 frozen,cùng2 calibration points,Clean và transfer |
| TASK3 audit18/11 | [Preflight lịch sử](task3_preflight_20261118.md) | STOP vì SSH timeout ở ngày đó; không phải kết quả TASK3A/C/D hoàn tất |

[Bảng máy đọc](current_fixed_results.csv) lấy trực tiếp từ JSON nguồn, chứa P/R/Hmean dưới dạng fraction cùng TP/FP/FN; không gộp common và expanded, pilot và full. B0 dùng setting riêng đã chọn; CRAFT có resize/default riêng. CSV này là chỉ mục, raw JSON/CI và source counts là bằng chứng chính.

## Kết luận hiện có

- Fine-tune miền dữ liệu là yếu tố lớn: B-Clean đã đạt LR2 75,51%. Mixed B1 đạt 76,90% và tiny recall cao hơn nhưng thêm FP, giảm Clean.
- Expanded calibration cải thiện B1 LR2 +1,610 pp so `.30/.40`; đó là hậu xử lý, không phải module kiến trúc.
- PRC, feature KD và residual/gated adapter sau FPNC chưa vượt matched controls. CRAFT standalone chưa cạnh tranh với B1 trong screen hiện tại; còn teacher QC chưa review.
- EDSR có lợi ích nhỏ ở operating point cũ, chưa đủ để kết luận là giải pháp cuối do thiếu Clean và transfer, có chi phí latency/FP.
- Fusion đã hoàn tất: Hmean gần ngang B1; ASF tăng tiny recall so B1 với FP/chữ lớn tradeoff. Tiny gain so matched Nearest còn chưa chắc. Không tự đổi primary endpoint hoặc chọn final winner từ phân tích hậu kiểm. [Workflow tiếp](../../research/current-workflow.md) nêu các quyết định còn thiếu.
