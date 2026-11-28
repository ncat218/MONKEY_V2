# Monkey: đối chiếu đầy đủ kết quả đã có,28/11/2026

Chưa hoàn tất mọi thí nghiệm. Job166501 FAILED; phần scorer đã hoàn tất và hợp lệ. Job189496 phục hồi đã RUNNING sau khi quota ghi được trở lại;probe GPU đã PASS và đang tiếp tục đánh giá B1_s43.
Job mới: đánh giá seed43 còn thiếu,zoom+EDSR±ASF±Q,B-Clean zoom và PRC2 sửa GT snapshot. Chưa có số của các nhánh mới.

2262 nguồn development,78221 GT words,5158 tiny GT words. Tiny là chiều cao GT quy về native LR2,giữ cố định khi upscale.
F dùng mask .40/score .60. Q dùng13 đặc trưng,raw pool>=.20 và cutoff .3315908613227198; không fit hay tune lại.
P/R/H/tiny ghi theo %. FP là số dự đoán sai theo evaluator đã khóa. Không dùng TEST.

## 1. Baseline,fusion,latent,phục hồi và scorer

| Phương pháp | Clean H | LR2 H | LR2 P | LR2 R | Tiny R | FP |
|---|---:|---:|---:|---:|---:|---:|
| B-Clean:100% Clean | 80.86 | 75.51 | 88.66 | 65.75 | 15.96 | 6,578 |
| B1:50% Clean +50% LR2 | 79.43 | 76.90 | 82.45 | 72.05 | 45.79 | 11,997 |
| B1 + continued FPNC/DBHead (Nearest) | 79.53 | 76.94 | 82.42 | 72.14 | 46.26 | 12,035 |
| B1 + ASF | 79.44 | 76.95 | 82.32 | 72.24 | 46.53 | 12,134 |
| B1 + Bilinear | 79.12 | 76.69 | 82.29 | 71.81 | 45.97 | 12,086 |
| B1 + DySample | 79.50 | 76.94 | 82.39 | 72.16 | 46.22 | 12,065 |
| B1 + head-only (10000 updates) | 79.47 | 76.87 | 82.40 | 72.04 | 45.89 | 12,037 |
| B1 + residual adapter (10000 updates) | 79.48 | 76.86 | 82.39 | 72.03 | 45.89 | 12,046 |
| B1 + gated adapter (10000 updates) | 79.47 | 76.86 | 82.39 | 72.03 | 45.91 | 12,042 |
| B1 + head-only (1000 updates) | 79.39 | 76.71 | 82.09 | 71.98 | 45.25 | 12,282 |
| B1 + residual adapter (1000 updates) | 79.39 | 76.71 | 82.09 | 71.99 | 45.21 | 12,282 |
| B1 + gated adapter (1000 updates) | 79.39 | 76.72 | 82.10 | 71.99 | 45.29 | 12,279 |
| B1/bicubic_x2/F | 79.38 | 77.15 | 82.28 | 72.62 | 45.97 | 12,236 |
| B1/edsr_x2/F | 79.07 | 77.79 | 81.93 | 74.05 | 49.75 | 12,774 |
| ASF/bicubic_x2/F | 79.41 | 77.25 | 82.17 | 72.88 | 47.44 | 12,366 |
| ASF/edsr_x2/F | 79.02 | 77.87 | 81.75 | 74.35 | 50.74 | 12,986 |
| B1/native/Q | 79.89 | 77.13 | 83.22 | 71.87 | 44.96 | 11,338 |
| B1/bicubic_x2/Q | 79.86 | 77.39 | 83.09 | 72.43 | 45.17 | 11,533 |
| B1/edsr_x2/Q | 79.50 | 78.10 | 82.90 | 73.83 | 48.58 | 11,915 |
| ASF/native/Q | 79.95 | 77.23 | 83.10 | 72.13 | 45.72 | 11,471 |
| ASF/bicubic_x2/Q | 79.88 | 77.50 | 83.04 | 72.65 | 46.45 | 11,606 |
| ASF/edsr_x2/Q | 79.50 | 78.23 | 82.71 | 74.22 | 49.92 | 12,132 |

Nearest/Bilinear/ASF/DySample đều train tiếp10000 updates trên B1,cùng FPNC/DBHead trainable; latent1k/10k đóng băng backbone/FPNC.
Chênh lệch ASF so Nearest vẫn chưa rõ; không quy toàn bộ mức tăng cho ASF.

## 2. Đối chứng ngưỡng raw của Trí (khác cột Q)

| Pipeline | Clean H | LR2 H | Tiny R LR2 | FP LR2 |
|---|---:|---:|---:|---:|
| B1/native/RAW | 79.41 | 76.90 | 45.91 | 12,096 |
| B1/bicubic_x2/RAW | 79.36 | 77.14 | 46.12 | 12,331 |
| B1/edsr_x2/RAW | 79.04 | 77.76 | 49.81 | 12,878 |
| ASF/native/RAW | 79.40 | 76.95 | 46.63 | 12,231 |
| ASF/bicubic_x2/RAW | 79.38 | 77.23 | 47.50 | 12,464 |
| ASF/edsr_x2/RAW | 78.98 | 77.87 | 50.81 | 13,081 |

RAW dùng score .5976406401908236 đã khóa trên TRAIN; đây là đối chứng ngưỡng,không phải model scorer.

## 3. Zoom B1 đã hoàn tất (7524 task gồm stress; bảng dưới cùng2262 nguồn Clean/LR2)

| Biến thể | Clean H | LR2 H | LR2 P | LR2 R | Tiny R LR2 | FP LR2 |
|---|---:|---:|---:|---:|---:|---:|
| global | 79.43 | 76.90 | 82.45 | 72.05 | 45.79 | 11,997 |
| guided | 77.92 | 75.34 | 80.77 | 70.58 | 46.10 | 13,143 |
| guided_edsr | 77.36 | 76.64 | 78.64 | 74.74 | 59.38 | 15,883 |
| guided_union | 77.28 | 76.35 | 77.79 | 74.97 | 55.45 | 16,747 |
| guided_verify | 80.13 | 76.31 | 85.20 | 69.10 | 37.11 | 9,386 |
| poly | 77.55 | 75.32 | 80.76 | 70.56 | 43.20 | 13,145 |
| tile2x2 | 77.80 | 75.15 | 80.92 | 70.15 | 45.19 | 12,941 |
| tile2x2_union | 77.09 | 76.28 | 77.91 | 74.71 | 54.59 | 16,569 |

Oracle windows dùng GT,chỉ là chẩn đoán: LR2 H=75.78%,tiny=46.43%;không phải phương án triển khai.
guided_edsr phục hồi crop,khác EDSR toàn ảnh. Phương án gộp mới phải giữ cùng cửa sổ crop và đối chứng để tách hiệu ứng.

## 4. Seed43 và phần dang dở

| Mô hình | Clean H | LR2 H | LR2 R | Tiny R LR2 | FP LR2 | Trạng thái |
|---|---:|---:|---:|---:|---:|---|
| B1 seed43,checkpoint cuối10000 updates | 79.47 | 76.74 | 71.69 | 42.38 | 11,849 | Đủ record tại ngưỡng cố định;chọn5k/10k và calibration chưa xong |

B-Clean seed43 train xong10000 updates,có checkpoints5k/10k,đánh giá chưa xong. B-Clean zoom chưa xong.
PRC2 cũ dừng7/200 active microbatches. CPU đã tái hiện lỗi target DBNet sửa polygon tại chỗ;run snapshot mới chưa có kết quả.
Zoom+EDSR+ASF±Q chưa có số. Không gắn số0 hoặc số dự đoán cho thí nghiệm chưa hoàn tất.

## 5. Thí nghiệm lịch sử: ngưỡng .30/.40,để riêng

| Phương pháp | Clean H | LR2 H | LR2 FP |
|---|---:|---:|---:|
| B0-cal |73.30|61.95|5129|
| B-Clean-cal |78.79|74.36|10457|
| B1-cal |77.55|75.29|16487|
| PRC M-cal |77.57|75.27|—|
| B1 train tiếp1k (KD control) |77.33|75.24|16766|
| Feature KD1k |77.32|75.24|16825|
| B1+bicubic x2 |77.42|75.24|16972|
| B1+EDSR x2 |77.01|75.89|17439|

Các hàng này không được ghép trực tiếp vào bảng .40/.60 để tính cải thiện. PRC v1 chỉ active608/20000 microbatches.
CRAFT độc lập dùng default resize/threshold riêng: Clean H70.35%,LR2 H47.00%,P86.14%,R32.32%,FP4068. Đây không phải B1+CRAFT.
Native .30/.30 của B0/B-Clean/B1 xem paired_adaptation_eda_report.md và calibration_report.md.

## 6. Kết luận có CI trên LR2

- ASF+EDSR so B1:H+0.970pp[0.569,1.385];tiny+4.944pp[3.572,6.397];FP+989.
- ASF+EDSR+Q so B1:H+1.334pp[1.003,1.680];R+2.163pp[1.825,2.531];tiny+4.130pp[2.686,5.646];FP+135[-257,543].
- Q bổ sung trên ASF+EDSR:H+0.363pp[0.204,0.578];FP-854;tiny-0.814pp[-1.299,-0.426].
- Zoom+EDSR so B1:tiny+13.591pp[10.785,16.895];H-0.264pp[-2.105,1.126];FP15883 so11997.
- ASF+EDSR so B-Clean:H+2.364pp[-0.510,4.959];tiny+34.781pp[19.645,49.667]. Không khẳng định H tăng rõ khi CI chứa0.

Tất cả là development đã dùng nhiều lần;bootstrap theo nguồn không đo biến động seed hay selection bias.
Ứng viên cân bằng hiện tại: B1+ASF+EDSR+Q. Ứng viên ưu tiên tiny recall:zoom+EDSR,đánh đổi FP/cost.
Không tự chọn winner hay chạy TEST từ bảng này.

## Nguồn

- ../artifacts/fusion_completed_111683/fusion_summary.json
- ../artifacts/research_20261122/latent_10k/results/latent_summary.json và latent_pilot/
- ../artifacts/quality_completed_166501/quality_combo_summary.json và report/complete receipts
- ../artifacts/comparison_20261128/comparison.json: machine-readable rows; zoom/checkpoint SHA from verified HPC snapshots.
