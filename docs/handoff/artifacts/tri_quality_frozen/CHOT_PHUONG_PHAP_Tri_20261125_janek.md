# Chốt phần của Trí · Phương pháp tốt nhất: learned box-quality scorer (v1) · 25/11/2026

## 1. Phương pháp
**Pipeline** (một checkpoint B1, một forward, detector không train thêm):
```
ảnh → B1 (frozen, FP32, batch 1) → P-map
    → DB post-processor của Tuấn, geometry mask_040 (mask .40, unclip 1,5, min width 5), min_text_score = 0
    → ứng viên raw ≥ 0,20 + 13 đặc trưng tính từ P-map trong và quanh box
    → logistic L2 (fit trên TRAIN) → giữ box p ≥ 0,3316 → evaluator chính thức
```
- **Đặc trưng:**
  - phân bố P trong mask của box: raw, std, q10/q50/q90, max;
  - P ở vành quanh box (`ring_mean`) và trong quad (`quad_mean`);
  - độ lấp đầy: polygon/quad, polygon/hull;
  - kích thước: cạnh ngắn chuẩn hóa, log aspect, số đỉnh.
  - Không dùng condition, ID hay GT.
- **Học:**
  - nhãn mô phỏng evaluator (IoU ≥ 0,5 với GT hợp lệ = 1);
  - 1.600 nguồn TRAIN để fit;
  - ngưỡng chọn trên 400 nguồn TRAIN-calibration (mục tiêu mean Hmean Clean/LR2, ràng buộc tiny);
  - không dùng A/B/TEST.
- **Bản chất:** chấm lại độ tin cậy định vị của box bằng thông tin không gian của P-map. Đây là adaptation của ý "học độ tin cậy định vị" (Jiang et al. ECCV 2018; Huang et al. CVPR 2019, chỉ đọc ở mức abstract), **không** phải kiến trúc mới.
- **Code:**
  - `mixab/boxquality_janek.py` (fit, trích);
  - `mixab/boxquality_devA_janek.py` (đánh giá dev A);
  - model đóng băng `docs/handoff/artifacts/janek/boxquality_v1_T4/model_janek.npz` (SHA `81867a49…`).

## 2. Kết quả (dev A, 1.130 nguồn, Clean + native LR2, Kaggle T4, đánh giá một lần với model đóng băng)
| | Clean H | LR2 H | mean | Clean TP / FP | LR2 TP / FP |
|---|---|---|---|---|---|
| B1 `.40/.60` (baseline tốt nhất của Tuấn) | 79,13 | 76,80 | 77,96 | 29.726 / 7.987 | 27.577 / 6.815 |
| **Box-quality v1** | **79,39** | **76,97** | **78,18** | 29.674 / 7.661 | 27.545 / 6.607 |

- **Δ mean +0,22 pp [+0,13; +0,30]**, Clean +0,26 [+0,16; +0,38], LR2 +0,17 [+0,07; +0,27]. CI bootstrap ghép cặp theo nguồn, 2.000 lần.
- **Cơ chế:** chủ yếu bớt FP (−326 Clean / −208 LR2), mất ít TP (−52 / −32). Tiny −0,93 / −0,43 pp.
- **Tổng quát hóa:** so với ngưỡng raw chọn lại trên cùng TRAIN là +0,22 trên A, so với +0,14 trên calibration. Phần lợi đến từ đặc trưng, không phải từ ngưỡng.
- Bỏ ảnh trang báo chi phối FP thì vẫn dương.
- **Chi phí:** latency median 1,065× [1,034; 1,086] (+32 ms/ảnh), VRAM không đổi.
- **Nhãn khóa trước:** `SMALL_USEFUL`. Nguồn: `docs/research/janek/boxquality_devA_result_20261125_janek.md`, `docs/handoff/artifacts/janek/boxquality_devA_20261125/`.

## 3. Vì sao là tốt nhất
| Hướng | Kết quả tốt nhất | Vì sao xếp sau |
|---|---|---|
| **Box-quality v1** | **+0,22 [+0,13; +0,30] mean, Clean và LR2 đều CI > 0** | — |
| CS-B1@0.7 | LR2 +0,18 [−0,21; +0,58] so với B1 `.40/.60`; Clean +0,39 [0,01; 0,81] | CI LR2 chứa 0; 2 checkpoint; latency 1,99×; trượt luật dừng A |
| Flip-TTA | +0,27 [−0,16; 0,66] LR2 | `INCONCLUSIVE`, tương đương nâng ngưỡng, 1,95× |
| MSB1 | +0,22 [−0,40; 0,81] | CI chứa 0 |
| Mix A, E-sel, E-BN, QRoute, DualAgree v1, ngưỡng theo kích thước, XV-B1, zoom | — | Âm hoặc trượt tiny |
| Learned box refinement | −0,82 pp trên TRAIN-calibration | `NO_SIGNAL`, A không đánh giá |

Box-quality là hướng duy nhất có cả hai điều kiện dương với CI > 0 so với baseline đã calibration, tốn chi phí gần như bằng 0, và có spec khóa trước khi chạy.

## 4. Được và không được tuyên bố
**Được tuyên bố:**
> "Trên development split A (1.130 nguồn, Clean + native LR2), box-quality scorer học trên TRAIN và đóng băng hơn B1 `.40/.60` +0,22 pp mean Hmean (CI95 [+0,13; +0,30]), với Clean +0,26 và LR2 +0,17. Chi phí là 1,065× latency median, không thêm VRAM, một checkpoint và một forward."

**Không được tuyên bố:**
- **Đạt cổng nhóm:** chưa đạt. g1 mean thiếu 0,32 pp so với B-Clean + 0,5; g2 Clean thiếu 1,13 pp so với B-Clean − 0,3; g3 tiny đạt.
- Cải tiến trên TEST: chưa mở. B: chưa mở.
- Kết quả chạy trên T4, nên gắn `PENDING_REFERENCE_CHECK`.

**Giới hạn đo:**
- Giai đoạn 0 cho thấy phần lớn FP còn lại là chữ thật thiếu nhãn (20/24 crop trong mẫu teacher-background). Hmean có thể đang phạt sai.
- Khoảng cách tới cổng g2 phần lớn đến từ việc B-Clean ít bắn vào chữ không nhãn.

## 5. Bàn giao
- **Báo cáo nháp:** đã cập nhật bảng §6.1 và §6.4, xem `docs/research/janek/BAO_CAO_KET_QUA_Tri_20261124_janek.md`.
- **Còn lại, cần user/Tuấn quyết:**
  1. gửi Tuấn: tin dự thảo cần cập nhật số dev A và refinement;
  2. có đăng ký box-quality v1 (model + ngưỡng đã khóa) vào benchmark TEST cuối của nhóm hay không. Claude không tự mở.
