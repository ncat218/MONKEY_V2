# Kết quả box-quality scorer đóng băng trên development A (Kaggle T4) · 25/11/2026

- **Spec bổ sung** (khóa trước khi chạy): `docs/research/janek/boxquality_devA_spec_20261125_janek.md`.
- **Bản chính:** `docs/handoff/artifacts/janek/boxquality_devA_20261125/`.
- **Output gốc:** `runtime/kaggle_downloads/boxquality_devA_T4_20261125-084951/`.

Đây là lượt chạy theo **quyết định của user sau khi xem pilot**. Pilot TRAIN-calibration vẫn giữ nhãn `POSITIVE_BELOW_GATE`.

- Scorer và ngưỡng **không fit, không chọn lại** trên A.
- A đã được xem nhiều lần trong các hướng trước, nên kết quả là **mô tả trên development**, không phải xác nhận trên TEST.
- Số chạy trên T4, nên gắn `PENDING_REFERENCE_CHECK` so với A100/H200.

## 1. Tiếp nhận (đều đạt)
- **Code manifest** `5693b621…`, trùng gói đã chuẩn bị.
- **Môi trường:** Pillow 11.3.0, DBPostprocessor `9bfff8aa…`, Tesla T4 (chỉ GPU 0), torch 2.9.0+cu128, MONKEY_V2 `dc19b85` sạch.
- **Chuẩn bị dữ liệu:** 1.130 nguồn, 2.260 task. Mọi ảnh trùng SHA đã khóa (Clean theo bundle, LR2 theo task). Test trong notebook: 149 OK (skip 9).
- **Trích:** 2.260/2.260 task. Đường đặc trưng trùng tuyệt đối `capture_candidates` ở **cả 2.260 task**.
- **Đánh giá:** evaluator chính thức trùng đường nhanh ở 2.260 task × 3 cánh. Không có đặc trưng không hữu hạn.
- **So với dump T4:** ứng viên trùng tuyệt đối 1.130/1.130 ở mỗi điều kiện. Hmean F tính từ bản ghi sống bằng đúng F tính từ dump.
- **Latency:** box của cả hai cánh trong lượt đo trùng lượt trích, 200/200 ảnh.
- **Tính lại ở local** từ bản ghi đã tải: `devA_result.json` trùng tuyệt đối, chỉ khác đường dẫn file latency.

## 2. Kết quả chính: A, 1.130 nguồn

| Cánh | Clean H | Clean P / R | Clean TP / FP / FN | LR2 H | LR2 P / R | LR2 TP / FP / FN | mean |
|---|---|---|---|---|---|---|---|
| F = B1 `.40/.60` | 79,13 | 78,82 / 79,43 | 29.726 / 7.987 / 7.697 | 76,80 | 80,18 / 73,69 | 27.577 / 6.815 / 9.846 | 77,96 |
| RAW (raw ≥ 0,5976) | 79,12 | 78,74 / 79,49 | 29.749 / 8.032 / 7.674 | 76,80 | 80,10 / 73,76 | 27.604 / 6.857 / 9.819 | 77,96 |
| **FULL** (p ≥ 0,3316) | **79,39** | 79,48 / 79,29 | 29.674 / 7.661 / 7.749 | **76,97** | 80,65 / 73,60 | 27.545 / 6.607 / 9.878 | **78,18** |

| Hiệu | mean (CI95) | Clean (CI95) | LR2 (CI95) | tiny Clean | tiny LR2 |
|---|---|---|---|---|---|
| **FULL − F** | **+0,215 [+0,134; +0,304]** | +0,26 [+0,16; +0,38] | +0,17 [+0,07; +0,27] | −0,93 pp | −0,43 pp |
| FULL − RAW | +0,219 [+0,135; +0,309] | +0,27 [+0,17; +0,40] | +0,17 [+0,06; +0,27] | −0,97 pp | −0,54 pp |
| RAW − F | −0,004 [−0,020; +0,012] | −0,01 | +0,00 | +0,04 pp | +0,12 pp |

CI bootstrap ghép cặp theo nguồn, 2.000 lần, seed 431.

**Recall theo chiều cao GT** (LR2-equivalent, px), F → FULL:

| Bin | GT | Clean | LR2 |
|---|---|---|---|
| < 4 (tiny) | 2.583 | 64,00 → 63,07 | 52,11 → 51,68 |
| 4–8 | 14.036 | 81,09 → 80,72 | 73,18 → 72,96 |
| 8–16 | 12.911 | 84,14 → 84,08 | 79,89 → 79,82 |
| ≥ 16 | 7.893 | 73,84 → 74,23 | 71,52 → 71,77 |

## 3. Latency và bộ nhớ (cùng lượt, T4, 200 ảnh × 2 lượt, xen kẽ)
| | F | FULL |
|---|---|---|
| Tổng median / p95 / mean (ms) | 432,6 / 592,8 / 448,5 | 460,7 / 621,8 / 480,3 |
| Post-process median / p95 (ms) | 10,8 / 38,8 | 24,7 / 161,5 |
| Scorer (logistic) median | — | 0,1 ms |
| VRAM đỉnh allocated / reserved | 1.017 / 2.732 MiB | 1.017 / 2.732 MiB |

- **FULL/F median 1,065× [1,034; 1,086]**, dưới 1,10.
- p95 1,049× [1,032; 1,352].
- Thêm trung bình 31,8 ms/ảnh [27,8; 36,1].
- Theo điều kiện: Clean 1,062×, LR2 1,075×.
- RSS tiến trình 3.288 MiB (chung cho cả lượt).
- Đường FULL đo nguyên bản cài đặt, chưa tối ưu.

## 4. Cổng nhóm trên A (spec §5; hệ thống = FULL cho mọi ảnh)
| Cổng | Giá trị | Kết quả |
|---|---|---|
| g1: mean ≥ model đơn tốt nhất + 0,5 | model đơn tốt nhất là B-Clean `.40/.60` (78,50); FULL − đó = **−0,32 pp** [−3,00; +1,88] | **không đạt** |
| g2: Clean ≥ B-Clean − 0,3 | 79,39 − 80,52 = **−1,13 pp** | **không đạt** |
| g3: tiny LR2 ≥ B1 `.40/.60` − 3 | −0,43 pp | đạt |

- Mốc lấy từ dump T4 của cùng ảnh: B-Clean Clean 80,52 / LR2 76,48; B1 `eps005/.50` 79,30 / 77,03.
- **Đọc (mô tả):**
  - F (B1) trên Clean đã thấp hơn B-Clean 1,39 pp. Đó là khoảng cách giữa hai checkpoint.
  - Một bộ hậu xử lý chỉ chạy trên đầu ra B1 cần thêm hơn +1,1 pp Clean mới qua g2, và khoảng +1 pp mean mới qua g1.
  - FULL chỉ tăng +0,26 / +0,22. Hướng này không có đường hợp lệ để đạt cổng nhóm.

## 5. Nhãn khóa: **`SMALL_USEFUL`** = tăng nhỏ nhưng hữu ích, chưa đạt cổng nhóm
FULL − F có Δ > 0 và cận dưới CI > 0, và đạt đủ các điều kiện của `SMALL_USEFUL`:
- Clean và LR2 đều tăng, CI không chứa 0;
- tiny giảm ít hơn 3 pp (Clean −0,93, LR2 −0,43);
- latency median 1,065× ≤ 1,10.

**Câu hỏi tổng quát hóa:** đọc là **"tổng quát hóa"**.
- FULL − RAW trên A là +0,22 [+0,13; +0,31]. Pilot TRAIN-calibration là +0,14 [+0,0004; +0,29].
- Phần lợi đến từ đặc trưng, không phải từ ngưỡng: RAW − F ≈ 0, vì ngưỡng RAW 0,5976 gần như bằng 0,60.

## 6. Cơ chế và độ tập trung (mô tả, hậu kiểm; không đổi nhãn)
Nguồn: `concentration_janek.json`, script `scripts/boxquality_devA_concentration_janek.py`.

**FULL chủ yếu bớt FP:**
- Clean: FP −326, TP −52 (Pred −378).
- LR2: FP −208, TP −32 (Pred −240).
- TP mất nằm ở box nhỏ (< 8 px). Ở GT ≥ 16 px, recall lại tăng nhẹ.

**Thắng / hòa / thua theo nguồn** (đóng góp ròng):
- Clean: 230 / 763 / 137.
- LR2: 207 / 784 / 139.

**Độ tập trung:**
- Top 10 nguồn chiếm 46% lợi ròng trên Clean và 63% trên LR2.
- Ảnh trang báo `6e340fcbb1e0a877` chiếm 3% lợi Clean và 16% lợi LR2. Bỏ riêng ảnh này thì Clean +0,27, LR2 +0,15: **vẫn dương**, không phải do một ảnh.
- Các nguồn đứng đầu đều có dạng "bớt 10–25 box, không mất TP". Chưa xem bằng mắt đó là nền hay chữ thiếu nhãn (pilot từng gặp chữ thật không nhãn).

## 7. Được và không được tuyên bố
**Được tuyên bố:**
> "Trên development split A (1.130 nguồn, Clean + native LR2), scorer học trên TRAIN và đóng băng hơn B1 `.40/.60` +0,22 pp mean Hmean (CI95 [+0,13; +0,30]), với Clean +0,26 và LR2 +0,17. Chi phí là 1,065× latency median, không thêm VRAM, một checkpoint và một forward. Scorer chưa đạt cổng nhóm, vì B1 vẫn kém B-Clean trên Clean."

**Không được tuyên bố:**
- cải tiến đã xác nhận trên TEST;
- vượt cổng nhóm;
- kết quả trên B (chưa mở).

**Giới hạn:**
- A đã được dùng cho các hướng trước. Ý tưởng scorer (đặc trưng P-map, bớt FP) hình thành sau khi xem chẩn đoán FP trên A (Flip-TTA).
- T4 → `PENDING_REFERENCE_CHECK`.
- Có rủi ro thiếu nhãn: một phần "FP" bị bớt có thể là chữ thật không có nhãn. Nếu vậy, Hmean đang phạt sai và mức lợi thật có thể **khác**.
