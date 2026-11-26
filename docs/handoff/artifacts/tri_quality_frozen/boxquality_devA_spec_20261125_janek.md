# Spec bổ sung · Đánh giá box-quality đã khóa trên development A (Clean + LR2) · 25/11/2026

> **Đây là quyết định của user sau khi xem pilot, không phải vì pilot đã vượt cổng.** Pilot TRAIN-calibration giữ nguyên nhãn **`POSITIVE_BELOW_GATE`** và kết luận cũ (`docs/research/janek/boxquality_result_20261125_janek.md`). Spec gốc §7 chỉ cho mở lượt dev khi có `PASS_CALIBRATION_SCREEN`. Lượt này **nằm ngoài luật đã khóa**. User yêu cầu ngày 25/11 để biết tín hiệu từ TRAIN-calibration có tổng quát hóa sang development hay không.
>
> **Giới hạn đã biết trước:**
> - Split A đã được xem nhiều lần (CS-B1, MSB1, zoom, Flip-TTA, XV-B1). Kết quả chỉ **mô tả**, không xác nhận được cải tiến.
> - Không mở B, không mở TEST. Không chỉnh gì sau khi xem A.
>
> **Trạng thái:** spec viết trước khi chạy audit ROI/ignore và trước khi chạy notebook. Mọi số đều chưa có.

## 1. Đối tượng đóng băng (không fit, không chọn lại)
| Đối tượng | Giá trị | Nguồn |
|---|---|---|
| Model scorer | `model_janek.npz`, SHA-256 `81867a49…` | `docs/handoff/artifacts/janek/boxquality_v1_T4/` |
| Kết quả fit (để đối chiếu ngưỡng) | `boxquality_fit_janek.json`, SHA-256 `76386825…` | cùng thư mục |
| Normalizer | `mean`/`std` trong `model_janek.npz` (fold fit) | như trên |
| Đặc trưng | 13 đặc trưng `FEATURES`, cùng hàm `capture_with_features` | `mixab/boxquality_janek.py` (SHA `0e8d5e53…`, **không sửa**) |
| Ứng viên | B1, geometry `mask_040` (mask .40, unclip 1,5, min width 5, eps .01), `min_text_score = 0`, raw ≥ 0,20 | spec gốc §4 |
| Ngưỡng FULL | `cut_full = 0,3315908613227198` | `model_janek.npz` = `boxquality_fit_janek.json` |
| Ngưỡng RAW đã khóa | `cut_raw = 0,5976406401908236` | như trên |
| Checkpoint | B1 `alpha_1p00.pth`, SHA theo `merge_manifest.json` | như lượt pilot |

Notebook kiểm SHA của `model_janek.npz` và kiểm hai ngưỡng trùng `boxquality_fit_janek.json`. Sai thì **dừng**.

## 2. Dữ liệu
- **A** = nguồn có `split_of(id) == 'A'` (`mixab/reselect_split_janek.py:54`): **1.130 nguồn**, mỗi nguồn có Clean và native LR2, tổng 2.260 task.
- Ảnh Clean đúng byte theo bundle (`clean_sha256`). LR2 dựng bằng `native_lr2` của Tuấn (Pillow 11.3.0) và phải trùng `image_sha256` của task đã khóa.
- Nhãn dùng row dev đã khóa (`validation_remaining.jsonl.gz`, ROI đã khóa) và `textvqa_evaluation_protocol.json` của bundle. **Không** dùng ROI toàn ảnh trên dev.
- Thiếu hoặc lỗi một task thì báo. **Không âm thầm loại.**

## 3. Các cánh (cùng ảnh, cùng pool ứng viên, cùng evaluator)
| Cánh | Giữ box khi | Vai trò |
|---|---|---|
| **F** = B1 `.40/.60` | raw ≥ 0,60 | baseline đã calibration của Tuấn; **phép so chính** |
| **RAW** | raw ≥ `cut_raw` | đối chứng ngưỡng raw đã khóa trên TRAIN |
| **FULL** | p_FULL ≥ `cut_full` | scorer đã khóa |

- Counts của mỗi cánh tính bằng **evaluator chính thức** `lowres.evaluate_shared.evaluate_image(gt_clean, roi_clean, prediction, policy)` trên tập box đã giữ.
- Đường nhanh (`counts_at`, như lượt fit) cũng được tính, và **phải trùng** evaluator chính thức ở mọi task, mọi cánh. Lệch thì kết quả không hợp lệ.
- **Mốc cho cổng nhóm** (B1 `eps005/.50`, B-Clean `.40/.60`, và B1 `.40/.60` để đối chiếu): tính lại bằng `CandidateMatcher` trên **dump T4 đã có** của cùng ảnh. LR2 dùng `dualagree_dump_v1_T4` (Colab T4); Clean dùng `csb1_clean` (Kaggle T4). Đây cũng là cách đã làm cho Clean CS-B1.
- Có kiểm F sống so với F tính từ dump: số task trùng tuyệt đối (polygon và raw), và chênh Hmean. **Chỉ báo, không chặn**, vì Kaggle và Colab có thể khác build.

## 4. Báo cáo (đầy đủ, không chọn chỉ số thuận lợi)
Mỗi cánh, mỗi điều kiện (Clean, LR2):
- TP / FP / FN / GT / Pred, precision, recall, Hmean;
- recall theo 4 bin chiều cao GT (< 4, 4–8, 8–16, ≥ 16 px LR2-equivalent; tiny = < 4);
- mean Hmean (Clean, LR2).

Các hiệu, mỗi hiệu cho Clean, LR2 và mean, kèm CI95 bootstrap ghép cặp theo nguồn (2.000 lần, seed 431, giữ cặp Clean/LR2):
- **FULL − F** (chính);
- FULL − RAW (phần do đặc trưng; so với pilot +0,14);
- RAW − F.

Ngoài ra:
- số nguồn FULL thắng, hòa, thua so với F;
- tỉ phần lợi ròng từ 10 nguồn đóng góp lớn nhất (mô tả, như phân tích CS-B1);
- FULL − mốc cho từng cổng nhóm.

## 5. Cổng và nhãn kết luận (khóa trước)
**Cổng nhóm trên A** (giữ nguyên như spec Clean CS-B1; hệ thống = FULL áp cho **mọi** ảnh, không router):
- g1: mean(FULL) ≥ mean của model đơn tốt nhất + 0,5 pp. Model đơn xét trong {B1 `.40/.60`, B1 `eps005/.50`, B-Clean `.40/.60`}, tính từ dump T4.
- g2: Clean(FULL) ≥ Clean(B-Clean `.40/.60`) − 0,3 pp.
- g3: tiny LR2(FULL) ≥ tiny LR2(B1 `.40/.60`) − 3 pp.

**Nhãn** (xét theo thứ tự; dựa trên FULL − F, mean Hmean):

| Nhãn | Điều kiện | Nghĩa cho user |
|---|---|---|
| `INVALID_RUN` | một kiểm ở §6 sai | không kết luận |
| `GROUP_GATE_PASS` | g1, g2, g3 đều đạt **và** cận dưới CI của FULL − F > 0 | **đạt cổng** |
| `SMALL_USEFUL` | Δ > 0 và cận dưới CI > 0; không điều kiện nào xấu đi có ý nghĩa (cận trên CI của Δ Clean và của Δ LR2 đều ≥ 0); tiny Clean và tiny LR2 của FULL đều ≥ F − 3 pp; latency median đầu-cuối FULL/F ≤ 1,10 | **tăng nhỏ nhưng hữu ích** (chưa đạt cổng) |
| `POSITIVE_NOT_USEFUL` | Δ > 0 và cận dưới CI > 0, nhưng trượt ít nhất một điều kiện của `SMALL_USEFUL` (ghi rõ điều kiện nào) | có tăng nhưng phải trả giá; **không có lợi** để dùng |
| `INCONCLUSIVE` | Δ > 0 nhưng cận dưới CI ≤ 0 | **không chứng minh được lợi** |
| `NO_BENEFIT` | Δ ≤ 0 | **không có lợi** |

- Ngưỡng latency 1,10× là mục tiêu thiết kế đã ghi ở spec gốc §8, không đặt mới.
- **Đọc câu hỏi tổng quát hóa** (mô tả, không đổi nhãn), dựa trên FULL − RAW trên A:
  - "tổng quát hóa": Δ > 0 và cận dưới CI > 0;
  - "một phần": Δ > 0 nhưng CI chứa 0;
  - "không": Δ ≤ 0.

## 6. Kiểm hợp lệ (dừng nếu sai)
1. SHA của dataset, của `model_janek.npz`, của B1; hai ngưỡng trùng file fit.
2. Mọi ảnh trùng SHA đã khóa (Clean theo bundle, LR2 theo task).
3. Đường đặc trưng trùng tuyệt đối `capture_candidates` của Tuấn (polygon và raw) ở **mọi** task A.
4. Đường nhanh = evaluator chính thức ở mọi task × cánh (§3).
5. Đủ 2.260 task, không có đặc trưng không hữu hạn.
6. Smoke (Ô 4, 10 nguồn A = 20 task, chọn theo hash `20261125:bq-devA:smoke:`) chạy hết các kiểm trên và in thời gian dự kiến. Dừng nếu một kiểm sai hoặc dự kiến > 3 giờ.

## 7. Latency công bằng
- **Ảnh:** 100 nguồn A chọn theo hash (`20261125:bq-devA:latency:`) × 2 điều kiện = 200 ảnh đo, cộng 10 ảnh warm-up khác. 2 lượt, xen kẽ thứ tự cánh (lượt 2 đảo lại). `torch.cuda.synchronize` trước và sau mỗi cánh. Đọc file trước khi đo (page cache ấm cho cả hai); kiểm SHA nằm ngoài vùng đo.
- **F (như triển khai):** decode + resize (pipeline MMOCR) → forward B1 → DB post-process của Tuấn với `mask_040` và `min_text_score = 0,60` → polygon.
- **FULL (như triển khai):** decode + resize → forward B1 → `capture_with_features` (post-process `min_text_score = 0` + 13 đặc trưng) → lọc raw ≥ 0,20 → chuẩn hóa + logistic → giữ p ≥ `cut_full` → polygon.
  - Đường này **chưa tối ưu**: có thêm một lần kiểm mean trong mask. Vẫn đo nguyên như vậy, không tối ưu riêng cho phép đo.
- **Kiểm đúng:** polygon F trong lượt đo phải trùng `select_candidates(…, 0,60)` của đường thường. Polygon FULL phải trùng tập FULL của lượt trích (cùng ảnh).
- **Báo:**
  - median, p95, mean (ms) cho tổng và từng giai đoạn (decode, forward, post-process, scorer), tách theo điều kiện;
  - tỉ số median FULL/F kèm CI bootstrap theo ảnh; tỉ số p95; thời gian thêm trung bình;
  - **bộ nhớ:** VRAM đỉnh (allocated và reserved) của từng cánh, đo riêng trên ảnh warm-up (reset peak trước mỗi cánh); RSS đỉnh của tiến trình (`ru_maxrss`, chung cho cả lượt);
  - môi trường (GPU, driver, phiên bản).
- Số latency chỉ so được **trong cùng lượt chạy**.

## 8. Kiểm ROI/ignore của nhãn TRAIN (trước khi chạy notebook)
- **Bối cảnh đã kiểm** (`rev4/training_data.py:84-162`, `rev4/train_recovery.py:43-70` @ `b9eaf37`):
  - B1 được train từ manifest TRAIN đóng băng `manifest__train__…jsonl` (SHA `ca1b9163…`).
  - ROI của manifest đó là trường nhìn thấy sau perspective, lấy từ CSV job 8323 (`/datastore/...`, **chỉ có trên HPC**). File không có trên máy này (đã tìm trong workspace; Drive H không mount).
  - Luật valid/ignore là cùng `gt_for_source` + `DEFAULT_SHARED_CONFIG`. Bản vendored đã tái tạo dev 2.262/2.262 (P0).
  - Nhãn của scorer dùng **ROI = toàn ảnh**. Vì vậy, chỗ khác nhau duy nhất có thể có là ROI: GT bị bỏ (ngoài trường nhìn), GT ignore do `cropped_below_visibility`, và box bị evaluator lọc theo ROI.
- **Không so trực tiếp được với manifest TRAIN gốc.** Thay vào đó đo **ước lượng** trên dev A, nơi có cả ROI đã khóa: lấy ứng viên B1 `mask_040` (raw ≥ 0,20) từ dump T4 Clean và LR2, rồi gán nhãn theo đúng luật `task_labels` với hai row:
  - row đã khóa (ROI lịch sử);
  - row dựng lại bằng `gt_for_source` với ROI toàn ảnh.
  - So nhãn từng ứng viên theo trọng số của fit (mỗi task 0,5).
  - Phép đo này chỉ đọc GT A để đo nhiễu nhãn. Nó không chạm scorer hay ngưỡng nào.
  - **Suy luận:** TRAIN và dev cùng sinh bởi một generator (`CONFIG_SHA256 141df1cf…`), nên tỉ lệ trên dev là ước lượng hợp lý cho TRAIN. **Chưa kiểm** trên TRAIN.
- **Ngưỡng "đáng kể" (khóa trước):** tổng trọng số ứng viên có nhãn khác nhau giữa hai luật ≥ **2,0%**. Tính cả đổi 1↔0 và đổi "bị bỏ" ↔ "có nhãn".
  - **Dưới 2%:** ghi nhận, chạy notebook với scorer cũ.
  - **Từ 2% trở lên:** **báo user trước khi chạy**. Không sửa nhãn, không fit lại. Nếu user muốn sửa, phải là scorer **mới** có tên mới và spec mới; scorer cũ giữ nguyên kết quả.

## 9. Sau khi có kết quả
- Báo nhãn §5 và toàn bộ bảng §4. Không đổi ngưỡng, đặc trưng, model hay pool. Không chạy thêm biến thể. Không mở B/TEST.
- Chỉ đề xuất sửa tiếp khi chỉ ra được **một cơ chế lỗi cụ thể** có bằng chứng (ví dụ một loại box bị chấm sai có hệ thống). Mỗi đề xuất kèm spec riêng; không quét hàng loạt.

## Phụ lục · Kết quả kiểm §8 (chạy 25/11, sau khi khóa spec; chưa chạy notebook)
Script `scripts/boxquality_devA_roi_audit_janek.py` → `docs/handoff/artifacts/janek/boxquality_devA_20261125/roi_audit_janek.json`. Mẫu: ứng viên B1 `mask_040` từ dump T4 trên A, 1.130 nguồn.

| Điều kiện | Ứng viên | Task có thay đổi | pos→bị bỏ | neg→bị bỏ | neg→pos | pos→neg | Lệch có trọng số |
|---|---|---|---|---|---|---|---|
| LR2 | 48.652 | 62 / 1.130 | 181 | 83 | 8 | 0 | 0,75% |
| Clean | 54.936 | 71 / 1.130 | 197 | 97 | 10 | 0 | 0,78% |
| **Chung** | | | | | | | **0,77%** |

(Chiều đọc: nhãn theo ROI toàn ảnh → nhãn theo ROI đã khóa.)

- **Kết luận theo luật khóa:** 0,77% < 2% → **không đáng kể**. Chạy notebook với scorer cũ; không sửa nhãn, không fit lại.
- **Mô tả:**
  - Phần lệch chủ yếu là box gần mép ảnh. Theo ROI toàn ảnh thì có nhãn; theo ROI lịch sử thì bị evaluator bỏ (không tính vào Pred).
  - Không có ca TP thành FP. Có 18 ca FP thành TP: GT bị cắt bởi ROI thì khớp IoU tốt hơn.
  - Đây là ước lượng trên dev. Áp sang TRAIN là **suy luận** (cùng generator); **chưa kiểm** trên TRAIN.
