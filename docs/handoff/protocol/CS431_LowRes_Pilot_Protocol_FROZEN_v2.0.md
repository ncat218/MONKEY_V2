# CS431 – Giao thức pilot Low-Resolution (FROZEN v2.0)

Đối tượng: MonkeyOCRv2-AS + DBNet. Trạng thái: FROZEN v2.0, đã khóa; bắt đầu Phase 1. Ngày: 07/11/2026.

Các ngưỡng số trong tài liệu là đề xuất thực dụng của nhóm, chưa được xác nhận khoa học. Các ngưỡng này đã được khóa trong v2.0 và không sửa sau khi xem kết quả.

## 0. Đã chốt và chưa chốt

**Đã chốt:** MonkeyOCRv2-AS + DBNet; nghiên cứu low-resolution robustness; TextVQA FULL_R1 là môi trường chính; ICDAR2015 là đánh giá ngoài miền; baseline B0, B_up, B0-cal, B1 và M.

**Chưa chốt:** cơ chế cải tiến M là Scale-Aware Training (SAT) hay Paired Response Consistency (PRC). Quyết định này chỉ được đưa ra sau pilot, theo cổng ở mục 6. Không huấn luyện cả hai.

## 1. Dữ liệu và phân chia

- Dataset: TextVQA FULL_R1 + TextOCR word polygons; metric và ignore policy giữ nguyên bản đã đóng băng (maximum matching, IoU ≥ 0,5).
- Diagnostic subset: 500 ảnh nguồn lấy ngẫu nhiên từ validation, seed 20261107. Dùng để phân tích lỗi và chọn hướng.
- Remaining validation: 2.762 ảnh nguồn. Dùng để tune threshold, hyperparameter, chọn checkpoint.
- Test: chỉ dùng cho benchmark cuối, không dùng để chọn phương pháp hay ngưỡng. Test TextVQA lịch sử đã từng được xem nên kết quả được gọi là exploratory.
- Mọi biến thể (Clean, LR2, LR4, LR8) của cùng một nguồn nằm cùng một nhóm.
- Ghi số ảnh, valid GT, ignored GT và số GT theo bin chiều cao.
- GT validation chỉ dùng cho phân tích và chọn mô hình, không dùng làm supervision. GT và prediction của teacher dùng khi train chỉ lấy từ training split.

## 2. Điều kiện suy thoái và hệ tọa độ

- Sinh LR bằng `generate_lr_only.py` (bicubic, PNG, Pillow 11.3.0). Hai protocol suy luận cần báo riêng:
  - **Native LR**: ảnh nhỏ thật (W//s × H//s).
  - **B_up**: ảnh LR được bicubic upsample về canvas gốc trước khi suy luận.
- Map prediction về hệ tọa độ ảnh clean bằng tỉ lệ theo từng trục: clean_width / lr_width và clean_height / lr_height (lấy từ `manifest.csv`), không dùng hằng số scale. GT và evaluator giữ nguyên.
- Ghi hai kích thước chữ riêng cho mỗi GT: **native LR height** (thông tin còn lại sau downsample) và **network-input height** (scale mà detector nhận sau resize/padding của MMOCR).
- Hạn chế cần nêu trong báo cáo: protocol là downsample bicubic sạch, không có blur, nhiễu hay nén JPEG.

## 3. Kiểm tra kỹ thuật bắt buộc (trước khi chạy pilot)

1. Xác minh resize, interpolation, padding và tham số postprocessing của DBNet (mask threshold, box_thresh, unclip, min size). Giữ cố định trong mọi điều kiện chẩn đoán.
2. **Identity test:** chạy pipeline mới với ảnh clean (LR1, không đổi) phải tái lập đúng số E0 clean đã có.
3. **Round-trip tọa độ:** map GT clean → LR → clean phải cho IoU gần 1.
4. Dùng một chế độ precision duy nhất (FP32 hoặc BF16) cho mọi điều kiện; ghi rõ.
5. Xác minh xuất được probability map trước postprocessing, kèm cách restore về hệ tọa độ nguồn và mask padding/ignored/vùng không hợp lệ.
6. Lưu raw outputs của diagnostic subset: polygon và score trước cutoff, P-map dạng float16.

## 4. Metric chẩn đoán

- Tổng thể: Precision, Recall, Hmean, TP, FP, FN, cho Clean/LR2/LR4/LR8.
- Robustness: Hmean drop, Recall drop, **clean-TP survival** (tỷ lệ clean TP vẫn là TP ở LR).
- Bin native LR height: <4, 4–<8, 8–<16, ≥16 px; kiểm tra độ nhạy ngưỡng 4/6/8 px.
- **Kernel response coverage:** C_GT = tỷ lệ pixel trong valid shrunk GT kernel có P > mask threshold. Báo C_GT(clean), C_GT(LR) và ΔC_GT cho từng GT. Kernel shrink theo đúng tham số của DBNet.
- **B_up recovery ratio** (LR4) = (Hmean_B_up − Hmean_B0) / (Hmean_clean − Hmean_B0). Ghi N/A khi mẫu số gần 0 (CI của hiệu Hmean_clean − Hmean_B0 chứa 0). Chỉ là một chỉ báo, đọc cùng P-map, kích thước chữ và phân tích lỗi.

## 5. Phân loại lỗi

Chỉ phân loại chính trên nhóm GT Clean TP → LR FN. Áp dụng theo thứ tự, mỗi GT rơi vào nhóm đầu tiên thỏa:

1. **Kernel không hợp lệ:** diện tích kernel < 4 pixel trên lưới P-map (không gán vào nhóm response).
2. **Response yếu:** C_GT(LR) < 0,30 (c0 = 0,30 là ngưỡng chính; 0,50 chỉ dùng cho sensitivity analysis; đồng thời xem C_GT(clean) và ΔC_GT).
3. **Contour/postprocess:** có tín hiệu nhưng không tạo được contour/box hợp lệ (kiểm tra min size, contour filtering, unclip).
4. **Score filtering:** có candidate hình học phù hợp nhưng bị loại bởi score threshold.
5. **Geometry:** có prediction giữ lại nhưng không đạt IoU matching.

Trường hợp không phân loại được hoặc có nhiều cơ chế cùng tác động được đánh dấu riêng. Các tỷ lệ chỉ là bằng chứng gợi ý; cần kiểm tra bằng ảnh mẫu trước khi gọi là nguyên nhân. Chỉ số response suy giảm chưa chứng minh encoder mất feature; muốn khẳng định điều đó cần thêm bằng chứng ở backbone/FPNC trên một số mẫu.

## 6. Cổng quyết định (ngưỡng đề xuất)

| Cổng | Điều kiện | Ý nghĩa |
|---|---|---|
| G1: LR4 đáng nghiên cứu | Hmean giảm ≥ 5 pp, cận dưới CI của mức giảm > 0 | Có bài toán thật |
| G2: mất mát đủ lớn | ≥ 15% clean TP chuyển FN ở LR4 | Có lượng detection mất đáng kể |
| G3: còn đủ thông tin | ≥ 30% GT thuộc clean-TP→LR4-FN có native height ≥ 6 px (độ nhạy 4/8) | Có nhóm lỗi khả thi để cải thiện |
| G4: không giải quyết được bằng upsample | Recovery ratio, P-map, kích thước chữ và phân tích lỗi cho thấy B_up chưa đủ (đánh giá tổng hợp, không dùng ngưỡng cố định) | Nếu B_up gần như đủ, báo kết quả và xem lại phạm vi |
| Hướng SAT | Tổng hợp bằng chứng cho thấy lệch scale chiếm ưu thế (recovery ratio cao, P-map còn tín hiệu, lỗi theo network-input height); không dùng ngưỡng riêng trên recovery ratio | Chọn SAT |
| Hướng PRC | Tổng hợp bằng chứng cho thấy response yếu chiếm ưu thế (> 50% trong GT kernel hợp lệ, kiểm tra c0 = 0,50), native height còn đủ, recovery ratio thấp hoặc N/A, và ≥ 80% GT mất có C_GT(clean) ≥ c0; không dùng ngưỡng riêng trên recovery ratio | Chọn PRC |
| Nhóm khác chiếm ưu thế | Contour/score/geometry chiếm đa số | Dừng, xem lại hướng (ví dụ calibration hoặc geometry) |

Nếu không đạt cổng thì báo kết quả và xem xét lại phạm vi. Không đổi ngưỡng sau pilot để hợp thức hóa kết quả. Chọn SAT hay PRC dựa trên tổng hợp recovery ratio, P-map, native và network-input height, phân tích lỗi và ảnh mẫu; recovery ratio không tự quyết định. Khi ở sát ranh giới, được dùng bằng chứng thị giác nhưng phải ghi lý do trong tài liệu.

## 7. Baseline và phương pháp

| Ký hiệu | Mô tả |
|---|---|
| B0 | Monkey official checkpoint, native LR |
| B_up | B0 + bicubic upsample LR về canvas gốc |
| B0-cal | B0 với threshold/box_thresh tune trên remaining validation cho LR, cùng quy trình với B1-cal và M-cal |
| B1 | Fine-tune Clean + LR thông thường |
| M | B1 + đúng một cơ chế (SAT hoặc PRC) |

**Calibration công bằng:** B0, B1 và M đều được tune `thresh` và `box_thresh` bằng cùng một quy trình trên remaining validation (cùng lưới tham số, cùng tiêu chí chọn, khóa trước khi chạy). Báo cả kết quả native (threshold mặc định) và calibrated, ký hiệu B0-cal, B1-cal, M-cal. Endpoint chính lấy theo calibrated.

**Protocol huấn luyện (B1 và M dùng chung):**

- Cùng initialization (B0), dữ liệu khả dụng, số optimizer updates, learning rate, quy trình chọn checkpoint trên remaining validation.
- Dữ liệu: Clean + LR2 + LR4. **LR8 không đưa vào train**, chỉ dùng để đánh giá ngoài phân phối.
- Tỷ lệ mẫu clean replay (đề xuất 50%) giống nhau ở B1 và M.
- Sinh LR on-the-fly phải dùng đúng hàm bicubic và Pillow 11.3.0 của `generate_lr_only.py`.
- SAT: chỉ thay quy tắc sampling/crop; báo số GT instances và phân bố kích thước GT thực sự được thấy khi train.
- PRC: giữ supervised detection loss và clean replay như B1, chỉ thêm masked response consistency trên P-map; chỉ áp dụng ở vùng teacher khớp GT ở clean (training split). Teacher là B0 đóng băng. P-map clean và LR phải được căn chỉnh về cùng hệ tọa độ, loại bỏ padding và vùng ignored.
- Nếu đủ GPU: 2 training seeds cho B1 và M.

## 8. Đánh giá cuối (một lần, sau khi khóa mọi lựa chọn)

- **Primary endpoint:** Hmean LR4 trên TextVQA test.
- Mục tiêu đề xuất: M − B1 ≥ 2 pp, cận dưới CI 95% của gain > 0, clean Hmean giảm không quá 1 pp so với B1. Ghi rõ đây là phát biểu "gain dương có bằng chứng, ước lượng ≥ 2 pp"; nếu muốn khẳng định "ít nhất 2 pp" thì cận dưới CI phải > 2.
- M cũng phải được so với B0, B_up và B1 ở cả hai chế độ native và calibrated (M-cal so với B0-cal và B1-cal), không chỉ B1.
- LR2 không được thấp hơn B1; báo cả LR8 kể cả khi không cải thiện.
- ICDAR2015 (xác nhận phụ): LR4 gain > 0 và clean giảm ≤ 1 pp. Chạy một lần.
- Bootstrap theo ảnh nguồn, resample cùng lúc mọi biến thể của một nguồn.
- Báo Precision, Recall, Hmean, TP/FP/FN, small recall, clean-TP survival, latency, VRAM.

**Khóa ngưỡng và báo cáo bắt buộc:** c0 = 0,30 là ngưỡng chính để phân loại response yếu; 0,50 chỉ dùng cho sensitivity analysis. Diện tích kernel tối thiểu là 4 pixel. Các tiêu chí clean (giảm ≤ 1 pp), LR2 và ICDAR2015 phải được báo cáo đầy đủ dù đạt hay không đạt.

## 9. Thời gian và quy tắc dừng (khoảng 3 tuần)

- Tuần 1: audit pipeline (mục 3), khóa subset, chạy pilot E0 và B_up, ra quyết định theo mục 6.
- Tuần 2: train B1 và M, tune trên remaining validation.
- Tuần 3: đánh giá cuối và viết báo cáo.
- Nếu pilot không đạt cổng, kết luận và chuyển hướng (calibration, geometry/word separation, hoặc một corruption đơn lẻ khác). Không kéo dài pilot.
- Kết quả âm được đo sạch (B0, B_up, B0-cal, B1, M, CI) vẫn là kết quả hợp lệ.

## 10. Điều nhóm cần xác nhận trước khi chạy

- [ ] Cấu hình inference thực tế của Monkey (resize, padding, postprocessor)
- [ ] Cách xuất P-map trước postprocessing
- [ ] Giá trị đã khóa trong v2.0: c0 = 0,30 (chính) và 0,50 (sensitivity), kernel tối thiểu 4 pixel, h* = 6 px (độ nhạy 4/8), G1–G4, clean replay 50%
- [ ] Precision dùng cho B0 (FP32 hoặc BF16)
- [ ] Cả nhóm xác nhận FROZEN v2.0 trước khi chạy pilot
