# Workflow hiện tại · bổ sung 28/11/2026

**Mới nhất:** [ICDAR2015 gốc192199 đã hoàn tất](../handoff/results/icdar2015_transfer_192199.md), 12 setting/500 ảnh. B0 H88,78%, B-Clean82,20%, B1 75,80%, tổ hợp ASF+EDSR+Q78,09%. Có cải thiện so B1 nhưng chưa giữ năng lực B0 hay tăng small recall ở benchmark này. [TextOCR development28/11](../handoff/results/comparison_20261128.md) là bảng riêng. Không chọn lại phương pháp/ngưỡng theo ICDAR test.

Đọc [handoff](../../handoff.md) cho số liệu, metric và ràng buộc. Đây là trạng thái nghiên cứu sau các extension; không sửa protocol frozen.

| Giai đoạn | Trạng thái có bằng chứng | Việc tiếp theo |
|---|---|---|
| Reproduce nhánh detector | Có B0 AS+FPNC+DBNet, config và checkpoint | Phân biệt với reproduce toàn bộ paper parsing |
| Data/GT/split/degradation audit | TRAIN 15.222, VAL 3.262; LR native, EXIF/round-trip/precision/hash receipts | Audit khả năng tái lập TEST trước benchmark cuối; không đọc TEST để chọn hướng |
| S1/S2 degradation pilot | Có report Clean/LR/blur/JPEG và gate; đã được xem | Không gọi S1/S2 là fresh holdout |
| B1 mixed fine-tuning + calibration | Hoàn thành, checkpoint công khai | Giữ B1 là baseline mạnh cho các extension |
| B-Clean clean-only control | Hoàn thành cùng init/sources/updates | Báo riêng domain adaptation và lợi ích LR mixing |
| EDA / PR / quad grid | Có per-source counts/CI; job 63985 và fixed diagnosis 73827 | Hoàn tất review 127 paired cards ở ngưỡng mới; tránh trộn với 160 cards cũ |
| PRC / feature KD | Đã chạy; chưa vượt matched controls | PRC mask low-activity còn cần chẩn đoán, không train lại mù |
| EDSR | Recheck163012 và EDSR×ASF163064 COMPLETED; quality transfer166501 hoàn tất phase riêng | Đọc bảng28/11 và CI; không trộn ngưỡng lịch sử hay benchmark |
| CRAFT | Screen độc lập + 200 TRAIN teacher maps; chưa có auxiliary head | Review 32 cards, đánh giá độ tin cậy ký tự trước khi cân nhắc head |
| Residual/gated latent sau FPNC | Pilot 1k và full 10k hoàn thành; chưa có lợi ích | Dừng lặp lại cùng cấu hình; chỉ mở lại khi có giả thuyết và bằng chứng mới |
| ASF/DySample fusion | Recovery111683 COMPLETED; đủ4 nhánh10k và mọi điều kiện | [Kết quả](../handoff/artifacts/fusion_completed_111683/README.md): Hmean gần ngang B1; tiny recall có lợi ích/đánh đổi |
| Multi-seed / final benchmark | Chưa hoàn tất | Chỉ sau khi khóa method; không dùng TEST chọn hướng |
| ICDAR2015 gốc192199 | COMPLETED; input/B0 per-image audit PASS, đủ12 setting | Báo negative transfer so B0/B-Clean; không retune trên test |

## Hành động ngay

1. Đọc [TextOCR28/11](../handoff/results/comparison_20261128.md) và [ICDAR192199](../handoff/results/icdar2015_transfer_192199.md) theo đa mục tiêu. Các job EDSR/fusion/ICDAR này đã có kết quả; không submit trùng. Dừng phân tích nếu identity/count/precision/checkpoint FAIL; giữ log và báo lỗi.
2. Nếu hoàn tất: công bố toàn bộ nearest/ASF/bilinear/DySample, checkpoint cuối, fixed threshold, paired source CI, FP và 4 height bins. ASF phải so với nearest; DySample phải so với bilinear để tách interpolation khỏi learned sampling. Cùng FPNC+DBHead trainable và stream; A0 cũ chỉ DBHead không đủ làm control.
3. Đọc transfer LR4/LR8/blur/JPEG trên cùng 500 nguồn và clean/LR2 anchors; ghi đây là stress trên development đã xem, chưa chứng minh real-world LR. Báo cả suy giảm, không chọn corruption tốt nhất.
4. Song song hoàn tất manual review. Người review ghi nhãn/ghi chú ở bản sao CSV riêng có version, giữ blank nếu chưa review; không sửa raw snapshot. Dùng paired card + GT + P-map để phân biệt localization, chữ nhỏ yếu, merge/split, ignore/annotation và nền giống chữ.
5. Nếu module có tín hiệu vượt matched control mà giữ Clean/FP/latency hợp lý: đề xuất chốt trước tiêu chí và seed lặp lại. Nếu không có tín hiệu, kết thúc hướng đó và trình bày negative results; không kéo dài train cùng giả thuyết mà thiếu bằng chứng.

## Khóa phương pháp và benchmark cuối

Trước đánh giá cuối, nhóm cần xác nhận checkpoint, operating point, single/two-detector deployment, seed budget, các metric và dataset được phép đọc. Ngưỡng thành công và chấp nhận đánh đổi là quyết định cần chốt trước, không suy ra từ điểm TEST.

TextVQA/TextOCR: P/R/Hmean và TP/FP/FN theo evaluator nội bộ đã khóa, source-paired CI, Clean retention, FP, tiny recall, latency có scope rõ. Audit lại manifest TEST, train overlap và lịch sử exposure; đừng gọi TEST hoàn toàn chưa từng xem nếu protocol cũ ghi khác.

ICDAR2015: đã đánh giá nguyên500 ảnh gốc với weights/settings khóa, dùng MMOCR vanilla IoU>.5 và ignore>.5, tách khỏi TextOCR matcher. [Kết quả192199](../handoff/results/icdar2015_transfer_192199.md) xác nhận B0 reproduce và cho thấy giới hạn chuyển giao của model thích nghi TextOCR. B0 đã train ICDAR TRAIN; benchmark đã xem khi reproduce, không phải fresh zero-shot. <8px có0 GT hợp lệ; nhóm<16px có137, không đồng nhất với tiny<4px LR2 TextOCR.

Giữ kết quả common `.30/.40` và expanded `.40/.60` ở bảng riêng. Không cộng lợi ích EDSR tại ngưỡng cũ vào baseline ngưỡng mới. Không gọi calibration là kiến trúc mới, không dùng CI source để khẳng định chênh lệch seed cỡ vài phần trăm điểm.

## Nhánh nghiên cứu có điều kiện

ASF/DySample đang kiểm tra fusion thay vì tiếp tục adapter đầu ra FPNC không có tín hiệu. Character/affinity supervision chỉ mở lại nếu teacher QC đáng tin và có complementary TP hữu ích. C2/C3 refinement, alignment loss khác hoặc hard negatives chỉ nên thiết kế sau khi manual EDA chỉ ra lỗi mục tiêu; chưa phải kết quả đã làm.

Mix A/B, QRoute, DualAgree thuộc workflow cộng tác viên: đánh giá và deployment budget riêng; chưa gộp vào kết luận của main flow. Các [đề xuất có ngày cũ](../archive/README.md) được giữ để truy nguồn, không đại diện ưu tiên hiện tại.

## Nhánh kết hợp được yêu cầu26/11

[EDSR × ASF](../edsr-asf-v1.md) đã COMPLETED và phase Q đã hoàn tất: xem [bảng TextOCR28/11](../handoff/results/comparison_20261128.md). [ICDAR2015 gốc192199](../handoff/results/icdar2015_transfer_192199.md) dùng cùng các checkpoint đã khóa; không cho thấy lợi ích Hmean riêng rõ của ASF/Q tại dataset này. Không giả định hai lợi ích cộng tuyến tính hoặc chọn lại cutoff để cứu test.
