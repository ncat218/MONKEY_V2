# Đề xuất phục hồi ảnh LR2 trước khi phát hiện chữ

**Trạng thái: đề xuất để nhóm thảo luận, chưa chạy thí nghiệm.** Đây là một phần mở rộng sau [Rev.4](../rev4/README.md). Câu hỏi nghiên cứu là liệu super-resolution ×2 có giúp B1-cal phát hiện thêm chữ trên LR2 so với bicubic ×2, và lợi ích đó đi kèm bao nhiêu chi phí suy luận.

## Kết quả làm căn cứ

Trên cùng 2.262 nguồn validation TextVQA FULL_R1, B1-cal đạt **77,55% Hmean trên Clean** và **75,29% trên LR2**. LR2 vẫn thấp hơn Clean **5,72 điểm recall**. Trial PRC đạt **75,27% Hmean LR2**, chênh **−0,02 điểm** so với B1-cal. CI 95% ghép cặp theo nguồn trên validation là **[−0,064; +0,022] điểm**. Nguồn số liệu là job calibration `12396` và job PRC `12518`.

Trong 6.454 từ B1-cal tìm được trên Clean nhưng bỏ sót ở LR2, job `12434` xếp **2.791 ca** vào nhóm phản hồi yếu và **3.280 ca** vào nhóm lỗi hình học theo ngưỡng chẩn đoán chính `c0=0.30`. Có thể theo dõi riêng hai nhóm khi phân tích những từ được tìm lại sau phục hồi. Pilot S2-A từng ghi nhận bicubic trước B0 tăng **0,80 điểm Hmean LR2**. Cần đo lại đối chứng bicubic với B1-cal trên đúng tập validation của đề xuất này.

## Thiết kế thí nghiệm

| Đầu vào cho B1-cal | Phép so sánh |
|---|---|
| LR2 gốc | Tái lập mốc Hmean 75,29% từ job `12396`. |
| LR2 → bicubic ×2 | Đo tác động của việc tăng kích thước ảnh với cùng detector. |
| LR2 → EDSR baseline ×2 | Đo phần cải thiện thêm của phục hồi học được so với bicubic. |

- Cả ba nhánh dùng cùng **2.262 nguồn remaining validation**, checkpoint B1 ở 10.000 optimizer updates, `mask_thr=0.30`, `min_text_score=0.40` và cùng evaluator. Cả bicubic lẫn EDSR nhận cùng ảnh LR2 và xuất đúng `2 × width` và `2 × height` của LR2.
- Với ảnh Clean có cạnh lẻ, ảnh LR2 phóng ×2 có thể lệch một pixel so với canvas gốc. Dùng tỷ lệ kích thước thực theo từng trục để đưa polygon về hệ Clean trước khi matching. Ảnh Clean và GT phục vụ đánh giá và phân tích lỗi. Model phục hồi chỉ nhận ảnh LR2.
- Ứng viên đầu tiên là **EDSR baseline ×2** pretrained vì tác giả công bố checkpoint ×2 và bản baseline có khoảng 1,37 triệu tham số. Trước khi chạy, ghi rõ phiên bản mã, SHA-256 checkpoint, cách decode và cấu hình suy luận. Giữ cố định cả model phục hồi và detector trong thí nghiệm này. [Bài báo EDSR, CVPRW 2017](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.html), [mã và checkpoint của tác giả](https://github.com/sanghyun-son/EDSR-PyTorch#quickstart-demo).
- Scene Text Telescope là tài liệu đọc thêm về phục hồi hướng chữ. Bài báo đánh giá lợi ích cho nhận dạng chữ, còn đề xuất này đo khả năng phát hiện vùng chữ của DBNet trên ảnh toàn cảnh. [Scene Text Telescope, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Scene_Text_Telescope_Text-Focused_Scene_Image_Super-Resolution_CVPR_2021_paper.html).

## Đọc kết quả

- Chỉ số chính là **Hmean của EDSR ×2 trừ Hmean của bicubic ×2**. Báo thêm chênh lệch so với LR2 gốc. Tính CI 95% bằng 2.000 lần bootstrap ghép cặp theo ảnh nguồn và tính lại micro-Hmean ở mỗi lần lấy mẫu.
- Báo Precision, Recall, TP, FP, FN, recall theo chiều cao chữ native LR2 và latency đầu cuối theo ảnh. Dùng chiều cao chữ trước phục hồi để cùng một từ nằm trong cùng nhóm kích thước ở cả ba nhánh.
- Xem ví dụ từ được tìm lại, từ bị mất thêm và polygon mới bị tính FP. Đối chiếu các ca được tìm lại với chẩn đoán phản hồi yếu và hình học của B1-cal.
- Nếu EDSR vượt cả LR2 gốc và bicubic, xem mức tăng có đi kèm giảm recall ở nhóm chữ nhỏ hoặc tăng FP và latency hay không. Nếu bicubic và EDSR cho kết quả gần nhau, báo chênh lệch và CI để nhóm cân nhắc chi phí phục hồi. Đây là kết quả validation của phần mở rộng. Benchmark test Rev.4 là một mốc riêng.
