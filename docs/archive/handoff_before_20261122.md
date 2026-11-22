# Handoff CS431 · MonkeyOCRv2-AS + DBNet

Cập nhật artifacts 19/11/2026 · [Repo MONKEY_V2](https://github.com/ncat218/MONKEY_V2).

**Đọc cập nhật mới trước:** [Baseline B-Clean/B1, calibration mở rộng và artifacts cho Mix A](docs/handoff/artifacts/pr_quad_audit_v1/README.md).
Đã có counts 72 settings theo source, manifest remaining-validation, checkpoint B-Clean,
endpoint reference counts và 160 thẻ review. Job phục hồi `63985` COMPLETED; human review PENDING.
B-Clean/B1 sau calibration mở rộng đạt Clean/LR2 Hmean **80,86/75,51%** và **79,43/76,90%**.
Phần bên dưới giữ mô tả và kết quả lịch sử ngày 15/11; các số `.30/.40` không thay thế
baseline mở rộng mới. Các extension chưa thay protocol và benchmark cuối chưa chạy.

**Đề bài cho người review.** Đọc kết quả và đề xuất **một hướng nghiên cứu độc lập** để cải thiện phát hiện chữ ở LR2 so với B1-cal. Bạn tự chọn giả thuyết và phương pháp dựa trên bằng chứng hoặc tài liệu bạn tìm được. PRC và EDSR là các thử nghiệm đã chạy để tham khảo. Mục 5 ghi phần đề xuất cần gửi lại cho nhóm.

## 1. Mục tiêu, dữ liệu và các giai đoạn

Đo khả năng **phát hiện vùng chữ** khi ảnh giảm độ phân giải. Nhánh chính dùng **LR2** với MonkeyOCRv2-AS + DBNet, FP32. Rev.4 kế thừa các điều khoản còn áp dụng từ FROZEN v2.0. [Protocol Rev.4](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/protocol/CS431_Pilot_Protocol_v2.1_FINAL.md) · [FROZEN v2.0](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/protocol/CS431_LowRes_Pilot_Protocol_FROZEN_v2.0.md).

**Tình trạng đóng góp thực nghiệm.** Kết quả đã đo là mức tăng của baseline fine-tune B1 và kết quả của hai hướng thử nghiệm dùng PRC và EDSR. Hiệu quả của một phương pháp vượt B1-cal trên benchmark cuối vẫn là câu hỏi mở.

| Dữ liệu | Quy tắc |
|---|---|
| TextVQA FULL_R1 + polygon từ TextOCR | Split detection nội bộ từ official TextVQA train. Giữ mọi biến thể cùng source ID hoặc cùng clean SHA-256 trong một fold. |
| TRAIN · 15.222 nguồn | Train B1 và M với 50% Clean, 50% LR2; B-Clean dùng 100% Clean trên cùng nguồn. [CSV source IDs, đường dẫn và SHA ảnh](docs/handoff/data/README.md). |
| Validation · 3.262 nguồn | S1 dùng 500 nguồn, S2 dùng 500 nguồn khác. 2.262 nguồn còn lại dùng chọn checkpoint, calibration và chẩn đoán. |
| TEST · 3.263 nguồn | Chấm TextVQA sau khi khóa checkpoint và ngưỡng. ICDAR2015 test dùng xác nhận ngoài miền. |

| Giai đoạn | Việc thực hiện | Trạng thái |
|---|---|---|
| 1. Audit và pilot | Khóa nguồn, GT, hash, EXIF và phép đổi tọa độ. S1 đo LR ban đầu, S2 xác nhận LR2 qua G1–G3 và khảo sát Blur/JPEG. G4 xem B_up, P-map và lỗi hình học để chọn cải tiến. | Pilot đã xong. Có bằng chứng định lượng G4, báo cáo vẫn ghi chờ review ảnh. |
| 2. Huấn luyện | Train B1 rồi một nhánh M dùng PRC trên cùng TRAIN và cùng ngân sách. | Đã xong. |
| 3. Chọn checkpoint và ngưỡng | Chọn checkpoint và calibration trên 2.262 nguồn remaining validation. Chẩn đoán lỗi và thử thêm phục hồi ảnh trên tập này. | Đã xong. |
| 4. Benchmark cuối | Khóa checkpoint và ngưỡng. Chấm TextVQA TEST, sau đó chấm ICDAR2015 test để đánh giá ngoài miền. | Còn chờ. |
| 5. Hoàn thiện báo cáo | Tổng hợp bảng kết quả test, chênh lệch có CI, lỗi theo kích thước chữ, ví dụ và chi phí suy luận. | Còn chờ benchmark cuối. |

| Nhánh | Câu hỏi kiểm tra |
|---|---|
| B0 / B_up | B0 là checkpoint Monkey gốc. B_up phóng LR2 bằng bicubic trước B0 để đo lợi ích của nội suy. |
| B1 | Baseline fine-tune từ B0 với Clean + LR2. Đo lợi ích của việc học trực tiếp trên LR2. |
| M (PRC) | Phương án thử nghiệm với cùng recipe B1 và thêm loss khớp P-map với teacher B0 đóng băng, tại kernel chữ mà teacher phát hiện khớp GT TRAIN trên ảnh Clean cùng nguồn. |
| B0-cal / B1-cal / M-cal | Chọn cặp ngưỡng DBNet trên cùng 2.262 nguồn, tối đa hóa trung bình Hmean Clean và LR2. |

B1 và M dùng seed 42, AdamW `2e-5`, batch hiệu dụng 4 và **10.000 optimizer updates**. Chọn giữa mốc 5.000 và 10.000 theo trung bình Hmean native Clean/LR2, hòa thì lấy mốc sớm. Calibration dùng cùng lưới 15 cặp ngưỡng, ưu tiên native nếu hòa. Cả ba mô hình chọn `mask=0.30`, `score=0.40`. Native dùng `0.30/0.30`.

## 2. Checkpoint và bằng chứng tái lập

Hậu tố `-cal` chỉ cặp ngưỡng hậu xử lý và dùng cùng weights với bản native. B1 và M xuất dưới dạng `state_dict` cho suy luận. [Tệp tải, SHA-256 và cách load](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/checkpoints.md).

| Checkpoint | Bằng chứng tái lập |
|---|---|
| **B0 / E0** | Với 500 ảnh Clean S1, B0 khớp số đếm và matching của E0 lịch sử. [Tải B0 từ upstream](https://huggingface.co/HB16888/MonkeyOCRv2_det/blob/main/dbnet_mkv2vitae_icdar2015.pth). |
| **B1 / B1-cal** | Job `12396` tái lập điểm native của `12187`. Job `12434` tái lập số đếm từng ảnh Clean/LR2 và `13127` tái lập riêng nhánh LR2 gốc. Đây là đối chứng chính cho hướng mới. [Tải B1](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/weights/b1-inference.pth). |
| **M / M-cal (PRC)** | Job `12518` đối chiếu kết quả native giữa bước chọn checkpoint và bước calibration. [Tải M](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/weights/m-prc-inference.pth). |

Các file để thiết lập benchmark gồm [config inference](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/inference_config.py), [policy chấm TextVQA](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/textvqa_evaluation_protocol.json) và [mã chấm](https://github.com/ncat218/MONKEY_V2/blob/main/lowres/evaluate_shared.py). Tải dữ liệu từ nguồn phát hành. [TextVQA](https://textvqa.org/download/) · [TextOCR](https://textvqa.org/textocr/dataset/) · [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads).

## 3. Cách chạy và chấm benchmark cuối

| | TextVQA FULL_R1 | ICDAR2015 Incidental Scene Text |
|---|---|---|
| Vai trò | Benchmark chính trên 3.263 nguồn TEST nội bộ đã khóa, dùng GT polygon TextOCR. | Xác nhận ngoài miền trên 500 ảnh test chính thức, 2.077 vùng chữ hợp lệ và 3.153 vùng ignore theo audit lịch sử. |
| Đầu vào | Clean, LR2 chính, LR4/LR8 để đo mức suy thoái nặng và B_up ở LR2. | Clean, LR2 cùng điều kiện đã chọn, kèm LR4 để kiểm tra mức nặng. Dùng checkpoint và ngưỡng đã chốt từ TextVQA validation. |
| Cách matching | Evaluator FULL_R1 ghép một-một theo số cặp tối đa ở polygon IoU **≥0,50**. Loại prediction có dưới 50% diện tích trong ROI hoặc trên 50% diện tích sau cắt ROI chồng vùng ignore. | Đề xuất tái sử dụng evaluator ICDAR2015 lịch sử, MMOCR `HmeanIOUMetric` với `strategy=vanilla`, IoU **>0,50** và ignore overlap **>0,50**. Chốt evaluator trước job. Job `11339` kiểm chứng pipeline cũ với E0/A′. |
| Phép so sánh | B0, B_up, B1 và M. Báo cả native và calibrated cho B0/B1/M. Endpoint chính là `Hmean(M-cal) − Hmean(B1-cal)` trên LR2. | So B0/B1/M ở các ngưỡng đã khóa. Báo kết quả riêng cho từng điều kiện và kiểm tra Clean retention. |

1. **Chuẩn bị ảnh ghép cặp.** Với mỗi ảnh nguồn, tạo từng mức LR trực tiếp từ Clean bằng bicubic Pillow 11.3.0, kích thước mỗi cạnh là `max(1, size // scale)`. Đưa polygon về hệ Clean bằng tỷ lệ kích thước thực trên từng trục trước khi matching. Kiểm hash danh sách nguồn, ảnh, GT, checkpoint và evaluator. [Mã sinh LR](https://github.com/ncat218/MONKEY_V2/blob/b0b63b4b326aae6a0ba8d69919ac80ea3be10c8c/lowres/generate_lr_only.py) · [Evaluator TextVQA](https://github.com/ncat218/MONKEY_V2/blob/b0b63b4b326aae6a0ba8d69919ac80ea3be10c8c/lowres/evaluate_shared.py).
2. **Báo metric theo từng bộ và từng điều kiện.** Cộng TP/FP/FN toàn tập để tính micro-Precision, Recall và Hmean (`2TP / (2TP + FP + FN)`). Báo thêm recall chữ theo chiều cao native LR (`<4`, `4–<8`, `8–<16`, `≥16` px), tỷ lệ từ tìm được trên Clean vẫn tìm được trên LR, latency và VRAM. Tính CI 95% chênh Hmean bằng ít nhất 2.000 bootstrap ghép cặp theo ảnh nguồn, tính lại micro-Hmean mỗi lần.
3. **Kết luận theo tiêu chí đã khóa.** Trên TextVQA TEST LR2, mục tiêu là M-cal hơn B1-cal **≥1,00 điểm Hmean** theo ước lượng, cận dưới CI 95% **>0** và Clean giảm tối đa **1,00 điểm**. Với ICDAR2015, báo thêm tiêu chí phụ kế thừa cho LR4 là Hmean tăng và Clean giảm tối đa 1,00 điểm. Tập test chỉ dùng ở bước này.

## 4. Kết quả đã có

Báo cáo gốc theo từng job ở [thư mục kết quả](https://github.com/ncat218/MONKEY_V2/tree/main/docs/handoff/results). Đọc nhanh [B0/B1 calibration](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/results/calibration_report.md), [chẩn đoán B1-cal](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/results/b1_cal_diagnostics.md), [PRC](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/results/prc_report.md) và [phục hồi ảnh](https://github.com/ncat218/MONKEY_V2/blob/main/docs/handoff/results/restoration_report.md).

| Mốc | Kết quả |
|---|---|
| **S1 · 11492** | B0 đạt 70,76% Clean và 27,43% LR4. LR4 trượt G3 vì chỉ 22,59% từ bị mất cao ít nhất 6 px, dưới mốc 30%. |
| **Audit và S2-A · 11907** | B0 đạt 66,65% Clean, 57,26% LR2, 25,64% LR4 và 6,69% LR8. LR2 giảm 9,39 điểm, CI 95% `[3,86; 13,68]`, đạt G1. Có 29,84% Clean TP chuyển thành FN, đạt G2. Trong số từ mất, 57,88% cao ít nhất 6 px, đạt G3. Chọn LR2 để train. B_up tăng 0,80 điểm. |
| **S2-B · 12117** | Hmean Gaussian Blur `σ=1.88/3.75` là 34,11%/10,06%. JPEG `quality=30/10` là 60,83%/47,46%. Kết quả mô tả, G3 chưa đánh giá, `τ=N/A`. |
| **Khôi phục TRAIN · 12181** | Dựng lại từ metadata job `8323` và mã lịch sử `816c6d6`. Manifest 15.222 nguồn khớp SHA-256 gốc, cả 3.262 dòng validation tái lập chính xác. |
| **B1 · 12187 và calibration · 12396** | B1 chốt checkpoint 10.000 updates. B1 native tái lập đúng theo từng điều kiện khi calibration. B1-cal tăng **13,35 điểm Hmean LR2** so với B0-cal, CI 95% mô tả `[+10,25; +16,33]` tại ngưỡng cố định trên validation. |
| **Chẩn đoán B1-cal · 12434** | Trong 6.454 Clean TP → LR2 FN, có 3.280 lỗi hình học (50,82%) và 2.791 ca phản hồi yếu (43,24%) tại `c0=0.30`. Ở `c0=0.50`, phản hồi yếu chiếm 54,20%. Có ảnh review P-map. |
| **PRC · 12518** | M chốt checkpoint 10.000 updates. Chênh lệch M-cal so với B1-cal là +0,02 điểm Clean và −0,02 điểm LR2, CI mô tả LR2 `[−0,064; +0,022]`. PRC có mask hợp lệ ở **608/20.000 microbatch (3,04%)**. |

**Điểm trên cùng 2.262 nguồn validation**

| Mô hình | Clean Hmean | LR2 Hmean |
|---|---:|---:|
| B0 native | 71,03% | 59,78% |
| B0-cal | 73,30% | 61,95% |
| B1 native | 71,89% | 69,88% |
| B1-cal | 77,55% | 75,29% |
| M native | 71,93% | 69,88% |
| M-cal | 77,57% | 75,27% |

**Phục hồi ảnh trước B1-cal · job 13127, thí nghiệm mở rộng trên validation.** Giữ checkpoint và ngưỡng B1-cal. So bicubic và EDSR baseline ×2 của torchSR `v1.0.3` ở cùng kích thước đầu ra. Job đã tái lập số đếm LR2 gốc từng ảnh. [Thiết kế và checkpoint EDSR](https://github.com/ncat218/MONKEY_V2/blob/b0b63b4b326aae6a0ba8d69919ac80ea3be10c8c/docs/lr2-restoration-proposal.md).

| Đầu vào | Hmean | Recall chữ <4 px | FP | Latency trung vị |
|---|---:|---:|---:|---:|
| LR2 gốc | 75,29% | 45,02% | 16.487 | 136,9 ms |
| Bicubic ×2 | 75,24% | 45,35% | 16.972 | 161,6 ms |
| EDSR ×2 | **75,89%** | **48,86%** | 17.439 | 204,6 ms |

**Đánh giá.** Mức tăng **13,35 điểm Hmean LR2** đến từ baseline B1-cal so với B0-cal trên validation. Thử nghiệm PRC cho **−0,02 điểm** so với B1-cal và có tín hiệu loss ở 3,04% microbatch. EDSR là model pretrained có sẵn dùng làm tiền xử lý. Trên validation, EDSR tăng **0,60 điểm** so với LR2 gốc, CI 95% `[+0,24; +0,96]`, và **0,66 điểm** so với bicubic, CI `[+0,27; +1,04]`. Đổi lại, EDSR tăng 952 FP và thêm 67,7 ms latency trung vị so với LR2 gốc. Latency gồm phục hồi và suy luận detector, bỏ thời gian ghi PNG tạm và chấm metric.

## 5. Việc cần giải quyết và đề bài nghiên cứu độc lập

1. **Khôi phục manifest TEST.** Chưa thấy file manifest của 3.263 nguồn trong các thư mục HPC đã kiểm tra. Dựng lại từ audit `8323` theo split lịch sử như TRAIN rồi kiểm SHA-256 gốc `054f44643c7d3fa0006c8eb3e502c602953957c34cf957cc5d954dd04f4e079f` trước khi chấm.
2. **Hoàn thành benchmark Rev.4.** Dùng checkpoint và ngưỡng đã khóa ở Mục 1–2. Xác nhận evaluator ICDAR2015 và kiểm lại hash 500 ảnh cùng GT, rồi chạy benchmark như Mục 3. Ghi rõ TextOCR official train từng được dùng trong pretraining Monkey khi diễn giải TextVQA TEST.
3. **Ghi giới hạn của PRC và VRAM.** Nguyên nhân cụ thể của tỷ lệ mask PRC thấp vẫn chưa được xác định. Báo cáo G4 vẫn ghi chờ review định tính. Số peak VRAM 322 MiB trong summary B1 được đọc sau khi bộ đếm bị reset. Dùng log B1 ghi 4.965 MiB khi đối chiếu tài nguyên và nêu rõ nguồn đo.

### Phần gửi lại cho nhóm

**Câu hỏi mở.** Cách nào có thể giúp B1-cal phát hiện thêm chữ trên LR2 mà vẫn giữ chất lượng Clean và chi phí hợp lý? Dữ kiện để bắt đầu là 3.280 lỗi hình học, 2.791 ca phản hồi yếu và recall 45,02% ở nhóm chữ LR2 dưới 4 px. Người review có thể dùng dữ kiện này hoặc đưa ra giả thuyết khác có căn cứ. Nếu muốn nghiên cứu mức suy thoái nặng hơn hoặc loại suy thoái khác, ghi rõ câu hỏi và đề xuất protocol riêng.

Gửi lại một bản đề xuất ngắn gồm:

1. **Một giả thuyết và một phương pháp tự chọn.** Nêu nhóm lỗi muốn xử lý, bằng chứng hoặc bài báo liên quan, và lý do kỳ vọng có tác động.
2. **Một thí nghiệm có đối chứng.** Nêu thay đổi duy nhất cần đo, checkpoint đối chứng B1-cal, dữ liệu cần dùng, ngân sách chạy và điều kiện để quyết định tiếp tục. Nếu cần train hoặc chỉnh ngưỡng, dùng TRAIN và remaining validation theo vai trò ở Mục 1.
3. **Cách đọc kết quả.** Ghi chỉ số chính, CI, ảnh hưởng lên Clean, FP và latency. Chốt thiết kế trước khi dùng test. Với hướng ngoài Rev.4, ghi rõ đây là đề xuất nghiên cứu bổ sung.
