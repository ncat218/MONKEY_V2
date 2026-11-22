# Bàn giao MONKEY_V2 · CS431

Cập nhật **22/11/2026**, snapshot kết quả đã xác minh; job fusion **106090 đã FAILED ở cleanup đánh giá; recovery v2 job111683 đã submit và RUNNING ở lần kiểm tra khởi động**. Điểm bắt đầu cho người nhận: [workflow hiện tại](docs/research/current-workflow.md), [bảng kết quả/báo cáo gốc](docs/handoff/results/README.md), [checkpoint tải được](docs/handoff/checkpoints.md), [artifacts mới](docs/handoff/artifacts/research_20261122/README.md).

## 1. Câu hỏi và phạm vi báo cáo

- **RQ1:** Native LR2 làm detector gốc suy giảm thế nào, đặc biệt với chữ nhỏ?
- **RQ2:** Lợi ích fine-tune đến từ thích nghi miền TextOCR hay từ trộn độ phân giải? Đối chứng B-Clean là bắt buộc để phân biệt.
- **RQ3:** Sau khi có baseline fine-tune và hậu xử lý mạnh, module bổ sung có tăng chất lượng, giữ Clean và kiểm soát FP/latency không?

Pipeline là **MonkeyOCRv2-AS backbone → FPNC → DBHead → DBPostprocessor quad**. Backbone trích xuất đặc trưng đa tỉ lệ; FPNC hợp nhất thành tensor 256 kênh; DBHead dự đoán bản đồ, hậu xử lý tạo các tứ giác. Không có LLM sinh văn bản hoặc decoder tái tạo ảnh trong tác vụ này. Sự nhạy cảm LR là vấn đề thực nghiệm nhóm khảo sát, không mặc định là một trong bốn limitations mà tác giả tự nêu cho document parsing. Chỉ nói đã tái lập nhánh detection và thích nghi checkpoint; chưa tuyên bố reproduce toàn bộ paper.

| Tên | Ý nghĩa |
|---|---|
| B0 | Detector gốc, AS + FPNC + DBNet, checkpoint detection ICDAR2015 |
| B-Clean | Từ cùng B0, fine-tune 100% Clean trên đúng TRAIN của nhóm |
| B1 | Từ cùng B0, fine-tune 50% Clean / 50% native LR2 |
| `-cal` | Cùng trọng số, đổi ngưỡng hậu xử lý đã chọn trên development; không phải train thêm |
| M / PRC | B1 recipe thêm probability-map consistency; một seed |
| A0/A1/A2 | Từ B1, train tiếp DBHead / thêm residual adapter / thêm gated adapter sau FPNC |

## 2. Dữ liệu, split và suy thoái

Ảnh từ **TextVQA**, nhãn phát hiện dày đặc cấp từ từ **TextOCR**, không chỉ chữ liên quan câu trả lời VQA. FULL_R1 là split nội bộ của nhóm, không đồng nhất với tên official train/val/test của dataset.

| Split nội bộ | Nguồn | Vai trò / khả năng truy cập |
|---|---:|---|
| TRAIN | 15.222 | Huấn luyện; [CSV IDs, đường dẫn tương đối và SHA256 ảnh](docs/handoff/data/train_ids.csv) |
| S1 | 500 | Pilot; [IDs](docs/handoff/artifacts/research_20261122/data/S1_source_ids.txt) |
| S2 | 500 | Pilot; [IDs](docs/handoff/artifacts/research_20261122/data/S2_source_ids.txt) |
| Remaining-validation | 2.262 | Checkpoint selection, calibration, EDA và thử nghiệm tiếp theo |
| TEST | 3.263, số lịch sử | Không đọc trong các extension/snapshot này; chưa audit khả năng tái lập cuối ở lần bàn giao này |

[Validation IDs và hash ảnh](docs/handoff/artifacts/research_20261122/data/validation_ids.csv) có đủ 3.262 nguồn và role. TRAIN/VAL disjoint đã kiểm lại. Manifest TRAIN SHA256 `ca1b91636cbe025823041ce336ffb59affa826a073f666dd7ba386bcab6c983d`; VAL `5cbf2230b0f83ab30816aca8afc5cb466e0b3301030ccd25d7d6d62d3136dfff`. CSV TRAIN SHA `8ac5cbd98db0ffcb67536c6629ed4839ecf8c8e82b927796d0f9208102805f60`. Đây là các file khác nhau nên hash khác nhau. Xem [receipt](docs/handoff/artifacts/research_20261122/data/split_receipt.json). Không phát hành ảnh hoặc manifest đầy đủ chứa dữ liệu của dự án cũ.

**Native LR2:** RGB, Pillow 11.3.0, `resize((floor(W/2), floor(H/2)), BICUBIC)`. Không upscale lại về W×H trước detector. Detector vẫn resize giữ tỉ lệ theo `(4068,1024)`, pad bội 32 như config gốc. Đổi tọa độ GT/prediction bằng tỉ lệ thật từng trục; kích thước lẻ không được giả định scale chính xác 2. Blur sigma 1,88/3,75; JPEG quality 30/10 và LR4/LR8 là các stress conditions riêng.

**Giới hạn độc lập:** S1/S2 đã bị xem trong pilot, remaining-validation được dùng lặp lại; không gọi chúng là fresh holdout. Protocol v2.0 cũng ghi nhận TEST từng được xem ở lịch sử trước; cần công khai lịch sử đó và khóa đánh giá cuối, không tuyên bố toàn bộ TEST còn nguyên vẹn chưa từng bị xem. AS pretraining có TextOCR TRAIN; B0 detection đã học ICDAR2015. Cần phân biệt upstream pretraining exposure với fine-tune split của nhóm.

## 3. Huấn luyện và metric đã dùng

B-Clean/B1 cùng B0 init, source TRAIN, full detector trainable, seed 42, **10.000 optimizer updates**, microbatch 2, accumulation 2, LR 2e-5, weight decay 1e-4, clip 5, warmup 100. Hai candidate checkpoint tại 5k/10k được chọn bằng mean Clean/LR2 Hmean theo protocol. B-Clean chỉ đổi phân bố resolution, không thêm nguồn hay tăng số updates. Tên runner `iter_20000` ứng với 20.000 microbatches / 10.000 optimizer updates.

Metric chính là **micro Precision / Recall / Hmean**, TP/FP/FN từ matching polygon một-một maximum cardinality, IoU ≥0,5, xử lý ROI và ignore theo [evaluator đã khóa](docs/handoff/textvqa_evaluation_protocol.json). FN là GT chưa có prediction khớp đủ IoU, có thể là bỏ sót hoặc định vị không đạt. FP là prediction không được ghép, chưa đủ bằng chứng để gọi tất cả là nền giả.

Paired bootstrap theo **source_image_id**, 2.000 lần, seed 20261107, giữ cặp Clean/LR2 và các model cùng nguồn. CI này mô tả biến thiên nguồn, **không đo biến thiên seed huấn luyện** và không sửa được selection bias. Báo thêm recall theo chiều cao chữ, FP, Clean retention và latency. Runtime FP32, matmul TF32 off / cuDNN TF32 on theo chính sách đã audit. Đường thực tế là **quad**, không phải poly.

## 4. Baseline và vai trò calibration

Bảng chính dưới đây trên 2.262 remaining-validation. Grid mở rộng dùng cùng 72 combinations và quy tắc maximize mean Clean/LR2 Hmean cho từng model. B-Clean/B1 chọn mask `.40`, score `.60`, unclip 1,5, min width 5, epsilon `.01`; B0 chọn `.30/.45`, epsilon `.005`. Đây là lựa chọn exploratory trên validation đã xem.

| Model | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP | LR2 recall <4 px |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 74,09% | 62,38% | 90,16% | 47,68% | 4.069 | 1,01% |
| B-Clean | 80,86% | 75,51% | 88,66% | 65,75% | 6.578 | 15,96% |
| B1 | 79,43% | 76,90% | 82,45% | 72,05% | 11.997 | 45,79% |

B1 phát hiện thêm chữ nhỏ nhưng đánh đổi FP và Clean. B1 − B-Clean: LR2 **+1,394 pp**, CI **[−1,255; +3,719]**; Clean **−1,426 pp**, CI **[−2,114; −0,903]**. Chưa có bằng chứng B1 hơn toàn diện. B-Clean tăng rất mạnh so với B0, nên không thể quy toàn bộ lợi ích B1 cho mixed-resolution training.

| Model, common historical mask/score `.30/.40` | Clean H | LR2 H |
|---|---:|---:|
| B0-cal | 73,30% | 61,95% |
| B-Clean-cal | 78,79% | 74,36% |
| B1-cal | 77,55% | 75,29% |

Chỉ đổi hậu xử lý B1 từ common sang grid mới: LR2 **+1,610 pp**, CI [1,006; 2,301], FP **16.487 → 11.997**; Clean +1,882 pp. Đây là cải thiện calibration, không phải kiến trúc mới. Không thay các báo cáo frozen bằng bảng exploratory. [Artifacts PR/quad](docs/handoff/artifacts/pr_quad_audit_v1/README.md) và [chẩn đoán ở ngưỡng mới](docs/handoff/artifacts/research_20261122/fixed_diagnostics/results/fixed_diagnostics_report.md).

## 5. Những gì đã thử

| Thử nghiệm | Thiết kế / điểm so sánh | Kết quả và đánh giá |
|---|---|---|
| PRC | Thêm consistency, common `.30/.40` | M 77,57/75,27 vs B1 77,55/75,29 (Clean/LR2). Mask active 608/20.000 microbatches; chưa chứng minh lợi ích, cần chẩn đoán mask |
| Feature distillation, job 48499 | FPNC fused feature, Clean teacher, thêm 1k updates; có continued-training control | KD LR2 75,24; control 75,24; Δ −0,00048 pp, CI [−0,03483; 0,03482]. Không có tín hiệu vượt control |
| Bicubic x2 / EDSR x2 trước B1 | Common `.30/.40`, native LR2 baseline 75,29 | Bicubic 75,24; EDSR 75,89, Δ +0,60 pp, CI [0,24;0,96]. FP 16.487→17.439; median 136,9→204,6 ms. Chưa đo Clean, chưa kiểm ở calibration mới; SR pretrain bicubic trùng suy thoái tổng hợp |
| CRAFT độc lập, job 74220 | Default CRAFT `.7/.4/.4`, canvas1280, mag1,5; B1 `.40/.60` | Clean/LR2 70,35/47,00; B1 79,43/76,90. CRAFT-only LR2 TP 1.172, tiny recall 0/5.158. Khác resize policy, chưa phải so sánh kiến trúc công bằng hay kết quả fusion |
| Latent adapter pilot, job 81918 | Thêm 1k updates, A0/A1/A2 từ B1; freeze backbone+FPNC | LR2 76,71/76,71/76,72; không vượt B1 76,90 |
| Latent adapter full, job 88935 | Thêm 10k updates mỗi nhánh; cùng stream, detection loss; freeze backbone+FPNC | Xem bảng dưới: chưa vượt control hoặc B1 |

| Job 88935, fixed `.40/.60` | Clean H | LR2 H | LR2 FP |
|---|---:|---:|---:|
| B1 — trước train tiếp | 79,43% | 76,90% | 11.997 |
| A0 — chỉ DBHead | 79,47% | 76,87% | 12.037 |
| A1 — residual adapter sau FPNC | 79,48% | 76,86% | 12.046 |
| A2 — gated residual adapter | 79,47% | 76,86% | 12.042 |

A2 − A0 LR2 **−0,007 pp**, CI [−0,01845; 0,00367]; A2 − B1 **−0,037 pp**, CI [−0,11291; 0,05007]. Adapter có cập nhật trọng số và đủ Adam steps; stream source/augmentation giống nhau giữa ba nhánh. Chưa thấy lợi ích trong cấu hình này, không phủ định mọi latent method. Đây là feature refinement theo detection loss, không phải tái tạo RGB và không dùng trực tiếp checkpoint S1b của hệ Qwen khác.

CRAFT đã chạy screen và tạo teacher maps trên 200 TRAIN sources, có 32 cards chờ review; **chưa train character/affinity auxiliary head**. Word GT không đủ chứng minh pseudo character labels đúng. Không tự hợp detector rồi gọi số craft-only là mức tăng khả thi.

Latency ngắn của B1 ở job latent: mean của image-median khoảng Clean **128,49 ms**, LR2 **127,86 ms**, batch1 FP32 H200 MIG35GB. Chỉ 8 ảnh cố định, 2 warmups/5 repeats, model + CPU quad decode, không IO/preprocess/load; không phải trung bình toàn dataset. Không so trực tiếp với đo EDSR có scope khác.

## 6. Đang chạy và bước tiếp theo

**Fusion job106090 đã dừng ở cleanup ASF; [recovery v2](docs/fusion-recovery-v2.md) giữ nguyên checkpoint và chạy tiếp phần thiếu.** Thiết kế gốc: nearest control, ASF, bilinear control, DySample; cùng B1 init, freeze backbone, train **FPNC + DBHead** 10k updates. Đây là chế độ trainable khác A0 cũ nên phải có control mới. 1k pilot nằm trong cùng run 10k, giữ optimizer/scheduler; lỗi probe thì dừng toàn job, metric pilot âm không tự đổi mục tiêu. Probe kỹ thuật và frozen-corruption check đã PASS. Nearest/ASF đã train đủ10k; Clean/LR2 đủ2.262 nguồn, stress ASF còn thiếu; Bilinear/DySample chưa train. Chưa có báo cáo tổng kết đầy đủ. [Thiết kế và nguồn nghiên cứu](docs/fusion_research_and_generalization.md), [identity/probe snapshot](docs/handoff/artifacts/research_20261122/fusion_snapshot/).

Sau run: so ASF với nearest, DySample với bilinear và với B1; Clean/LR2 đủ 2.262 nguồn, fixed `.40/.60` chính, `.30/.40` phụ; transfer LR4/LR8/blur/JPEG trên 500 nguồn đã chọn bằng hash. Không chọn threshold riêng theo corruption để làm đẹp kết quả.

Việc còn thiếu quyết định chất lượng báo cáo:

1. **Review bằng mắt:** 127 paired cards ở `.40/.60` và 32 CRAFT TRAIN cards vẫn PENDING. Nhãn proxy geometry/weak response chưa phải nguyên nhân thật; không suy diễn FP đều do nền.
2. **Tổng quát hóa:** chưa chứng minh threshold/module tốt trên nhiều corruption hay ICDAR2015. S1/S2 chỉ là confirmation trên phần development đã xem, không fresh holdout.
3. **Độ ổn định:** một seed; cần thêm seed matched baseline/method nếu có tín hiệu đáng kể và ngân sách cho phép. CI theo nguồn không thay thế seed.
4. **Chọn cải tiến:** hiện có adaptation + calibration mạnh, EDSR có lợi ích nhỏ với chi phí rõ; PRC/KD/adapter chưa vượt control. Chưa có module kiến trúc mới được chứng minh tốt hơn baseline mạnh.
5. **Benchmark cuối:** nhóm khóa method/checkpoint/postprocessing/deployment trước, audit lại manifest và lịch sử TEST, rồi đánh giá internal TextVQA/TextOCR TEST và ICDAR2015 theo evaluator đã công bố. Không dùng hai bộ này chọn hướng. ICDAR evaluator và ignore/matching cần ghi riêng, không tự sửa protocol cho giống nhau.

Nhánh Mix A/B, QRoute/DualAgree của cộng tác viên được theo dõi riêng. Không gộp kết quả chưa xác minh vào bảng Monkey; hệ dùng hai checkpoint cần báo đủ chi phí và thống nhất ràng buộc triển khai. Repo này chưa chứng minh ưu thế của các nhánh đó.

## 7. Tái lập và bàn giao

- [Weights B0/B-Clean/B1/M/A0/A1/A2 + SHA256 + config](docs/handoff/checkpoints.md), tải qua Git LFS; file inference không có optimizer.
- [Artifacts gốc PR/quad](docs/handoff/artifacts/pr_quad_audit_v1/README.md) giữ nguyên; [snapshot mới](docs/handoff/artifacts/research_20261122/README.md) có per-source counts, CI, review ZIP và provenance. Không chứa toàn bộ P-map cache/ảnh dataset.
- [Protocol v2.0 và v2.1](docs/handoff/protocol/) được giữ nguyên. Các thử nghiệm sau đó là extension exploratory, không thay endpoint của protocol đã khóa.
- [Trang cũ đã lưu](docs/archive/README.md); dùng workflow hiện tại để quyết định bước tiếp, dùng raw report cho chi tiết thực nghiệm.

Đóng góp báo cáo có thể bảo vệ bằng bằng chứng là benchmark degradation có kiểm soát, clean-only control tách domain/resolution adaptation, calibration đối xứng và phân tích đánh đổi tiny recall/FP/Clean; các kết quả âm có đối chứng được trình bày đầy đủ. Nếu fusion không vượt control, báo đúng kết quả và giới hạn thay vì coi một module thêm vào là cải tiến đã thành công.
