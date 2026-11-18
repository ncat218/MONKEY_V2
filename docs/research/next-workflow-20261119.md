# Workflow cần làm tiếp: chẩn đoán, pilot latent, xác nhận

Ngày: 19/11/2026, Asia/Bangkok.

Đã đọc `MonkeyOCRv2_TextDetection_Progress_Results_Research_Roadmap.md` cùng đề xuất
S1b/CRAFT trong cuộc trao đổi, đối chiếu báo cáo job và source hiện có. Các bước
dưới đây là kế hoạch tiếp theo, không phải kết quả đã chạy hay sửa protocol đã khóa.
Lượt này chỉ viết tài liệu; không sửa code thực nghiệm hoặc submit job.

## 1. Mục tiêu và điểm xuất phát

**Câu hỏi chính:** một can thiệp nhẹ có giữ được recall chữ nhỏ của B1, đồng thời
giảm FP và giữ chất lượng Clean, vượt các đối chứng fine-tune/calibration không?

Đã có B0, B-Clean, B1, EDA ba mô hình, PRC, feature-KD và restoration controls.
Không cần chạy lại B-Clean/B1 hoặc dựng lại split để bắt đầu workflow này.

| Mốc | Clean Hmean | LR2 Hmean | LR2 FP |
|---|---:|---:|---:|
| B0-cal | 73,30% | 61,95% | 5.129 |
| B-Clean-cal | 78,79% | 74,36% | 10.457 |
| B1-cal | 77,55% | 75,29% | 16.487 |

B1−B-Clean: LR2 +0,93 pp, CI [-1,43; +3,14]; Clean -1,24 pp,
CI [-2,06; -0,64]. Recall LR2 <4 px tăng 19,31→45,02%, song FP tăng ròng 6.030.
Các số này xác định vấn đề, chưa chỉ ra nguyên nhân nằm ở backbone/FPNC.

## 2. Những chỗ cần cập nhật trong roadmap nguồn

1. **B-Clean đã được xác minh calibrated-optimal 0,30/0,40**, cùng B0/B1 trong
   original 15-pair grid. Bảng common và calibrated hiện trùng nhau.
2. **Bicubic ×2 control đã chạy**: LR2 75,29; bicubic ×2 75,24; EDSR ×2 75,89.
   EDSR−bicubic +0,66 pp, CI [0,27; 1,04]. Còn thiếu Clean cho EDSR và kiểm tra
   tổng quát hơn, không thiếu toàn bộ matched-output-size control. Nguồn:
   [restoration report](../handoff/results/restoration_report.md).
3. Native LR2 dùng Pillow 11.3.0 **bicubic downsampling**, không upscale trở lại.
   Kích thước luôn được lấy từ task thực tế; không mặc định feature grid cũng
   giảm hai lần vì detector còn resize/pad nội bộ. Dùng lại `input_shapes.csv`.
4. `geometry` và `weak_response` trong implementation là các nhãn rời nhau do
   thứ tự ưu tiên của classifier. Những nguyên nhân thực tế có thể đồng thời tồn
   tại; hai nhãn đó không phải phép phân rã nguyên nhân nhân quả.
5. DBHead hiện tại dự đoán các map phục vụ DBNet, không có một nhánh regression
   tọa độ hộp riêng mang tên geometry. Quad được tạo bởi postprocessor.
6. S1b Qwen weights không dùng trong nhánh Monkey: khác feature space/shape và
   TRAIN trùng 2.053 remaining-validation sources. Chỉ tái sử dụng ý tưởng kiến
   trúc, khởi tạo module mới. Xem [audit S1b/CRAFT](s1b-craft-monkey-feasibility-20261119.md).

## 3. Bước làm ngay: PR và quad audit, chưa train

**Câu hỏi:** ở cùng mức recall, B1 có còn nhiều FP hơn B-Clean? Chỉnh cách tạo/lọc
hộp có giải quyết một phần trade-off mà không thay feature không?

### Công việc

- Tái sử dụng original-grid counts, predictions, pre-score-cutoff candidates và
  source IDs từ các job calibration/EDA. Dùng cả Clean và LR2.
- Vẽ PR, Hmean–threshold, global-FP–tiny-recall cho B-Clean/B1; B0 là tham chiếu.
- Đối chiếu FP tại các mức global recall và tiny-text recall chung, trong phần
  khoảng giá trị mà hai model đều đạt được. Không gán precision theo size bin
  từ số GT-only recall khi chưa định nghĩa cách gán FP vào bin.
- Đọc và hash **DBPostprocessor thực cài trên HPC**; xác minh ảnh hưởng của
  `unclip_ratio`, `min_text_width`, `epsilon_ratio` trên đúng nhánh `quad`.
- Thiết kế một grid mở rộng có giới hạn chi phí, cùng grid/quy tắc chọn cho các
  model; lưu trong config exploratory riêng trước khi chạy. Không thay các kết
  quả `.30/.40` cũ, không tự chuyển `quad` sang `poly`.
- Khi đổi geometry hoặc mask threshold, cần P-map FP32 hoặc forward mới. EDA
  hiện không mặc định chứa toàn bộ raw FP32 P-map. Một forward có thể dùng cho
  nhiều cấu hình postprocessing, thay vì forward lại mỗi cấu hình.

Source MMOCR công khai giúp xác định logic cần kiểm tra, nhưng không thay thế
audit bản cài thực tế:
[DBPostprocessor](https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/postprocessors/db_postprocessor.py).

### Đầu ra

`pr_curves.csv`, `threshold_frontier.csv`, `quad_filter_audit.csv`,
`input_shapes_audit.csv`, `calibration_diagnostics.md`, kèm hash cấu hình và các
baseline cũ tái lập được. Grid mới được ghi là exploratory, không gọi là frozen
protocol ban đầu. Chọn theo mục tiêu Clean/LR2 chung, không tối ưu riêng LR2 rồi
bỏ qua Clean.

## 4. Làm cùng đợt: phân loại FP, regression và persistent FN

**Câu hỏi:** FP đang là nền giả, hộp trùng, lệch biên, gộp/tách từ hay vấn đề GT?

- Dùng `fp_analysis.csv`, `paired_transitions.csv`, `failure_distributions.json`
  và các ảnh/P-map đã có trước khi chạy thêm inference.
- Tăng ròng 6.030 FP không phải một danh sách 6.030 hộp mới. FP correspondence
  B-Clean→B1 trên LR2 gồm 9.697 newer-only, 3.667 older-only, 6.790 persistent FP
  dưới quy tắc correspondence IoU 0,5. Đây là đối chiếu FP, không đổi evaluator.
- Review riêng 1.453 GT regressions và 19.590 persistent FN, thêm rescued <4 px.
- Chọn mẫu mới theo seed cố định, phân tầng điều kiện/score/cỡ chữ/transition;
  lưu quần thể và xác suất lấy mẫu. Nếu ước lượng tỷ lệ toàn tập, dùng trọng số
  theo quần thể từng tầng. Không dùng 192 ảnh minh họa để suy rộng tỷ lệ lỗi.
- Dùng một taxonomy thống nhất: background; duplicate; split/merge; box misfit;
  candidate bị filter; annotation ambiguity; không đủ thông tin để kết luận.
- Với FN, trace P-map→candidate→filter→matching để phân biệt tín hiệu yếu và hộp
  sai. Proxy từ P-map chưa chứng minh latent feature bị thiếu thông tin.
- Nền có chữ nhưng không match GT không mặc định là negative để training.

**Đầu ra:** `fp_taxonomy.csv`, `regression_review.csv`, `diagnosis_report.md`.
Báo mẫu số rõ, tách nhãn heuristic với review ảnh và các ca còn bất đồng.

Hai bước trên là công việc tiếp theo cần triển khai. Chưa cần train adapter.

## 5. Quyết định sau chẩn đoán

| Bằng chứng quan sát | Hành động tiếp theo |
|---|---|
| PR/quad control đã cải thiện trade-off đáng kể | Giữ làm baseline mới; chưa cần thêm module để giải thích cùng lợi ích |
| Tín hiệu có nhưng nhiều box misfit/merge/split | Ưu tiên localization/postprocessing; không mặc định cần feature restoration |
| Persistent tiny FN còn phản hồi yếu, chưa được xử lý bởi controls | Short pilot residual sau FPNC là ứng viên hợp lý |
| Trade-off chủ yếu do phân bố training | Cân nhắc 75/25 Clean/LR2 như nhánh recipe đối chứng riêng, cùng B0 init và budget B1 |

Không bắt buộc chỉ có một loại lỗi. Chọn một can thiệp chính cho vòng nghiên cứu
tiếp theo và ghi rõ bằng chứng đang hỗ trợ giả thuyết nào.

## 6. Pilot latent tối thiểu, nếu quyết định theo hướng feature

**Vị trí đầu tiên:** sau FPNC, 256 kênh, trước DBHead. Dùng adapter residual mới
`F'=F+R(F)`, output zero-init để tại bước đầu tái lập B1. Chưa thêm gate, CRAFT,
Clean identity loss hay nhiều adapter ở C2/C3 đồng thời.

### Ba nhánh

| Nhánh | Khởi tạo | Backbone/FPNC | Trainable | Loss |
|---|---|---|---|---|
| Head-control | B1 | Frozen | DBHead | Detection |
| Residual | Cùng B1 | Frozen | DBHead + R mới | Detection |
| Residual+KD | Cùng B1; cùng R init như Residual | Frozen | DBHead + cùng R | Detection + aligned Clean feature guidance |

Đây là thiết kế frozen-feature cụ thể cho vòng sàng lọc, không phải mọi cấu hình
adapter. Nó khác pilot continued fine-tuning full-model cũ và khác nhánh C0/C1/M
thêm identity penalty trong ghi chú trước. Các ghi chú đó là lựa chọn đề xuất;
workflow này chọn ma trận trên để cô lập ít biến số hơn.

### Những điều phải cố định

- Cùng B1 khởi tạo, 15.222 TRAIN IDs, 50/50 Clean/native LR2, source order,
  augmentation, seed, microbatch/accumulation, optimizer/LR/scheduler và update
  budget. Không áp dụng câu "cùng B0 initialization" của full 10k baselines vào
  pilot này: đây là adaptation tiếp từ cùng B1.
- Đề xuất 1.000 optimizer updates/nhánh cho vòng đầu, không sweep checkpoint;
  ghi lại config mới trước chạy. Chưa cam kết GPU hours khi chưa profile.
- Frozen phải giữ cả parameters và buffers/trạng thái BatchNorm phù hợp; teacher
  frozen/eval/no_grad. Kiểm tra hash/state trước và sau probe.
- Teacher Clean dùng B1 fixed; căn chỉnh crop/flip/FOV, grid, ignore/padding.
  Chốt feature normalization, mask và lambda trước khi chạy. Không ép khớp feature
  ở các vị trí không tương ứng.
- Feature loss phải tác động lên `F'` sau R. Nếu đo trên F frozen trước R, loss
  đó không thể huấn luyện R. Đây là gradient gate cần kiểm tra bằng autograd.
- Probe: identity-init, finite FP32, shape/alignment, nonzero useful adapter
  gradients, teacher không grad và không cập nhật frozen state.

### Cách đọc kết quả

- Residual−Head-control: tác động thêm adapter trong frozen regime.
- Residual+KD−Residual: giá trị thêm của feature guidance.
- Cả hai phải được so với B1/B-Clean và baseline postprocessing mạnh, không chỉ
  thắng một head-only control để tuyên bố vượt full fine-tuning.
- Residual chưa vượt control không tự động bác bỏ Residual+KD; đánh giá nhánh đó
  bằng kết quả đối chứng thực tế. Cũng không mặc định KD sẽ sửa mọi pilot âm.
- Báo native/common thresholds; nếu calibration lại, cùng grid và selection
  procedure cho tất cả checkpoint mới. Báo Clean/LR2, FP, recall theo cỡ chữ,
  source-paired CI, latency và memory.
- Nếu không có tín hiệu thực dụng, không tự chạy lại nhiều vị trí/lambda đến khi
  có số đẹp. Một pilot âm được giữ trong báo cáo.

## 7. Chỉ mở rộng có điều kiện

### Gate và ASF

Nếu adapter có tín hiệu tốt nhưng còn sửa sai nhiều vùng Clean/nền, mới thử gate
học từ feature. So adapter thường và gated adapter trong cùng regime. Clean
identity penalty, nếu thêm, cũng cần control riêng; không gộp nhiều loss rồi quy
lợi ích cho gate.

Gate theo không gian trên feature đã hợp nhất và adaptive fusion giữa nhiều scale
là hai can thiệp khác nhau. Nếu chuyển sang multi-scale fusion mới, cần đối chứng
ASF có sẵn trong MMOCR; không bắt buộc train ASF trước một residual sau FPNC chỉ
vì cả hai cùng thao tác feature. Baseline hiện tại `asf_cfg=None`.
[Source FPNC](https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/necks/fpn_cat.py).

### CRAFT

Đánh giá bổ trợ riêng, chưa gộp vào pilot latent ban đầu. Đo CRAFT trên Clean/LR2,
GT B1-FN/CRAFT-TP, B1-TP/CRAFT-FN, FP bổ sung, tiny recall và latency. Chốt đúng
checkpoint/preprocessing và provenance trước khi chạy.

CRAFT trên LR2 không bổ trợ tốt sẽ làm yếu lý do dùng nó như detector/prior tại
inference. Điều đó chưa bác bỏ một hướng khác: teacher CRAFT trên Clean TRAIN để
tạo pseudo-character labels. Nhánh teacher cần audit độ tin cậy/coverage riêng,
không coi word-GT overlap là bằng chứng character labels hoàn toàn đúng.

Chỉ thêm character/affinity auxiliary head khi có nhãn phù hợp và giả thuyết được
hỗ trợ. Word mask thông thường chưa đủ để gọi một module là CRAFT-inspired.

## 8. Xác nhận phương pháp cuối và hoàn thành đồ án

1. Chọn một phương pháp từ development; chốt weights, preprocessing, ngưỡng và
   điều kiện triển khai (mọi ảnh hay có điều kiện) trước đánh giá cuối.
2. Thêm seed cho phương pháp và matched control. Seed thứ hai là kiểm tra ổn định
   bước đầu; ba seed trở lên hữu ích nếu muốn định lượng biến thiên training.
3. Kiểm tra LR4/LR8, blur/JPEG, ICDAR2015 theo kế hoạch riêng và fixed settings;
   giữ rõ nguồn data/condition nào đã được dùng để lựa chọn phương pháp.
4. S1/S2 đã dùng ở pilot: có thể xác nhận fixed settings nhưng không gọi là fresh
   holdout của toàn quá trình nghiên cứu. Không chia lại remaining-validation rồi
   tuyên bố phần mới chia chưa từng được xem.
5. Kiểm tra readiness của manifest/evaluator final TEST sau khi đã khóa phương
   pháp. Chạy một đợt final benchmark cho baseline và phương pháp đã chọn; không
   dùng kết quả đó để quay lại chọn phương pháp/ngưỡng. Chưa làm bước này bây giờ.
6. Báo contributions đã đo: replication detection, controlled Clean-only/mixed
   training, paired EDA; chỉ thêm contribution của module nếu ablation chứng minh.
   Giữ các pilot âm và giới hạn provenance/selection/seed trong báo cáo.

## 9. Điều cần chốt trước bất kỳ job mới nào

Tên stage/output version; hash inputs/source; grid calibration; sampling/review
rules; trainable policy; update/compute budget; loss weights; checkpoint selection;
latency protocol; mục tiêu Clean/LR2/tiny-FP trade-off và tiêu chí dừng thực dụng.
Ví dụ ±0,5 pp trong roadmap là gợi ý, chưa phải ngưỡng đã khóa.

Mọi thay đổi là extension riêng; không ghi đè protocol, checkpoint, báo cáo và số
liệu cũ. Không dùng trọng số S1b Qwen hoặc TEST trong các bước lựa chọn trên.
