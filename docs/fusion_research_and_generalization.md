# Fusion, upsampling và khả năng tổng quát · 22/11/2026

## Điểm xuất phát

B1 ở operating point mở rộng đạt Clean/LR2 Hmean 79,43/76,90% trên 2.262 nguồn
remaining-validation đã dùng nhiều lần. Adapter sau FPNC ở cả 1k và 10k updates
chưa hơn B1/control. Chưa có cơ sở tăng tiếp ngân sách cho đúng adapter đó.
Các kết quả này không phủ định mọi cách can thiệp đặc trưng, cũng không chứng
minh các lỗi proxy geometry/background là nguyên nhân đã được xác nhận bằng ảnh.

## Research từ nguồn chính thức

| Công trình | Cơ chế và liên quan | Quyết định trong đợt này |
|---|---|---|
| [DBNet++ / ASF](https://github.com/MhLiao/DB), [MMOCR FPNC](https://mmocr.readthedocs.io/en/stable/_modules/mmocr/models/textdet/necks/fpn_cat.html) | Chú ý theo scale/vị trí khi hợp nhất feature; đúng bài toán scene-text detection | Train ASF có sẵn; đối chứng cùng train FPNC+DBHead. Không gọi là kiến trúc mới của nhóm. |
| [DySample, ICCV 2023](https://github.com/tiny-smart/dysample) | Học offset lấy mẫu khi upsample, không cần CUDA extension riêng; bằng chứng gốc trên các tác vụ dense prediction, không phải kết quả Monkey/LR2 | Train tại cả top-down và output alignment của FPNC; có bilinear control. Pin source MIT, LP/group4/no scope; đổi offset init sang zero để khớp bilinear ban đầu. |
| [I3CL, IJCV 2022](https://github.com/ViTAE-Transformer/I3CL) | Ngữ cảnh trong/giữa text instances; giải quyết đứt đoạn và nền phức tạp | Hướng riêng nếu review xác nhận split/background; không ghép nguyên detector khác vào job này. Cần đối chứng và tích hợp nhãn/loss riêng. |
| [TextDS, ECCV 2026](https://arxiv.org/abs/2606.28077), [paper đầy đủ](https://arxiv.org/html/2606.28077v2) | Dual SAM2/DINOv3, step-wise LoRA và common-subspace fusion cho distribution shift | Liên quan trực tiếp câu hỏi tổng quát, nhưng không phải module drop-in của Monkey. Thay hai encoder sẽ thành flow/benchmark mới và tăng tải, nên chưa train trong đợt FPNC. |
| [Fine-tuning Gap / LTB, IJCAI 2025](https://ijcai.org/proceedings/2025/83) | Phân tích tối ưu riêng từng dataset và khả năng chuyển miền; đề xuất joint-dataset learning | Cơ sở để không đồng nhất điểm TextVQA với generalization. Chưa thêm dataset/condition vào TRAIN vì sẽ thay câu hỏi và confound lợi ích dữ liệu. |

CRAFT đã sàng lọc: standalone LR2 Hmean 47,00%, không có TP ở bin native LR2
<4 px, nhưng còn phần bổ trợ với B1. Teacher Clean TRAIN có 32 thẻ/200 nguồn
chưa được review ký tự. Không tự coi pseudo-label là đáng tin và chưa train
character/affinity head. Đây là nhánh có điều kiện, không phải đã bác bỏ ý tưởng.

## Một job: probe → pilot trong run → full → đánh giá

Chạy `submit_fusion_study.py`. Một allocation, một job ID; các process con chia
phase vẫn dùng chung allocation, không submit job pilot khác.

| Arm | Fusion / upsample | Các phần được train |
|---|---|---|
| nearest | FPNC gốc | FPNC + DBHead |
| asf | ASF chuẩn của MMOCR, nearest | FPNC + ASF + DBHead |
| bilinear | Bilinear tại top-down và output alignment | FPNC + DBHead |
| dysample | DySample tại cùng các vị trí của bilinear | FPNC + DySample + DBHead |

Tất cả dùng B1 shared weights, backbone đóng băng cả buffers; FPNC được học ở
mọi arm. Vì vậy không dùng A0 head-only của run cũ làm control duy nhất.
Same 15.222 TRAIN sources, 50/50 Clean/native LR2, augmentations, seed 42,
FP32, AdamW LR 2e-5, WD 1e-4, clip 5, microbatch2/accumulation2.
Warmup100 updates rồi Poly.9 tới LR×.01 ở update10k. Không thêm KD/CRAFT loss.

1. Kiểm hash checkpoint/data/FPNC thực cài; storage flush/fsync/read-back.
2. Test source DySample đã pin, shapes/gradients/configs và decoded pixels của
   LR4/LR8/Blur/JPEG đối chiếu scripts đã khóa trên hai nguồn TRAIN.
3. GPU probe cho bốn arm, mỗi arm hai scratch optimizer updates; scratch bỏ đi.
   Nearest phải tái lập B1 ban đầu; zero-offset DySample khớp bilinear trong sai
   số FP32 tối đa 1e-3 ở P-map; ASF chuẩn không được giả định identity-init.
4. Mỗi arm bắt đầu run 10k có lịch LR đầy đủ. **1k updates đầu chính là pilot**.
   Tại mốc này kiểm finite loss/optimizer count, backbone, module đã cập nhật,
   checkpoint đọc lại được và ghi receipt PASS. FAIL kỹ thuật dừng toàn job;
   PASS tiếp tục ngay cùng Runner/optimizer/dataloader/scheduler đến tổng10k.
   Không reset hoặc nối một scheduler pilot đã hết LR. Không dùng điểm Hmean
   pilot làm cổng loại model; điểm âm ở pilot không tự ngăn full run.
5. Lưu checkpoint pilot1k và 2.5k/5k/7.5k/10k; chỉ đánh giá final10k. Không search
   checkpoint. Hash toàn bộ chuỗi nguồn/augmentation phải giống giữa các arm.
6. Chạy B1/B-Clean cố định làm reference; nearest/asf/bilinear/dysample có ngân
   sách bằng nhau. So ASF−nearest và DySample−bilinear, đồng thời so với B1.

Một MIG H200 35GB, 8 CPU, 40GB RAM, wall limit12h. Cần bốn run10k, baseline,
stress inference và bootstrap nên giới hạn này cao hơn job latent cũ; không phải
cam kết thời gian hoàn tất hay ngân sách chỉ dành cho arm tốt nhất.

## Ngưỡng và generalization

Chọn ngưỡng trên validation rồi khóa là hợp lệ; đánh giá lại chính validation
không phải ước lượng ngoài mẫu. Không gọi calibration là cải tiến kiến trúc.
Trong đợt này **không chọn ngưỡng mới**:

- Primary chung: mask.40/score.60, geometry gốc unclip1.5/min-width5/epsilon.01.
- Secondary lịch sử .30/.40. Báo thêm score.30/.40/.50/.60/.70 ở mask.40, chỉ
  sensitivity; không chọn điểm đẹp nhất trên mỗi corruption hoặc dataset.
- Clean/LR2: đủ 2.262 nguồn development. Stress: 500 nguồn lấy theo hash nguồn
  từ remaining-validation trước khi đọc predictions, cùng cohort cho mọi model.
  Đây không phải holdout mới. Cùng500 nguồn có Clean/LR2 anchor để so mức giảm.
- LR4/LR8 bicubic, Blur1/2 sigma1.88/3.75, JPEG30/10 trực tiếp từ Clean; không
  dùng trong TRAIN, không tune ngưỡng riêng và không tự đặt SSIM tau/G3.
- P/R/Hmean, TP/FP/FN, recall bốn bin GT theo native LR2 reference, FP, paired
  source CI2k và latency model+decode ngắn trên tám input cố định. CI theo nguồn
  không đo training-seed variance và không sửa thiên lệch do development reuse.

Nếu chỉ tăng tại một ngưỡng hoặc chỉ trên Clean/LR2 TextVQA, kết luận giới hạn ở
setting đó. Không chọn lại ngưỡng trên LR4/blur/JPEG để che suy giảm. Trước khi
tuyên bố chuyển miền cần khóa weights/preprocessing/ngưỡng rồi benchmark ngoài
dataset; corruption transfer trên cùng ảnh chỉ đo một phần robustness.

**Chưa dùng TextVQA TEST hoặc ICDAR2015 TEST** để chọn phương pháp. Benchmark
cuối phải đóng băng lựa chọn trước; dùng cùng ngưỡng trên ICDAR2015, không tune
test. B0 vốn được train ICDAR2015 và upstream có TextOCR pretraining, nên phải
công bố provenance thay vì gọi tất cả ảnh là chưa từng thấy. S1/S2 cũng từng
được xem ở pilot. Không âm thầm đổi split hoặc lịch sử sử dụng dữ liệu.

```bash
python submit_fusion_study.py --plan-only
python submit_fusion_study.py
```

Output: `.runtime/fusion_study/v1/results/fusion_report.md`, JSON đầy đủ,
`stress_sources.json`, `preflight.json`, và mỗi arm có `pilot_gate.json`.
Review định tính các thẻ lỗi trước đây vẫn PENDING; job này không tự tạo nhãn
tay hoặc tuyên bố đã xác nhận nguyên nhân FP/geometry.
