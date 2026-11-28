# ICDAR2015 gốc: kết quả các pipeline đã khóa — job 192199

Ngày 28/11/2026. Job **COMPLETED, exit 0:0, 57 phút 18 giây**, H200 `h200node01`, MIG 2g.35gb.
**Kết luận:** tổ hợp ASF+EDSR+Q cải thiện Hmean so B1 đã fine-tune, nhưng chưa vượt B0 hay B-Clean trên ICDAR2015 gốc. Không có mức tăng Recall chữ nhỏ của tổ hợp cuối so B1 trong phép đo này.

## 1. Đã kiểm tra đúng bộ reproduce

- Đúng 500 ảnh gốc và manifest MMOCR; SHA256 ZIP/annotation/audit và cả 500 ảnh đều khớp. Kích thước, EXIF đều PASS.
- Có 2.077 GT hợp lệ, 3.153 GT ignore. Không tạo LR2 cho bảng này.
- B0 setting reproduce tái hiện TP1.800/FP178/FN277, Hmean88,7793%.
- Kiểm tra thêm raw predictions của job11339: **500/500 ảnh khớp TP/FP/FN và danh sách GT được match**. Một số score/quads không giống tuyệt đối về số học; không tuyên bố prediction bit-identical.
- Code, config, dữ liệu, 5 checkpoint detector, EDSR và evaluator giữ nguyên từ lúc submit đến cuối run. 6 kiểm thử CPU PASS trước inference.
- Chỉ inference; không train, chọn checkpoint, fit Q hay tune ngưỡng trên ICDAR test.

## 2. Kết quả chính

| Pipeline | P (%) | R (%) | Hmean (%) | R chữ <16 px (%) | FP | FN | Median ms/ảnh |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 — setting reproduce | 91.00 | 86.66 | 88.78 | 90.51 | 178 | 277 | 181.1 |
| B0 — setting chung `.40/.60` | 88.00 | 86.13 | 87.06 | 90.51 | 244 | 288 | 180.8 |
| B-Clean — fine-tune 100% Clean | 77.52 | 87.48 | 82.20 | 92.70 | 527 | 260 | 178.0 |
| B1 — fine-tune 50/50 Clean/LR2 | 67.10 | 87.10 | 75.80 | 90.51 | 887 | 268 | 177.7 |
| B1 + ASF | 67.42 | 87.19 | 76.04 | 89.78 | 875 | 266 | 181.9 |
| B1 + EDSR ×2 | 72.12 | 85.80 | 78.36 | 88.32 | 689 | 295 | 496.6 |
| B1 + ASF + EDSR ×2 | 72.01 | 85.60 | 78.22 | 89.78 | 691 | 299 | 500.7 |
| B1 + ASF + EDSR ×2 + Q | 71.76 | 85.65 | 78.09 | 89.78 | 700 | 298 | 503.0 |

**Chữ nhỏ:** chiều cao là y-extent polygon GT ở ảnh gốc, giữ nguyên trước mọi phép phóng. Nhóm <8 px có **0 GT**, nên tiny recall là **N/A**, không phải 0%. Nhóm <16 px có **137 GT**; các nhóm 16–<32 và ≥32 có 826 và 1.114 GT. Không đồng nhất nhóm này với tiny <4 px trên native LR2 của TextOCR.

**Setting:** B0 reproduce dùng mask `.30`, epsilon `.01`, score native `.30` rồi packed score cutoff `.60` đã khóa lịch sử. Hàng B0 common và mọi F dùng `.40/.60`, epsilon `.01`; B0 frozen riêng trong bảng đầy đủ dùng `.30/.45`, epsilon `.005`, đã chọn từ TextOCR. Q dùng raw pool ≥`.20`, cutoff quality `.3315908613227198` đã khóa trên TRAIN. Không chọn setting tốt nhất từ bảng ICDAR.

## 3. CI ghép cặp và đóng góp từng thành phần

| Phép so (mới − cũ) | ΔHmean pp [95% CI] | ΔRecall pp [95% CI] | ΔR <16 px pp [95% CI] | ΔFP | ΔFN |
|---|---:|---:|---:|---:|---:|
| ASF so B1 | +0.243 [-0.336; +0.785] | +0.096 [-0.464; +0.643] | -0.730 [-3.522; +1.460] | -12 | -2 |
| ASF so Nearest train tiếp | +0.030 [-0.425; +0.475] | +0.144 [-0.292; +0.585] | -0.730 [-3.598; +1.481] | +3 | -3 |
| EDSR+B1 so B1 | +2.563 [+1.532; +3.652] | -1.300 [-2.100; -0.519] | -2.190 [-4.795; +0.000] | -198 | +27 |
| EDSR+B1 so bicubic+B1 | +2.164 [+1.143; +3.127] | -0.915 [-1.728; -0.097] | -0.730 [-2.143; +0.000] | -160 | +19 |
| ASF+EDSR so EDSR+B1 | -0.142 [-0.688; +0.372] | -0.193 [-0.708; +0.313] | +1.460 [+0.000; +3.650] | +2 | +4 |
| Thêm Q trên ASF+EDSR | -0.128 [-0.488; +0.235] | +0.048 [-0.241; +0.334] | +0.000 [+0.000; +0.000] | +9 | -1 |
| Tổ hợp Q so B1 | +2.293 [+1.257; +3.385] | -1.444 [-2.277; -0.650] | -0.730 [-3.846; +2.400] | -187 | +30 |
| Tổ hợp Q so B-Clean | -4.104 [-5.757; -2.666] | -1.830 [-2.796; -0.834] | -2.920 [-7.640; +1.379] | +173 | +38 |
| Tổ hợp Q so B0 reproduce | -10.684 [-12.679; -8.756] | -1.011 [-2.423; +0.393] | -0.730 [-4.902; +2.598] | +522 | +21 |

2.000 bootstrap theo **ảnh**, seed20261107; khớp các CI đã xuất bởi worker. Không bootstrap từng từ như các mẫu độc lập. CI không đo biến thiên seed train; đây là các checkpoint seed42, benchmark đã có lịch sử xem lúc reproduce.

## 4. Diễn giải

1. **Cải thiện trên B1 có chuyển được một phần:** EDSR+B1 tăng Hmean 75,80→78,36%, FP887→689; Recall giảm 87,10→85,80%, FN268→295. Lợi ích tổng thể đi kèm giảm FP, không chứng minh phục hồi thêm chữ nhỏ ở ảnh ICDAR gốc.
2. **ASF chưa cho thấy đóng góp Hmean riêng rõ:** ASF native76,04% so Nearest76,01%. Thêm ASF khi đã có EDSR cho78,22% so EDSR+B1 78,36%. Xem CI để tránh quy tác dụng train thêm cho ASF.
3. **Q không lặp lại lợi ích TextOCR:** trên ASF+EDSR, thêm Q cho Hmean78,09% so78,22%; FP691→700, FN299→298. Đây là point estimate; CI trong bảng cho biết độ bất định. Không fit hoặc chọn lại cutoff để cứu kết quả test.
4. **B0 vẫn mạnh nhất ở benchmark gốc.** Tổ hợp Q thua B0 reproduce khoảng10,68pp Hmean; vẫn thua B0 common `.40/.60`. Chênh lệch không chỉ do đối chiếu hai điểm vận hành khác nhau. B-Clean82,20% cũng cao hơn các nhánh B1 trong lượt này.
5. **Không ngoại suy từ LR2 TextOCR sang mọi ảnh.** Trên development TextOCR LR2, tổ hợp Q từng tăng Hmean/tiny recall; trên ICDAR gốc, mức tăng so B1 chủ yếu thể hiện ở Precision/FP và chưa giữ được năng lực B0. Hai kết quả không được gộp thành tuyên bố cải thiện phổ quát.
6. **Chi phí:** B1 median177,7ms; tổ hợp Q503,0ms trong audit20 ảnh/3 warmup/2 lượt, GPU dùng chung. Tính SR+detector/postprocessing; không tính ghi PNG, hash, metric hoặc kiểm tra parity lặp. Không phải cam kết latency production.

## 5. Đối chiếu paper và phạm vi kết luận

Paper §4.3 ghi DBNet+MonkeyOCRv2 **88,5%**; README detection chính thức ghi **88,9%**. Giữ riêng hai con số công bố; neo đối chứng của lượt này là **B0 FP32 tái lập88,78%** và thêm hàng B0 common cùng ngưỡng. Đây là tái lập suy luận từ checkpoint phát hành, không phải tái huấn luyện toàn bộ paper.
[Paper chính thức](https://arxiv.org/html/2607.11562v1#S4.SS3), [README detection chính thức](https://github.com/Yuliang-Liu/MonkeyOCRv2/tree/main/detection/mmocr).

B0 từng train trên ICDAR TRAIN; 500 ảnh test đã xem trong reproduce và đánh giá trước đây. Lượt này đo **chuyển giao trở lại benchmark gốc sau thích nghi TextOCR**, không phải zero-shot trên một holdout chưa từng thấy. Chỉ dùng kết quả để mô tả chuyển giao/giới hạn, không chọn lại phương pháp hoặc ngưỡng.

Báo cáo đồ án nên giữ LR2 TextOCR là bài toán chính, trình bày cải tiến so B1 cùng đối chứng B-Clean và ablation. ICDAR2015 là bằng chứng về **giới hạn tổng quát hóa/giữ năng lực benchmark gốc**, không phải kết quả vượt paper. Nguyên nhân FP cụ thể cần review ảnh và matching; chưa khẳng định mọi FP là nền hay lỗi nhãn.

## 6. File và bằng chứng

- [Bảng đủ 12 setting và CI gốc](../artifacts/icdar2015_192199/icdar2015_report.md).
- [Summary JSON đầy đủ](../artifacts/icdar2015_192199/icdar2015_summary.json), [counts500 ảnh/cell](../artifacts/icdar2015_192199/source_counts.npz), [CI thành phần](../artifacts/icdar2015_192199/component_contrasts.json).
- [Audit dữ liệu](../artifacts/icdar2015_192199/data_audit.json), [B0 từng ảnh](../artifacts/icdar2015_192199/b0_per_image_metric_audit.json), [identity cuối](../artifacts/icdar2015_192199/final_identity_audit.json), [Slurm receipt](../artifacts/icdar2015_192199/slurm_job.txt).
- [Index đúng500 ảnh](../data/icdar2015_reproduce/source_ids.csv); index không thay thế GT polygons.
- [Spec khóa trước inference](../../icdar2015-transfer-v1.md), [launcher](../../../submit_icdar_transfer.py).
- Rebuild bảng/CI: `python analysis/icdar2015_transfer_report.py` (chỉ đọc artifacts; không inference/train).

Dữ liệu gốc còn trên HPC tại `/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/ICDAR2015__Monkey_E0_Aprime_FP32_transfer/inputs/` gồm `ch4_test_images.zip`, `dataset_audit.json`, `textdet_test__recovered_byte_identical_to_Phase1.json`. Manifest cuối là MMOCR JSON có `metainfo`/`data_list`, 500 hàng ảnh và `instances` chứa polygon/bbox/ignore. SHA256 annotation `fd1a9d52dbd2b3ed86a16fef27cda35e4c44624d07c4f1c25338ea2ca660f718`.
Raw predictions theo từng ảnh vẫn được giữ tại `.runtime/icdar2015_transfer/v1/records/` trên HPC. Không phân phối lại ảnh trong Git.
