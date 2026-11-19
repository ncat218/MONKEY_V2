# Baseline và artifacts cho Mix A · cập nhật 19/11/2026

Gói này chia sẻ kết quả PR/quad audit trên **2.262 nguồn remaining-validation**, paired
Clean/native LR2. Sweep job **54308** đã hoàn tất ba mô hình trước khi lỗi ở bước
review. Job phục hồi **63985 COMPLETED, exit 0**, elapsed **00:04:07**, xác minh lại
counts từng ảnh và hoàn thiện artifacts. [Receipt hoàn tất](results/complete.json) ·
[trạng thái Slurm](provenance/job_63985_sacct.txt) · [báo cáo gốc](results/decision_audit_report.md).

Đây là số exploratory trên development đã dùng cho chọn checkpoint, calibration
và phân tích trước đây. S1/S2 từng được xem ở pilot. CI là paired bootstrap theo
source sau selection, không phải CI giữa seed. Human review **PENDING**: đủ 160
thẻ ảnh/P-map nhưng chưa có nhãn tay. Gói này không phải kết quả Mix A/Mix B.

## Baseline hiện tại

Weights không đổi khi calibration; các cột mới dưới đây dùng cùng quy tắc chọn
trong grid 72 settings. Các số cũ được giữ riêng để đối chiếu.

| Model | Clean H cũ `.30/.40` | Clean H mở rộng | LR2 H cũ `.30/.40` | LR2 H mở rộng | LR2 FP mở rộng | LR2 recall <4 px mở rộng |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 73,30% | 74,09% | 61,95% | 62,38% | 4.069 | 1,01% |
| B-Clean | 78,79% | 80,86% | 74,36% | 75,51% | 6.578 | 15,96% |
| B1 | 77,55% | 79,43% | 75,29% | 76,90% | 11.997 | 45,79% |

| Model | mask_thr | min_text_score | unclip_ratio | min_text_width | epsilon_ratio |
|---|---:|---:|---:|---:|---:|
| B0 | 0.30 | 0.45 | 1.5 | 5 | 0.005 |
| B-Clean | 0.40 | 0.60 | 1.5 | 5 | 0.010 |
| B1 | 0.40 | 0.60 | 1.5 | 5 | 0.010 |

Tất cả là **FP32, quad**, `max_candidates=3000`, cùng evaluator và preprocessing.
Tham số/mô tả có thể đọc bằng máy: [baseline_settings.json](baseline_settings.json).
Số P/R/Hmean, TP/FP/FN và CI chi tiết: [summary gốc](results/decision_audit_summary.json).

Grid gồm **9 cấu hình geometry/mask × 8 ngưỡng score**, không phải tích Descartes
của mọi giá trị geometry. Xem [grid gốc](../../../../rev4/decision_plan.py),
[matcher](../../../../rev4/decision_core.py), [sweep worker](../../../../rev4/decision_audit.py).

**Đính chính tie-break:** `choose_setting` thực tế ưu tiên base `.30/.30` khi hòa,
sau đó theo thứ tự grid. Câu ưu tiên `.30/.40` trong báo cáo gốc là lỗi mô tả.
Đã kiểm tra bằng exact fractions: nghiệm tối ưu của B0/B-Clean/B1 trên toàn cohort
lần lượt là index 44/70/70, mỗi model chỉ có một nghiệm tối ưu, nên bảng hiện tại
không bị ảnh hưởng. Giữ nguyên file báo cáo gốc để bảo toàn hash.

## Checkpoint và config để bắt đầu

- [B-Clean state_dict](../../weights/bclean-inference.pth), newly published;
  [metadata xuất checkpoint](provenance/bclean_checkpoint_export.json). Toàn bộ 377
  tensor được kiểm tra bằng nhau với state_dict checkpoint gốc job 48833.
- [B1 state_dict](../../weights/b1-inference.pth), bản đã chia sẻ trước đây.
- [B0 upstream và bảng SHA-256/cách load](../../checkpoints.md).
- [Config inference](../../inference_config.py),
  [policy evaluator đúng bytes HPC](provenance/textvqa_evaluation_protocol.json),
  [evaluator](../../../../lowres/evaluate_shared.py),
  [DBPostprocessor thực cài](provenance/installed_db_postprocessor.py).

B-Clean và B1 đều khởi tạo B0, full fine-tune 10.000 optimizer updates, seed 42,
cùng 15.222 TRAIN source IDs; B-Clean dùng 100% Clean, B1 dùng 50/50 Clean/native
LR2. [B-Clean report](../../results/bclean_report.md) ·
[training/selection provenance](provenance/bclean_summary.json) ·
[EDA ba model](../../results/paired_adaptation_eda_report.md).

File tải là plain state_dict, không chứa optimizer. Hash tệp tải khác hash
checkpoint runner gốc; không dùng hai loại hash thay cho nhau. Nạp bằng
`torch.load(path, map_location='cpu', weights_only=True)`, rồi
`model.load_state_dict(state, strict=True)` vào kiến trúc đã dựng từ config.
Sau đó dùng `model.float().eval()` và các thông số trong bảng.

## Counts: đủ để reselect và đối chiếu endpoint Mix A

- [B-Clean_source_counts.npz](counts/B-Clean_source_counts.npz)
- [B1_source_counts.npz](counts/B1_source_counts.npz)
- [B0_source_counts.npz](counts/B0_source_counts.npz)

Mỗi NPZ có:

| Key | Ý nghĩa |
|---|---|
| `source_ids` | 2.262 source IDs, thứ tự tăng dần |
| `settings` | 72 JSON strings; dùng nguyên thứ tự này, có thể lấy key bằng `setting_key` |
| `counts` | int64, shape `(2262, 2, 72, 5)` |
| condition axis | index 0 = Clean, index 1 = LR2 |
| count axis | `tp, gt, pred, tiny_gt, tiny_tp` |

`FP=pred−tp`, `FN=gt−tp`, `Hmean=2tp/(gt+pred)`. Tiny là GT AABB height <4 px
quy về native LR2; Clean/LR2 dùng cùng nhóm GT. NPZ không có đủ ba size bins còn
lại, không suy ra chúng từ tiny count. Không đảo thứ tự theo CSV task khi ghép
NPZ: join bằng source ID và condition.

Đối chiếu EDA độc lập ở `.30/.40` theo từng ảnh:
[eda_common_counts.csv.gz](provenance/eda_common_counts.csv.gz), gồm
`model, source_id, condition, sample_id, tp, gt, pred, covered, mask_thr, min_text_score`.
Nguồn: các calibration-count caches đã khóa; [EDA completion/model hashes](provenance/eda_complete.json).

Grid mới của Mix A và R1/A-B selection trong đề xuất phối hợp chưa được chạy hay
chốt bởi gói này. Nếu chọn lại setting trên A, phải chọn lại cả hai endpoint theo
cùng rule; các optimum toàn cohort trong bảng chỉ là tham chiếu hiện tại. Chia A/B
từ development đã dùng không tạo holdout mới.

## Cohort, GT và ảnh

- [source_cohort.csv](data/source_cohort.csv): ID, clean relative path, dimensions,
  clean SHA-256, số valid/ignored GT cho 2.262 nguồn.
- [tasks_metadata.csv](data/tasks_metadata.csv): đủ 4.524 Clean/LR2 task, kích thước,
  SHA-256, relative path và loại root để ánh xạ sang nơi lưu ảnh của người chạy.
- [validation_remaining.jsonl.gz](data/validation_remaining.jsonl.gz): đúng bytes
  manifest GT/ROI đã khóa, chỉ remaining-validation. Giải nén rồi kiểm SHA-256
  `052857fba8b6cd7c78e48a3dcb53475730d03f2ca9d885b66a4a562eea38872c`.

Ảnh đầy đủ lấy từ [TextVQA](https://textvqa.org/download/), GT gốc từ
[TextOCR](https://textvqa.org/textocr/dataset/). Dùng `clean_relpath` và root của
người chạy. Native LR2 dùng Pillow 11.3.0 bicubic downsample trực tiếp từ Clean,
mỗi trục `max(1, size//2)`, không upscale lại trước detector.
[Generator](../../../../lowres/generate_lr_only.py) ·
[quy tắc task/ảnh đã dùng](../../../../rev4/training_data.py).

GT/matching dùng `gt_clean`/`roi_clean` trong manifest; đưa prediction native LR2
về hệ Clean bằng tỷ lệ kích thước thực từng trục. Detector có resize/pad nội bộ,
nên không suy grid feature chỉ từ kích thước file LR2. Split này dùng development.

## Review và provenance

- [160 thẻ paired RGB/P-map, ZIP](review/review_cards.zip), đi cùng
  [selected_review.json](review/selected_review.json),
  [human_review.csv](review/human_review.csv).
- [FP proxies toàn tập](review/fp_taxonomy_proxies.csv.gz),
  [GT transitions/diagnosis toàn tập](review/gt_diagnosis.csv.gz),
  [diagnosis_summary](review/diagnosis_summary.json).
- [PR curves](results/pr_curves.csv), [full frontier](results/threshold_frontier.csv),
  [quad filter audit](results/quad_filter_audit.csv).
- [Sweep identity](provenance/sweep_run_identity.json),
  [recovery identity](provenance/recovery_run_identity.json),
  [job 54308 stdout](provenance/job_54308.stdout.log),
  [job 63985 stdout](provenance/job_63985.stdout.log).

Review/proxy/FP correspondence hiện dùng **common `.30/.40`**, không phải ngưỡng
mở rộng `.40/.60`. Nhãn tay để trống, không xem proxy hoặc background/no-GT-overlap
là kết luận đã xác nhận. Nếu quyết định Mix B theo residual FP tại setting mới,
cần xác định lại quần thể tương ứng và review phù hợp.

## Tải và kiểm tra

Weights và ZIP thẻ review dùng Git LFS. Khi đã có checkout:

```bash
git pull --ff-only
git lfs pull --include="docs/handoff/weights/bclean-inference.pth,docs/handoff/weights/b1-inference.pth,docs/handoff/artifacts/pr_quad_audit_v1/review/review_cards.zip"
python docs/handoff/artifacts/pr_quad_audit_v1/verify_bundle.py --counts --weights
```

Verifier cần NumPy cho `--counts`; không nạp model, không inference, không submit
job. Nó kiểm tra hash file, report receipt, đủ source/counts, endpoint `.30/.40`
và totals tại setting đã chọn. [Manifest SHA-256/size](bundle_manifest.json).
Hash code và checkpoints nguyên gốc nằm trong provenance; các absolute paths ở
đó là lịch sử HPC, còn các link và file trong gói này tải trực tiếp từ repo.
