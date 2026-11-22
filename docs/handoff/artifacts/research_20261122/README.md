# Artifacts nghiên cứu · snapshot 22/11/2026

Gói bằng chứng sau PR/quad audit, lấy từ các run đã có. [Handoff](../../../../handoff.md) và [workflow hiện tại](../../../research/current-workflow.md) giải thích kết quả và giới hạn. Không chạy train/inference mới khi xuất gói; không đọc TEST.

| Thư mục | Nội dung |
|---|---|
| `latent_10k/` | Job88935: report/CI/summary/completion, trained receipts, config gốc, latency, validation JSON nén và source counts cho A0/A1/A2 |
| `latent_pilot/` | Job81918 recovery1k: cùng schema, probe và recovery provenance |
| `craft/` | Job74220: report/CI, complementary GT sets, counts/matched ann IDs; `train_teacher_qc.zip` có maps/metadata 200 TRAIN sources và 32 cards chờ review |
| `feature_kd/` | Job48499: report/CI, trained receipts và validation theo ảnh cho continued/KD; không thêm checkpoint pilot trùng lặp |
| `fixed_diagnostics/` | Job73827: P/R/H, per-source NPZ, height bins, transitions CSV nén, proxy taxonomy, 127 paired cards trong `review/paired_cards.zip` |
| `data/` | 3.262 VAL IDs/clean SHA/role; S1/S2 IDs và split receipt; TRAIN CSV ở [data đã công bố](../../data/README.md) |
| `technical_preflight/`, `s1/` | Audit split/EXIF/round-trip lịch sử và báo cáo pilot đầu |
| `weights/` | Metadata xuất B0/A0/A1/A2; per-tensor reference và receipt kiểm B1; weights tải tại [checkpoint index](../../checkpoints.md) |
| `fusion_snapshot/` | Job106090 RUNNING lúc lấy snapshot: identity, preflight, corruption probe, source selection, job ticket và stdout tail. **Không có final score hoặc checkpoint fusion** |

`job_status.psv` và `snapshot.json` là trạng thái có timestamp lúc xuất, không tự cập nhật. Các fail trước recovery được giữ trong lịch sử/provenance: job74219 lỗi config freeze; job77412 dừng, có cache/log rỗng nên chưa xác định nguyên nhân gốc; 81918 mới là recovery hoàn tất. Không coi một job fail là kết quả âm của phương pháp.

Các số đều từ development đã dùng lặp lại, một seed. S1/S2 đã xem ở pilot. CI theo nguồn không đo seed variability hoặc selection uncertainty. Không đổi protocol/evaluator hay ghi đè frozen reports. [PR/quad bundle trước đó](../pr_quad_audit_v1/README.md) vẫn là nguồn grid/calibration đầy đủ.

## Tải và kiểm

```bash
git lfs pull
python docs/handoff/artifacts/research_20261122/verify_bundle.py --counts --weights
```

Mặc định verifier dùng thư viện chuẩn để kiểm toàn bộ file size/SHA. `--counts` cần NumPy, kiểm split IDs và tái tổng hợp P/R/H/TP/FP/FN của latent/CRAFT/fixed diagnosis. `--weights` kiểm hash 7 checkpoint public. Không cần GPU, ảnh dataset hay Slurm. Chạy từ bất kỳ working directory nào với đường dẫn script đúng.

`bundle_manifest.json` là inventory cho gói công khai; `export_inventory.json` giữ hash export nguyên gốc từ HPC. ZIP qua Git LFS; CSV/JSON.gz/NPZ là dữ liệu tải trực tiếp. Verifier sẽ FAIL nếu nhận LFS pointer hoặc file thiếu. Không tự extract ZIP hay chạy code từ ZIP.

## Schema và cách đọc

- `latent_*/<arm>/source_counts.npz`: `source_ids` (2.262), `conditions` (clean,lr2), `threshold_pairs` ([.30,.40],[.40,.60]), `counts` shape `(2262,2,2,4)` theo **TP,GT,pred,covered**; `bins` `(2262,2,2,4,2)` theo GT/TP. Các bin dùng chiều cao native LR2 tham chiếu cùng nguồn để so Clean/LR2; xem raw training/evaluation code trước khi ghép với báo cáo cũ dùng condition-specific height.
- `fixed_diagnostics/results/paired_source_counts.npz`: IDs và mỗi model/condition array `(2262,4)` cũng TP/GT/pred/covered. Không coi covered là TP; đó là chỉ số coverage riêng.
- `craft/per_source_counts.csv.gz`: số đếm CRAFT theo source/condition và matched GT IDs. Số đếm B1 nằm ở fixed-diagnostics NPZ; complementary sets đã tổng hợp trong summary, không phải fused predictions.
- `validation.json.gz`: JSON task-level nguyên từ run, không phải danh sách ảnh train. JSON còn ghi path HPC để audit; đó không phải đường dẫn tải cho người nhận.
- `fixed_diagnostics/review/human_review.csv`: nhãn reviewer còn trống, **PENDING**. `selected_review.json` ghi sampling strata/probabilities; 127 cards không phải prevalence estimate chính xác.
- CRAFT word-level matching và soft maps không xác minh character/affinity label đúng. Chưa train character auxiliary head.

Ảnh review có chọn lọc phục vụ kiểm tra nghiên cứu, không thay cho bản phát hành TextVQA/TextOCR. Không phát hành toàn bộ ảnh, P-map cache, optimizer state, TEST manifest hay các artifact của dự án Q-LRG khác. Script xuất [export_research_snapshot.py](../../../../tools/export_research_snapshot.py) liệt kê nguồn có chọn lọc và receipts; không phải launcher huấn luyện.
