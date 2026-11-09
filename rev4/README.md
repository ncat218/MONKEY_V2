# Rev.4 · S2-A trên UIT-HPC

Nếu checkout đã có receipt, split và bằng chứng dry-run của TASK 1, cập nhật
code và submit lại từ thư mục MONKEY_V2.

```bash
cd /datastore/dungmt/textvqa/Tuan/MONKEY_V2
git pull --ff-only
python submit_s2a.py --plan-only
python submit_s2a.py
```

`--plan-only` đối chiếu SHA-256 của receipt GO, protocol, S2 source list,
manifest, generator và model. Nó in cấu hình job để kiểm tra. Lệnh thứ hai
submit một job Slurm H200 MIG. Launcher dùng các artifact đã kiểm chứng
trong `.runtime/rev4_task1/` và receipt đã khóa trong
`.runtime/rev4_task2a/handoff/`. Các generator gốc, protocol nhóm và
artifact dữ liệu được giữ trong runtime.

Worker kiểm lại identity trên 500 ảnh S1 rồi chạy B0 FP32 với clean, LR2,
LR4/LR8 stress và B_up LR2 trên S2. Báo cáo xét G1–G3 cho LR2 và dùng 2.000
paired source bootstraps. Kết quả ở `.runtime/rev4_task2a/run/`. Xem
`logs/job_<ID>.stdout.log` và `results/s2a_report.md`.

Với PNG có ICC profile vượt giới hạn Pillow, bước chuẩn bị dùng một bản
đọc tạm bỏ riêng chunk ICC. Worker giữ bytes PNG gốc và kiểm tra OpenCV
đọc PNG gốc ra cùng pixel RGB trước khi dùng cho inference. Ca recheck
được ghi trong `prepared/prepared.json`.

Mã audit và dry-run TASK 1 ở `rev4/audit_split.py` và
`rev4/run_task1_checks.sh`. Mã hiệu chuẩn TRAIN và test giả lập nằm trong
`rev4/ssim_calibration.py` và `tests/`. Nhánh Blur/JPEG và τ phụ thuộc
kết quả G1–G3 của LR2.

## TASK 2B mô tả

Job 11907 đã chọn LR2 theo thứ tự Rev.4. Trên cùng checkout, dùng:

```bash
git pull --ff-only
python submit_s2b.py --plan-only
python submit_s2b.py
```

Launcher đối chiếu hash báo cáo và identity S2-A, cả ba cổng LR2 và
hash S2. Nó dùng lại clean matching từ job 11907, rồi chạy B0 FP32 với
Blur 1.88/3.75 và JPEG 30/10 trên cùng 500 source. Báo cáo mô tả G1/G2
và ghi `G3 = NOT ASSESSED`, `τ = N/A`.

Kết quả nằm trong `.runtime/rev4_task2b/run/results/`, gồm
`s2b_report.md`, `s2b_summary.json`, `review_examples.json` và
P-map float16 cùng polygon/score trước cutoff theo từng mẫu.

## B1 Clean + LR2

Chạy từ checkout MONKEY_V2 có đủ bằng chứng S2-A và S2-B:

```bash
git pull --ff-only
python submit_b1.py --plan-only &&
python submit_b1.py
```

Nếu manifest `manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl`
nằm ở đường dẫn khác trên HPC, chỉ định bản gốc đã sao lưu:

```bash
train=/path/to/manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl
python submit_b1.py --plan-only --train-manifest "$train" &&
python submit_b1.py --train-manifest "$train"
```

Nếu còn bản sao metadata FULL_R1, dùng `--full-r1-root /path/to/full_textvqa_degraded`
ở cả hai lệnh thay cho `--train-manifest`. Thư mục cần `main/image_manifest.csv`,
`config/frozen_degradation_config.json` và thư mục `metadata/` nguyên bản,
gồm `split_policy.json`, `locked_eval_manifest.csv` và CSV nguồn khớp SHA-256
trong frozen config. Cách này dùng metadata và ảnh clean, nên có thể dùng bản
sao thiếu `main/images/`. `--plan-only` kiểm tra sự hiện diện của metadata.
Trong job, worker dùng mã lịch sử `816c6d6`, đối chiếu fold validation đã khóa
và chỉ chấp nhận manifest 15.222 nguồn TRAIN có SHA-256
`ca1b91636cbe025823041ce336ffb59affa826a073f666dd7ba386bcab6c983d`.

Job huấn luyện B1 từ B0 FP32, 10.000 optimizer updates, seed 42, AdamW
LR 2e-5, batch hiệu dụng 4 và clean replay đúng 50% theo microbatch.
LR2 tạo on-the-fly bằng RGB bicubic Pillow 11.3.0 với geometry theo tỉ
lệ kích thước thực. Giữ pipeline augment 640×640 của config đã dùng.
Checkpoint định kỳ kèm optimizer để có thể resume bằng cùng lệnh.

Hai checkpoint ở 5.000 và 10.000 updates được chấm với ngưỡng native
trên 2.262 nguồn remaining validation. Chọn mean Hmean clean/LR2 cao nhất,
hòa thì lấy checkpoint sớm. Báo cáo `.runtime/rev4_b1/run/b1_report.md`
và `b1_summary.json`. Calibration theo lưới chung là bước tiếp theo.

Để xem mẫu G4 trên HPC, chạy lệnh CPU dưới đây từ MONKEY_V2. Script dùng
lại ảnh và P-map S2-A đã có, rồi tạo bảng trong
`.runtime/rev4_b1/g4_review/lr2_pmap_review.png`.

```bash
stack=/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/environment_and_models__phase1_verified_stack
"$stack/python311_venv/bin/python" -m rev4.g4_review
```
