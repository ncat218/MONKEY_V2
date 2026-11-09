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
