# MONKEY_V2 · Low Resolution pilot

Pilot Phase 1 trên TextVQA FULL_R1, theo FROZEN v2.0. B0 là checkpoint gốc
của MonkeyOCRv2-AS + DBNet, suy luận FP32 như job E0 10326. Bộ sinh dùng
Pillow 11.3.0 và bicubic độc lập từ ảnh clean cho LR2/LR4/LR8. Ở B_up,
ảnh LR4 được bicubic upsample về canvas gốc trước khi đưa qua Monkey.

## Chạy trên UIT-HPC

Chạy từ tài khoản có quyền đọc dữ liệu và môi trường đã dùng cho job 10326.
Launcher tìm stack, checkpoint, fold validation, evaluator policy và E0
validation trong layout HPC cũ. Có thể chỉ đường dẫn qua
`--legacy-root`, `--clean-root` và `--runtime-root`.

```bash
git clone https://github.com/ncat218/MONKEY_V2.git
cd MONKEY_V2
python run_all.py --plan-only
python run_all.py
```

Lệnh cuối submit một job vào Slurm `normal` với `gpu:2g.35gb:1`,
`qos=mig2-35`, RAM 40 GB và 8 giờ. Job ID và thư mục logs xuất trên
terminal. Runtime mặc định ở
`/datastore/$USER/_workspace_cs431_monkey/lowres/MONKEY_V2__FULL_R1_LowRes_pilot_FROZEN_v2_0/`.
Lần chạy đầu cài riêng Pillow 11.3.0 cho bước sinh ảnh. Inference dùng
stack E0 đã có. Dung lượng đầu ra chủ yếu là 2.500 P-map float16 nén.

Khi job gián đoạn, chạy lại cùng lệnh để dùng các sample hoàn chỉnh.
Code, task, checkpoint và runtime inference được fingerprint. Nếu thay
đổi các đầu vào đã khóa, dùng một runtime root mới. Chờ Slurm
`COMPLETED` và `results/pilot_summary.json` trước khi đọc kết quả.

## Kết quả để review

- `frozen_v2_0/split_lock.json` khóa 500 nguồn validation theo seed
  20261107 và 2.762 nguồn còn lại cho calibration. Pilot dùng 500 nguồn.
- `frozen_v2_0/preflight.json` ghi hash, số GT, các bin native height và
  round-trip tọa độ. `results/clean_identity.json` chỉ xuất sau khi
  matching theo từng ảnh clean khớp E0. Hmean clean tham chiếu là 70,76%.
- `results/pmap_equivalence.json` xác minh kết quả decode từ P-map trùng
  native predict. Mỗi sample có polygon/score sau và trước score cutoff,
  kích thước ảnh, resize factor, P-map float16 và bitmap vùng hợp lệ.
  Bitmap vượt `mask_thr=0.30` lấy từ FP32 trước khi lưu. Để giải bitmap,
  dùng `np.unpackbits(array, count=p.size).reshape(p.shape)`.
- Polygon do DBPostprocessor trả về đã ở tọa độ ảnh đưa vào loader. Với
  Native LR, nhân thêm `clean_width / lr_width` và
  `clean_height / lr_height`. Để đọc P-map trong hệ clean, giới hạn vùng
  theo `img_shape`, chia theo `scale_factor` rồi nhân theo `lr_to_clean`
  trên từng trục. Bitmap hợp lệ loại phần padding, vùng ngoài ROI và
  ignored GT. Metric detection dùng postprocessor và evaluator đã khóa.
- `results/pilot_report.md` tóm tắt điểm và cổng. `results/pilot_summary.json`
  có P/R/Hmean, TP/FP/FN, paired bootstrap CI,
  clean-TP survival, native-height bins, taxonomy lỗi, độ nhạy theo 4/6/8
  px và response cutoff 0,30/0,50, recovery ratio B_up ở LR4 và G1–G3.
  Đọc thêm ảnh và P-map của các GT mất để đánh giá G4 và chọn SAT hoặc PRC.

Repo chứa mã. Ảnh, checkpoint, manifest FULL_R1, raw predictions và
kết quả ở runtime riêng. Evaluator lấy nguyên bytes từ CS431 tại commit
`1844bcf`, được kiểm SHA-256 trước khi chạy.
