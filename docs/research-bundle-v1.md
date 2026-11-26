# Research bundle v1 · một job: zoom theo vùng yếu, seed lặp, PRC2

Ngày 26/11/2026. Extension exploratory trên 2.262 nguồn remaining-validation (development, đã dùng nhiều lần) và
500 nguồn stress đã chọn bằng hash của fusion 111683. Không đọc TEST, không chọn ngưỡng mới; protocol FROZEN
v2.0/v2.1 và các báo cáo cũ giữ nguyên.

## So với cái gì
Mọi biến thể so với **B1 đã fine-tune (seed 42)**, hoặc B-Clean với nhánh B-Clean, ở **cùng điểm cố định
`.40/.60`** (phụ `.30/.40`). Ngoài Hmean, báo cáo có recall chữ < 4 px và < 8 px (chiều cao quy về LR2), FP,
FP của box nhỏ, CI bootstrap ghép cặp theo nguồn, và *recall chữ nhỏ mà chính B1 đạt khi dời score cutoff tới
cùng FP* (matched-FP, nội suy trên counts fusion, không ngoại suy). Có thêm cột Hmean khi bỏ nguồn
`6e340fcbb1e0a877` (trang báo không gán nhãn, Trí phát hiện).

## Vì sao làm
- 15.194 FN của LR2 cũng là FN ở Clean; chỉ 6.667 FN là do LR2 (61% là chữ < 8 px). Recall theo cỡ chữ có
  dạng chữ U ngược; chữ < 4 px chỉ 61% (Clean) và 46% (LR2).
- Detector resize mọi ảnh về cùng canvas (cạnh ngắn 1024), nên phóng to cả ảnh không làm chữ lớn hơn trong mạng.
  Muốn chữ lớn hơn phải cắt vùng rồi phóng to.
- Fusion 111683: ASF tăng tiny recall so B1, nhưng Nearest (cùng chế độ train tiếp) tăng gần bằng; phần lớn lợi
  ích đến từ train tiếp FPNC + DBHead. Cần seed thứ hai để biết chênh lệch cỡ này có vượt nhiễu train không.
- PRC (M) gần như không chạy: job 12518 chỉ active 608/20.000 microbatch.

## Không nằm trong bundle
EDSR/bicubic toàn ảnh trên B1 đã fine-tune là job **163012** (`submit_edsr_recheck.py`): Clean, LR2, stress, cả
hai điểm. Không chạy lại ở đây. Biến thể "chỉ phục hồi ảnh mà detector phải phóng ≥ 1,5×" chỉ phụ thuộc kích
thước ảnh, nên tính hậu kiểm từ record của 163012. Bundle chỉ thêm EDSR trên crop zoom (`guided_edsr`).

## Cơ chế zoom (mượn từ Vision-RL² / SD-RPN / ViCrop, AutoFocus)
Các hệ VQA chọn vùng liên quan câu hỏi bằng heatmap nội tại rồi cắt và mã hóa lại ở độ phân giải cao.
Bài detection không có câu hỏi, nên "câu hỏi" được thay bằng: *vùng nào detector đang phản hồi chữ yếu hoặc nhỏ?*
1. Pass toàn ảnh như B1/B-Clean.
2. Từ P-map toàn ảnh lấy component yếu hoặc nhỏ (P > 0,20; chiều cao quy về LR2 < 8 px, hoặc < 16 px nhưng
   đỉnh < 0,60). Chọn tối đa 4 cửa sổ nửa kích thước ảnh, tham lam theo số component phủ được.
3. Cửa sổ được detector resize về cùng canvas nên chữ lớn gấp đôi. `guided_edsr` phục hồi crop ×2 bằng EDSR trước.
4. Hợp nhất theo cỡ chữ: `replace` để pass zoom phụ trách chữ < 8 px trong cửa sổ; `union` chỉ thêm box nhỏ
   không trùng; `verify` chỉ giữ box nhỏ của pass toàn ảnh khi pass zoom cũng thấy.
5. Đối chứng: tiling 2×2 (cùng trần chi phí; `tile2x2_union` lặp lại ZB1 của Trí), oracle đặt cửa sổ lên chữ
   nhỏ của GT (trần lợi ích, không triển khai được), hậu xử lý `poly` trên cùng P-map.

B1: Clean, LR2 và 6 điều kiện stress (stress chỉ chạy global, tile2x2, tile2x2_union, guided, guided_verify).
B-Clean: Clean và LR2.

## Seed lặp
`B1_s43` và `BClean_s43` dùng đúng recipe B1/B-Clean (B0 init, 10k update), chỉ đổi `cfg.randomness.seed = 43`.
MMEngine truyền seed runner vào sampler, nên seed đổi cả thứ tự rút nguồn lẫn RNG augment. Probe kiểm tra sampler
nhận seed 43, thứ tự rút khác seed 42, và B-Clean s43 rút cùng thứ tự với B1 s43.

## PRC2
Giữ nguyên căn chỉnh Clean → lưới augment LR2 và masked MSE của PRC. Thay cách chọn kernel: dùng trực tiếp
độ phủ của teacher B0 trên kernel GT (C_GT ≥ 0,30, ≥ 4 px), không qua postprocessor + ghép polygon.
Probe phát lại luồng TRAIN để đo căn chỉnh, đường v1 (số dự đoán, số từ ghép được, rơi do ROI/ignore), nền của
teacher và độ phủ v2; chỉ train khi đạt ngưỡng. Hook dừng train nếu 200 microbatch đầu có < 50% microbatch active.
Đây là amendment hậu nghiệm, phải báo cùng kết quả PRC v1.

## Pha trong một allocation
`probe` (unit test, cấu hình 3 arm, zoom trên 4 task thật khớp counts cũ, probe PRC2) → `zoom B1` →
`train B1_s43`, `BClean_s43`, `PRC2` → `evaluate` từng arm (5k/10k theo quy tắc cũ, lưới 15 cặp, `.40/.60`,
stress 500 nguồn) → `zoom B-Clean` → `report`. Pha độc lập lỗi thì ghi `status.json` và chạy tiếp; probe kỹ
thuật phải PASS trước. Ước lượng ~22 h trong giới hạn 40 h (train ~1,6 h/arm theo B1 12187 và B-Clean).

```bash
python submit_research_bundle.py --plan-only
python submit_research_bundle.py
```
Kết quả: `.runtime/research_bundle/v1/results/bundle_report.md` và `bundle_summary.json`.
