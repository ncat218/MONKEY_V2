# CS431 — MonkeyOCRv2-AS + DBNet: Pilot v2.1 (FINAL / Execution Protocol)

**Ngày:** 08/11/2026 · **Bản cập nhật:** Rev.4 (pool TRAIN 20K, pilot S2 có điều kiện; trước mọi kết quả S2)  
**Phạm vi:** Scene Text Detection under Single-Factor Degradation: LR2, Gaussian Blur, JPEG recompression.  
**Trạng thái:** **FINAL Rev.4 về thiết kế**, chưa xác nhận hoàn tất preflight. **Được chạy pha S2-LR2 sau khi preflight kỹ thuật PASS**, không cần chờ τ. **τ chỉ cần thiết nếu LR2 không qua G1–G3 và Blur/JPEG cần xét G3**; lúc đó phải hiệu chuẩn mù trên TRAIN và khóa τ **trước khi mở kết quả Blur/JPEG trên S2**. Không tự ghi FROZEN/GO khi thiếu bằng chứng. Thay đổi này là quyết định được đăng ký **trước khi chạy/đọc S2**; nếu S2 đã bị mở thì không được hồi tố áp dụng.  
**Nguồn quy tắc duy nhất cho v2.1:** **file này** (`CS431_Pilot_Protocol_v2.1_FINAL.md`, Rev.4). Các bản v2.1 DRAFT / Claude Docs trước đó đã được thay thế; không dùng song song để chỉnh hoặc thực thi. **Kế thừa:** `CS431_LowRes_Pilot_Protocol_FROZEN_v2.0.md` đối với các điều khoản không bị thay thế. Các quy tắc không thay đổi vẫn áp dụng. Mọi điều chỉnh phát sinh sau S2 phải công bố như amendment hậu nghiệm, không thay số lịch sử.

## 0. Quyết định đã khóa

| Nội dung | Giá trị/quy tắc |
|---|---|
| Model | MonkeyOCRv2-AS (ViTAEv2-S) + FPNC + DBNet; không chuyển backbone, không dùng nhánh Paddle |
| Dataset | TextVQA FULL_R1 + TextOCR word polygons; official TextVQA train có detection train/val/test nội bộ theo source đã audit |
| Pilot S1 | 500 source v2.0; **không sử dụng lại để chọn hướng v2.1** |
| Pilot S2 | 500 source mới trong **2.762 source validation còn lại**, `seed = 20261108` |
| Validation sau S2 | 2.262 source; chỉ dùng calibration/checkpoint/hyperparameters |
| Ứng viên chính | **A = LR2**, **B = Gaussian blur mức 1**, **C = JPEG quality 30** |
| Mức nặng để đánh giá | LR4, LR8; blur mức 2; JPEG quality 10 |
| Thứ tự chọn train nếu nhiều ứng viên đạt | **A (LR2) → B (Blur) → C (JPEG)**, có quyết định theo giai đoạn ở §6; không xếp hạng bằng proxy G3 khác đơn vị |
| Train | Chỉ **Clean + một condition chính được chọn**; clean replay **50%**; severity nặng không đưa vào train |
| Mục tiêu chính | ΔHmean = Hmean(M-cal) − Hmean(B1-cal) trên **TextVQA test, condition chính**; ước lượng **≥ +1,00 pp** và **cận dưới CI 95% > 0** |
| Clean retention | Hmean clean của M-cal không thấp hơn B1-cal quá **1,00 pp**; các kết quả âm cũng phải báo |
| G1 | Hmean(E0 clean) − Hmean(E0 condition) ≥ **5,00 pp** và CI 95% cho mức giảm có cận dưới > 0 |
| G2 | Tối thiểu **15% clean-TP chuyển thành FN** ở condition chính |
| G3 A (LR2) | Tối thiểu **30% GT clean-TP→LR2-FN** có chiều cao native LR2 ≥ **6 px**; sensitivity 4/8 px |
| G3 B/C (Blur/JPEG) | Tối thiểu **30% GT clean-TP→condition-FN** có SSIM crop ≥ τ đã hiệu chuẩn trước khi mở kết quả Blur/JPEG trên S2; crop SSIM không hợp lệ đếm không đạt và báo riêng |
| G4 | Chẩn đoán chi phí thấp (upsample/unsharp, P-map, geometry); **không có ngưỡng hồi phục cứng** và không tự thay thế G1–G3 |
| Tiết kiệm đọc mù | **Chỉ hiệu chuẩn τ nếu LR2 không đạt ít nhất một trong G1–G3**. Dù LR2 đạt vẫn chạy E0 Blur/JPEG **mô tả** trên S2 để thử cả hai nhóm; không gọi đó là G3 Blur/JPEG đầy đủ |
| Đánh giá cuối | TextVQA test (primary) + ICDAR2015 test (external), báo cả native và calibrated cùng FP/FN, P/R/Hmean, size bins, latency/VRAM |

**Lưu ý thống kê:** “Ước lượng gain ≥1 pp và CI dưới >0” **không** có nghĩa “đã chứng minh gain tối thiểu 1 pp”. Muốn phát biểu đó, CI dưới phải >1 pp. Bootstrap **theo source**, lấy lại cùng source ở mọi condition, tính lại TP/FP/FN và micro-Hmean, tối thiểu 2.000 bootstrap replicates; lưu bootstrap seed và code.

## 1. Bảo toàn kết quả v2.0 và giải thích giới hạn

Job **11492** hoàn tất exit code 0:0; identity clean trên S1 khớp E0 lịch sử. Hmean E0 **70,76% clean → 27,43% LR4**, giảm **43,33 pp** (CI 95% **[37,08; 49,78]**). Clean TP→LR4 FN **8.901/11.941 = 74,54%**. G3: **2.011/8.901 = 22,59%** có native height ≥6 px, thấp hơn ngưỡng 30%, nên **LR4 trượt G3 và dừng theo v2.0**. Bicubic upsample chỉ tăng 1,78 pp; 97,05% GT bị mất có valid kernel được phân vào “response yếu” theo phép đo khóa trước.

Số liệu exploratory trên S1 cho LR2: Hmean giảm 10,78 pp; **2.103/3.383 = 62,16%** GT bị mất có native height ≥6 px. Vì đã thấy số đó, S2 là **đánh giá xác nhận thứ hai trong nội bộ validation**, không được quảng bá là lựa chọn hoàn toàn không chịu ảnh hưởng exploratory.

**Không suy diễn quá mức:** native height <6 px và P-map yếu gợi ý sự mất chi tiết/response, **không chứng minh** chữ không thể phục hồi hoặc encoder mất feature. SSIM là proxy cho chất lượng crop, **không tự chứng minh** từ còn đọc được.

## 2. Preflight S2 — lấy mẫu hoàn toàn tái lập (chưa dùng detector)

1. Đọc danh sách **source ID** trong remaining validation của FULL_R1; loại S1 theo source ID và nhóm clean SHA (nếu có). Kiểm tra số phần tử hợp lệ **2.762**, không có duplicate. Khóa ID→relative path mapping + phiên bản/hash manifest FULL_R1.
2. Thứ tự và thuật toán bắt buộc (Python phiên bản ghi trong receipt):

   ```python
   import random
   ids_sorted = sorted(set(remaining_source_ids))  # Python lexicographic Unicode order
   assert len(ids_sorted) == 2762
   sampled = random.Random(20261108).sample(ids_sorted, 500)
   s2_ids = sorted(sampled)  # canonical file order, independent of sampled order
   assert len(s2_ids) == len(set(s2_ids)) == 500
   ```

3. Ghi `S2_source_ids.txt`: một source ID/mỗi dòng, UTF-8 **không BOM**, newline LF, **có LF ở dòng cuối**, thứ tự như `s2_ids`. SHA256 của **file source ID** là hash chuẩn lựa chọn S2 (không dùng absolute path). Ghi thêm `S2_sources.txt` đường dẫn tương đối ứng với **đúng cùng thứ tự ID**; UTF-8 LF và SHA256 riêng. `S2_sources.txt` là đầu vào `--source-list`.
4. Kiểm tra S2 và S1 không trùng source ID, không trùng nhóm clean SHA (nếu manifest cung cấp); tất cả biến thể của một source ở cùng tập. Báo số valid/ignored GT, số GT phân theo chiều cao và số ảnh nguồn.
5. **Tuyệt đối không mở predictions E0/B0/đầu ra detector có sẵn cho S2**, không xem điểm/chẩn đoán S2 trước `PRE-S2 LOCK RECEIPT`.

## 3. Định nghĩa GT height, mức suy thoái và script

### 3.1. Height và severity

- Codex **phải truy nguyên định nghĩa chính xác** của `native GT height` dùng cho cổng G3 Job 11492 (script/module và đoạn code). Nếu không thể xác định, **FAIL và dừng**; không thay bằng một định nghĩa mới rồi so sánh với S1.
- Từ **valid word GT trên S2 clean**, tính median height `m` theo cùng định nghĩa đã tìm được; không xem detector predictions. Báo n, min/q25/median/q75/max và kích thước ảnh (width/height/long side min/median/max).
- Gaussian blur: `sigma1 = round(0.1*m, 2)`; `sigma2 = round(0.2*m, 2)` **đơn vị px gốc**; dùng đúng hai giá trị 2 chữ số sau dấu thập phân đó khi truyền CLI, không làm tròn số nguyên hoặc tự đổi do thấy performance. Nếu σ<=0 hoặc hai mức trùng nhau: **FAIL**. Xem ảnh minh họa **không có prediction** để phát hiện lỗi kỹ thuật/EXIF; nếu severity vô lý, ghi issue và sửa protocol **trước S2**, không âm thầm thay.
- JPEG: `quality=30` (primary), `quality=10` (stress). Ảnh JPEG nguồn được decode RGB rồi JPEG recompress, không gọi là nén JPEG lần đầu.
- LR: **native LR** dùng `generate_lr_only.py` với downsample bicubic, PNG, Pillow==11.3.0, kích thước `W//scale × H//scale`. Dùng ratio chính xác `clean_w/lr_w` và `clean_h/lr_h` khi map prediction về tọa độ clean; `B_up` là upsample LR về canvas clean trước detector, không tự coi bằng internal resize.
- Blur/JPEG: output cùng width/height; tuyệt đối kiểm tra pixel coordinate/orientation khớp với GT nguồn. Không mặc định đúng chỉ vì kích thước giống nhau.

### 3.2. Các script và hash

- **Bản blur/JPEG đúng được xác định bằng SHA256, bất kể tên file:** `79a1e49a68821e50fecfd8caa0c6627e5a840f276600ab75dd592bd5fdc0cffb`. File `generate_blur_jpeg_s2_audited.py` trong kit có hash đó; một file `generate_blur_jpeg.py` bên ngoài kit **có thể trùng từng byte** và được dùng nếu hash đúng. **Mọi file cùng tên nhưng hash khác** (ví dụ bản sơ khai không có `--source-list`) **không được dùng**. Kiểm tra hash trên chính HPC trước chạy.
- `generate_lr_only.py`: hash **người dùng cung cấp**, **chưa được kiểm chứng trong môi trường soạn protocol**: `6cdc66a712aac431af4d2f79da6bd196e6a240b7964283688830b136834aa4f3`. **Phải kiểm tra hash trên HPC trước dùng**; lệch → FAIL.
- Pillow **11.3.0**. Lưu `python --version`, `pip freeze` / môi trường conda, Git commit, config DBNet/MMOCR/checkpoint SHA, script hash vào receipt.
- `generate_blur_jpeg_s2_audited.py` nhận `--input-dir`, `--source-list`, `--output-dir`, `--blur-sigmas sigma1 sigma2`, `--jpeg-qualities 30 10`, `--expect-count 500`, `--strict-exif`. Output dir phải **chưa tồn tại**; nếu chạy lỗi giữa chừng, không dùng manifest/output dở dang, làm sạch một cách có kiểm tra rồi chạy lại.
- Manifest dự kiến **10 cột**: `source,condition,parameter,width,height,out_path,pillow_version,source_sha256,output_sha256,source_exif_orientation`. Kiểm tra đúng **500 × 4 = 2.000** biến thể, đường dẫn/hash có tồn tại, unique `(source,condition,parameter)`, EXIF và nguồn ID khớp S2. Không tạo các biến thể thực của S2 cho tới khi σ đã tính và khóa.

## 4. Audit kỹ thuật và nguyên tắc STOP

- Tra Job **11492**: config, checkpoint, model dtype loaded, inference autocast scope, FP32/BF16, postprocessing params; không suy từ tên checkpoint. Dùng **chính chế độ đã tạo identity E0** cho mọi condition trên S2.
- Identity test Clean/LR1 (pipeline tương ứng) phải khớp E0 đã lưu khi chạy trên cohort giống nhau; test round-trip polygon Clean → LR → Clean (IoU gần 1, báo error tối đa); resize/interpolation/padding/postprocessor đã khóa; xuất P-map trước postprocess và mask padding/ignored; lưu candidate polygon/score trước cutoff.
- Đếm EXIF orientation khác 1 trên cả S1 và S2. Nếu >0, liệt kê ID/filename và **dừng preflight** để audit chính sách reader/GT; không tự xoay hoặc xóa nguồn.
- Chỉ dùng train GT cho teacher supervision; validation GT chỉ phân tích/chọn, test GT chỉ benchmark cuối. Không xem test trong lựa chọn hướng, và ghi lịch sử test đã dùng exploratory.
- Nếu có **FAIL** ở sample ID/hash, GT-height provenance, precision, EXIF, script hash, số lượng/geometry hoặc split: **dừng trước S2**, báo lỗi và chứng cứ, không tự sửa protocol.

## 5. Hiệu chuẩn SSIM τ bằng đọc mù — **chỉ kích hoạt khi LR2 trượt** (và trước khi xét G3 Blur/JPEG trên S2)

### 5.1. Bộ mẫu (TRAIN only) — **chỉ Blur1 và JPEG30, theo nhánh có điều kiện**

- **Quyết định đã chốt:** chỉ hiệu chuẩn `τ_Blur1` và `τ_JPEG30` **nếu LR2 trượt G1, G2 hoặc G3**. Nếu LR2 đạt cả ba cổng, kết luận chọn LR2 trước; không đọc mù, ghi `τ=N/A (LR2 selected by registered priority)` và vẫn chạy Blur/JPEG **mô tả inference** trên S2. Không hiệu chuẩn τ riêng cho Blur2/JPEG10 (stress-only). Không được gọi Blur/JPEG `G3 PASS` nếu chưa có τ.
- Từ pool TRAIN cố định **tối đa 20.000 unique `(source_id,word_id)`**, chỉ chọn tối đa **200 word IDs/condition** để đọc mù. Seed pool `20261109`; seed stratified selection `20261108` (ghi thuật toán, sorted population list, Python version, canonical file SHA256). Chọn crop đọc mù **từ chính pool 20K** để `p_b>0, n_b=0` không do hai frame độc lập. Mỗi ID tạo một cặp clean/degraded; không dùng S1/S2/validation/test, không nhân bản. Ưu tiên hạn chế trùng word ID giữa conditions và báo overlap thực tế.
- **Phân tầng cố định và tăng mẫu ở vùng τ cao:** 6 tầng `[0.30,0.40)`, `[0.40,0.50)`, `[0.50,0.60)`, `[0.60,0.70)`, `[0.70,0.80)`, `[0.80,0.90)` chọn **20 word mỗi tầng** (120 tổng), cộng tầng `[0.90,1.00]` chọn **80 word**. Như vậy các mức `τ=0.90` có cơ hội có `N(τ)≥30` sau khi kiểm tra đọc clean. Không bảo đảm trước khi người đọc hoàn tất.
- **Quy tắc thiếu mẫu (khóa trước):** nếu một tầng có ít word hợp lệ hơn quota, lấy tất cả word của tầng đó; chuyển phần thiếu sang các tầng **có SSIM cao nhất còn dư** theo thứ tự `[0.90,1.00]` rồi `[0.80,0.90)` xuống `[0.30,0.40)`; không lấy lặp. Nếu tổng hợp lệ <200, dùng toàn bộ và báo tổng thực có cùng phân bố. Không tự tạo crop giả hoặc hạ ngưỡng để đạt quota. Nếu không có crop `SSIM≥0.30`, **FAIL** hiệu chuẩn condition.
- Mẫu phân tầng theo SSIM **không đại diện phân bố SSIM tự nhiên**. Phải hiệu chỉnh trọng số theo phân bố SSIM của word GT hợp lệ trên TRAIN khi ước lượng tỷ lệ đọc được. G3 trên S2 vẫn đo trực tiếp trên **toàn bộ** clean-TP→FN thực tế, không suy ra tỷ lệ G3 từ bộ hiệu chuẩn.
- Tính SSIM từ cặp crop **cùng GT word**: bounding rectangle trục của word polygon theo hệ clean; `(x_min,y_min)=floor`, `(x_max,y_max)=ceil`, clamp vào canvas; **không resize**. Gray BT.601 bằng Pillow `convert('L')`; `skimage.metrics.structural_similarity(..., data_range=255, win_size=7)`. Cạnh ngắn crop <7 px là **invalid** (không tính SSIM, báo số lượng trong khung mẫu trước lựa chọn). Dùng cùng bytes image trước và sau corruption theo version khóa.
- SSIM ngoài `[0.30,1.00]` **không được chọn vào bộ đọc hiệu chuẩn**, nhưng phải báo số lượng/đặc tính trong population train. SSIM âm/ngoài `[0,1]` báo riêng, không âm thầm clamp. Chỉ đọc crop có SSIM hợp lệ. Lưu SSIM full precision, source ID, word ID, condition, GT transcription **ở bảng đáp án tách biệt**.
- Xuất **phiếu đọc mù**: trộn ngẫu nhiên crop clean và corrupted của cùng word với mã ẩn; không hiển thị condition, source ID, GT/chuỗi đúng, SSIM hay thứ tự cặp. Hai người đọc độc lập tự gõ chuỗi. Sau đó mới giải mã và đối chiếu với TextOCR. Báo inter-rater agreement và số crop clean gốc không đọc được.
- **Ngân sách đọc:** 2 conditions × 200 words × 2 versions (clean/degraded) × 2 raters = **1.600 lượt đọc tối đa**, bằng ngân sách thiết kế cũ 4 × 100 × 2 × 2 nhưng tập trung vào điều kiện chính. Kiểm định thủ công 60 crop/condition trên S2 là **tùy chọn sau khi chọn hướng**, không phải điều kiện GO của preflight.

### 5.2. Tỷ lệ đọc được có trọng số và quy tắc chọn τ (Rev.4)

**Mục tiêu đo:** xác suất một word *vẫn đọc đúng sau suy thoái*, trong số word đọc đúng ở clean và có SSIM ≥ τ, khi phân bố SSIM được đưa về phân bố TRAIN. Việc cố ý lấy 80/200 word ở bin `[0.90,1.00]` **không** được phép làm bin này chiếm trọng số 40% ở mọi τ.

**Population tham chiếu và sampling frame chung:** lấy ngẫu nhiên **20.000 unique word IDs từ toàn bộ valid TextOCR word GT TRAIN** (hoặc toàn bộ nếu ít hơn), `sorted((source_id,word_id))` rồi `random.Random(20261109).sample`, khóa SHA256 của ID list. Với **cùng pool này**, tính SSIM clean→Blur1 và clean→JPEG30; với mỗi condition, `p_b` là phân bố theo bảy bin `[0.30,1.00]` **có SSIM hợp lệ**, chuẩn hóa trong phần population SSIM≥0.30. Báo thêm counts invalid, <0.30, >1 và số ở từng bin. Chọn 200 word IDs để đọc mù **bên trong cùng pool**, phân tầng theo §5.1 và `random.Random(20261108)` sau khi sort word IDs theo từng bin; báo SHA256 selection mỗi condition. Đây là thiết kế lấy mẫu hai giai đoạn, 20K là ước lượng population TRAIN có sai số, không phải census. Không nhìn prediction hoặc các ảnh từ S1/S2/test khi lập pool/phiếu.

**Quy tắc bin hiếm:** bin trống trong **pool 20K** có `p_b=0`, không được lấy sample đọc từ bin đó và **không tự FAIL vì bin vắng mặt**; ghi `not observed in pool`, không suy luận tỷ trọng thật bằng 0. Với bin không trống `p_b>0`, sampler phải chọn ≥1 word và phiếu đọc phải đầy đủ (`n_b>0`); nếu không đáp ứng thì FAIL condition vì lỗi sampling/annotation, không bịa trọng số. Không tự bổ sung word sau khi biết câu trả lời người đọc. Quy tắc quota, bổ sung từ bin khác khi thiếu, N/N_eff tại mọi τ và điều kiện FAIL thống kê vẫn như §5.1–§5.2.

**Đọc mù:** chuẩn hóa transcript và GT bằng `unicodedata.normalize('NFC', text).lower().strip()`; so khớp chính xác, không bỏ dấu hay dấu câu. `C_i=1` khi **cả hai người** đọc đúng clean; `D_i=1` khi **cả hai** đọc đúng degraded; hai bản phải cùng word ID. Chỉ được tính τ sau khi nhận đầy đủ phiếu độc lập; bất kỳ bản thiếu/không hợp lệ nào đều cần báo và condition FAIL nếu không thể truy vết.

Với mỗi condition, gọi `b(i)` là bin SSIM của word `i` trong bộ đọc, `n_b` là số word **được đánh giá đầy đủ** trong bin b, `p_b` là tỷ trọng population của bin đó. Đặt `w_i=p_{b(i)}/n_{b(i)}` (nhân mọi trọng số với cùng một hằng số không làm đổi tỷ lệ). Với threshold `t` trong lưới:

\[
R_{\mathrm{read,w}}(t)=
\frac{\sum_{i:\,SSIM_i\ge t} w_i C_i D_i}
{\sum_{i:\,SSIM_i\ge t} w_i C_i}.
\]

- Để kiểm tra, cũng báo **unweighted** `N(t)=Σ C_i`, `B(t)=Σ C_i D_i`, `R_unweighted=B/N`; nhưng **chỉ dùng `R_read,w` để chọn τ**.
- Bắt buộc báo theo **từng bin**: `p_b`, `n_b`, `#clean_correct`, `#correct_both`, unweighted readable rate, weight `p_b/n_b`; không chỉ báo giá trị cộng dồn. Phân bố TRAIN được ước lượng nên mọi tỷ lệ có thể chịu sai số sampling.
- Kiểm tra cỡ mẫu tại từng t bằng `N(t)≥30` **word clean được cả hai người đọc đúng (không trọng số)** và số mẫu hiệu dụng Kish trên các word clean-correct: `N_eff(t)=(Σ w_i C_i)^2 / Σ (w_i C_i)^2 ≥30`. Nếu mẫu số bằng 0 hoặc điều kiện nào không đạt, ghi `UNDERPOWERED`, không xem là đạt.
- Lưới cố định `0.30,0.35,...,0.90` (bước `0.05`). Chọn **τ nhỏ nhất** mà `R_read,w(t')≥0.80` tại **mọi t'≥τ trên lưới**; **mọi t' này phải đủ cả `N≥30` và `N_eff≥30`**. Nếu không, condition **FAIL** hiệu chuẩn và không tự tăng mẫu hay nới ngưỡng sau khi đã có kết quả đọc. Báo đặc biệt `N(0.90)`, `N_eff(0.90)` và toàn bộ bảng threshold. Điều kiện này bảo đảm không xem thiếu mẫu ở đầu cao của lưới là đạt ngầm.
- Nếu không có τ thỏa: candidate Blur1/JPEG30 đó `not evaluable` cho G3 và không được chọn train; **không chặn LR2**. Blur2/JPEG10 luôn `τ=N/A (stress-only)`.

**Giới hạn diễn giải:** trọng số điều chỉnh phân bố SSIM của **TRAIN valid words**, **không tự hiệu chuẩn trên nhóm clean-TP→FN** của S2 (nhóm bị mất có thể có phân bố và độ đọc được khác). SSIM vẫn là proxy, không là ground truth đọc chữ. Có thể kiểm định mù tùy chọn 60 crop clean-TP→FN trên S2 **sau khi τ đã khóa**, chỉ để báo độ tin cậy/giới hạn, không dùng chỉnh τ hoặc G3.

**Định nghĩa G3 Blur/JPEG ở S2 không đổi:** mẫu số = toàn bộ GT clean TP→condition FN; tử số = số GT trong nhóm đó có SSIM crop hợp lệ ≥τ đã khóa. Crop <7 px hoặc không tính được SSIM được ghi riêng và **tính là không đạt** trong G3. Không áp trọng số calibration vào phép đếm G3 S2.

## 6. Pilot E0 trên S2 — triển khai theo giai đoạn đã đăng ký trước S2

**Stage S2-A (LR trước):** chỉ được mở kết quả S2 sau khi `PRE-S2 TECHNICAL GO` (mục 8) PASS: nguồn S2/manifest, GT height, sigma, hash script, precision/identity, EXIF, coordinate mapping và fixed settings. Chạy cùng B0 và evaluator trên **Clean, LR2 và LR4/LR8 stress**, kèm B_up cho LR2; tính G1/G2/G3 LR2, G4 định tính, báo raw TP/FP/FN, Hmean, paired source-bootstrap CI và clean-TP transitions. Không dùng S2 để tune tham số. **Không cần τ để chạy S2-A.**

- **Nếu LR2 qua G1+G2+G3:** chọn **A=LR2** làm condition train theo thứ tự ưu tiên đã khóa; không hiệu chuẩn τ Blur/JPEG. Để đáp ứng mục tiêu **thử cả A và B**, vẫn chạy E0 inference trên **Blur1/Blur2, JPEG30/JPEG10 của cùng S2** (sigma/JPEG đã khóa trước S2), báo P/R/Hmean, G1/G2, size/P-map và ví dụ; đánh dấu `G3 = NOT ASSESSED (no τ; LR2 selected)` và **không** tuyên bố Blur/JPEG đã qua pilot đầy đủ. Không thay quyết định LR2 nếu một severity khác có Hmean đẹp hơn.
- **Nếu LR2 trượt bất kỳ G1/G2/G3:** chưa xem score/prediction Blur/JPEG trên S2. Thực hiện hiệu chuẩn mù τ_Blur1/τ_JPEG30 bằng **TRAIN-only** đúng §5; khi có phiếu, khóa τ hoặc FAIL theo condition trước khi đọc kết quả Blur/JPEG trên S2. Sau đó chạy E0 Blur1/2, JPEG30/10 **trên chính S2**, đánh giá G1/G2/G3 tương ứng; chọn B Blur1 trước C JPEG30 nếu cả hai qua. Nếu không ứng viên nào qua, không train M.

**Dữ liệu và bảo toàn:** tất cả conditions dùng cùng S2 source IDs, B0 checkpoint/precision, GT matching/ignore policy, calibration defaults đã khóa; raw P-map float16, pre-cutoff polygon/score và logs. Với bất kỳ condition nào được đánh giá đầy đủ: báo P/R/Hmean, TP/FP/FN, ΔHmean/CI, G1/G2/G3, clean TP→FN, native/network-input size, kernel-invalid (`area<4`), weak response (`c0=0.30`; sensitivity 0.50), contour/score/geometry. G4 được báo như bằng chứng bổ sung, **không** tự làm cổng veto mới. Blur unsharp baseline chỉ khi tham số đã được chốt từ TRAIN/remaining validation trước S2 (không được optimize bằng S2); nếu chưa có, ghi `N/A`. JPEG cheap-recovery baseline `N/A`.

**Chỉ sau khi condition được chọn** mới quyết định SAT (bằng chứng về scale) hoặc PRC (weak response + clean teacher đủ tin cậy); SAT không mặc định phù hợp Blur/JPEG. Nếu không có cơ chế M phù hợp, báo B1 so với các đối chứng, không ép thiết kế thêm module. Mọi thay đổi sau khi đã nhìn S2 là post-hoc, không gọi là áp dụng đúng pre-registration.
## 7. Training (chỉ một loại corruption) và final benchmark

| Nhánh | Định nghĩa |
|---|---|
| B0 | Monkey official checkpoint (không train target), native default postprocess |
| B_up / B_unsharp | Cheap preprocessing control hợp loại corruption, **không training** |
| B0-cal | B0, mask threshold + score threshold được chọn trên **remaining validation** |
| B1 | Fine-tune Clean + chosen degradation, clean replay 50%, sampling thông thường |
| M | B1 + **một** can thiệp: SAT (một thay đổi sampling/crop) **hoặc** PRC (một masked response consistency loss) |
| B1-cal, M-cal | Calibration **cùng lưới, cùng điều kiện chọn** trên remaining validation; báo cả native/calibrated |

- Cùng B0 initialization, scheduler, LR, optimizer updates, augmentation severity, clean replay, decode recipe trừ biến can thiệp, reporting cost/VRAM. Nếu SAT làm thay GT instance count, ghi count/size distribution; PRC teacher B0 frozen chỉ dùng train GT-validated region, warp mask chuẩn cùng geometry, không bỏ supervised detection loss. Tối thiểu 1 seed/nhánh; 2 seeds B1/M nếu tài nguyên cho phép, không coi 1 seed là ổn định training.
- Primary endpoint **calibrated Hmean của M vs B1 trên condition chính TextVQA test**. Báo CI source-bootstrap 95%, Δ≥1,00 pp ước lượng + lower bound >0. Báo thêm B0/B0-cal/B_up, clean loss ≤1 pp, condition nặng và LR2/ICDAR phù hợp; không ẩn regression. ICDAR2015 test chỉ để external confirmation, **không tune/tối ưu**.
- TextOCR official train từng được dùng trong nguồn pretraining Monkey nên local test không mặc nhiên unseen-pretraining; kết quả TextVQA test trước đó đã xem mang tính exploratory, nêu rõ trong báo cáo.

## 8. PRE-S2 LOCK RECEIPT (tệp phải do Codex điền từ bằng chứng, không bịa)

Lưu `outputs/pre_s2_lock_receipt.md` với các trường sau và bằng chứng filepath/hash/log cho từng dòng:

| Mục | Giá trị cần ghi từ dữ liệu | PASS criterion |
|---|---|---|
| Manifest FULL_R1 | path + content SHA256 + counts | phiên bản khớp split đã audit |
| S1 | source count + hash + clean SHA grouping | 500 nguồn |
| S2 | `S2_source_ids.txt` hash + `S2_sources.txt` hash, sampling Python version | 500 unique; không trùng S1 |
| GT height | exact Job11492 code path, function, line + median height | định nghĩa được truy nguyên |
| Blur σ1/σ2 | giá trị thô và round(.,2) | hữu hạn >0, khác nhau |
| JPEG | 30/10 | PASS |
| B0 precision | dtype weights/forward/autocast + config path | trùng E0 identity mode |
| EXIF | số ảnh orientation!=1 S1/S2 + danh sách | 0 hoặc có audit geometry/GT được duyệt trước S2 |
| Scripts | sha256 audited blur/JPEG và LR generator | khớp expected / lập discrepancy report |
| Resize & P-map | identity + round-trip + export/checkpoint hashes | PASS |
| τ Blur1/JPEG30 | `NOT REQUIRED` nếu LR2 qua G1–G3; ngược lại pool TRAIN 20K + grid + p_b/n_b, N/B/R_unweighted/R_weighted/N_eff + selected τ/FAIL + reader agreement | conditional theo §5–6 |
| τ Blur2/JPEG10 | `N/A (stress-only)`, không đọc hiệu chuẩn | PASS (not required for G3) |
| Diagnostic run readiness | không mở S2 predictions trước technical GO, sigma locked, clean ids hash frozen; ghi riêng Stage A / Blur-JPEG trạng thái | PASS |

**GO/NO-GO hai giai đoạn:** `PRE-S2 TECHNICAL GO` cho phép **chỉ** S2-A khi các kiểm tra kỹ thuật (hash, split, identity, precision, EXIF, tọa độ, σ) PASS, **không cần τ**. Nếu LR2 pass G1–G3: có thể chạy Blur/JPEG mô tả nhưng `G3=N/A`, chọn LR2. Nếu LR2 fail: `BLUR/JPEG G3 GO` yêu cầu τ hợp lệ từ TRAIN **trước khi đọc** output Blur/JPEG S2; condition FAIL τ không được xét shortlist. Không cần hoàn tất cả hai condition hiệu chuẩn nếu một condition FAIL; đánh giá condition còn hợp lệ. Chụp snapshot bất biến protocol/receipt/source IDs/script/config hashes tại từng stage. Không sửa mục tiêu, ngưỡng hoặc nguồn dữ liệu sau khi xem S2.

## 9. Lộ trình ngắn

1. **TASK 1 — Technical preflight (không inference S2):** chọn/khóa S2 và hash; truy nguyên GT height của Job11492, tính σ, audit precision/EXIF/scripts/identity/GT mapping, xác minh tạo Blur/JPEG; triển khai code chọn pool TRAIN 20K và scoring SSIM (chưa cần thực thi tính SSIM hay đọc mù nếu LR2 chưa trượt). Xuất `PRE-S2 TECHNICAL GO` từ bằng chứng.
2. **TASK 2A — LR pilot trên S2:** chạy Clean/LR2 và LR4/LR8 stress, B_up; kết luận G1/G2/G3 LR2 trước khi quyết định nhánh tiếp theo.
3. **TASK 2B — Blur/JPEG:** nếu LR2 đạt, chỉ inference mô tả cùng S2, `G3=N/A`. Nếu LR2 trượt, kích hoạt đọc mù TRAIN và khóa τ, sau đó chạy Blur/JPEG và xét G1–G3 để chọn tối đa một condition train.
4. **Train:** B1 và tối đa một M (nếu có bằng chứng), cùng ngân sách; chọn checkpoint trên remaining validation.
5. **Final:** TextVQA test, ICDAR2015 test, source-bootstrap CI, latency/VRAM, ablation, báo cả kết quả âm.

**Không chạy TASK 2A khi technical preflight chưa PASS. Không dùng S2/test để sửa protocol. Đây là amendment Rev.4 trước khi xem S2; không áp dụng hồi tố.**
