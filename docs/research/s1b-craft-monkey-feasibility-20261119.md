# Khả năng dùng S1b và CRAFT cho MonkeyOCRv2-AS + DBNet

Ngày kiểm tra: 19/11/2026, Asia/Bangkok. Đây là ghi chú nghiên cứu và audit chỉ đọc,
không phải sửa đổi protocol hay kết quả thí nghiệm. Chưa viết module thực nghiệm,
chưa chạy inference, chưa train, chưa submit job. Không đọc TEST.

## Kết luận

- Checkpoint S1b được chỉ định **không ghép trực tiếp được** vào sau FPNC: nó xử lý
  token Qwen 2.5-VL-3B 2.048 chiều, còn FPNC của detector hiện tại xuất 256 kênh.
- TRAIN của checkpoint này chứa **2.053/2.262** nguồn remaining-validation Monkey,
  khớp cả source ID và SHA256 ảnh Clean. Do đó không dùng trọng số này, kể cả như
  teacher, để tuyên bố kết quả tuân thủ split Monkey hiện tại. Chồng lấn này chỉ
  liên quan đề xuất dùng checkpoint ngoài; các kết quả B0/B-Clean/B1 đã có không đổi.
- Có thể nghiên cứu **kiến trúc residual restoration tương tự**, khởi tạo mới,
  đặt sau FPNC và huấn luyện trên đúng 15.222 nguồn TRAIN Monkey.
- CRAFT là detector ký tự/affinity từ ảnh RGB. CRAFT nguyên bản không phải module
  phục hồi latent hay cơ chế chọn token cho LLM. Nó có thể làm prior/teacher cho
  một phương pháp mới, nhưng cần chứng minh giá trị bổ sung trước khi tích hợp.
- Không có kết quả mới chứng minh các phương án sau vượt B1-cal hoặc B-Clean-cal.

## 1. Audit checkpoint S1b thực tế

Checkpoint trên HPC:

```text
/datastore/dungmt/textvqa/Tuan/Q-LRG-Robust-Scene-Text-VQA/runs/s1-20261107T051836Z-wymhdtwk/s1/variants/S1b/checkpoint.pt
```

Đã đọc tensor bằng `torch.load(map_location='cpu', weights_only=True)`. Không chạy
forward, không nạp Qwen đầy đủ, không sử dụng GPU. Checkpoint có dung lượng
13.083.707 bytes; adapter có 1.029.152 tham số FP32.

| Hạng mục | Giá trị kiểm tra | Nguồn | Kết luận |
|---|---|---|---|
| SHA256 checkpoint | `e94d739466192bf61bb5ae158ea8a70841df84fe3fdb81dce359d905f951d77f` | Bytes checkpoint | Đã ghi nhận |
| Variant | `S1b` | Checkpoint `arm` | PASS |
| Run binding | `1d902fc3134098331a0ae77d4ab1146ebe714b66112f4b04d7b2fe40fb0633f6` | Checkpoint và logical hash `s1/run.json` | PASS |
| TRAIN binding | `851ddc3e488a60088a5e22da3e1d69678baf67b006ea3347a69d0ba04e1e5c4a` | Logical hash TRAIN manifest và run binding | PASS |
| Model nền | `Qwen/Qwen2.5-VL-3B-Instruct` | Run metadata | Đã xác minh |
| Điểm chèn | Sau visual Merger, trước language model | `qlrg/feature_injection.py` | Đã xác minh |
| Input adapter | `[H×W, 2048]` | LayerNorm `[2048]`, Linear `[224,2048]` trong checkpoint | Không tương thích trực tiếp FPNC |
| Bottleneck | 224; hai spatial block | Tensor checkpoint và `s1.yaml` | Đã xác minh |
| Output adapter | `[H×W, 2048]` | Linear output `[2048,224]` | Không tương thích trực tiếp DBHead 256 kênh |
| Điều kiện học | CLEAN/LR4; config `clean_fraction=0.4` theo lựa chọn ảnh | Training receipt, source, config | Khác native LR2 hiện tại |
| Suy thoái | Bicubic floor-div4, upscale về kích thước gốc | Run preprocessing identity | Khác native LR2 downsample-only |
| TRAIN sources | 19.758 unique image IDs | `s1/train_manifest.json` | Khác split Monkey |
| Giao TRAIN S1b với remaining-validation Monkey | **2.053/2.262 ID và Clean SHA256** | So sánh hai manifest | **FAIL cho việc dùng trọng số theo split hiện tại** |
| Chuyển giao chất lượng detection | Chưa đo | Không chạy model | UNKNOWN |

Tên S1b này thuộc dự án Q-LRG, không phải cohort S1 500 nguồn của Monkey.

Kiến trúc đọc được từ source và tensor:

```text
Qwen merged tokens, width 2048
  → LayerNorm → Linear 2048→224 → GELU
  → 2 × residual [depthwise Conv3×3 → GELU → pointwise Conv1×1]
  → Linear 224→2048
  → cộng residual vào token ban đầu
  → Qwen language model
```

Loss S1b trong source đã xác minh gồm answer cross-entropy qua Qwen, MSE giữa
feature đã sửa và feature Clean, cùng penalty residual; config có
`feature_weight=1`, `residual_weight=0.01`, `clean_residual_weight=0.1`.
Đây là adapter phục vụ VQA trong không gian Qwen. Monkey detection không có LLM
để giữ nguyên thành phần answer loss này.

Chỉ thêm phép chiếu 256→2048→256 có thể làm các tensor nối được về mặt kích thước.
Nó không chứng minh các chiều feature có cùng ý nghĩa, không bảo đảm có lợi cho
detection và không giải quyết chồng lấn dữ liệu.

### Provenance của source

- Q-LRG HEAD khi đọc: `5230c5a8ba80c4539ce9e3d89542caf3e8960a3e`.
- Hash toàn bộ source Q-LRG hiện tại:
  `9f533d8eb49f812a4797eba02f7ff8bcd1b28cb2ddd757f9d62b824f3012438d`.
- Hash này khớp `s1/storage.json:runtime_source_hash` của run đã resume.
- `run.json:shared.source_hash` còn giữ hash binding cũ
  `c485bef6a0d1e9d442654354ba0196326dd9a23c7728243d0f0f4995cbe4e529`;
  không đánh đồng nó với source runtime đã resume. Đã đối chiếu cả hai.
- Logical hash dùng serializer của Q-LRG; khác SHA256 bytes file JSON.
- TRAIN manifest bytes SHA256:
  `dff086a0b8e8b11a72b227a746d866229ddc9c60608365fcdf99a61f198c3adc`.
- Remaining-validation Monkey manifest bytes SHA256:
  `052857fba8b6cd7c78e48a3dcb53475730d03f2ca9d885b66a4a562eea38872c`.

Không đọc manifest hoặc pixels TEST để thực hiện kiểm tra giao này. Việc chỉ dùng
209 nguồn còn lại sẽ thay cohort; ghi chú này không đề xuất tự thay split như vậy.

## 2. CRAFT làm gì và can thiệp ở đâu?

CRAFT: **Character Region Awareness for Text Detection**, Youngmin Baek và cộng sự,
CVPR 2019. Paper: [arXiv](https://arxiv.org/abs/1904.01941),
[PDF](https://arxiv.org/pdf/1904.01941). Nó dự đoán vùng ký tự và liên kết giữa các
ký tự; paper dùng nhãn ký tự tổng hợp và pseudo-label cho ảnh thực có word GT.

[Code chính thức](https://github.com/clovaai/CRAFT-pytorch) dùng VGG16-BN, các block
upsampling/skip connection kiểu U-Net, và đầu ra hai kênh region/affinity:
[craft.py](https://github.com/clovaai/CRAFT-pytorch/blob/master/craft.py).
[Hậu xử lý](https://github.com/clovaai/CRAFT-pytorch/blob/master/craft_utils.py)
threshold hai map, nối thành phần liên thông rồi tạo hộp/vùng chữ.

```text
Ảnh RGB → CRAFT backbone/fusion → character map + affinity map → vùng chữ
```

Việc chiếu hộp CRAFT lên grid token và hợp mask để giữ token cho LLM là một bước
tích hợp thêm. Đoạn mô tả pipeline của Danh do người dùng cung cấp được coi là
context; chưa kiểm tra source của Danh trong lượt nghiên cứu này. Nếu thao tác đúng
như mô tả, nó thay tập token được giữ, không học sửa giá trị của từng vector.

Detector Monkey hiện tại giữ grid feature dày đặc; không có SD-RPN/token-pruning
như mô tả đó. Vì vậy không có sẵn một tập token bị loại để chỉ cần hợp mask CRAFT.

### Ba cách dùng ý tưởng CRAFT với Monkey

| Cách dùng | Vị trí | Lợi ích giả thuyết | Đối chứng/rủi ro |
|---|---|---|---|
| Detector phụ hoặc soft prior | CRAFT chạy RGB song song; map được căn chỉnh sang FPNC | Tín hiệu ký tự bổ sung ở vùng DBNet yếu | Thêm một detector và latency; CRAFT cũng có thể bỏ sót chữ LR |
| Teacher chỉ trong training | CRAFT tạo pseudo-target trên TRAIN; hướng dẫn nhánh phụ | Có giám sát bổ sung, có thể bỏ teacher khi inference | Pseudo-label sai; phải phân biệt với GT thật và đo coverage |
| Head lấy cảm hứng character/affinity | Sau FPNC Monkey, thay/ghép nhánh dự đoán | Học phân tách ký tự và cách nhóm từ | Cần target ký tự/affinity; word polygon TextOCR chưa đủ; thay đổi lớn hơn |

Giữ soft prior/residual thay vì hard-mask triệt tiêu mọi vùng CRAFT không thấy.
Hợp thẳng các hộp CRAFT với DBNet có thể tăng FP; việc gộp detector cần baseline và
xử lý dự đoán trùng. Không gọi cách dùng cơ chế có sẵn là kỹ thuật mới.

README chính thức cung cấp các bộ weight khác nhau: General (SynthText/IC13/IC17),
IC15 (SynthText/IC15), và LinkRefiner (CTW1500). Phải chốt đúng weight và hash trước
khi thử. [Model provenance](https://github.com/clovaai/CRAFT-pytorch#test-instruction-using-pretrained-model).
Repo không phát hành đầy đủ training code. Chưa tải/kiểm tra weight CRAFT cục bộ,
chưa đo tính tương thích của code cũ với stack hiện tại.

## 3. Hướng nên ưu tiên: residual restoration mới sau FPNC

Vị trí đề xuất:

```text
RGB → Monkey backbone → FPNC → F [B,256,Hf,Wf]
                                 ↓
                      residual module R, khởi tạo mới
                                 ↓
                         F' = F + R(F)
                                 ↓
                       DBHead → P-map → hộp chữ
```

Đây là đề xuất, chưa được triển khai. Có thể dùng bottleneck 256→64→256 với spatial
block kiểu S1b; chỉ chuyển cấu trúc, không mang tensor trọng số Qwen sang. Khởi tạo
nhánh output bằng 0 để tại thời điểm đầu, F'=F và detector phải tái lập đúng B1.
Nếu cần gate học được, đó là ablation tiếp theo, không gộp thêm vào pilot đầu.

Vị trí khác để nghiên cứu sau đó là res2/res3 trước FPNC, tương ứng 64/128 kênh
trong config hiện tại. Giả thuyết là sửa đặc trưng không gian trước khi hợp nhất
có thể hữu ích hơn cho chữ nhỏ. Đây chưa phải bằng chứng vị trí đó tốt hơn; cần
so sánh với sau FPNC dưới budget/module capacity được ghi rõ và đo actual shapes.
Không gắn đồng thời nhiều adapter ngay từ pilot đầu vì khó xác định nguồn lợi ích.

Điểm khác feature-distillation pilot đã làm: pilot cũ thêm loss trên fused FPNC,
không có restoration module phụ trong graph inference. Đề xuất mới thêm một module
thực sự xử lý feature khi inference. Điều này tăng khả năng mô hình nhưng cũng thêm
chi phí; không bảo đảm tốt hơn. Pilot cũ chưa cho gain so với continued fine-tuning.

Teacher ban đầu nên dùng B1 frozen trên Clean để bám không gian feature của student
B1. Có thể tái sử dụng quy trình căn chỉnh crop/flip/FOV và mask của pilot cũ để
giảm số yếu tố thay đổi. Không mặc định feature Clean và LR có cùng grid, hoặc
kích thước backbone luôn chênh đúng hai lần; phải kiểm tra tensor và phép chiếu.

Loss đề xuất gồm:

```text
L = L_DB_detection
  + λ_feature × L_feature(F'_LR, stopgrad(F_Clean_teacher))
  + λ_identity × ||F'_Clean − F_Clean||²
```

Mask feature chỉ dùng GT TRAIN, loại ignore/padding. Feature loss chỉ là ràng buộc
phụ; detection loss vẫn quyết định tác vụ. Cần chuẩn hóa/kiểm tra gradient và chốt
λ trước khi chạy; không sao chép hệ số Qwen vì hai không gian/loss khác nhau.
Clean identity là giả thuyết để giảm biến đổi không cần thiết, không bảo đảm giữ
nguyên Clean Hmean sau khi các thành phần student tiếp tục được train.

### Pilot tối thiểu, chưa chạy

Trước pilot: hoàn thiện đối chứng hậu xử lý hình học trên remaining-validation và
phân loại FP để có baseline mạnh, không thay kết quả cũ.

| Arm đề xuất | Khởi tạo | Module mới | Loss |
|---|---|---|---|
| C0 | B1 | Không | Detection; continued fine-tuning |
| C1 | B1 | R khởi tạo mới | Detection + Clean identity |
| M | B1 | Cùng R như C1 | Detection + Clean identity + feature consistency |

- M so với C1 cô lập tác động feature consistency trong cùng kiến trúc/identity loss.
- C1 so với C0 đo gói thay đổi capacity + identity regularization, chưa tách riêng
  đóng góp của hai thành phần. Nếu có tín hiệu tốt mới thêm ablation identity.
- Cả ba cùng 1.000 optimizer updates cho pilot, cùng source draw, augmentation,
  50% Clean/50% LR2, seed và optimizer schedule; khởi tạo từ cùng B1, teacher frozen.
  Giữ cùng chính sách train backbone/FPNC/DBHead ở cả ba arm và ghi rõ policy.
- Đây là mở rộng exploratory; phải lưu config riêng và chốt learning rate,
  bottleneck/loss weights trước khi chạy. Không thay recipe B1 đã khóa.
- Probe bắt buộc: identity ở init, FP32/finite, mask hợp lệ, teacher không có grad,
  grad adapter khác 0 sau bước thích hợp, alignment FOV và grid đúng.
- Đánh giá native và common operating point đã khóa; báo Clean/LR2 P/R/Hmean,
  TP/FP/FN, source-paired CI, recall theo cỡ chữ, actual latency và memory.
- Một seed và 1.000 updates chỉ sàng lọc giả thuyết. CI chứa 0 không chứng minh
  phương pháp vô tác dụng; mức tăng nhỏ chưa đủ lý do tự chạy full trial.
- Chỉ cân nhắc full trial khi lợi ích so với C1 có ý nghĩa thực tiễn và đánh đổi
  Clean/FP/latency phù hợp ngân sách. Cần seed bổ sung trước tuyên bố bền vững.

Tham chiếu chi phí đã đo ở pilot cũ trên H200 MIG: continued 1.000 updates ~587 s,
feature KD ~859 s. Đây không phải dự báo thời gian của module R mới; phải profile
minibatch trước khi chốt GPU hours. Không có benchmark latency cho R hoặc CRAFT.

## 4. Pilot CRAFT nên đi độc lập trước

Chưa train hay gộp CRAFT vào R. Đầu tiên, chốt pretrained weight/preprocessing,
đánh giá CRAFT như một detector phụ trên cùng cohort development và evaluator:

1. P/R/Hmean và latency của CRAFT trên Clean/LR2, với cấu hình được ghi rõ.
2. Bao nhiêu GT là B1-FN nhưng CRAFT-TP theo cùng matching một-một/IoU.
3. Bao nhiêu GT là B1-TP nhưng CRAFT-FN; bao nhiêu FP tăng thêm nếu kết hợp.
4. Kiểm tra theo chữ <4 px, 4–<8 px và nhóm geometry/weak-response của B1.
5. Nếu tín hiệu bổ sung ít, không tích hợp. Nếu có, mới thử teacher training-only
   hoặc soft prior, với baseline đơn giản dùng output CRAFT trước khi thêm mạng mới.

Các điểm số này là proposal; chưa có kết quả cho thấy CRAFT tốt hơn B1 ở chữ nhỏ.
S1b-style R và CRAFT có thể nhắm hai việc khác nhau: sửa feature và cung cấp tín
hiệu định vị. Chỉ gộp khi ablation độc lập cho thấy lợi ích bổ sung; không mặc định
hai khối ghép lại sẽ tốt hơn.

## 5. Liên hệ kết quả hiện tại và giới hạn kết luận

- B-Clean-cal: Clean/LR2 Hmean 78,79/74,36; B1-cal: 77,55/75,29.
- B1−B-Clean LR2 Hmean +0,93 pp, CI [-1,43; +3,14]; Clean -1,24 pp,
  CI [-2,06; -0,64]. B1 LR2 tăng ròng 4.697 TP và 6.030 FP.
- Recall <4 px: B-Clean 19,31%, B1 45,02%. Nhóm geometry/weak-response là
  chẩn đoán quy tắc, chưa phải nguyên nhân đã xác lập cho toàn tập.
- Câu hỏi cho module mới: giữ lợi ích chữ nhỏ của B1, giảm FP/lỗi hộp và giữ Clean.
  Đưa feature gần Clean bằng MSE không tự bảo đảm cải thiện cả ba mục tiêu.
- Remaining-validation đã dùng nhiều lần để phát triển phương pháp. Các CI là
  mô tả theo source, không phản ánh seed variance hoặc selection uncertainty.
- S1/S2 đã được xem ở pilot; không gọi chúng là fresh independent holdout.
- LR4/blur/JPEG và ICDAR2015 là hướng xác nhận tổng quát riêng sau khi chốt phương
  pháp. Không mở rộng TRAIN hoặc dùng TEST để chọn từ ghi chú này.

## Phần đã làm trong lượt nghiên cứu này

Đã đọc source/metadata/checkpoint S1b, xác minh shape và binding, so sánh TRAIN với
remaining-validation theo ID/SHA256, đọc paper/code CRAFT chính thức và đề xuất
các đối chứng. Chỉ tạo ghi chú nghiên cứu cục bộ này. Không sửa protocol, source,
checkpoint hay artifacts trên HPC; không submit bất kỳ job nào.
