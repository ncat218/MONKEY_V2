# Checkpoint và cấu hình suy luận

Cập nhật 22/11/2026. **B0 và B1 đều tải trực tiếp từ repo này qua Git LFS.** Các tệp là plain `state_dict` dùng suy luận, không có optimizer/Runner state. `-cal` thay hậu xử lý, không tạo checkpoint khác.

| Model | File | Config portable | SHA256 tệp tải |
|---|---|---|---|
| B0 — detector gốc | [b0-inference.pth](weights/b0-inference.pth) | [B0 expanded](configs/b0_expanded.py) | `45460883fbfbcc278cad449b94635f4e03d858ce8dfc11ebae07b7b4288dc49e` |
| B-Clean — 100% Clean | [bclean-inference.pth](weights/bclean-inference.pth) | [B1/B-Clean expanded](configs/b1_expanded.py) | `94f0e4ecbb6881b4430c841845994c6568c145af7bb54fc41514eb0c78f3a6aa` |
| B1 — 50% Clean/LR2 | [b1-inference.pth](weights/b1-inference.pth) | [B1/B-Clean expanded](configs/b1_expanded.py) | `9c3814e494a73a0a6089ba5a6585f079ea8f5a6defdc9347213bb98a64337147` |
| M — PRC, lịch sử | [m-prc-inference.pth](weights/m-prc-inference.pth) | [Common `.30/.40`](configs/common_cal.py) | `905eba90d92f6ed2d3940a2091901d792055d15835db4886d7413c1e0cf82a78` |
| A0 — DBHead +10k | [latent10k-head-inference.pth](weights/latent10k-head-inference.pth) | [head](configs/latent10k_head.py) | `d6959adaaa323b3152b284f208c764e755fa062259ea5e1f9f717d9a9f9bad96` |
| A1 — residual +10k | [latent10k-plain-inference.pth](weights/latent10k-plain-inference.pth) | [plain](configs/latent10k_plain.py) | `fce82ebb147c8ca0762742773f9f6cd1ef8bff0c48778ba58083cc192d49e4d9` |
| A2 — gated +10k | [latent10k-gated-inference.pth](weights/latent10k-gated-inference.pth) | [gated](configs/latent10k_gated.py) | `52ba9896b63c8d3973447474d98f32892315b5d848ee4f246c6a89c371a7e872` |

A0/A1/A2 là kết quả **âm** của job88935, chưa vượt B1 hoặc matched control. Công khai để tái lập, không khuyến nghị thay B1. M/PRC cũng chưa có lợi ích xác nhận. Checkpoint fusion chưa công bố: job106090 đã dừng ở cleanup sau khi Nearest/ASF train đủ10k; [recovery v2](../fusion-recovery-v2.md) sẽ hoàn tất phần còn lại.

## Tải đủ weights

```bash
git lfs install
git clone https://github.com/ncat218/MONKEY_V2.git
cd MONKEY_V2
git lfs pull --include="docs/handoff/weights/*.pth"
python docs/handoff/artifacts/research_20261122/verify_bundle.py --weights
```

Verifier cũng kiểm các ZIP review: nếu chỉ tải weights, chạy `git lfs pull` trước khi kiểm toàn bundle. Trên GitHub có thể dùng nút Download của từng file. Một file chỉ vài dòng `version https://git-lfs...` là pointer, chưa tải đủ. B0 khoảng 46,7 MB, B1 khoảng 88,6 MB; khác kích thước vì dtype được giữ từ checkpoint nguồn, không phải thiếu tensor.

## Môi trường và cách nạp

1. Cài nhánh [MonkeyOCRv2 detection/MMOCR chính thức](https://github.com/Yuliang-Liu/MonkeyOCRv2), gồm đăng ký `MonkeyOCRv2ViTAEBackbone`; tải code/config encoder từ [MonkeyOCRv2-AS](https://huggingface.co/zenosai/MonkeyOCRv2-AS). Stack đã dùng: Torch 2.9.0+cu128, MMCV 2.0.1, MMEngine 0.10.7, MMOCR 1.0.1, OpenCV 4.10.0.84, Pillow 11.3.0.
2. Chọn config trong bảng, kế thừa [base inference config](inference_config.py). Thay `cfg.model.backbone.model_dir` và `weights` bằng vị trí encoder hợp lệ của máy chạy. Config này còn chứa training/dataloader kế thừa từ upstream để truy nguồn; khi suy luận chỉ build `cfg.model`, không khởi động Runner/dataloader TEST.
3. Đặt repo root trên PYTHONPATH. Register MMOCR modules, import `cfg.custom_imports` nếu có, rồi build detector. Với A0/A1/A2 cần `rev4.latent_plugins.LatentPilotDBNet`; adapter schema không tương thích với plain DBNet A1/A2.
4. Nạp **plain** weights bằng `model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)`; gọi `model.float().eval()` và đưa lên thiết bị. FP32 runtime, matmul TF32 off, cuDNN TF32 on. B0 export giữ dtype nguồn (có half); phải cast model sang FP32 sau load.
5. Giữ resize `(4068,1024)` keep ratio, pad32, RGB/BGR policy và actual-coordinate transforms như base config/evaluator. Chỉ đổi config portable đường dẫn không tự tạo ra evaluator benchmark hoàn chỉnh. Muốn chấm đúng FULL_R1 cần manifest/GT đã khóa và scripts tương ứng, không gọi trực tiếp upstream ICDAR dataloader thay thế.

Common `.30/.40` dùng [config riêng](configs/common_cal.py); native `.30/.30` có cùng weights. B0 expanded `.30/.45`, epsilon `.005`; B-Clean/B1 expanded `.40/.60`, epsilon `.01`. Không sử dụng threshold đã chọn riêng trên từng corruption/test dataset.

## Provenance và giấy phép

[B0 nguồn](https://huggingface.co/HB16888/MonkeyOCRv2_det/blob/main/dbnet_mkv2vitae_icdar2015.pth) SHA `4b13e26ad5c2e1b13fef45ac608298254d083062d2b9cafba706064bd8e2cf82`. B0 export có đủ 377 tensor, giá trị/dtype bằng nguồn; hash file khác do bỏ Runner/optimizer. [Metadata B0](artifacts/research_20261122/weights/b0-export.json).

B1 runner SHA `6df7a3fea16e1d3170af92212d060aab8040bb128fcfb63fb81eb7c8572ec58f`; [kiểm chứng 377 tensor](artifacts/research_20261122/weights/b1-export-verification.json) so từng shape/dtype/byte hash với [reference từ runner](artifacts/research_20261122/weights/b1-tensor-reference.json). B-Clean runner SHA `861bbde2aaf0b83cc98fab148d374276aa84012a7599a43dc1e7a227628aacfc`; [metadata](artifacts/pr_quad_audit_v1/provenance/bclean_checkpoint_export.json). A0/A1/A2 có metadata xuất và source runner hash trong [weights provenance](artifacts/research_20261122/weights/).

B-Clean/B1 cùng TRAIN 15.222 nguồn, B0 init và 10k updates; A0/A1/A2 thêm 10k từ B1, backbone/FPNC frozen. File inference không dùng để resume optimizer chính xác. Không công bố checkpoint trung gian/pilot trùng lặp; code và receipts được giữ.

[Apache License 2.0](weights/LICENSE.txt), nguồn [MonkeyOCRv2 Detection](https://huggingface.co/HB16888/MonkeyOCRv2_det) và [AS](https://huggingface.co/zenosai/MonkeyOCRv2-AS). Ảnh TextVQA/TextOCR tải từ bên phát hành và tuân theo điều khoản dữ liệu riêng.
