# Checkpoint để tái lập suy luận

| Model | Tệp tải | SHA-256 của tệp tải |
|---|---|---|
| B0 / E0 | [Checkpoint gốc](https://huggingface.co/HB16888/MonkeyOCRv2_det/blob/main/dbnet_mkv2vitae_icdar2015.pth) | `4b13e26ad5c2e1b13fef45ac608298254d083062d2b9cafba706064bd8e2cf82` |
| B-Clean | [bclean-inference.pth](weights/bclean-inference.pth) | `94f0e4ecbb6881b4430c841845994c6568c145af7bb54fc41514eb0c78f3a6aa` |
| B1 | [b1-inference.pth](weights/b1-inference.pth) | `9c3814e494a73a0a6089ba5a6585f079ea8f5a6defdc9347213bb98a64337147` |
| M (PRC) | [m-prc-inference.pth](weights/m-prc-inference.pth) | `905eba90d92f6ed2d3940a2091901d792055d15835db4886d7413c1e0cf82a78` |

B-Clean/B1/M chứa tensor từ `state_dict` checkpoint 10.000 optimizer updates. Tệp xuất dùng để suy luận và được lưu bằng Git LFS. Dùng nút **Download** trên trang file GitHub hoặc clone với Git LFS.

**Cập nhật 19/11/2026:** đã chia sẻ [baseline và artifacts PR/quad mới](artifacts/pr_quad_audit_v1/README.md)
cho Mix A. B-Clean/B1 chọn `mask_thr=0.40`, `min_text_score=0.60` trong grid mở rộng;
B0 chọn `.30/.45`, `epsilon_ratio=.005`. Các ngưỡng `.30/.40` bên dưới mô tả calibration
lịch sử, vẫn được giữ làm common operating point. Các bản calibrated dùng cùng weights.

B-Clean là đối chứng 100% Clean, cùng B0 init, TRAIN source IDs và 10.000 updates như B1.
Đã xác minh 377 tensor của bản xuất bằng checkpoint runner gốc; xem
[metadata xuất](artifacts/pr_quad_audit_v1/provenance/bclean_checkpoint_export.json).
SHA-256 checkpoint runner gốc B-Clean là
`861bbde2aaf0b83cc98fab148d374276aa84012a7599a43dc1e7a227628aacfc`,
B1 là `6df7a3fea16e1d3170af92212d060aab8040bb128fcfb63fb81eb7c8572ec58f`.
Các hash này khác hash tệp state_dict tải ở bảng trên do khác cách đóng gói.

1. Cài [MonkeyOCRv2 phần detection/MMOCR](https://github.com/Yuliang-Liu/MonkeyOCRv2) và tải [encoder MonkeyOCRv2-AS](https://huggingface.co/zenosai/MonkeyOCRv2-AS). Môi trường đã kiểm chứng dùng Torch 2.9.0+cu128, MMCV 2.0.1, MMEngine 0.10.7, MMOCR 1.0.1 và OpenCV 4.10.0.84.
2. Dùng [config inference](inference_config.py) và trỏ `model_dir`, `weights` trong `model.backbone` đến encoder đã tải. Config được xuất từ bản FP32 dùng trên HPC với đường dẫn encoder tương đối.
3. Sau khi tạo detector từ config, nạp B1 hoặc M bằng `model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)` rồi gọi `model.float().eval()`. Dùng `mask_thr=0.30`, `min_text_score=0.40` cho calibrated và `0.30/0.30` cho native.

Checkpoint B1 và M được fine-tune trên phần TRAIN của TextVQA FULL_R1 từ B0. Xem giấy phép đi kèm và nguồn weights gốc. [Apache License 2.0](weights/LICENSE.txt) · [MonkeyOCRv2 Detection](https://huggingface.co/HB16888/MonkeyOCRv2_det) · [MonkeyOCRv2-AS](https://huggingface.co/zenosai/MonkeyOCRv2-AS).

Để chấm lại **đúng split FULL_R1**, cần các manifest khóa của nhóm và ảnh từ nguồn phát hành. [Bàn giao và trạng thái manifest TEST](../../handoff.md).
