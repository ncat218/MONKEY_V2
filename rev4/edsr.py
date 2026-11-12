"""EDSR baseline x2 inference using the torchSR weight layout."""
import torch
from torch import nn


def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(conv(64, 64), nn.ReLU(), conv(64, 64))

    def forward(self, image):
        return image + self.body(image)


class EDSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_mean = nn.Conv2d(3, 3, 1)
        self.head = nn.Sequential(conv(3, 64))
        self.body = nn.Sequential(*(ResidualBlock() for _ in range(16)), conv(64, 64))
        self.tail = nn.Sequential(nn.Sequential(conv(64, 256), nn.PixelShuffle(2)), conv(64, 3))
        self.add_mean = nn.Conv2d(3, 3, 1)

    def forward(self, rgb):
        features = self.head(self.sub_mean(rgb))
        return self.add_mean(self.tail(self.body(features) + features))


@torch.inference_mode()
def restore(model, rgb, device, tile=256, halo=40):
    """Restore RGB uint8 using tile context wider than the receptive field."""
    import numpy as np

    h, w = rgb.shape[:2]
    data = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    output = torch.empty((1, 3, 2*h, 2*w), dtype=torch.float32, device=device)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            bottom, right = min(y+tile, h), min(x+tile, w)
            top_pad, left_pad = max(0, y-halo), max(0, x-halo)
            patch = model(data[:, :, top_pad:min(bottom+halo, h), left_pad:min(right+halo, w)])
            output[:, :, 2*y:2*bottom, 2*x:2*right] = patch[:, :, 2*(y-top_pad):2*(bottom-top_pad),
                                                           2*(x-left_pad):2*(right-left_pad)]
    return np.ascontiguousarray(output[0].clamp_(0, 255).round_().to(torch.uint8).permute(1, 2, 0).cpu().numpy())
