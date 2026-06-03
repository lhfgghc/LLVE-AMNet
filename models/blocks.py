import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """Layer normalization over the channel dimension for (B, C, H, W) tensors."""

    def __init__(self, C):
        super().__init__()
        self.ln = nn.LayerNorm(C)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvBlock(nn.Module):
    """Conv-BN-ReLU block for projection and downsampling."""

    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with pre-norm LayerNorm."""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.body(self.norm1(x))


class DownBlock(nn.Module):
    """Downsample with stride-2 conv, then refine with ResBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_ch, out_ch, ks=3, stride=2, padding=1),
            ResBlock(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample, concatenate skip connection, fuse, and refine."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_fuse = ConvBlock(in_ch + skip_ch, out_ch, ks=3, stride=1, padding=1)
        self.res_body = ResBlock(out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_fuse(x)
        x = self.res_body(x)
        return x
