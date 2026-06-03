"""RetinexFormer-based encoder and decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from .retinex import IGAB, IlluminationEstimator, trunc_normal_
from .blocks import ResBlock


class RetinexImageEncoder(nn.Module):
    """IGAB-based image encoder with illumination-guided attention at each scale."""

    def __init__(self, in_ch=3, encoder_channels=[64, 128, 256, 256],
                 num_blocks=[1, 1, 2, 2], dim_head=32):
        super().__init__()

        assert len(encoder_channels) >= 1
        ch0, ch1, ch2, ch3 = encoder_channels
        self.channels = encoder_channels
        self.out_ch = encoder_channels[-1]

        self.illu_estimator = IlluminationEstimator(n_fea_middle=ch0, n_fea_in=4, n_fea_out=3)

        self.enc0_embed = nn.Conv2d(in_ch, ch0, 3, 1, 1, bias=False)
        self.enc0_igab = IGAB(dim=ch0, dim_head=dim_head, heads=ch0 // dim_head,
                              num_blocks=num_blocks[0])

        self.down1 = nn.Conv2d(ch0, ch1, 4, 2, 1, bias=False)
        self.enc1_igab = IGAB(dim=ch1, dim_head=dim_head, heads=ch1 // dim_head,
                              num_blocks=num_blocks[1])
        self.illu_down1 = nn.Conv2d(ch0, ch1, 4, 2, 1, bias=False)

        self.down2 = nn.Conv2d(ch1, ch2, 4, 2, 1, bias=False)
        self.enc2_igab = IGAB(dim=ch2, dim_head=dim_head, heads=ch2 // dim_head,
                              num_blocks=num_blocks[2])
        self.illu_down2 = nn.Conv2d(ch1, ch2, 4, 2, 1, bias=False)

        self.down3 = nn.Conv2d(ch2, ch3, 4, 2, 1, bias=False)
        self.enc3_igab = IGAB(dim=ch3, dim_head=dim_head, heads=ch3 // dim_head,
                              num_blocks=num_blocks[3])
        self.illu_down3 = nn.Conv2d(ch2, ch3, 4, 2, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, use_checkpoint=False):
        """Returns encoder features and illumination features at 4 scales."""
        illu_fea, illu_map = self.illu_estimator(x)
        x_enhanced = x * illu_map + x

        x0 = self.enc0_igab(self.enc0_embed(x_enhanced), illu_fea)

        illu_fea_1 = self.illu_down1(illu_fea)
        x1 = self.enc1_igab(self.down1(x0), illu_fea_1)

        illu_fea_2 = self.illu_down2(illu_fea_1)
        x2 = self.enc2_igab(self.down2(x1), illu_fea_2)

        illu_fea_3 = self.illu_down3(illu_fea_2)
        x3 = self.enc3_igab(self.down3(x2), illu_fea_3)

        return [x0, x1, x2, x3], [illu_fea, illu_fea_1, illu_fea_2, illu_fea_3]


class RetinexDecoder(nn.Module):
    """IGAB-based decoder with skip connections and illumination guidance."""

    def __init__(self, encoder_channels, latent_dim, out_ch=3,
                 num_blocks=[2, 2, 1, 1], dim_head=32):
        super().__init__()
        ch0, ch1, ch2, ch3 = encoder_channels

        self.bottleneck = nn.Conv2d(latent_dim, ch3, 3, 1, 1, bias=False)

        self.up2 = nn.ConvTranspose2d(ch3, ch2, stride=2, kernel_size=2, padding=0)
        self.fuse2 = nn.Conv2d(ch2 * 2, ch2, 1, 1, bias=False)
        self.dec2_igab = IGAB(dim=ch2, dim_head=dim_head, heads=ch2 // dim_head,
                              num_blocks=num_blocks[0])
        self.illu_up2 = nn.ConvTranspose2d(ch3, ch2, stride=2, kernel_size=2, padding=0)

        self.up1 = nn.ConvTranspose2d(ch2, ch1, stride=2, kernel_size=2, padding=0)
        self.fuse1 = nn.Conv2d(ch1 * 2, ch1, 1, 1, bias=False)
        self.dec1_igab = IGAB(dim=ch1, dim_head=dim_head, heads=ch1 // dim_head,
                              num_blocks=num_blocks[1])
        self.illu_up1 = nn.ConvTranspose2d(ch2, ch1, stride=2, kernel_size=2, padding=0)

        self.up0 = nn.ConvTranspose2d(ch1, ch0, stride=2, kernel_size=2, padding=0)
        self.fuse0 = nn.Conv2d(ch0 * 2, ch0, 1, 1, bias=False)
        self.dec0_igab = IGAB(dim=ch0, dim_head=dim_head, heads=ch0 // dim_head,
                              num_blocks=num_blocks[2])
        self.illu_up0 = nn.ConvTranspose2d(ch1, ch0, stride=2, kernel_size=2, padding=0)

        self.out_conv = nn.Conv2d(ch0, out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, h, enc_feats, enc_illu_feas, use_checkpoint=False):
        """Decode latent features with skip connections. Returns residual image."""
        x0, x1, x2, x3 = enc_feats
        illu_fea_0, illu_fea_1, illu_fea_2, illu_fea_3 = enc_illu_feas

        x = self.bottleneck(h)

        x = self.dec2_igab(self.fuse2(torch.cat([self.up2(x), x2], dim=1)), illu_fea_2)
        x = self.dec1_igab(self.fuse1(torch.cat([self.up1(x), x1], dim=1)), illu_fea_1)
        x = self.dec0_igab(self.fuse0(torch.cat([self.up0(x), x0], dim=1)), illu_fea_0)

        return self.out_conv(x)


class LightweightEventEncoder(nn.Module):
    """Lightweight encoder for event data, outputs only deepest features."""

    def __init__(self, in_ch=10, base_ch=32, latent_dim=256, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.enc0 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1, bias=False), ResBlock(base_ch))
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False)
        self.enc1 = ResBlock(base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False)
        self.enc2 = ResBlock(base_ch * 4)
        self.down3 = nn.Conv2d(base_ch * 4, latent_dim, 4, 2, 1, bias=False)
        self.enc3 = ResBlock(latent_dim)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            x0 = ckpt(self.enc0, x, use_reentrant=False)
            x1 = ckpt(self.enc1, ckpt(lambda y: self.down1(y), x0, use_reentrant=False), use_reentrant=False)
            x2 = ckpt(self.enc2, ckpt(lambda y: self.down2(y), x1, use_reentrant=False), use_reentrant=False)
            x3 = ckpt(self.enc3, ckpt(lambda y: self.down3(y), x2, use_reentrant=False), use_reentrant=False)
        else:
            x0 = self.enc0(x)
            x1 = self.enc1(self.down1(x0))
            x2 = self.enc2(self.down2(x1))
            x3 = self.enc3(self.down3(x2))
        return x3


class LightweightIREncoder(nn.Module):
    """Lightweight encoder for IR data, outputs only deepest features."""

    def __init__(self, in_ch=1, base_ch=32, latent_dim=256, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.enc0 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1, bias=False), ResBlock(base_ch))
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False)
        self.enc1 = ResBlock(base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False)
        self.enc2 = ResBlock(base_ch * 4)
        self.down3 = nn.Conv2d(base_ch * 4, latent_dim, 4, 2, 1, bias=False)
        self.enc3 = ResBlock(latent_dim)

    def _forward_stage0(self, x): return self.enc0(x)
    def _forward_stage1(self, x0): return self.enc1(self.down1(x0))
    def _forward_stage2(self, x1): return self.enc2(self.down2(x1))
    def _forward_stage3(self, x2): return self.enc3(self.down3(x2))

    def forward(self, x):
        if self.use_checkpoint and self.training:
            x0 = ckpt(self._forward_stage0, x, use_reentrant=False)
            x1 = ckpt(self._forward_stage1, x0, use_reentrant=False)
            x2 = ckpt(self._forward_stage2, x1, use_reentrant=False)
            x3 = ckpt(self._forward_stage3, x2, use_reentrant=False)
        else:
            x0 = self.enc0(x)
            x1 = self.enc1(self.down1(x0))
            x2 = self.enc2(self.down2(x1))
            x3 = self.enc3(self.down3(x2))
        return x3
