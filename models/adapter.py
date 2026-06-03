import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock
from torch.utils.checkpoint import checkpoint as ckpt


class HallucinationModule(nn.Module):
    """ResBlock-based module for modality hallucination."""

    def __init__(self, channels, n_resblocks=2):
        super().__init__()
        self.head = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_resblocks)])
        self.tail = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        nn.init.constant_(self.tail.weight, 0)
        nn.init.constant_(self.tail.bias, 0)

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint and self.training:
            x = ckpt(self.head, x, use_reentrant=False)
            x = ckpt(self.body, x, use_reentrant=False)
            return ckpt(self.tail, x, use_reentrant=False)
        else:
            return self.tail(self.body(self.head(x)))


class FusionModule(nn.Module):
    """Fuses features from image, event, and IR modalities."""

    def __init__(self, latent_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(latent_ch * 3, latent_ch, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, feat_img, feat_event, feat_ir):
        return self.fuse(torch.cat([feat_img, feat_event, feat_ir], dim=1))


class SpectralGatingUnit(nn.Module):
    """Frequency-domain gating: masking + amplitude scaling."""

    def __init__(self, dim):
        super().__init__()
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, 1),
            nn.Sigmoid()
        )
        self.freq_scale = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, 1),
            nn.Tanh()
        )

    def _forward_impl(self, fft_feat):
        mask = self.freq_gate(fft_feat)
        feat_filtered = fft_feat * mask
        scale = self.freq_scale(feat_filtered)
        return feat_filtered * (1 + scale)

    def forward(self, fft_feat, use_checkpoint=False):
        if use_checkpoint and self.training:
            return ckpt(self._forward_impl, fft_feat, use_reentrant=False)
        return self._forward_impl(fft_feat)


class S2DGTranslator(nn.Module):
    """Spatial-to-frequency Domain Guided Translator for modality hallucination."""

    def __init__(self, encoder_channels, latent_dim):
        super().__init__()

        self.adapters = nn.ModuleList()
        total_in_ch = 0
        for ch in encoder_channels:
            self.adapters.append(nn.Sequential(
                nn.Conv2d(ch, latent_dim // 4, 1),
                nn.BatchNorm2d(latent_dim // 4),
                nn.ReLU(inplace=True)
            ))
            total_in_ch += (latent_dim // 4)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_in_ch, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )

        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.spatial_denoiser = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // 2, latent_dim, 1),
            nn.Sigmoid()
        )

        self.spectral_gate = SpectralGatingUnit(latent_dim)

        self.ctx_guide = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.Sigmoid()
        )
        self.reconstruct_head = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def _forward_impl(self, enc_feats_list, use_checkpoint=False):
        target_h, target_w = enc_feats_list[-1].shape[2:]

        processed = []
        for i, feat in enumerate(enc_feats_list):
            f = self.adapters[i](feat)
            if f.shape[2:] != (target_h, target_w):
                f = F.adaptive_avg_pool2d(f, (target_h, target_w))
            processed.append(f)

        x = self.fusion_conv(torch.cat(processed, dim=1))

        x_low = self.low_pass(x)
        x_high_raw = x - x_low

        spatial_weight = self.spatial_denoiser(x_low)
        x_high_clean = x_high_raw * spatial_weight

        x_high_fft = torch.fft.rfft2(x_high_clean, norm='ortho')
        fft_cat = torch.cat([x_high_fft.real, x_high_fft.imag], dim=1)
        fft_processed = self.spectral_gate(fft_cat, use_checkpoint=use_checkpoint)

        c_real, c_imag = torch.chunk(fft_processed, 2, dim=1)
        x_high_fft_c = torch.complex(c_real, c_imag)
        x_high_rec = torch.fft.irfft2(x_high_fft_c, s=(target_h, target_w), norm='ortho')

        guide_map = self.ctx_guide(x_low)
        return self.reconstruct_head(x_high_rec * guide_map + x_low)

    def forward(self, enc_feats_list, use_checkpoint=False):
        if not isinstance(enc_feats_list, (list, tuple)):
            raise TypeError("enc_feats_list must be a list/tuple of feature tensors")

        if (not use_checkpoint) or (not self.training):
            return self._forward_impl(list(enc_feats_list), use_checkpoint=False)

        feats = tuple(enc_feats_list)

        def _run(*fs):
            return self._forward_impl(list(fs), use_checkpoint=True)

        return ckpt(_run, *feats, use_reentrant=False)
