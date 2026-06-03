"""RetinexFormer core: illumination estimation and illumination-guided attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.utils.checkpoint import checkpoint as ckpt

from .blocks import ChannelLayerNorm, ResBlock


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):
    """Layer normalization wrapper."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class IlluminationEstimator(nn.Module):
    """Estimates illumination features and illumination map from RGB input."""

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IG_MSA(nn.Module):
    """Illumination-Guided Multi-head Self-Attention."""

    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q, k, v, illu_attn = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp, illu_fea_trans.flatten(1, 2))
        )

        v = v * illu_attn

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    """Feed-forward network with depthwise convolution."""

    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    """Illumination-Guided Attention Block: stacked IG_MSA + FeedForward."""

    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class MultiModalIlluminationEstimator(nn.Module):
    """Illumination estimator supporting RGB + Event + IR inputs."""

    def __init__(self, n_fea_middle, rgb_ch=3, event_ch=0, ir_ch=0, n_fea_out=3):
        super().__init__()
        self.event_enable = (event_ch > 0)
        self.ir_enable = (ir_ch > 0)
        self.event_ch = event_ch
        self.ir_ch = ir_ch

        n_fea_in = rgb_ch + 1
        if self.event_enable:
            n_fea_in += event_ch
        if self.ir_enable:
            n_fea_in += ir_ch
        self.n_fea_in = n_fea_in

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, rgb, event=None, ir=None):
        B, _, H, W = rgb.shape
        mean_c = rgb.mean(dim=1).unsqueeze(1)
        inputs = [rgb, mean_c]

        if self.event_enable:
            if event is not None:
                inputs.append(event)
            else:
                inputs.append(torch.zeros(B, self.event_ch, H, W, device=rgb.device, dtype=rgb.dtype))

        if self.ir_enable:
            if ir is not None:
                inputs.append(ir)
            else:
                inputs.append(torch.zeros(B, self.ir_ch, H, W, device=rgb.device, dtype=rgb.dtype))

        input_cat = torch.cat(inputs, dim=1)
        x_1 = self.conv1(input_cat)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class SNRMapGenerator(nn.Module):
    """Generates an SNR map from enhanced and blurred images."""

    def __init__(self, snr_factor=1.0):
        super().__init__()
        self.snr_factor = snr_factor

    def forward(self, enhanced_img, enhanced_img_blur):
        dark = (enhanced_img[:, 0:1] * 0.299 +
                enhanced_img[:, 1:2] * 0.587 +
                enhanced_img[:, 2:3] * 0.114)
        light = (enhanced_img_blur[:, 0:1] * 0.299 +
                 enhanced_img_blur[:, 1:2] * 0.587 +
                 enhanced_img_blur[:, 2:3] * 0.114)
        noise = torch.abs(dark - light)
        snr = torch.div(light, noise + 1e-4)
        batch_size = snr.shape[0]
        snr_max = torch.max(snr.view(batch_size, -1), dim=1)[0].view(batch_size, 1, 1, 1)
        snr_map = snr * self.snr_factor / (snr_max + 1e-4)
        return torch.clamp(snr_map, min=0.0, max=1.0)


class SingleScaleSNRAwareFusion(nn.Module):
    """SNR-guided feature fusion at latent space (H/8)."""

    def __init__(self, latent_dim, snr_threshold=0.5, depth=1,
                 high_snr_weight=0.7, low_snr_weight=0.3,
                 use_event=True, use_ir=True, use_checkpoint=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.threshold = snr_threshold
        self.depth = depth
        self.high_snr_weight = high_snr_weight
        self.low_snr_weight = low_snr_weight
        self.use_event = use_event
        self.use_ir = use_ir
        self.use_checkpoint = use_checkpoint

        self.img_extractor = nn.ModuleList([ResBlock(latent_dim) for _ in range(depth)])
        if use_event:
            self.ev_extractor = nn.ModuleList([ResBlock(latent_dim) for _ in range(depth)])
        if use_ir:
            self.ir_extractor = nn.ModuleList([ResBlock(latent_dim) for _ in range(depth)])

        self.fea_align = nn.Sequential(
            nn.Conv2d(latent_dim * 4, latent_dim, kernel_size=3, stride=1, padding=1),
            ChannelLayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )

    def _fuse_impl(self, rgb_feat, snr_map, att_feat, event_feat, ir_feat):
        snr_weight = snr_map.clone()
        snr_weight[snr_weight <= self.threshold] = self.low_snr_weight
        snr_weight[snr_weight > self.threshold] = self.high_snr_weight
        snr_reverse_weight = 1.0 - snr_weight

        ch = rgb_feat.shape[1]
        snr_w = snr_weight.repeat(1, ch, 1, 1)
        snr_rw = snr_reverse_weight.repeat(1, ch, 1, 1)

        rgb_out = rgb_feat
        for i in range(self.depth):
            rgb_out = self.img_extractor[i](rgb_out)

        if self.use_event and event_feat is not None:
            ev_out = event_feat
            for i in range(self.depth):
                ev_out = self.ev_extractor[i](ev_out)
        else:
            ev_out = torch.zeros_like(rgb_out)

        if self.use_ir and ir_feat is not None:
            ir_out = ir_feat
            for i in range(self.depth):
                ir_out = self.ir_extractor[i](ir_out)
        else:
            ir_out = torch.zeros_like(rgb_out)

        out_rgb = torch.mul(rgb_out, snr_w)
        out_ev = torch.mul(ev_out, snr_rw)
        out_ir = torch.mul(ir_out, snr_rw)
        return self.fea_align(torch.cat([out_rgb, out_ev, out_ir, att_feat], dim=1))

    def forward(self, rgb_feat, snr_map, att_feat, event_feat=None, ir_feat=None):
        event_input = event_feat if event_feat is not None else torch.zeros_like(rgb_feat)
        ir_input = ir_feat if ir_feat is not None else torch.zeros_like(rgb_feat)

        if self.use_checkpoint and self.training:
            return ckpt(
                self._fuse_impl, rgb_feat, snr_map, att_feat,
                event_input if self.use_event else None,
                ir_input if self.use_ir else None,
                use_reentrant=False
            )
        else:
            return self._fuse_impl(
                rgb_feat, snr_map, att_feat,
                event_input if self.use_event else None,
                ir_input if self.use_ir else None
            )
