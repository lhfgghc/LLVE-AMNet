import torch
import torch.nn as nn
import pytorch_msssim


def calculate_psnr(pred, target):
    """Calculate PSNR between two tensors. Input: (3,H,W) or (1,3,H,W)."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_ssim(pred, target):
    """Calculate SSIM between two tensors. Input: (3,H,W) or (1,3,H,W)."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    return pytorch_msssim.ssim(pred, target, data_range=1.0).item()


def ssim_loss(pred, tgt):
    """SSIM loss for 4D (B,C,H,W) or 5D (B,T,C,H,W) inputs."""
    if pred.dim() == 5:
        B, T = pred.size(0), pred.size(1)
        loss_sum = 0.0
        for t in range(T):
            loss_sum += 1 - pytorch_msssim.ssim(pred[:, t], tgt[:, t], data_range=1.0)
        return loss_sum / T
    return 1 - pytorch_msssim.ssim(pred, tgt, data_range=1.0)


def charbonnier_loss(pred, target, eps=1e-6):
    """Charbonnier loss (robust L1 variant)."""
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def temporal_loss(pred, target):
    """Temporal gradient consistency loss for video sequences."""
    if pred.dim() != 5:
        return torch.tensor(0.0, device=pred.device)
    B, T, C, H, W = pred.shape
    if T < 2:
        return torch.tensor(0.0, device=pred.device)
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    return charbonnier_loss(pred_diff, target_diff)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss module."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.sqrt((diff * diff) + (self.eps * self.eps)).mean()


class DistillNLLLoss(nn.Module):
    """Gaussian NLL loss for probabilistic distillation."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, dist_params, target):
        if dist_params is None or target is None:
            return torch.tensor(0.0).to(target.device if target is not None else 'cuda')
        mu, logvar = dist_params
        target = target.detach()
        mse = (mu - target) ** 2
        inv_var = torch.exp(-logvar)
        return (0.5 * (logvar + mse * inv_var)).mean()
