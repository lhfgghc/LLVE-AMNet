import torch
from torch.optim.lr_scheduler import (LinearLR, CosineAnnealingLR,
                                       SequentialLR, CosineAnnealingWarmRestarts)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Build LR scheduler: warmup + cosine/step/linear decay."""
    scheduler_cfg = cfg["train"].get("scheduler", {})
    if not scheduler_cfg.get("use", False):
        return None

    num_epochs = cfg["train"]["num_epochs"]
    total_steps = num_epochs * steps_per_epoch
    warmup_epochs = scheduler_cfg.get("warmup_epochs", 0)
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = total_steps - warmup_steps
    base_lr = optimizer.param_groups[0]['lr']
    min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
    mode = scheduler_cfg.get("type", "cosine")
    num_cycles = scheduler_cfg.get("num_cycles", 8)

    print(f"[Scheduler] Type: {mode} | Warmup: {warmup_epochs} epochs | Min LR: {min_lr}")

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )

    decay_scheduler = None
    if mode == "cosine":
        if num_cycles > 1:
            cycle_steps = decay_steps // num_cycles
            decay_scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=cycle_steps, T_mult=1, eta_min=min_lr
            )
            print(f"[Scheduler] Cosine with Restarts: {num_cycles} cycles, {cycle_steps} steps/cycle")
        else:
            decay_scheduler = CosineAnnealingLR(
                optimizer, T_max=decay_steps, eta_min=min_lr
            )
    elif mode == "step":
        milestones_epochs = scheduler_cfg.get("milestones", [50, 80])
        gamma = scheduler_cfg.get("gamma", 0.1)
        milestones_steps = [
            (m - warmup_epochs) * steps_per_epoch
            for m in milestones_epochs if m > warmup_epochs
        ]
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones_steps, gamma=gamma
        )
    elif mode == "linear":
        end_factor = min_lr / base_lr if base_lr > 0 else 0.0
        decay_scheduler = LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor, total_iters=decay_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {mode}")

    return SequentialLR(
        optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
    )
