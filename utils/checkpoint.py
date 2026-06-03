import os
import glob
import re
import torch


def find_latest_checkpoint(ckpt_dir):
    """Find the checkpoint with the highest epoch number."""
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if len(ckpts) == 0:
        return None

    def extract_ep(p):
        m = re.search(r"epoch_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    ckpts = sorted(ckpts, key=extract_ep)
    return ckpts[-1]


def load_checkpoint(path, model, optimizer, device, scheduler=None, ema=None):
    """Load checkpoint with model, optimizer, scheduler, and EMA state."""
    print(f"[Checkpoint] Loading from {path} ...")
    ckpt = torch.load(path, map_location=device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
        print("[Checkpoint] Scheduler state loaded.")

    if ema is not None:
        if "model_ema" in ckpt:
            ema.module.load_state_dict(ckpt["model_ema"])
            print("[Checkpoint] EMA state loaded.")
        else:
            ema.set(model)
            print("[Checkpoint] EMA not found in checkpoint, initialized with current model.")

    return ckpt.get("epoch", 0) + 1, ckpt.get("global_step", 0)


def save_checkpoint(path, model, optimizer, epoch, global_step, scheduler=None, ema=None):
    """Save checkpoint with model, optimizer, scheduler, and EMA state."""
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if scheduler is not None:
        state_dict["scheduler"] = scheduler.state_dict()
    if ema is not None:
        state_dict["model_ema"] = ema.module.state_dict()
    torch.save(state_dict, path)
