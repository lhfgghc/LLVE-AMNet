import os
import glob
import shutil
import time
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from timm.utils import ModelEmaV3
from datetime import datetime

from utils.metrics import calculate_psnr, calculate_ssim
from utils.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint


def make_event_vis(voxel, robust_max=5.0, debug=False):
    """Convert event voxel grid to RGB visualization (red=positive, blue=negative)."""
    img_2d = torch.sum(voxel, dim=0)
    H, W = img_2d.shape
    device = img_2d.device

    abs_img = torch.abs(img_2d)
    if abs_img.max() > 0:
        p95 = torch.quantile(abs_img, 0.95).item()
        p90 = torch.quantile(abs_img, 0.90).item()
        abs_max = abs_img.max().item()
        abs_mean = abs_img.mean().item()
        if p95 > 0:
            robust_max = max(p95, abs_mean * 2, 0.1)
        elif p90 > 0:
            robust_max = max(p90, abs_mean * 2, 0.1)
        else:
            robust_max = max(abs_max, abs_mean * 2, 0.1)
    else:
        robust_max = max(robust_max if robust_max else 5.0, 1.0)

    vis_img = torch.full((3, H, W), 0.5, device=device)
    pos_mask = img_2d > 0
    neg_mask = img_2d < 0

    val_pos = (img_2d[pos_mask].abs() / robust_max) * 0.5 if pos_mask.any() else torch.tensor([], device=device)
    val_neg = (img_2d[neg_mask].abs() / robust_max) * 0.5 if neg_mask.any() else torch.tensor([], device=device)
    val_pos = torch.clamp(val_pos, 0, 0.5)
    val_neg = torch.clamp(val_neg, 0, 0.5)

    vis_img[0][pos_mask] += val_pos
    vis_img[1][pos_mask] -= val_pos
    vis_img[2][pos_mask] -= val_pos
    vis_img[0][neg_mask] -= val_neg
    vis_img[1][neg_mask] -= val_neg
    vis_img[2][neg_mask] += val_neg
    return vis_img


class Trainer:
    """Training loop with EMA, AMP, checkpointing, and multi-modal evaluation."""

    def __init__(self, cfg, model, optimizer, logger, scheduler=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler

        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main = (self.rank == 0)

        self.cfg["exp"].setdefault("eval_before_train", False)
        self.cfg["exp"].setdefault("num_vis_samples", 3)
        self.cfg["exp"].setdefault("random_vis_samples", True)
        self.cfg["exp"].setdefault("save_visuals", True)

        self.start_epoch = 0
        self.global_step = 0

        self.base_model = self.model
        while hasattr(self.base_model, "module"):
            self.base_model = self.base_model.module
        self.device = next(self.base_model.parameters()).device

        if self.is_main:
            print(f"[Trainer] Model: {type(self.base_model).__name__}")

        self.out_dir = os.path.join(cfg["exp"]["out_dir"], cfg["exp"]["experiment_name"])
        self.ckpt_dir = f"{self.out_dir}/checkpoints"
        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.out_dir, "metrics_log.jsonl") if self.is_main else None

        train_cfg = cfg.get("train", {})
        loss_config = train_cfg.get("loss_weight", {}) if train_cfg.get("loss_weight") is not None else {}
        self.lambda_robust = loss_config.get("lambda_robust", 1.0)
        self.lambda_real = loss_config.get("lambda_real", 0.5)
        self.lambda_distill_event = loss_config.get("lambda_distill_event", 0.1)
        self.lambda_distill_ir = loss_config.get("lambda_distill_ir", 0.5)
        self.lambda_ssim = loss_config.get("lambda_ssim", 0.2)
        self.distill_warmup_epochs = loss_config.get("distill_warmup_epochs", 2)

        self.train_use_real_event = train_cfg.get("train_use_real_event", True)
        self.train_use_real_ir = train_cfg.get("train_use_real_ir", True)

        self.freeze_aux_encoders = train_cfg.get("freeze_aux_encoders", False)
        if self.freeze_aux_encoders:
            for name in ["event_encoder", "ir_encoder"]:
                if hasattr(self.base_model, name):
                    for p in getattr(self.base_model, name).parameters():
                        p.requires_grad = False
                    if self.is_main:
                        print(f"[Trainer] Frozen: {name}")

        self.use_ema = cfg["train"].get("use_ema", True)
        self.ema_decay = cfg["train"].get("ema_decay", 0.999)
        self.ema_start_epoch = cfg["train"].get("ema_start_epoch", 0)
        if self.use_ema:
            self.ema = ModelEmaV3(self.base_model, decay=self.ema_decay,
                                  device=self.device, use_warmup=True)
            if self.is_main:
                print(f"[Trainer] EMA enabled (decay={self.ema_decay})")
        else:
            self.ema = None

        self.use_amp = cfg["train"].get("use_amp", False) and (self.device.type == "cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            if self.is_main:
                print("[Trainer] AMP enabled (Float16)")
        else:
            self.scaler = None

        resume_path = None
        pretrain_path = None

        if self.is_main:
            manual_resume = cfg["exp"].get("resume", None)
            if manual_resume and os.path.exists(manual_resume):
                resume_path = manual_resume
            elif os.path.exists(os.path.join(self.ckpt_dir, "latest.pth")):
                resume_path = os.path.join(self.ckpt_dir, "latest.pth")
            else:
                resume_path = find_latest_checkpoint(self.ckpt_dir)

            if not resume_path:
                pretrain_path = cfg["model"].get("pretrain_model", None)

        if self.distributed:
            resume_path = self._broadcast_path(resume_path)
            pretrain_path = self._broadcast_path(pretrain_path)

        if resume_path:
            print(f"[Rank {self.rank}] Loading checkpoint: {resume_path}")
            self.start_epoch, self.global_step = load_checkpoint(
                resume_path, self.base_model, self.optimizer, self.device,
                scheduler=self.scheduler, ema=self.ema
            )
            if self.is_main:
                print(f"[Resume] Epoch {self.start_epoch - 1}, Step {self.global_step}")
        elif pretrain_path:
            self._load_pretrain(pretrain_path)
            if self.use_ema:
                self.ema.set(self.base_model)
            if self.is_main:
                print("[Pretrain] Weights loaded, fresh optimizer")
        else:
            if self.is_main:
                print("[Trainer] Training from scratch")

        if self.distributed:
            dist.barrier()

        self.max_keep = cfg["train"].get("max_ckpt", 3)
        self.best_ckpts = []
        if self.is_main:
            self._init_best_checkpoints()

    def _broadcast_path(self, path):
        """Broadcast file path from rank 0 to all ranks."""
        if not self.distributed:
            return path
        if self.is_main:
            path_len = len(path) if path else 0
        else:
            path_len = 0
        len_tensor = torch.tensor([path_len], device=self.device, dtype=torch.long)
        dist.broadcast(len_tensor, src=0)
        path_len = len_tensor.item()
        if path_len == 0:
            return None
        if self.is_main:
            path_bytes = list(path.encode('utf-8'))
        else:
            path_bytes = [0] * path_len
        path_tensor = torch.tensor(path_bytes, device=self.device, dtype=torch.uint8)
        dist.broadcast(path_tensor, src=0)
        return bytes(path_tensor.cpu().numpy()).decode('utf-8')

    def _init_best_checkpoints(self):
        """Restore best checkpoint list from disk."""
        pattern = os.path.join(self.ckpt_dir, "epoch_*_psnr*.pth")
        for f_path in glob.glob(pattern):
            try:
                fname = os.path.basename(f_path)
                parts = fname.split("_")
                epoch = int(parts[1])
                metric = float(parts[2].replace(".pth", "").replace("psnr", ""))
                self.best_ckpts.append((metric, epoch, f_path))
            except Exception:
                pass
        self.best_ckpts.sort(key=lambda x: x[0], reverse=True)
        while len(self.best_ckpts) > self.max_keep:
            _, _, worst_path = self.best_ckpts.pop()
            if os.path.exists(worst_path):
                try:
                    os.remove(worst_path)
                except OSError:
                    pass

    def _load_pretrain(self, path):
        """Load pretrained weights (model only, no optimizer state)."""
        if not os.path.exists(path):
            if self.is_main:
                print(f"[Warn] Pretrain path not found: {path}")
            return
        print(f"[Pretrain] Loading from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint)
        if state_dict and list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.base_model.load_state_dict(state_dict, strict=False)
        if self.is_main:
            if missing:
                print(f"[Pretrain] Missing keys: {len(missing)}")
            if unexpected:
                print(f"[Pretrain] Unexpected keys: {len(unexpected)}")

    def _save_metrics_to_file(self, epoch, dataset_name, metrics_dict):
        """Append metrics to JSONL log file."""
        if not self.is_main or self.metrics_file is None:
            return
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "epoch": int(epoch),
                "dataset": dataset_name,
                **metrics_dict
            }
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Warn] Failed to save metrics: {e}")

    def _save_vis_samples(self, vis_results, epoch, prefix):
        """Save visualization samples to disk."""
        if not self.is_main or not vis_results:
            return
        if not self.cfg["exp"].get("save_visuals", True):
            return

        vis_dir = os.path.join(self.out_dir, "vis_samples")
        epoch_str = f"epoch_{epoch}" if epoch >= 0 else "zeroshot"
        os.makedirs(vis_dir, exist_ok=True)

        for sid, sample_dict in vis_results.items():
            sample_dir = os.path.join(vis_dir, epoch_str, f"sample_{sid}")
            os.makedirs(sample_dir, exist_ok=True)

            tensor_list, captions = [], []
            for k in ["gt", "input", "event", "ir", "pred_fakeE_fakeIR",
                       "pred_realE_fakeIR", "pred_fakeE_realIR", "pred_realE_realIR"]:
                if k in sample_dict:
                    tensor_list.append(sample_dict[k])
                    captions.append(k)

            if not tensor_list:
                continue

            for t, cap in zip(tensor_list, captions):
                t = t.detach().cpu().clamp(0, 1)
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                elif t.shape[0] > 3:
                    t = t[:3]
                arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(sample_dir, f"{cap}.png"))

    def train_one_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        device = self.device
        accum_steps = self.cfg["train"].get("accumulate_grad_batches", 1)

        warmup_epochs = self.distill_warmup_epochs
        current_distill_factor = epoch / max(1, warmup_epochs) if epoch < warmup_epochs else 1.0

        progress = tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Train Epoch {epoch} [rank {self.rank}]",
            leave=True, disable=(not self.is_main)
        )
        self.optimizer.zero_grad(set_to_none=True)

        for idx, batch in progress:
            if len(batch) != 6:
                raise ValueError(f"Dataset must return 6 items, got {len(batch)}.")
            x, y, event, ir, s_event, s_ir = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            has_event = (event is not None) and (event.dim() > 1) and (event.max() > -0.9) and (s_event.sum() > 0)
            has_ir = (ir is not None) and (ir.dim() > 1) and (ir.max() > -0.9) and (s_ir.sum() > 0)
            event = event.to(device, non_blocking=True) if has_event else None
            ir = ir.to(device, non_blocking=True) if has_ir else None

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                loss_cfg = {
                    "lambda_robust": self.lambda_robust,
                    "lambda_real": self.lambda_real,
                    "lambda_ssim": self.lambda_ssim,
                    "lambda_distill_event": self.lambda_distill_event,
                    "lambda_distill_ir": self.lambda_distill_ir,
                    "distill_factor": current_distill_factor,
                    "train_use_real_event": (self.train_use_real_event and has_event),
                    "train_use_real_ir": (self.train_use_real_ir and has_ir),
                }
                total_loss, loss_dict = self.model(
                    x, y=y, event=event, ir=ir, loss_cfg=loss_cfg
                )
                loss_robust = loss_dict.get("loss_fakeE_fakeIR", loss_dict.get("loss_robust", 0.0))
                loss_distill_weighted_val = loss_dict["loss_distill_weighted"]
                loss_distill_event_val = loss_dict["loss_distill_event"]
                loss_distill_ir_val = loss_dict["loss_distill_ir"]

                combo_losses = {}
                for key in ["loss_fakeE_fakeIR", "loss_realE_fakeIR",
                            "loss_fakeE_realIR", "loss_realE_realIR"]:
                    if key in loss_dict:
                        combo_losses[key] = loss_dict[key]

            loss_for_backward = total_loss / accum_steps

            if self.use_amp:
                self.scaler.scale(loss_for_backward).backward()
                do_step = ((idx + 1) % accum_steps == 0) or ((idx + 1) == len(train_loader))
                if do_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    scale_after = self.scaler.get_scale()
                    if self.use_ema and (scale_after >= scale_before):
                        if epoch < self.ema_start_epoch:
                            self.ema.set(self.base_model)
                        else:
                            self.ema.update(self.base_model, step=self.global_step)
                    if self.scheduler is not None and (scale_after >= scale_before):
                        self.scheduler.step()
            else:
                loss_for_backward.backward()
                do_step = ((idx + 1) % accum_steps == 0) or ((idx + 1) == len(train_loader))
                if do_step:
                    torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_ema:
                        if epoch < self.ema_start_epoch:
                            self.ema.set(self.base_model)
                        else:
                            self.ema.update(self.base_model, step=self.global_step)
                    if self.scheduler is not None:
                        self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.is_main:
                progress.set_postfix({
                    "L": f"{total_loss.item():.3f}",
                    "fakeE_fakeIR": f"{loss_robust:.3f}",
                    "Dst": f"{loss_distill_weighted_val:.4f}",
                    "LR": f"{current_lr:.2e}",
                })

                if idx % self.cfg["train"]["log_interval"] == 0:
                    log_dict = {
                        "train/loss_total": float(total_loss.item()),
                        "train/loss_fakeE_fakeIR": float(loss_robust),
                        "train/loss_distill_weighted": float(loss_distill_weighted_val),
                        "train/loss_distill_event": float(loss_distill_event_val),
                        "train/loss_distill_ir": float(loss_distill_ir_val),
                        "train/lr": float(current_lr),
                        "epoch": int(epoch),
                    }
                    for key, val in combo_losses.items():
                        log_dict[f"train/{key}"] = float(val)
                    self.logger.log(log_dict, step=int(self.global_step))

            self.global_step += 1

    def evaluate(self, loader, epoch, prefix="val"):
        """Evaluate model with multiple modality combinations."""
        if self.use_ema and self.ema is not None:
            eval_model = self.ema.module
        else:
            eval_model = self.model
            while hasattr(eval_model, 'module'):
                eval_model = eval_model.module
        eval_model = eval_model.to(self.device)
        eval_model.eval()

        local_metrics = torch.zeros(9, device=self.device, dtype=torch.float64)
        vis_results = {}
        num_vis = self.cfg["exp"].get("num_vis_samples", 3)
        random_vis = self.cfg["exp"].get("random_vis_samples", True)
        save_visuals = self.cfg["exp"].get("save_visuals", True)

        vis_indices = set()
        if save_visuals and num_vis > 0:
            total_samples = len(loader)
            if random_vis and total_samples > num_vis:
                vis_indices = set(torch.randperm(total_samples)[:num_vis].tolist())
            else:
                vis_indices = set(range(min(num_vis, total_samples)))

        iterator = enumerate(loader)
        if self.is_main:
            iterator = tqdm(iterator, total=len(loader),
                           desc=f"Eval {prefix.upper()} (ep {epoch})", leave=False)

        with torch.no_grad():
            for idx, batch in iterator:
                if len(batch) != 6:
                    continue
                x, y, event, ir, s_event, s_ir = batch
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                has_event = (event is not None) and (event.dim() > 1) and (event.max() > -0.9) and (s_event.sum() > 0)
                has_ir = (ir is not None) and (ir.dim() > 1) and (ir.max() > -0.9) and (s_ir.sum() > 0)
                event = event.to(self.device, non_blocking=True) if has_event else None
                ir = ir.to(self.device, non_blocking=True) if has_ir else None

                gt_vid = y[0]
                T = gt_vid.size(0)

                # RGB only (hallucinated Event + IR)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    pred = eval_model.inference_video(
                        x[0], event=event[0] if has_event else None,
                        ir=ir[0] if has_ir else None,
                        event_use_real=False, ir_use_real=False,
                    )
                pred = torch.clamp(pred, 0.0, 1.0)
                for t in range(T):
                    local_metrics[0] += calculate_psnr(pred[t], gt_vid[t])
                    local_metrics[1] += calculate_ssim(pred[t], gt_vid[t])

                # Real Event + hallucinated IR
                pred_re = None
                if has_event:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                        pred_re = eval_model.inference_video(
                            x[0], event=event[0], ir=ir[0] if has_ir else None,
                            event_use_real=True, ir_use_real=False,
                        )
                    pred_re = torch.clamp(pred_re, 0.0, 1.0)
                    for t in range(T):
                        local_metrics[2] += calculate_psnr(pred_re[t], gt_vid[t])
                        local_metrics[3] += calculate_ssim(pred_re[t], gt_vid[t])

                # Hallucinated Event + real IR
                pred_ri = None
                if has_ir:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                        pred_ri = eval_model.inference_video(
                            x[0], event=event[0] if has_event else None, ir=ir[0],
                            event_use_real=False, ir_use_real=True,
                        )
                    pred_ri = torch.clamp(pred_ri, 0.0, 1.0)
                    for t in range(T):
                        local_metrics[4] += calculate_psnr(pred_ri[t], gt_vid[t])
                        local_metrics[5] += calculate_ssim(pred_ri[t], gt_vid[t])

                # Real Event + real IR
                pred_both = None
                if has_event or has_ir:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                        pred_both = eval_model.inference_video(
                            x[0],
                            event=event[0] if has_event else None,
                            ir=ir[0] if has_ir else None,
                            event_use_real=True if has_event else None,
                            ir_use_real=True if has_ir else None,
                        )
                    pred_both = torch.clamp(pred_both, 0.0, 1.0)
                    for t in range(T):
                        local_metrics[6] += calculate_psnr(pred_both[t], gt_vid[t])
                        local_metrics[7] += calculate_ssim(pred_both[t], gt_vid[t])

                local_metrics[8] += T

                if self.is_main and idx in vis_indices:
                    sample_dict = {"input": x[0, 0].cpu().clone(), "gt": gt_vid[0].cpu().clone()}
                    sample_dict["pred_fakeE_fakeIR"] = pred[0].cpu().clone()
                    if pred_re is not None:
                        sample_dict["pred_realE_fakeIR"] = pred_re[0].cpu().clone()
                    if pred_ri is not None:
                        sample_dict["pred_fakeE_realIR"] = pred_ri[0].cpu().clone()
                    if pred_both is not None:
                        sample_dict["pred_realE_realIR"] = pred_both[0].cpu().clone()
                    if has_event:
                        sample_dict["event"] = make_event_vis(event[0, 0]).cpu().clone()
                    if has_ir:
                        iv = ir[0, 0].cpu().clone()
                        i_min, i_max = iv.min(), iv.max()
                        if (i_max - i_min) > 1e-6:
                            iv = (iv - i_min) / (i_max - i_min)
                        sample_dict["ir"] = iv.repeat(3, 1, 1)
                    vis_results[idx] = sample_dict

        if self.distributed:
            dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)

        global_m = local_metrics.cpu().numpy()
        count = max(1.0, global_m[8])

        avg_psnrs = [global_m[i] / count if global_m[i] > 0 else 0.0 for i in [0, 2, 4, 6]]
        avg_ssims = [global_m[i] / count if global_m[i] > 0 else 0.0 for i in [1, 3, 5, 7]]

        names = ["fakeE_fakeIR", "realE_fakeIR", "fakeE_realIR", "realE_realIR"]
        metrics_dict = {}
        for i, name in enumerate(names):
            metrics_dict[f"{name}_psnr"] = float(avg_psnrs[i])
            metrics_dict[f"{name}_ssim"] = float(avg_ssims[i])

        if self.is_main:
            main_psnr = next((p for p in avg_psnrs if p > 0), 0.0)
            print(f"[{prefix.upper()}] Epoch {epoch} | PSNR: {main_psnr:.4f} | Count: {int(count)}")

            log_dict = {}
            for i, name in enumerate(names):
                if avg_psnrs[i] > 0:
                    log_dict[f"{prefix}/{name}_psnr"] = avg_psnrs[i]
                    log_dict[f"{prefix}/{name}_ssim"] = avg_ssims[i]
            self.logger.log(log_dict, step=int(epoch))
            self._save_vis_samples(vis_results, epoch, prefix)

        if self.distributed:
            dist.barrier()

        if self.is_main:
            self._save_metrics_to_file(epoch, prefix, metrics_dict)

        main_psnr = next((p for p in avg_psnrs if p > 0), 0.0)
        main_ssim = next((s for s in avg_ssims if s > 0), 0.0)
        return main_psnr, main_ssim

    def save_ckpt(self, epoch, metric=None):
        """Save latest and best checkpoints."""
        assert self.is_main
        latest_path = os.path.join(self.ckpt_dir, "latest.pth")
        save_checkpoint(latest_path, self.base_model, self.optimizer, epoch,
                        self.global_step, scheduler=self.scheduler, ema=self.ema)

        if epoch % 5 == 0:
            ms_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}_milestone.pth")
            shutil.copyfile(latest_path, ms_path)

        if metric is not None:
            if len(self.best_ckpts) < self.max_keep or metric > self.best_ckpts[-1][0]:
                filename = f"epoch_{epoch}_psnr{metric:.4f}.pth"
                best_path = os.path.join(self.ckpt_dir, filename)
                shutil.copyfile(latest_path, best_path)
                print(f"[Save] Best: epoch_{epoch} PSNR={metric:.4f}")
                self.best_ckpts.append((metric, epoch, best_path))
                self.best_ckpts.sort(key=lambda x: x[0], reverse=True)
                if len(self.best_ckpts) > self.max_keep:
                    _, _, worst_path = self.best_ckpts.pop()
                    if os.path.exists(worst_path):
                        try:
                            os.remove(worst_path)
                        except OSError:
                            pass
            else:
                print(f"[Save] Epoch {epoch} PSNR={metric:.4f}")
        else:
            print(f"[Save] Epoch {epoch}")

    def train(self, train_loader, val_loader):
        """Main training loop."""
        if self.is_main:
            print("[Notice] Start Training...")
            print(f"[Notice] Distributed={self.distributed}, Rank={self.rank}")

        if self.cfg["exp"].get("eval_before_train", False):
            if self.distributed:
                dist.barrier()
            self.evaluate(val_loader, epoch=-1, prefix="zeroshot")
            if self.distributed:
                dist.barrier()

        for epoch in range(self.start_epoch, self.cfg["train"]["num_epochs"]):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                try:
                    train_loader.sampler.set_epoch(epoch)
                except Exception:
                    pass

            if self.is_main:
                print(f"\n[Epoch {epoch}]")

            self.train_one_epoch(train_loader, epoch)
            if self.distributed:
                dist.barrier()

            if epoch % self.cfg["exp"].get("eval_interval", 1) == 0:
                if self.is_main:
                    print("[Eval] Running evaluation...")
                val_psnr, _ = self.evaluate(val_loader, epoch, prefix="val")

                if self.is_main:
                    self.save_ckpt(epoch, metric=val_psnr)
            elif self.is_main:
                self.save_ckpt(epoch, metric=None)

            if self.distributed:
                dist.barrier()

        if self.is_main:
            print("[Notice] Training completed!")
