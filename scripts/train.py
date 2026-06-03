"""AMNet training script."""

import os
import sys
import math
import argparse
import yaml
import torch

current_path = os.path.dirname(os.path.abspath(__file__))
amnet_path = os.path.dirname(current_path)
sys.path.insert(0, amnet_path)

from scheduler.build_scheduler import build_scheduler
from datasets.did_video_dataset_rgb import DIDVideoDatasetRGB
from datasets.sde_video_dataset_rgb import SDEVideoDatasetRGB
from datasets.sde_video_dataset_mm import SDEVideoDatasetMM
from datasets.sdsd_video_dataset_rgb import SDSDVideoDatasetRGB
from models.amnet import AMNet
from engine.trainer import Trainer
from utils.seed import set_seed
from utils.logger import Logger


if __name__ == "__main__":
    MAX_TRAIN_SAMPLES = None
    NUM_EPOCHS = None

    print("=" * 60)
    print(" " * 15 + "AMNet Training")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/experiments/amnet/did/rgb_only_ft.yaml",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r"))
    print(f"Config: {args.cfg}")
    print(f"Experiment: {cfg['exp']['experiment_name']}")

    set_seed(cfg["exp"]["seed"])

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("Loading dataset...")

    dataset_map = {
        "did": DIDVideoDatasetRGB,
        "sde_in": SDEVideoDatasetRGB,
        "sde_out": SDEVideoDatasetRGB,
        "sde_in_mm": SDEVideoDatasetMM,
        "sde_out_mm": SDEVideoDatasetMM,
        "sdsd_in": SDSDVideoDatasetRGB,
        "sdsd_out": SDSDVideoDatasetRGB,
    }

    dataset_type = cfg["dataset"].get("type", "did")
    DatasetClass = dataset_map.get(dataset_type, DIDVideoDatasetRGB)
    print(f"Using dataset class: {DatasetClass.__name__} (type={dataset_type})")

    dataset_kwargs = {
        "root": cfg["dataset"]["root"],
        "clip_len": cfg["dataset"]["clip_len"],
        "clip_stride": cfg["dataset"]["clip_stride"],
        "crop_size": tuple(cfg["dataset"]["crop_size"]),
        "base_size": cfg["dataset"].get("base_size", 256),
        "split_json": cfg["dataset"].get("split_json"),
        "split": "train",
    }

    if dataset_type in ("sde_in_mm", "sde_out_mm"):
        dataset_kwargs.update({
            "use_event": cfg["dataset"].get("use_event", True),
            "use_ir": cfg["dataset"].get("use_ir", False),
            "num_bins": cfg["dataset"].get("num_bins", 10),
            "mask_mode": cfg["dataset"].get("mask_mode", "strict"),
            "p_mask_event": cfg["dataset"].get("p_mask_event", 0.0),
            "p_mask_ir": cfg["dataset"].get("p_mask_ir", 0.0),
        })

    train_set = DatasetClass(**dataset_kwargs)

    if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES > 0:
        train_set = torch.utils.data.Subset(train_set, list(range(min(MAX_TRAIN_SAMPLES, len(train_set)))))
        print(f"[Notice] Training set limited to {len(train_set)} samples for quick testing")

    test_kwargs = {
        "root": cfg["dataset"]["root"],
        "clip_len": cfg["dataset"]["clip_len"],
        "clip_stride": cfg["dataset"]["clip_stride"],
        "crop_size": tuple(cfg["dataset"]["crop_size"]),
        "base_size": cfg["dataset"].get("base_size", 256),
        "split_json": cfg["dataset"].get("split_json"),
        "split": "test",
    }

    if dataset_type in ("sde_in_mm", "sde_out_mm"):
        test_kwargs.update({
            "use_event": cfg["dataset"].get("use_event", True),
            "use_ir": cfg["dataset"].get("use_ir", False),
            "num_bins": cfg["dataset"].get("num_bins", 10),
            "mask_mode": cfg["dataset"].get("mask_mode", "strict"),
            "p_mask_event": cfg["dataset"].get("p_mask_event", 0.0),
            "p_mask_ir": cfg["dataset"].get("p_mask_ir", 0.0),
        })

    test_set = DatasetClass(**test_kwargs)

    if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES > 0:
        test_set = torch.utils.data.Subset(test_set, list(range(min(50, len(test_set)))))

    print(f"Train: {len(train_set)} samples | Test: {len(test_set)} samples")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=4,
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
    )

    print("Building model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMNet(**cfg["model"]).to(device)

    train_cfg = cfg.get("train", {})
    if train_cfg.get("freeze_backbone", False):
        frozen_modules = ["encoder", "decoder", "ln_img", "proj_img"]
        frozen_count = 0
        for name in frozen_modules:
            if hasattr(model, name):
                module = getattr(model, name)
                for p in module.parameters():
                    p.requires_grad = False
                frozen_count += sum(p.numel() for p in module.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Freeze] Backbone frozen: {frozen_modules}")
        print(f"[Freeze] Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M ({100*trainable_params/total_params:.1f}%)")

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"])
    )

    accum_steps = cfg["train"].get("accumulate_grad_batches", 1)
    effective_steps = math.ceil(len(train_loader) / accum_steps)
    print(f"Steps/epoch: {len(train_loader)} | LR: {cfg['train']['lr']}")
    scheduler = build_scheduler(optimizer, cfg, effective_steps)

    logger = Logger(cfg)

    trainer = Trainer(cfg, model, optimizer, logger, scheduler)

    if NUM_EPOCHS is not None and NUM_EPOCHS > 0:
        cfg["train"]["num_epochs"] = NUM_EPOCHS
        print(f"[Notice] Quick testing mode: overriding num_epochs to {NUM_EPOCHS}")

    print("=" * 60)
    print(f"Total Epochs: {cfg['train']['num_epochs']}")
    print("=" * 60)

    trainer.train(
        train_loader,
        test_loader,
    )

    print("=" * 60)
    print("Training Completed!")
    print("=" * 60)
