#!/usr/bin/env python3
"""Training entrypoint for every model:

    python -m team_detect.train --config configs/fasterrcnn_film.yaml --seed 43

The two-stage detectors train on detection loss + role loss and are
validated on mAP plus their val losses; the standalone classifier trains on
role loss alone, on ground-truth boxes. Both share team_detect.engine.trainer.fit.
scripts/run_experiments.py runs every experiment and seed through this.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from team_detect.data.robot_jersey import ROLE_NAMES, build_dataloader
from team_detect.engine.metrics import norm_cxcywh_to_xyxy
from team_detect.engine.trainer import build_optimizer, evaluate_detector, fit, move_batch, train_one_epoch
from team_detect.models.factory import STANDALONE, build_model
from team_detect.robot_config import RobotConfig, load_robot_config, save_robot_config
from team_detect.utils.seed import set_seed


def _standalone_loss(model, batch: dict, img_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], img_size)
    labels = batch["cls"].long()
    logits = model(batch["img"], boxes_xyxy, batch["batch_idx"].long(), batch["embedding"])
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc, labels.shape[0]


def _standalone_train_epoch(model, loader: DataLoader, optimizer, device: str, img_size: int, log_interval: int, epoch: int) -> dict[str, float]:
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        loss, acc, n = _standalone_loss(model, batch, img_size)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * n
        total_acc += acc.item() * n
        total_n += n
        if log_interval and step % log_interval == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} role_loss={loss.item():.4f} role_acc={acc.item():.4f}")
    return {"role_loss": total_loss / max(total_n, 1), "role_acc": total_acc / max(total_n, 1)}


@torch.no_grad()
def _standalone_evaluate(model, loader: DataLoader, device: str, img_size: int) -> dict[str, float]:
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for batch in loader:
        loss, acc, n = _standalone_loss(model, move_batch(batch, device), img_size)
        total_loss += loss.item() * n
        total_acc += acc.item() * n
        total_n += n
    return {"role_loss": total_loss / max(total_n, 1), "role_acc": total_acc / max(total_n, 1)}


def run_training(config: RobotConfig) -> dict:
    set_seed(config.train.seed)
    device = config.train.device
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_robot_config(config, output_dir / "config.yaml")

    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")
    test_loader = build_dataloader(config, "test") if config.data.test_manifest else None

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    t, img_size = config.train, config.data.img_size

    if config.model.architecture == STANDALONE:
        train_epoch_fn = lambda epoch: _standalone_train_epoch(model, train_loader, optimizer, device, img_size, t.log_interval, epoch)
        eval_fn = lambda loader: _standalone_evaluate(model, loader, device, img_size)
    else:
        model.configure(epochs=t.epochs)
        train_epoch_fn = lambda epoch: train_one_epoch(model, train_loader, optimizer, device, t.log_interval, epoch)
        eval_fn = lambda loader: evaluate_detector(model, loader, device, img_size, ROLE_NAMES, t.conf_thres)

    return fit(model, train_epoch_fn, eval_fn, val_loader, test_loader, t, output_dir, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a YAML config, see configs/")
    parser.add_argument("--seed", type=int, default=None, help="Override config.train.seed")
    parser.add_argument("--output-dir", default=None, help="Override config.train.output_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.train.epochs")
    parser.add_argument("--device", default=None, help="Override config.train.device, e.g. cuda:0")
    args = parser.parse_args()

    config = load_robot_config(args.config)
    if args.seed is not None:
        config.train.seed = args.seed
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.epochs:
        config.train.epochs = args.epochs
    if args.device:
        config.train.device = args.device

    run_training(config)


if __name__ == "__main__":
    main()
