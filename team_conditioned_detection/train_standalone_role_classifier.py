#!/usr/bin/env python3
"""Training entrypoint for the standalone (non-shared-backbone) role
classifier - see team_conditioned_detection/models/standalone_role_classifier.py.

Doesn't reuse team_conditioned_detection.engine.trainer.run_training_loop: that loop is
built around the Detector interface (compute_loss/predict, box metrics/
map50), and this model has no detection concept at all - just role
classification on already-known boxes (GT during training, matching how
every other role_acc in this project is computed). So it gets its own
much simpler loop, but mirrors the SAME output layout (checkpoints/,
metrics/history.csv, metrics/epoch_NNNN.json, metrics/test.json,
metrics/final.json) for easy comparison against the shared-backbone role
heads' results.

    python -m team_conditioned_detection.train_standalone_role_classifier --config configs/robot_jersey_standalone_classifier.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from team_conditioned_detection.engine.metrics import norm_cxcywh_to_xyxy
from team_conditioned_detection.engine.trainer import _write_history_csv, build_optimizer
from team_conditioned_detection.models.standalone_role_classifier import StandaloneRoleClassifier
from team_conditioned_detection.robot_config import RobotConfig, load_robot_config, save_robot_config
from team_conditioned_detection.train_robots import build_robot_dataloader
from team_conditioned_detection.utils.seed import set_seed


def _step(model: StandaloneRoleClassifier, batch: dict, device: str, img_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    img = batch["img"].to(device)
    boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], img_size).to(device)
    batch_idx = batch["batch_idx"].long().to(device)
    labels = batch["cls"].long().to(device)
    embedding = batch["embedding"].to(device)
    logits = model(img, boxes_xyxy, batch_idx, embedding)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc, labels.shape[0]


def train_one_epoch(model, loader: DataLoader, optimizer, device: str, img_size: int, log_interval: int, epoch: int) -> dict[str, float]:
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        loss, acc, n = _step(model, batch, device, img_size)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * n
        total_acc += acc.item() * n
        total_n += n
        if step % log_interval == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} role_loss={loss.item():.4f} role_acc={acc.item():.4f}")
    return {"role_loss": total_loss / max(total_n, 1), "role_acc": total_acc / max(total_n, 1), "loss": total_loss / max(total_n, 1)}


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str, img_size: int) -> dict[str, float]:
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for batch in loader:
        loss, acc, n = _step(model, batch, device, img_size)
        total_loss += loss.item() * n
        total_acc += acc.item() * n
        total_n += n
    return {"role_loss": total_loss / max(total_n, 1), "role_acc": total_acc / max(total_n, 1), "loss": total_loss / max(total_n, 1)}


def run_standalone_training(config: RobotConfig) -> dict:
    set_seed(config.train.seed)
    device = config.train.device
    output_dir = Path(config.train.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    save_robot_config(config, output_dir / "config.yaml")

    train_loader = build_robot_dataloader(config, "train")
    val_loader = build_robot_dataloader(config, "val")
    test_loader = build_robot_dataloader(config, "test") if config.data.test_manifest else None

    model = StandaloneRoleClassifier(
        embed_dim=6,
        hidden_dim=config.model.film_hidden_dim,
        conditioning_method=config.model.conditioning_method,
        crop_size=config.model.crop_size,
        encoder_channels=config.model.encoder_channels,
        use_distance_feature=config.model.role_head_use_distance_feature,
    ).to(device)

    optimizer = build_optimizer(model, config)

    history_path = output_dir / "metrics" / "history.csv"
    history_rows: list[dict] = []
    best_metric_name = config.train.checkpoint_metric
    metric_mode = config.train.checkpoint_metric_mode
    best_score = float("-inf") if metric_mode == "max" else float("inf")
    patience = config.train.early_stopping_patience
    min_delta = config.train.early_stopping_min_delta
    stopped_early_at = None

    early_stop_names = config.train.early_stopping_metrics or [best_metric_name]
    early_stop_modes_list = config.train.early_stopping_modes or [metric_mode] * len(early_stop_names)
    early_stop_modes = dict(zip(early_stop_names, early_stop_modes_list))
    best_early_stop_scores = {name: (float("-inf") if early_stop_modes[name] == "max" else float("inf")) for name in early_stop_names}
    evals_without_improvement = {name: 0 for name in early_stop_names}

    for epoch in range(1, config.train.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config.data.img_size, config.train.log_interval, epoch)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

        if epoch % config.train.eval_interval == 0 or epoch == config.train.epochs:
            val_metrics = evaluate(model, val_loader, device, config.data.img_size)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            with open(output_dir / "metrics" / f"epoch_{epoch:04d}.json", "w") as f:
                json.dump({"epoch": epoch, "train": train_metrics, "val": val_metrics}, f, indent=2)

            score = val_metrics.get(best_metric_name, 0.0)
            improved = score > best_score + min_delta if metric_mode == "max" else score < best_score - min_delta
            if improved:
                best_score = score
                torch.save({"model": model.state_dict(), "config": row}, output_dir / "checkpoints" / "best.pt")

            for name in early_stop_names:
                s = val_metrics.get(name, 0.0)
                s_improved = s > best_early_stop_scores[name] + min_delta if early_stop_modes[name] == "max" else s < best_early_stop_scores[name] - min_delta
                if s_improved:
                    best_early_stop_scores[name] = s
                    evals_without_improvement[name] = 0
                else:
                    evals_without_improvement[name] += 1

            watch_status = ", ".join(f"{n}={evals_without_improvement[n]}/{patience}" for n in early_stop_names)
            print(f"epoch {epoch}: {best_metric_name}={score:.4f} (best so far {best_score:.4f})" + ("" if patience is None else f" [{watch_status}]"))

        history_rows.append(row)
        torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / "checkpoints" / "last.pt")
        _write_history_csv(history_path, history_rows)

        if patience is not None and all(v >= patience for v in evals_without_improvement.values()):
            stopped_early_at = epoch
            print(f"Early stopping: no improvement in any of {early_stop_names} for {patience} eval(s), stopping at epoch {epoch}")
            break

    final_summary = {
        "best_" + best_metric_name: best_score,
        "epochs_configured": config.train.epochs,
        "epochs_run": stopped_early_at or config.train.epochs,
        "stopped_early": stopped_early_at is not None,
    }

    if test_loader is not None:
        best_ckpt = output_dir / "checkpoints" / "best.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        test_metrics = evaluate(model, test_loader, device, config.data.img_size)
        with open(output_dir / "metrics" / "test.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        final_summary["test"] = test_metrics
        print(f"Test set ({best_metric_name}={test_metrics.get(best_metric_name, 0.0):.4f}): {test_metrics}")

    with open(output_dir / "metrics" / "final.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"Training complete. best {best_metric_name}={best_score:.4f}. Outputs in {output_dir}")
    return final_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_robot_config(args.config)
    run_standalone_training(config)


if __name__ == "__main__":
    main()
