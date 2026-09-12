"""Generic train/eval loop. Deliberately doesn't know anything about YOLO,
FiLM, or the specific conditioning signal: it only talks to the
team_conditioned_detection.models.base.Detector interface and to plain batch dicts, so a
different architecture registered under a different name
(team_conditioned_detection.models.registry.register_model) or a different dataset both
plug straight in.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from team_conditioned_detection.engine.metrics import DetectionMetrics, batch_to_targets


def move_batch(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_one_epoch(model, loader: DataLoader, optimizer, device: str, log_interval: int, epoch: int) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        loss, loss_dict = model.compute_loss(batch)
        loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

        if log_interval and step % log_interval == 0:
            comps = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            print(f"  epoch {epoch} step {step}/{len(loader)} {comps}")

    elapsed = time.time() - t0
    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avg["epoch_time_sec"] = elapsed
    return avg


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str, img_size: int, class_names: list[str], conf_thres: float) -> dict[str, float]:
    model.eval()
    metrics = DetectionMetrics(class_names, iou_thres=0.5, conf_thres=conf_thres)
    loss_totals: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        preds = model.predict(batch["img"], batch.get("embedding"), conf_thres=conf_thres)
        targets = batch_to_targets(batch, img_size)
        metrics.update(preds, targets)
        # A second forward pass (predict() already did one) - compute_loss
        # needs GT in the batch and runs its own forward internally (see
        # Detector.compute_loss's docstring), so it can't reuse predict()'s
        # pass. Worth the cost: unlike map_50 (thresholded - can sit at a
        # hard 0.0 for many epochs even while the model is genuinely
        # improving, e.g. early in a from-scratch multi-instance/multi-class
        # run before any prediction clears the IoU/confidence bar), loss is
        # smooth and never floors out, so it's the more reliable early-
        # stopping/checkpoint-selection signal when map is sparse - see
        # TrainConfig.checkpoint_metric_mode. Every component of loss_dict
        # (not just the total) is kept, not just for visibility - a
        # detector with several component losses that converge at
        # different rates (e.g. robot_role.py's detection_loss + role_loss)
        # needs each one individually watchable by early stopping, see
        # TrainConfig.early_stopping_metrics.
        _, loss_dict = model.compute_loss(batch)
        for k, v in loss_dict.items():
            loss_totals[k] = loss_totals.get(k, 0.0) + v
        n_batches += 1
    out = metrics.compute()
    for k, v in loss_totals.items():
        out[k] = v / max(n_batches, 1)
    out["loss"] = out.get("total_loss", 0.0)  # alias - every compute_loss sets "total_loss"
    return out


def build_optimizer(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    if config.train.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.train.lr, weight_decay=config.train.weight_decay)
    if config.train.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.train.lr, weight_decay=config.train.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{config.train.optimizer}'")


def run_training_loop(
    model,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    train_config,
    img_size: int,
    class_names: list[str],
    output_dir: Path,
    device: str,
) -> dict[str, float]:
    """The epoch loop, checkpointing, early stopping, and final test eval -
    identical regardless of what dataset/embedding feeds the model, so
    team_conditioned_detection.train_robots's run_robot_training (jersey-colour
    conditioning) calls this directly rather than duplicating it.
    """
    history_path = output_dir / "metrics" / "history.csv"
    history_rows: list[dict] = []
    best_metric_name = train_config.checkpoint_metric
    metric_mode = train_config.checkpoint_metric_mode
    if metric_mode not in ("max", "min"):
        raise ValueError(f"Unknown checkpoint_metric_mode {metric_mode!r} (expected 'max' or 'min')")
    best_score = float("-inf") if metric_mode == "max" else float("inf")
    patience = train_config.early_stopping_patience
    min_delta = train_config.early_stopping_min_delta
    stopped_early_at = None

    # Checkpointing ("what's the best model so far") always uses the single
    # checkpoint_metric - keeps "best.pt" well-defined. Early stopping can
    # watch a wider set independently (early_stopping_metrics) so a
    # metric that's still improving isn't dragged down by one that's
    # plateaued inside the same combined total - see TrainConfig docstring.
    early_stop_names = train_config.early_stopping_metrics or [best_metric_name]
    early_stop_modes_list = train_config.early_stopping_modes or [metric_mode] * len(early_stop_names)
    if len(early_stop_modes_list) != len(early_stop_names):
        raise ValueError(f"early_stopping_modes (len {len(early_stop_modes_list)}) must match early_stopping_metrics (len {len(early_stop_names)})")
    early_stop_modes = dict(zip(early_stop_names, early_stop_modes_list))
    best_early_stop_scores = {name: (float("-inf") if early_stop_modes[name] == "max" else float("inf")) for name in early_stop_names}
    evals_without_improvement = {name: 0 for name in early_stop_names}

    for epoch in range(1, train_config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, train_config.log_interval, epoch)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

        if epoch % train_config.eval_interval == 0 or epoch == train_config.epochs:
            val_metrics = evaluate(model, val_loader, device, img_size, class_names, train_config.conf_thres)
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
        "epochs_configured": train_config.epochs,
        "epochs_run": stopped_early_at or train_config.epochs,
        "stopped_early": stopped_early_at is not None,
    }

    if test_loader is not None:
        best_ckpt = output_dir / "checkpoints" / "best.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        test_metrics = evaluate(model, test_loader, device, img_size, class_names, train_config.conf_thres)
        with open(output_dir / "metrics" / "test.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        final_summary["test"] = test_metrics
        print(f"Test set ({best_metric_name}={test_metrics.get(best_metric_name, 0.0):.4f}): {test_metrics}")

    with open(output_dir / "metrics" / "final.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"Training complete. best {best_metric_name}={best_score:.4f}. Outputs in {output_dir}")
    return final_summary


def _write_history_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
