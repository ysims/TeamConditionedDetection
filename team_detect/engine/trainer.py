"""Generic train/eval loop. Doesn't know which model is underneath: callers
pass a train-one-epoch function and an evaluate function, so the two-stage
detectors and the standalone classifier (see team_detect.train) share the
same epoch loop,
checkpointing, early stopping, and output layout:

    <output_dir>/config.yaml, checkpoints/{best,last}.pt,
    metrics/epoch_NNNN.json, metrics/history.csv, metrics/test.json,
    metrics/final.json
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from team_detect.engine.metrics import DetectionMetrics, batch_to_targets


def move_batch(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_one_epoch(model, loader: DataLoader, optimizer, device: str, log_interval: int, epoch: int) -> dict[str, float]:
    """For any model with compute_loss(batch) -> (loss, {name: value})."""
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

    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avg["epoch_time_sec"] = time.time() - t0
    return avg


@torch.no_grad()
def evaluate_detector(model, loader: DataLoader, device: str, img_size: int, class_names: list[str], conf_thres: float) -> dict[str, float]:
    """mAP/P/R/F1 over predict()'s output, plus every compute_loss component
    averaged over batches (the val losses early stopping watches)."""
    model.eval()
    metrics = DetectionMetrics(class_names, iou_thres=0.5, conf_thres=conf_thres)
    loss_totals: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        preds = model.predict(batch["img"], batch.get("embedding"), conf_thres=conf_thres)
        metrics.update(preds, batch_to_targets(batch, img_size))
        # A second forward pass - compute_loss needs GT during its own forward
        # (see Detector.compute_loss), so it can't reuse predict()'s pass.
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


class EarlyStopping:
    """Tracks the checkpoint metric (which decides best.pt) and, separately,
    every early_stopping_metrics entry - training stops only once all of
    those have gone `patience` evals without improving by min_delta.
    """

    def __init__(self, train_config):
        self.metric = train_config.checkpoint_metric
        self.mode = train_config.checkpoint_metric_mode
        if self.mode not in ("max", "min"):
            raise ValueError(f"Unknown checkpoint_metric_mode {self.mode!r} (expected 'max' or 'min')")
        self.patience = train_config.early_stopping_patience
        self.min_delta = train_config.early_stopping_min_delta
        self.best_score = self._worst(self.mode)

        names = train_config.early_stopping_metrics or [self.metric]
        modes = train_config.early_stopping_modes or [self.mode] * len(names)
        if len(modes) != len(names):
            raise ValueError(f"early_stopping_modes (len {len(modes)}) must match early_stopping_metrics (len {len(names)})")
        self.watch_modes = dict(zip(names, modes))
        self.watch_best = {n: self._worst(m) for n, m in self.watch_modes.items()}
        self.evals_without_improvement = {n: 0 for n in names}

    @staticmethod
    def _worst(mode: str) -> float:
        return float("-inf") if mode == "max" else float("inf")

    def _improved(self, score: float, best: float, mode: str) -> bool:
        return score > best + self.min_delta if mode == "max" else score < best - self.min_delta

    def update(self, val_metrics: dict[str, float]) -> bool:
        """Returns True if the checkpoint metric improved (save best.pt)."""
        score = val_metrics.get(self.metric, 0.0)
        improved = self._improved(score, self.best_score, self.mode)
        if improved:
            self.best_score = score
        for name, mode in self.watch_modes.items():
            s = val_metrics.get(name, 0.0)
            if self._improved(s, self.watch_best[name], mode):
                self.watch_best[name] = s
                self.evals_without_improvement[name] = 0
            else:
                self.evals_without_improvement[name] += 1
        status = ", ".join(f"{n}={c}/{self.patience}" for n, c in self.evals_without_improvement.items())
        print(f"{self.metric}={score:.4f} (best so far {self.best_score:.4f})" + ("" if self.patience is None else f" [{status}]"))
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and all(c >= self.patience for c in self.evals_without_improvement.values())


def fit(
    model,
    train_epoch_fn: Callable[[int], dict[str, float]],
    eval_fn: Callable[[DataLoader], dict[str, float]],
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    train_config,
    output_dir: Path,
    device: str,
) -> dict:
    """train_epoch_fn(epoch) trains one epoch and returns its metrics;
    eval_fn(loader) evaluates the model on a loader. Saves best.pt by the
    checkpoint metric, stops early, then evaluates best.pt on test_loader.
    """
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    history_rows: list[dict] = []
    stopper = EarlyStopping(train_config)
    stopped_early_at = None

    for epoch in range(1, train_config.epochs + 1):
        train_metrics = train_epoch_fn(epoch)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

        if epoch % train_config.eval_interval == 0 or epoch == train_config.epochs:
            val_metrics = eval_fn(val_loader)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            with open(output_dir / "metrics" / f"epoch_{epoch:04d}.json", "w") as f:
                json.dump({"epoch": epoch, "train": train_metrics, "val": val_metrics}, f, indent=2)
            print(f"epoch {epoch}: ", end="")
            if stopper.update(val_metrics):
                torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / "checkpoints" / "best.pt")

        history_rows.append(row)
        torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / "checkpoints" / "last.pt")
        _write_history_csv(output_dir / "metrics" / "history.csv", history_rows)

        if stopper.should_stop:
            stopped_early_at = epoch
            print(f"Early stopping: no improvement in any of {list(stopper.watch_modes)} for {stopper.patience} eval(s), stopping at epoch {epoch}")
            break

    final_summary = {
        "best_" + stopper.metric: stopper.best_score,
        "epochs_configured": train_config.epochs,
        "epochs_run": stopped_early_at or train_config.epochs,
        "stopped_early": stopped_early_at is not None,
    }

    if test_loader is not None:
        best_ckpt = output_dir / "checkpoints" / "best.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        test_metrics = eval_fn(test_loader)
        with open(output_dir / "metrics" / "test.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        final_summary["test"] = test_metrics
        print(f"Test set: {test_metrics}")

    with open(output_dir / "metrics" / "final.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"Training complete. best {stopper.metric}={stopper.best_score:.4f}. Outputs in {output_dir}")
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
