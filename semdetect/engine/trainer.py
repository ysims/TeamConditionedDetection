"""Generic train/eval loop. Deliberately doesn't know anything about YOLO,
FiLM, or balls: it only talks to the semdetect.models.base.Detector
interface and to plain batch dicts, so a different architecture registered
under a different name (semdetect.models.registry.register_model) or a
different dataset (same manifest schema, different images/descriptions)
both plug straight in.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from semdetect.config import Config, save_config
from semdetect.data.dataset import DetectionDataset, collate_fn
from semdetect.data.embedders import build_text_embedder
from semdetect.engine.metrics import DetectionMetrics, batch_to_targets
from semdetect.models.registry import build_model
from semdetect.utils.seed import set_seed


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
    for batch in loader:
        batch = move_batch(batch, device)
        preds = model.predict(batch["img"], batch.get("embedding"), conf_thres=conf_thres)
        targets = batch_to_targets(batch, img_size)
        metrics.update(preds, targets)
    return metrics.compute()


def build_dataloader(config: Config, split: str, text_embedder) -> DataLoader:
    manifest = {"train": config.data.train_manifest, "val": config.data.val_manifest, "test": config.data.test_manifest}[split]
    dataset = DetectionDataset(
        manifest_path=manifest,
        descriptions_csv=config.data.descriptions_csv,
        class_names=config.data.class_names,
        text_embedder=text_embedder,
        description_field=config.data.description_field,
        description_mode=config.data.description_mode,
        img_size=config.data.img_size,
        augment=config.data.augment and split == "train",
        seed=config.train.seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=(split == "train"),
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )


def build_optimizer(model, config: Config):
    params = [p for p in model.parameters() if p.requires_grad]
    if config.train.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.train.lr, weight_decay=config.train.weight_decay)
    if config.train.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.train.lr, weight_decay=config.train.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{config.train.optimizer}'")


def run_training(config: Config) -> dict[str, float]:
    set_seed(config.train.seed)
    device = config.train.device

    output_dir = Path(config.train.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    text_embedder = build_text_embedder(config.embedding)

    train_loader = build_dataloader(config, "train", text_embedder)
    val_loader = build_dataloader(config, "val", text_embedder)

    model = build_model(
        config.model.architecture,
        num_classes=len(config.data.class_names),
        class_names=config.data.class_names,
        variant=config.model.variant,
        img_size=config.data.img_size,
        pretrained=config.model.pretrained,
        use_film=config.model.use_film,
        embed_dim=text_embedder.embed_dim,
        film_hidden_dim=config.model.film_hidden_dim,
        film_layer_indices=config.model.film_layer_indices,
        film_early=config.model.film_early,
    ).to(device)
    if hasattr(model, "configure"):
        model.configure(epochs=config.train.epochs)

    optimizer = build_optimizer(model, config)

    history_path = output_dir / "metrics" / "history.csv"
    history_rows: list[dict] = []
    best_score = -1.0
    best_metric_name = config.train.checkpoint_metric
    patience = config.train.early_stopping_patience
    min_delta = config.train.early_stopping_min_delta
    evals_without_improvement = 0
    stopped_early_at = None

    for epoch in range(1, config.train.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config.train.log_interval, epoch)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

        if epoch % config.train.eval_interval == 0 or epoch == config.train.epochs:
            val_metrics = evaluate(model, val_loader, device, config.data.img_size, config.data.class_names, config.train.conf_thres)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            with open(output_dir / "metrics" / f"epoch_{epoch:04d}.json", "w") as f:
                json.dump({"epoch": epoch, "train": train_metrics, "val": val_metrics}, f, indent=2)

            score = val_metrics.get(best_metric_name, 0.0)
            improved = score > best_score + min_delta
            print(
                f"epoch {epoch}: {best_metric_name}={score:.4f} (best so far {best_score:.4f})"
                + ("" if patience is None else f" [{evals_without_improvement if not improved else 0}/{patience} evals without improvement]")
            )
            if improved:
                best_score = score
                evals_without_improvement = 0
                torch.save({"model": model.state_dict(), "config": row}, output_dir / "checkpoints" / "best.pt")
            else:
                evals_without_improvement += 1

        history_rows.append(row)
        torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / "checkpoints" / "last.pt")
        _write_history_csv(history_path, history_rows)

        if patience is not None and evals_without_improvement >= patience:
            stopped_early_at = epoch
            print(f"Early stopping: no {best_metric_name} improvement > {min_delta} in {patience} eval(s), stopping at epoch {epoch}")
            break

    final_summary = {
        "best_" + best_metric_name: best_score,
        "epochs_configured": config.train.epochs,
        "epochs_run": stopped_early_at or config.train.epochs,
        "stopped_early": stopped_early_at is not None,
    }

    if config.data.test_manifest:
        best_ckpt = output_dir / "checkpoints" / "best.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        test_loader = build_dataloader(config, "test", text_embedder)
        test_metrics = evaluate(model, test_loader, device, config.data.img_size, config.data.class_names, config.train.conf_thres)
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
