#!/usr/bin/env python3
"""Training entrypoint for the robot jersey-colour conditioning experiment.

    python -m team_conditioned_detection.train_robots --config configs/robot_jersey_film.yaml
    python -m team_conditioned_detection.train_robots --config configs/robot_jersey_baseline.yaml

Parallel to team_conditioned_detection/train.py (ball descriptions), not a variant of it -
see team_conditioned_detection/robot_config.py and team_conditioned_detection/data/robot_jersey.py for what's
different. Everything else (the actual epoch loop, checkpointing, early
stopping, metrics) is the exact same code as the ball pipeline, via
team_conditioned_detection.engine.trainer.run_training_loop.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Imports for registration side-effects (@register_model).
import team_conditioned_detection.models.robot_role  # noqa: F401
import team_conditioned_detection.models.rtdetr_film  # noqa: F401
import team_conditioned_detection.models.torchvision_film  # noqa: F401
import team_conditioned_detection.models.yolo_film  # noqa: F401
from team_conditioned_detection.data.robot_jersey import EMBED_DIM, ROLE_NAMES, RobotJerseyDataset, robot_collate_fn, worker_init_fn
from team_conditioned_detection.engine.trainer import build_optimizer, run_training_loop
from team_conditioned_detection.models.registry import build_model
from team_conditioned_detection.robot_config import RobotConfig, load_robot_config, save_robot_config
from team_conditioned_detection.utils.seed import set_seed


def build_robot_dataloader(config: RobotConfig, split: str) -> DataLoader:
    manifest = {"train": config.data.train_manifest, "val": config.data.val_manifest, "test": config.data.test_manifest}[split]
    dataset = RobotJerseyDataset(
        manifest_path=manifest,
        img_size=config.data.img_size,
        augment=config.data.augment and split == "train",
        # Randomized role<->colour assignment only on the split the model
        # trains on - val/test get one fixed assignment per image so
        # metrics are comparable epoch to epoch, see robot_jersey.py.
        randomize_roles=(split == "train"),
        seed=config.train.seed,
        wrong_conditioning=config.data.wrong_conditioning,
    )
    num_workers = config.data.num_workers
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=robot_collate_fn,
        drop_last=(split == "train"),
        persistent_workers=num_workers > 0,
        timeout=180 if num_workers > 0 else 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def run_robot_training(config: RobotConfig) -> dict[str, float]:
    set_seed(config.train.seed)
    device = config.train.device

    output_dir = Path(config.train.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    save_robot_config(config, output_dir / "config.yaml")

    train_loader = build_robot_dataloader(config, "train")
    val_loader = build_robot_dataloader(config, "val")
    test_loader = build_robot_dataloader(config, "test") if config.data.test_manifest else None

    model = build_model(
        config.model.architecture,
        num_classes=len(ROLE_NAMES),
        class_names=ROLE_NAMES,
        variant=config.model.variant,
        img_size=config.data.img_size,
        pretrained=config.model.pretrained,
        use_film=config.model.use_film,
        embed_dim=EMBED_DIM,
        film_hidden_dim=config.model.film_hidden_dim,
        film_layer_indices=config.model.film_layer_indices,
        film_early=config.model.film_early,
        conditioning_method=config.model.conditioning_method,
        auxiliary_loss_weight=config.model.auxiliary_loss_weight,
        roi_output_size=config.model.roi_output_size,
        role_head_use_deep_feature=config.model.role_head_use_deep_feature,
        role_head_use_distance_feature=config.model.role_head_use_distance_feature,
    ).to(device)
    if hasattr(model, "configure"):
        model.configure(epochs=config.train.epochs)

    if config.model.init_checkpoint:
        ckpt = torch.load(config.model.init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded weights from {config.model.init_checkpoint}")

    if config.model.freeze_detector:
        # By parameter name rather than a fixed "detector"/"model"
        # attribute - team_conditioned_detection.models.robot_role's two architectures name
        # their underlying network differently (self.detector for the
        # Ultralytics-backed one, self.model for the torchvision-backed
        # one, matching each family's existing convention elsewhere in
        # this codebase) - "everything except role_head" is the one thing
        # true of both, and is what "sequential training" (detector
        # converges first, then only the role head trains on top of
        # frozen features) actually means regardless of architecture.
        for name, p in model.named_parameters():
            if not name.startswith("role_head."):
                p.requires_grad = False
        n_frozen = sum(1 for n, p in model.named_parameters() if not p.requires_grad)
        n_total = sum(1 for _ in model.named_parameters())
        print(f"Froze {n_frozen}/{n_total} parameter tensors outside role_head - only role_head will train")

    optimizer = build_optimizer(model, config)

    return run_training_loop(
        model, optimizer, train_loader, val_loader, test_loader,
        config.train, config.data.img_size, ROLE_NAMES, output_dir, device,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config, see configs/robot_jersey_*.yaml")
    parser.add_argument("--output-dir", default=None, help="Override config.train.output_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.train.epochs")
    parser.add_argument("--device", default=None, help="Override config.train.device, e.g. cuda:0")
    args = parser.parse_args()

    config = load_robot_config(args.config)
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.epochs:
        config.train.epochs = args.epochs
    if args.device:
        config.train.device = args.device

    run_robot_training(config)


if __name__ == "__main__":
    main()
