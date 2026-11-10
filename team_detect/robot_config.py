"""Config schema for the robot jersey-colour role experiments.

The conditioning signal is a fixed-size (teammate_rgb, opponent_rgb) pair
computed straight from the manifest at data-loading time (see
team_detect.data.robot_jersey).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelConfig:
    # See team_detect.models.factory: "fasterrcnn_robot_role",
    # "rtdetr_robot_role", "yolo_robot_role", or "standalone_role_classifier".
    architecture: str = "fasterrcnn_robot_role"
    # An Ultralytics yaml stem for yolo/rtdetr (e.g. "yolo26n", "rtdetr-l");
    # a torchvision backbone name for fasterrcnn ("resnet50"). Unused by the
    # standalone classifier.
    variant: str = "resnet50"
    # Start from pretrained detection weights (COCO for Ultralytics, an
    # ImageNet backbone for Faster R-CNN).
    pretrained: bool = True
    # "film", "film_shift" or "cross_attention" - see team_detect.models.conditioners -
    # or "concat" (role-head detectors only; see RoleHead).
    conditioning_method: str = "film"
    # Hidden width of the conditioner's context MLP (for "concat", of the classifier MLP).
    hidden_dim: int = 256
    # Weight on the role cross-entropy, added to the detector's own loss.
    role_loss_weight: float = 1.0
    # Spatial resolution of the RoIAlign crop the role head pools from (both
    # the conditioned feature and the raw-pixel colour-distance feature). A
    # jersey covers only a small torso patch of the box - too coarse a grid
    # blends it into the surrounding chassis.
    roi_output_size: int = 8
    # Append the min-over-cells RGB distance to each supplied colour to the
    # conditioned feature before the role classifier.
    use_distance_feature: bool = False
    # standalone_role_classifier only: size of the raw-pixel RoIAlign crop
    # fed to its own CNN, and that CNN's output channels.
    crop_size: int = 64
    encoder_channels: int = 128


@dataclass
class TrainConfig:
    # Upper bound on training length - early stopping decides how long a run
    # actually goes.
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "adamw"
    device: str = "cuda"
    seed: int = 42
    eval_interval: int = 1
    conf_thres: float = 0.25
    output_dir: str = "outputs/run"
    # best.pt is chosen by this single val metric.
    checkpoint_metric: str = "loss"
    checkpoint_metric_mode: str = "min"
    # Training stops once EVERY listed val metric has gone
    # early_stopping_patience evals without improving by min_delta - needed
    # because detection_loss (fast) and role_loss (slow) converge at very
    # different rates. None = watch checkpoint_metric only.
    early_stopping_metrics: list[str] | None = None
    # Per-metric direction ("max"/"min"), parallel to early_stopping_metrics.
    # None = every watched metric uses checkpoint_metric_mode.
    early_stopping_modes: list[str] | None = None
    log_interval: int = 20
    # None disables early stopping.
    early_stopping_patience: int | None = 50
    early_stopping_min_delta: float = 1e-2


@dataclass
class RobotDataConfig:
    train_manifest: str
    val_manifest: str
    test_manifest: str | None = None
    img_size: int = 640
    augment: bool = True
    num_workers: int = 2
    # Seed for the fixed teammate/opponent assignment on val/test (training
    # re-randomises it every access). Deliberately independent of
    # train.seed, so every training seed is evaluated on the same test labels.
    eval_seed: int = 0


@dataclass
class RobotConfig:
    name: str = "robot_jersey_run"
    data: RobotDataConfig = field(default_factory=RobotDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Configs saved by older versions of this code (outputs/*/config.yaml) still
# load: renamed keys are mapped, and removed keys are dropped only if they
# hold the value that made them a no-op - anything else means the run used a
# feature this code no longer has.
_RENAMED_KEYS = {
    "model": {
        "film_hidden_dim": "hidden_dim",
        "auxiliary_loss_weight": "role_loss_weight",
        "role_head_use_distance_feature": "use_distance_feature",
    },
}
_REMOVED_KEYS = {
    "model": {
        "use_film": True,
        "film_layer_indices": None,
        "film_early": False,
        "init_checkpoint": None,
        "freeze_detector": False,
        "role_head_use_deep_feature": True,
    },
    "data": {"wrong_conditioning": False},
}


def _upgrade_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    raw = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}
    model = raw.get("model", {})
    if "film_hidden_dim" in model and "role_head_use_distance_feature" not in model:
        # Saved before the distance feature could be switched off - it was always on.
        model["role_head_use_distance_feature"] = True
    for section, renames in _RENAMED_KEYS.items():
        for old, new in renames.items():
            if old in raw.get(section, {}):
                raw[section][new] = raw[section].pop(old)
    for section, removed in _REMOVED_KEYS.items():
        for key, inert_value in removed.items():
            if key in raw.get(section, {}):
                value = raw[section].pop(key)
                if value != inert_value:
                    raise ValueError(f"{section}.{key}={value!r} uses a removed feature (only {inert_value!r} is supported)")
    return raw


def _from_dict(cls, d: dict[str, Any]):
    if not is_dataclass(cls):
        return d
    field_names = {f.name for f in fields(cls)}
    unknown = set(d) - field_names
    if unknown:
        raise ValueError(f"Unknown config key(s) {sorted(unknown)} for {cls.__name__}")

    type_hints = get_type_hints(cls)
    kwargs = {}
    for key, value in d.items():
        field_type = type_hints.get(key)
        if is_dataclass(field_type) and isinstance(value, dict):
            value = _from_dict(field_type, value)
        kwargs[key] = value
    return cls(**kwargs)


def load_robot_config(path: str | Path) -> RobotConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(RobotConfig, _upgrade_legacy(raw))


def config_to_dict(config: RobotConfig) -> dict[str, Any]:
    return asdict(config)


def save_robot_config(config: RobotConfig, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)
