"""Config schema for the robot jersey-colour conditioning experiment.

The embedding here is a fixed-size (teammate_rgb, opponent_rgb) pair
computed straight from the manifest at data-loading time (see
team_conditioned_detection.data.robot_jersey), not a frozen external text embedder, so
there's no EmbeddingConfig/provider/centering concept needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelConfig:
    # Registry key, see team_conditioned_detection.models.registry: "yolo", "rtdetr",
    # "fasterrcnn", or "fcos". The training loop only depends on the
    # Detector interface (see team_conditioned_detection/models/base.py).
    architecture: str = "yolo"
    # Meaning depends on architecture: an Ultralytics detection yaml stem
    # for "yolo" (e.g. "yolo26n", "yolo11n", "yolov8n") or "rtdetr" (e.g.
    # "rtdetr-l"); a torchvision backbone name for "fasterrcnn"/"fcos"
    # (e.g. "resnet50", "mobilenet_v3_large" - fasterrcnn only).
    variant: str = "yolo26n"
    pretrained: bool = False
    # Master switch for the whole feature this repo exists to test: with
    # use_film=False the conditioning signal is computed but never
    # touches the network, giving a true apples-to-apples baseline.
    use_film: bool = True
    film_hidden_dim: int = 256
    # "yolo"/"rtdetr" only: which layer indices (into the Ultralytics
    # model's internal nn.Sequential) get FiLM-modulated. None = auto: the
    # exact layers feeding the detection head (late - after backbone+neck
    # fusion). Pass explicit indices to condition on raw backbone features
    # instead (early - before cross-scale fusion); see the architecture's
    # module docstring for its layer-by-layer map. Ignored when use_film
    # is False or architecture is "fasterrcnn"/"fcos" (see film_early).
    film_layer_indices: list[int] | None = None
    # "fasterrcnn"/"fcos" only: False (default) = FiLM on the FPN outputs
    # feeding the head (late, uniform 256ch). True = FiLM on the backbone
    # body's per-stage outputs before the FPN fuses them (early).
    film_early: bool = False
    # "yolo"/"rtdetr" only: which mechanism generates the modulation at
    # each film_layer_indices - "film" (default), "cross_attention",
    # "conditional_batchnorm", "conditional_layernorm", "adain", or
    # "gated". All share the same injection points and hook mechanics -
    # see team_conditioned_detection/models/conditioners.py.
    conditioning_method: str = "film"
    # Weight on any conditioner's auxiliary loss, added to the detection
    # loss when a conditioner's pop_auxiliary_loss() returns non-None (no
    # current mechanism does; see Conditioner.pop_auxiliary_loss docstring).
    # Also reused by team_conditioned_detection.models.robot_role's two-stage detectors as
    # the role-classification loss weight (same idea: a secondary loss on
    # top of detection) - set to 0.0 there for detector-only warm-start
    # training, see robot_jersey_role_head_seq_phase1.yaml.
    auxiliary_loss_weight: float = 0.1
    # team_conditioned_detection.models.robot_role only: load a previous run's full
    # state_dict before training starts (e.g. a detector-only warm-start
    # checkpoint - see freeze_detector below). None = random init.
    init_checkpoint: str | None = None
    # team_conditioned_detection.models.robot_role only: freeze detector.* parameters after
    # construction/init_checkpoint loading, so the optimizer (which only
    # ever collects params with requires_grad=True, see
    # engine/trainer.py's build_optimizer) only trains the role head -
    # for sequential training (detector converges first, then the role
    # head trains on top of frozen, already-good detection features).
    freeze_detector: bool = False
    # team_conditioned_detection.models.robot_role only: spatial resolution of the RoIAlign
    # crop the role head pools from (both the conditioned deep feature and
    # the raw-pixel colour-distance feature). A jersey typically covers
    # only a small torso patch of a robot's full bounding box (confirmed
    # visually) - too coarse a grid blends that patch into the
    # surrounding grey/dark chassis before the classifier ever sees it.
    roi_output_size: int = 4
    # team_conditioned_detection.models.robot_role only: which features feed the role
    # head's final classifier - the FiLM/cross-attention-conditioned deep
    # feature, the hand-crafted colour-distance feature, or (default) both.
    # At least one must stay True. Ablation knobs: distance-only isolates
    # whether the hand-crafted nearest-colour feature alone explains most
    # of the accuracy; deep-only isolates whether the learned conditioned
    # pathway needs the hand-crafted feature's help at all.
    role_head_use_deep_feature: bool = True
    role_head_use_distance_feature: bool = True
    # team_conditioned_detection.train_standalone_role_classifier only: spatial size of the
    # RoIAlign'd raw-pixel crop fed to the standalone classifier's own
    # lightweight CNN (see StandaloneRoleClassifier - this crop is on raw
    # pixels, unrelated to roi_output_size above, which pools a detector's
    # internal feature map instead).
    crop_size: int = 64
    # team_conditioned_detection.train_standalone_role_classifier only: channel width of
    # the standalone classifier's own dedicated feature (LightweightCropEncoder's
    # output) - kept small deliberately, see standalone_role_classifier.py.
    encoder_channels: int = 128


@dataclass
class TrainConfig:
    # Upper bound on training length - early_stopping_patience is what
    # actually decides how long a run goes, so this just needs to be
    # generous enough not to be the binding constraint.
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "adamw"
    device: str = "cpu"
    seed: int = 42
    eval_interval: int = 1
    conf_thres: float = 0.25
    output_dir: str = "outputs/run"
    checkpoint_metric: str = "map_50"
    # "max" for metrics where higher is better (map_50 and friends - the
    # default). "min" for loss-like metrics (e.g. checkpoint_metric:
    # "loss", val loss computed in engine/trainer.py's evaluate()) - use
    # this when the "max" metric is too sparse/thresholded to be a
    # reliable early-stopping signal, e.g. map_50 can sit at a hard 0.0
    # for many epochs on a small/hard multi-instance task even while the
    # model is genuinely improving.
    checkpoint_metric_mode: str = "max"
    # Early stopping normally just watches checkpoint_metric (default:
    # unset, None). Set this to watch several val metrics independently
    # instead, all compared in checkpoint_metric_mode's direction -
    # training only stops once EVERY listed metric has gone
    # early_stopping_patience evals without improving, not just their sum.
    # Needed when checkpoint_metric bundles component losses that converge
    # at different rates (e.g. robot_role.py's "loss" = detection_loss
    # (fast) + role_loss (slow) - stopping on the combined total cuts the
    # slower one off as soon as the faster one's plateau makes the *sum*
    # look stalled, even while the slow one is still improving underneath).
    early_stopping_metrics: list[str] | None = None
    # Per-metric direction, parallel to early_stopping_metrics ("max"/
    # "min" each). None (default) = every watched metric uses
    # checkpoint_metric_mode - only needed when the watched metrics don't
    # all point the same way, e.g. watching role_acc (higher is better)
    # alongside role_loss/detection_loss (lower is better) in the same run.
    early_stopping_modes: list[str] | None = None
    log_interval: int = 10
    # Stop once every metric in early_stopping_metrics (defaults to just
    # checkpoint_metric) hasn't improved (by min_delta) for this many
    # *eval* epochs (i.e. eval_interval * early_stopping_patience training
    # epochs). None disables early stopping entirely.
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 1e-4


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


@dataclass
class RobotDataConfig:
    train_manifest: str
    val_manifest: str
    test_manifest: str | None = None
    img_size: int = 640
    augment: bool = True
    num_workers: int = 2
    # Ablation: replace the injected (teammate_rgb, opponent_rgb) with a
    # fresh random RGB pair, decoupled from the actual jersey colours -
    # ground-truth labels stay correct, only the conditioning signal
    # becomes uninformative. Tests whether *training* can still reach a
    # non-trivial role_acc without any real colour signal to learn from
    # (the training-time complement to the swap test run at eval time via
    # scripts/evaluate_role_conditioning.py). See RobotJerseyDataset.
    wrong_conditioning: bool = False


@dataclass
class RobotConfig:
    name: str = "robot_jersey_run"
    data: RobotDataConfig = field(default_factory=RobotDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_robot_config(path: str | Path) -> RobotConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(RobotConfig, raw)


def save_robot_config(config: RobotConfig, path: str | Path) -> None:
    def _to_dict(obj):
        if is_dataclass(obj):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
        return obj

    with open(path, "w") as f:
        yaml.safe_dump(_to_dict(config), f, sort_keys=False)
