"""Config schema for a training run.

Everything a run needs - data, model architecture, FiLM on/off, optimisation -
lives in one YAML file. Keeping this as plain dataclasses (rather than e.g.
pydantic) avoids a dependency and keeps the fields self-documenting.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    train_manifest: str
    val_manifest: str
    test_manifest: str | None = None
    descriptions_csv: str = "data/ball_descriptions.csv"
    # Which descriptions_csv column to embed. "Description" is the free-text
    # sentence; "Colours" is the shorter comma-separated colour list the
    # user flagged as possibly more useful signal for FiLM.
    description_field: str = "Description"
    # Ablation: "correct" is the real instance_id -> description mapping.
    # "shuffled" deterministically remaps every instance to a *different*
    # instance's description (a fixed derangement, seeded so it's stable
    # across runs) - i.e. FiLM gets a real, well-formed embedding, just of
    # the wrong ball. This isolates "does the extra FiLM capacity help
    # regardless of content" from "does correct semantic content matter";
    # see also model.use_film=False (no embedding at all) and
    # embedding.provider="random" (right shape, no text content).
    description_mode: str = "correct"
    class_names: list[str] = field(default_factory=lambda: ["ball"])
    img_size: int = 640
    augment: bool = True
    num_workers: int = 2


@dataclass
class EmbeddingConfig:
    # "clip" (open_clip), "bert" or "e5" (transformers, mean-pooled), or
    # "random" (ablation: a fixed per-text random vector - right shape and
    # per-instance consistency, zero semantic content). See
    # semdetect/data/embedders.py.
    provider: str = "clip"
    # Meaning depends on provider: open_clip model name for "clip"
    # (default pairs with pretrained="openai"), a HuggingFace model id for
    # "bert"/"e5" (e.g. "bert-base-uncased", "intfloat/e5-base-v2").
    # Unused for "random".
    model_name: str = "ViT-B-32-quickgelu"
    pretrained: str = "openai"  # clip only
    device: str = "cpu"
    random_dim: int = 512  # provider="random" only


@dataclass
class ModelConfig:
    # Registry key, see semdetect.models.registry: "yolo", "rtdetr",
    # "fasterrcnn", or "fcos". The training loop only depends on the
    # Detector interface (see semdetect/models/base.py), so more can be
    # added without touching train.py.
    architecture: str = "yolo"
    # Meaning depends on architecture: an Ultralytics detection yaml stem
    # for "yolo" (e.g. "yolo26n", "yolo11n", "yolov8n") or "rtdetr" (e.g.
    # "rtdetr-l"); a torchvision backbone name for "fasterrcnn"/"fcos"
    # (e.g. "resnet50", "mobilenet_v3_large" - fasterrcnn only).
    variant: str = "yolo26n"
    pretrained: bool = False
    # Master switch for the whole feature this repo exists to test: with
    # use_film=False the semantic embedding is computed but never touches
    # the network, giving a true apples-to-apples baseline.
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
    log_interval: int = 10
    # Stop once checkpoint_metric hasn't improved (by min_delta) for this
    # many *eval* epochs (i.e. eval_interval * early_stopping_patience
    # training epochs). None disables early stopping entirely.
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 1e-4


@dataclass
class Config:
    name: str = "semdetect_run"
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


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


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(Config, raw)


def save_config(config: Config, path: str | Path) -> None:
    def _to_dict(obj):
        if is_dataclass(obj):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
        return obj

    with open(path, "w") as f:
        yaml.safe_dump(_to_dict(config), f, sort_keys=False)
