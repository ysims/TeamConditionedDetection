# TeamConditionedDetection

Bounding-box object detection conditioned on team/role context, for
multi-instance scenes where identical-class objects must be told apart by
role rather than appearance alone. Motivating case: detecting robots on a
field and telling teammates from opponents by jersey colour, where the two
roles are visually identical except for that colour and the colour itself
changes every frame.

Built so the detection architecture, the conditioning mechanism (FiLM,
cross-attention, conditional norm, AdaIN, gated), and whether conditioning
happens at the whole-detector level or a separate role head are all
swappable independently via config.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
source .venv/bin/activate
```

`pyproject.toml` lists plain `torch`/`torchvision`, so `uv sync` pulls
whatever build PyPI serves for your platform (CUDA-enabled on a machine
with an NVIDIA GPU). On a CPU-only machine, install the CPU wheels instead:
`uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Dataset

`scripts/prepare_robot_jersey_dataset.py` turns one or more NUpbr synthetic
render runs (`raw/*.png` + `meta/*.yaml`, with `robots: [{id,
jersey_colour, bbox}, ...]` per image) into `train.csv` / `val.csv` /
`test.csv` manifests:

```sh
uv run python scripts/prepare_robot_jersey_dataset.py \
    --run-dir /path/to/NUpbr/outputs/run_4 /path/to/NUpbr/outputs/run_5 \
    --output data/manifest_robots
```

Each robot's jersey colour is drawn fresh per frame rather than being a
fixed per-instance property, so there's no colour vocabulary to hold out
across splits the way the old ball-description approach held out object
instance ids - images are partitioned randomly instead, with each image's
robots kept together in one split. The renderer guarantees at most 2
distinct jersey colours per image regardless of robot count, which is
exactly what the role-assignment scheme in
`team_conditioned_detection/data/robot_jersey.py` needs: group robots by
colour, randomly assign one group teammate and the other opponent.

## Training

Two architecture families, both driven by the same config schema
(`team_conditioned_detection/robot_config.py`) and epoch loop
(`team_conditioned_detection/engine/trainer.py`):

```sh
# Two-stage: unconditioned detector + a separate FiLM/cross-attention-
# conditioned role head (see "Architectures" below for why).
uv run python -m team_conditioned_detection.train_robots --config configs/robot_jersey_fasterrcnn_role_head.yaml

# Single-stage: the whole detector is conditioned on (teammate_rgb, opponent_rgb).
uv run python -m team_conditioned_detection.train_robots --config configs/robot_jersey_film.yaml
```

Every run trains for up to `train.epochs` (a generous ceiling, default
200) but stops early once every metric in `train.checkpoint_metric` /
`train.early_stopping_metrics` hasn't improved for
`train.early_stopping_patience` eval rounds (default 10 - set to `null`
to disable and always run the full ceiling).

Each run writes to `train.output_dir`:

- `config.yaml` - the resolved config used for the run
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `metrics/epoch_XXXX.json` - train/val losses + mAP/precision/recall per eval epoch
- `metrics/history.csv` - the same, one row per epoch, for quick plotting
- `metrics/final.json` - best validation score, whether/when it stopped early, and test-set metrics

### Architectures

Registered under `team_conditioned_detection.models.registry`
(`model.architecture` in config):

| `model.architecture` | approach | file |
| --- | --- | --- |
| `yolo_robot_role` | two-stage: unconditioned YOLO detector + conditioned role head | `team_conditioned_detection/models/robot_role.py` |
| `rtdetr_robot_role` | two-stage, RT-DETR detector | `team_conditioned_detection/models/robot_role.py` |
| `fasterrcnn_robot_role` | two-stage, Faster R-CNN detector | `team_conditioned_detection/models/robot_role.py` |
| `standalone_role_classifier` | role classification only, on GT/external boxes, no detector | `team_conditioned_detection/models/standalone_role_classifier.py` |
| `yolo` | single-stage: whole detector conditioned | `team_conditioned_detection/models/yolo_film.py` |
| `rtdetr` | single-stage, RT-DETR | `team_conditioned_detection/models/rtdetr_film.py` |
| `fasterrcnn` / `fcos` | single-stage, torchvision | `team_conditioned_detection/models/torchvision_film.py` |

The two-stage (`*_robot_role`) architectures exist because the single-stage
approach has a diagnosed failure mode: conditioning the whole detection
head means box regression is exposed to the same hard-to-learn
conditioning signal as the role decision, and a swap test (same image,
roles reversed) showed predictions barely changing either way. The
two-stage split makes detection completely unconditioned and puts a small
FiLM/cross-attention-conditioned classifier head on each detected box's
RoIAlign'd feature instead - see `robot_role.py`'s module docstring for
the full diagnosis.

`model.conditioning_method` picks the modulation mechanism at each
conditioning point - `film` (default), `cross_attention`,
`conditional_batchnorm`, `conditional_layernorm`, `adain`, or `gated` -
see `team_conditioned_detection/models/conditioners.py`.

Add another architecture by implementing the `Detector` interface
(`team_conditioned_detection/models/base.py`) and `@register_model("name")`;
`engine/trainer.py` only depends on that interface.

### Ablations

- `model.use_film: false` - master switch, no conditioning at all.
- `data.wrong_conditioning: true` - inject a random RGB pair decoupled
  from the actual jersey colours (ground truth stays correct); tests
  whether training can still reach non-trivial role accuracy without a
  real colour signal to learn from.
- `model.roi_output_size` / `role_head_use_deep_feature` /
  `role_head_use_distance_feature` (two-stage only) - isolate whether the
  learned conditioned feature or the hand-crafted colour-distance feature
  is doing the work.
- `model.init_checkpoint` + `model.freeze_detector` - sequential training:
  warm-start from a detector-only run, then freeze everything but the role
  head (see `configs/robot_jersey_role_head_seq_phase1.yaml` /
  `_phase2.yaml`).

### Post-hoc analysis

```sh
# Nearest-colour heuristic vs. the learned role head, plus a swap test
# scored against the original ground truth.
uv run python scripts/evaluate_role_conditioning.py \
    --config outputs/robot_jersey/<run>/config.yaml \
    --checkpoint outputs/robot_jersey/<run>/checkpoints/best.pt

# Swap test on GT boxes scored against the flipped label - does swapping
# the injected colours actually flip the prediction the way it should?
uv run python scripts/evaluate_swap_consistency.py \
    --config outputs/robot_jersey/<run>/config.yaml \
    --checkpoint outputs/robot_jersey/<run>/checkpoints/best.pt

# Pure rule-based baseline, no trained model: classify each GT box by
# nearest-colour distance alone.
uv run python scripts/evaluate_colour_rule_baseline.py --manifest data/manifest_robots/test.csv
```
