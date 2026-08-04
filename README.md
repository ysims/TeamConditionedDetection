# SemDetect

Bounding-box object detection with an optional FiLM-conditioned semantic
embedding of the target's text description, for objects with high
intra-class appearance variance (e.g. balls that vary wildly in colour
and pattern). Built so the detection architecture, the text embedding
provider, whether/where FiLM is applied, and the dataset are all
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

`scripts/prepare_nupbr_dataset.py` turns one or more NUpbr synthetic
render runs (`raw/*.png` + `meta/*.yaml`, one target-object bbox per
image) plus a descriptions CSV (`data/ball_descriptions.csv`: one row
per object instance id, with `Description` and `Colours` text columns)
into `train.csv` / `val.csv` / `test.csv` manifests:

```sh
uv run python scripts/prepare_nupbr_dataset.py \
    --run-dir /path/to/NUpbr/outputs/run_1 /path/to/NUpbr/outputs/run_2 /path/to/NUpbr/outputs/run_3 \
    --descriptions-csv data/ball_descriptions.csv \
    --object-key ball --class-name ball \
    --output data/manifest
```

`--run-dir` takes one or more runs and pools them before splitting, so a
later run that introduces new object instances (e.g. `run_2` adding new
ball ids) merges straight into the same dataset - see the script's
docstring. Instances present in a run but missing a description row are
skipped with a warning (not silently dropped-but-uncounted).

The split holds out whole object instance ids (e.g. all images of
`ball_007`) for val/test rather than splitting randomly, so val/test
contain appearances the model never saw during training - the split
that actually tests whether the semantic embedding helps generalize
across intra-class variance.

Swapping in a different object: rerun with `--object-key <key>
--class-name <name>` against meta yaml that has a
`<key>: {id, bbox}` entry and a matching descriptions CSV, no code
changes needed.

`scripts/build_full_dataset.py` is the separate "archive the whole
dataset" path: copies every image with a valid bbox (regardless of
description coverage) across one or more runs into one self-contained
folder with flattened annotations - meant for publishing/sharing as
dataset evidence, not for training directly. `--exclude-instance <id>...`
drops instances that shouldn't be in the dataset at all (as opposed to
one that's just temporarily missing a description).

## Training

```sh
uv run python -m semdetect.train --config configs/ball_baseline.yaml
uv run python -m semdetect.train --config configs/ball_film_late.yaml
```

Every run trains for up to `train.epochs` (a generous ceiling, default
200) but stops early once `train.checkpoint_metric` (default `map_50`)
hasn't improved for `train.early_stopping_patience` eval rounds (default
10 - set to `null` to disable and always run the full ceiling).

Each run writes to `train.output_dir`:

- `config.yaml` - the resolved config used for the run
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `metrics/epoch_XXXX.json` - train/val losses + mAP/precision/recall per eval epoch
- `metrics/history.csv` - the same, one row per epoch, for quick plotting
- `metrics/final.json` - best validation score, whether/when it stopped early, and test-set metrics

### Architectures

Registered under `semdetect.models.registry` (`model.architecture` in
config), each wrapping a real library implementation rather than a
reimplementation - see each file's module docstring for its layer map
and exactly where FiLM hooks in:

| `model.architecture` | `model.variant` examples | file | library |
| --- | --- | --- | --- |
| `yolo` | `yolo26n`, `yolo11n`, `yolov8n` | `semdetect/models/yolo_film.py` | Ultralytics |
| `rtdetr` | `rtdetr-l` | `semdetect/models/rtdetr_film.py` | Ultralytics |
| `fasterrcnn` | `resnet50`, `mobilenet_v3_large`, `mobilenet_v3_large_320` | `semdetect/models/torchvision_film.py` | torchvision |
| `fcos` | `resnet50` | `semdetect/models/torchvision_film.py` | torchvision |

`yolo`/`rtdetr` place FiLM via `model.film_layer_indices` (`null` = auto:
the layers feeding the detection head, "late"; explicit indices, e.g.
`[4, 6, 8]` for yolo26n, condition on raw backbone features instead,
"early"). `fasterrcnn`/`fcos` use the simpler `model.film_early: true/false`
(FPN outputs vs. the backbone body before the FPN) since torchvision's
multi-scale features are dict-keyed, not index-addressable the same way.
See `configs/*_film.yaml` / `configs/*_film_early.yaml` for one example
of each.

Add another architecture by implementing the `Detector` interface
(`semdetect/models/base.py`) and `@register_model("name")`;
`engine/trainer.py` only depends on that interface.

### Text embedding providers

`embedding.provider` in config, all exposing the same `.encode(text) ->
(embed_dim,) tensor` interface (`semdetect/data/embedders.py` +
`semdetect/data/clip_embedder.py`):

| `provider` | what it is |
| --- | --- |
| `clip` | open_clip (default `ViT-B-32-quickgelu`/`openai`) |
| `bert` | plain `bert-base-uncased`, mean-pooled |
| `e5` | `intfloat/e5-base-v2`, mean-pooled with its `"query: "` prefix convention |
| `random` | ablation - a fixed per-text random unit vector, see below |

### Ablations

Two ways to isolate "is FiLM using the *semantic content*, or just the
extra parameters / a per-instance identifier":

- `embedding.provider: random` (`configs/ball_film_random_embedding.yaml`)
  - a deterministic per-text random vector: right shape, per-instance
  consistent, zero semantic content.
- `data.description_mode: shuffled` (`configs/ball_film_wrong_descriptor.yaml`)
  - every instance gets a real CLIP embedding, but of a *different*
  ball's description (a fixed derangement, `build_shuffled_lookup` in
  `semdetect/data/dataset.py`) - real content, wrong instance.

If FiLM performs about the same under either ablation as with the real,
correctly-matched embedding, the gains aren't coming from semantic
content specifically.

### Comparing against zero-shot open-vocabulary detectors

Grounding DINO and OWL-ViT are natively text-conditioned (image + text
prompt in, boxes out), so they don't fit the FiLM-on-a-closed-set-detector
pattern above - they're run zero-shot (pretrained weights, no
fine-tuning) as reference points instead:

```sh
uv run python scripts/zero_shot_eval.py --model grounding_dino \
    --manifest data/manifest/test.csv
uv run python scripts/zero_shot_eval.py --model owlvit \
    --manifest data/manifest/test.csv
```

Prompted with `"a {description} {class_name}"` per image (both models
need the object noun present, not just a bare attribute phrase - see the
script's module docstring for the measurement behind that). Images are
letterboxed identically to training, so `outputs/zero_shot/<model>/metrics.json`
is directly comparable to a trained run's `metrics/test.json`.

### Parameter counts

```sh
uv run python scripts/params_report.py
```

Writes `outputs/params_report.csv`: every architecture above (FiLM on/off)
plus every text embedder plus the zero-shot foundation models, in one
table - e.g. `yolo26n` is ~2.5M params vs. Grounding DINO tiny's ~172M -
the evidence for whether a giant open-vocab model is actually a
reasonable fit for a low-resource robot. Foundation models are built from
their HuggingFace config only (no multi-GB weight download needed just to
count parameters).
