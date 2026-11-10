# Team Conditioned Detection

This repository contains methods for bounding-box object detection conditioned on sports team/role context, for multi-instance scenes where identical-class objects must be told apart by role. The problem considered here is the detection of teammate robots and opponent robots on a soccer field, where team is designated by jersey colour. All robots, whether teammate or opponent, may be any robot platform, and the only feature distinguishing the two is the jersey colour. Each team may be any colour, and training swaps teammate and opponent designation on the same image in different batches.

The code is built so that the detection architecture, the conditioning mechanism (FiLM, cross-attention, conditional norm, AdaIN, gated), and whether conditioning happens at the whole-detector level or a separate role head are all swappable independently via config.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

## Dataset

The dataset used is https://huggingface.co/datasets/Ysobel/robot_jersey_dataset. This dataset is semi-synthetic, created with [NUpbr](https://github.com/NUbots/NUpbr), which enabled the automatic generation and labelling of robots and their jerseys. The background images are 360 degree images from various real RoboCup fields, while the robots are synthetically added using robot models and the torso section coloured with a random jersey colour.

This repository supports the generation of a new NUpbr dataset, where `scripts/prepare_robot_jersey_dataset.py` turns one or more NUpbr synthetic render runs (`raw/*.png` + `meta/*.yaml`, with `robots: [{id,
jersey_colour, bbox}, ...]` per image) into `train.csv` / `val.csv` /
`test.csv` manifests:

```sh
uv run python scripts/prepare_robot_jersey_dataset.py \
    --run-dir /path/to/NUpbr/outputs/run_1 /path/to/NUpbr/outputs/run_2 \
    --output data/manifest_robots
```

Each robot's jersey colour is generated randomly per frame, and there is no fixed list of colours to use as a dataset partitioning mechanism. Images are partitioned randomly instead, with each image's robots kept together in one set.

## Running the experiments

One script trains every experiment for every seed, evaluates them, and aggregates the results:

```sh
uv run python scripts/run_experiments.py
```

It runs three stages, each of which skips work that's already done, so an interrupted run picks up where it left off:

1. **train**: each experiment and seed (42, 43, 44 by default) trains into `outputs/paper/<experiment>/seed<N>/`, with its full log in `train.log`. A run trained with a config that differs from the current one is reported as stale and is never mixed into the results. `--force` retrains it.
2. **eval**: each run's `best.pt` is evaluated on the test set (`eval.json`), along with the non-learned colour rules (`outputs/paper/colour_rules.json`).
3. **tables**: results are aggregated as mean ± sample standard deviation over seeds into `outputs/paper/results.json`, and summary tables are written to `--tables-dir` (default `tables/`).

```sh
uv run python scripts/run_experiments.py --stage tables                 # re-aggregate existing results
uv run python scripts/run_experiments.py --experiments yolo26n_film --seeds 42
```

All reported metrics use the test split with one fixed teammate/opponent assignment (`data.eval_seed`), the same for every training seed:

| metric | meaning |
| --- | --- |
| `role_acc` | role accuracy on ground-truth boxes |
| `swap_acc` | accuracy with the teammate/opponent colours exchanged, scored against the flipped label |
| `both_correct` | fraction of boxes correct under both contexts |
| `cond_swap_acc` | P(swap correct \| original correct) |
| `map_50` | mAP50 over predicted boxes and roles (detectors only) |
| `role_params` | parameters added for role classification, excluding any shared detector |

### Experiments

One config per experiment, in `configs/`:

| config | model |
| --- | --- |
| `fasterrcnn_film` | Faster R-CNN (ResNet-50-FPN) + FiLM role head |
| `rtdetr_film` | RT-DETR-L + FiLM role head, role-loss weight 0.1 |
| `rtdetr_film_rw1` | as `rtdetr_film`, role-loss weight 1.0 (as Faster R-CNN) |
| `yolo26n_film` | YOLO26n + FiLM role head, role-loss weight 0.1 |
| `yolo26n_film_rw1` | as `yolo26n_film`, role-loss weight 1.0 |
| `fasterrcnn_film_shift` | shift-only FiLM (scale fixed at 1) |
| `fasterrcnn_concat` | concatenation + MLP role head, parameter-matched to FiLM |
| `fasterrcnn_film_distance` | FiLM role head + colour-distance feature |
| `fasterrcnn_crossattn` | cross-attention role head (8 context tokens) |
| `fasterrcnn_crossattn_distance` | cross-attention role head + colour-distance feature |
| `standalone_classifier` | standalone FiLM-conditioned CNN on raw-pixel crops, no detector |

The non-learned baselines (`team_detect/evaluation.py`) assign each ground-truth box the role whose supplied colour is closest to the box's observed colour: mean RGB or minimum RGB distance over the whole box or a centre crop, mean hue or k-means dominant colour over the torso band, and hue/HSV models calibrated on the training set to map supplied colours to their rendered appearance. Test-set accuracy is also broken down by the hue separation of the two supplied colours.

Other scripts: `scripts/benchmark_latency.py` times the role heads and detectors on GPU and CPU, and `scripts/make_swap_figure.py` draws predictions on test frames under the original and swapped colour contexts.

## Training a single run

```sh
uv run python -m team_detect.train --config configs/fasterrcnn_film.yaml --seed 43 --output-dir outputs/scratch
uv run python -m team_detect.evaluation --run-dir outputs/scratch
```

Training runs for up to `train.epochs`, but stops early once every metric in `train.early_stopping_metrics` has gone `train.early_stopping_patience` validation rounds without improving. `best.pt` is selected by `train.checkpoint_metric`. Each run writes `config.yaml`, `checkpoints/{best,last}.pt`, and `metrics/` (per-epoch JSON, `history.csv`, `final.json`).

## Models

Every detector is two-stage: an unconditioned single-class ("robot") detector, plus a role head that RoIAligns an intermediate detector feature at each box and classifies teammate/opponent conditioned on the 6-d (teammate RGB, opponent RGB) context. Box regression never sees the context. See `team_detect/models/robot_role.py`.

| `model.architecture` | role feature | file |
| --- | --- | --- |
| `fasterrcnn_robot_role` | finest FPN level | `team_detect/models/robot_role.py` |
| `rtdetr_robot_role` | P3 backbone feature (layer 3), before hybrid-encoder fusion | `team_detect/models/robot_role.py` |
| `yolo_robot_role` | P3/8 backbone feature (layer 4), before the neck | `team_detect/models/robot_role.py` |
| `standalone_role_classifier` | its own small CNN on raw-pixel crops | `team_detect/models/standalone_role_classifier.py` |

`model.conditioning_method` is `film` (channel-wise scale and shift) or `cross_attention` (spatial positions attend to 8 context-derived tokens; an additive residual only). See `team_detect/models/conditioners.py`. `model.use_distance_feature` appends the minimum RGB distance over a grid of box cells to each supplied colour.

The single-stage (whole-detector-conditioned) models, the other conditioning mechanisms, and the earlier ablations were removed. They are in git history before this cleanup.
