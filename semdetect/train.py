#!/usr/bin/env python3
"""Generic training entrypoint.

    python -m semdetect.train --config configs/ball_film_late.yaml
    python -m semdetect.train --config configs/ball_baseline.yaml

Everything about the run (dataset paths, model architecture, whether FiLM
is on, hyperparameters) lives in the YAML config - this script just wires
config -> dataset -> model -> engine.trainer.run_training. See config.py
for the full schema and configs/ for annotated examples.

Registering a new model architecture (semdetect.models.registry) or
swapping the dataset (a different manifest/descriptions CSV pair, see
scripts/prepare_nupbr_dataset.py) doesn't require touching this file -
just add its import below for registration side-effects.
"""
import argparse

# Imports for registration side-effects (@register_model) - "yolo",
# "rtdetr", "fasterrcnn", "fcos".
import semdetect.models.rtdetr_film  # noqa: F401
import semdetect.models.torchvision_film  # noqa: F401
import semdetect.models.yolo_film  # noqa: F401
from semdetect.config import load_config
from semdetect.engine.trainer import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config, see configs/")
    parser.add_argument("--output-dir", default=None, help="Override config.train.output_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.train.epochs")
    parser.add_argument("--device", default=None, help="Override config.train.device, e.g. cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.epochs:
        config.train.epochs = args.epochs
    if args.device:
        config.train.device = args.device

    run_training(config)


if __name__ == "__main__":
    main()
