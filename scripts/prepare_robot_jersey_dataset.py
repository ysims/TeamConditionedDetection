#!/usr/bin/env python3
"""
Build a robot-detection manifest (teammate/opponent jersey-colour
conditioning experiment) from one or more NUpbr render runs.

Unlike scripts/prepare_nupbr_dataset.py (single ball instance per image,
held out by a fixed description vocabulary), this handles *multiple*
robots per image and has no instance vocabulary to hold out - each
robot's jersey_colour is drawn fresh per frame (see meta yaml: robots:
[{id, jersey_colour, bbox}, ...] - "id" is just a scene-slot label, the
colour differs every frame even for the same slot), so there's nothing
instance-specific to leak across a split; images are partitioned randomly
instead, per-image (an image's robots always stay together in one split).

Any number of robots per image is kept (0-robot images have nothing to
detect here, so those are still skipped) - the renderer guarantees at
most 2 distinct jersey colours per image regardless of robot count (a
3+-robot scene just means some colour is shared by more than one robot),
which is exactly what team_conditioned_detection/data/robot_jersey.py's role-assignment
scheme needs: group robots by colour, randomly assign one group teammate
and the other opponent. Images that somehow violate that guarantee are
skipped with a warning rather than silently mis-labeled.

Output layout at --output (default data/manifest_robots/):
    train.csv, val.csv, test.csv
        columns: image_path,width,height,x1,y1,x2,y2,jersey_colour,source_run
        one row per robot bbox - multiple rows share an image_path for
        multi-robot images.
    split_summary.json
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml
from PIL import Image

TARGET_TRAIN_FRAC = 0.70
TARGET_VAL_FRAC = 0.15
TARGET_TEST_FRAC = 0.15
MAX_DISTINCT_COLOURS = 2

FIELDNAMES = ["image_path", "width", "height", "x1", "y1", "x2", "y2", "jersey_colour", "source_run"]


def load_examples(run_dir: Path, top_crop_margin: float = 2.0) -> list[list[dict]]:
    """One entry per qualifying image, holding all of that image's robot
    rows together so a multi-robot image's group never gets split across
    partitions.

    Robots cropped at the top of frame are dropped before counting/
    qualifying: visual inspection showed a top-cropped box usually shows
    only legs, with the jersey-bearing torso cut off entirely, while
    bottom/side-cropped boxes still show the torso fine (~17.5% of raw
    boxes were top-cropped vs ~0.8% bottom-cropped) - keeping them would
    hand the role head boxes with no jersey signal to learn from at all,
    an unfixable floor no architecture change can work around. Only the
    top edge is filtered for this reason.
    """
    examples = []
    group_size_counts: dict[int, int] = {}
    zero_robot = 0
    dropped_top_cropped = 0
    skipped_too_many_colours = 0
    for meta_path in sorted((run_dir / "meta").glob("*.yaml")):
        entry = yaml.safe_load(meta_path.read_text())
        raw_robots = entry.get("robots") or []
        robots = [r for r in raw_robots if r["bbox"][1] > top_crop_margin]
        dropped_top_cropped += len(raw_robots) - len(robots)
        n = len(robots)
        if n == 0:
            zero_robot += 1
            continue
        n_colours = len({r["jersey_colour"] for r in robots})
        if n_colours > MAX_DISTINCT_COLOURS:
            skipped_too_many_colours += 1
            continue
        group_size_counts[n] = group_size_counts.get(n, 0) + 1
        image_path = run_dir / "raw" / f"{meta_path.stem}.png"
        if not image_path.exists():
            sys.exit(f"Image missing for {meta_path.name}: expected {image_path}")
        with Image.open(image_path) as im:
            width, height = im.size
        rows = []
        for r in robots:
            x1, y1, x2, y2 = r["bbox"]
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "width": width,
                    "height": height,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "jersey_colour": r["jersey_colour"],
                    "source_run": run_dir.name,
                }
            )
        examples.append(rows)
    group_summary = ", ".join(f"{k}-robot={v}" for k, v in sorted(group_size_counts.items()))
    print(
        f"  {run_dir.name}: {zero_robot} zero-robot, {group_summary}, "
        f"{skipped_too_many_colours} skipped (>{MAX_DISTINCT_COLOURS} distinct colours), "
        f"{dropped_top_cropped} top-cropped robot(s) dropped before counting"
    )
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        default=[
            Path("/home/ysi/code/NUbots/NUpbr/outputs/run_4"),
            Path("/home/ysi/code/NUbots/NUpbr/outputs/run_5"),
        ],
        # run_1/2/3 excluded: too dark/poorly lit for jersey colour to be
        # reliably legible (see the visual sanity check that diagnosed the
        # first FiLM/cross-attention attempts - both failed identically in
        # a way traced back to jersey colour often not being visible in
        # those renders). run_4/run_5 were generated later and are lit
        # better; run_5 also explicitly guarantees <=2 distinct jersey
        # colours per image even at 3+ robots.
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "manifest_robots")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for the random train/val/test split")
    args = parser.parse_args()

    for run_dir in args.run_dir:
        if not run_dir.exists():
            sys.exit(f"Run directory not found: {run_dir}")

    args.output.mkdir(parents=True, exist_ok=True)

    per_image_examples: list[list[dict]] = []
    for run_dir in args.run_dir:
        print(f"Scanning {run_dir} ...")
        per_image_examples.extend(load_examples(run_dir))

    print(f"{len(per_image_examples)} qualifying images total across {len(args.run_dir)} run(s)")
    if not per_image_examples:
        sys.exit("No qualifying images found")

    rng = random.Random(args.seed)
    rng.shuffle(per_image_examples)

    n = len(per_image_examples)
    n_train = round(n * TARGET_TRAIN_FRAC)
    n_val = round(n * TARGET_VAL_FRAC)
    partitions = {
        "train": per_image_examples[:n_train],
        "val": per_image_examples[n_train : n_train + n_val],
        "test": per_image_examples[n_train + n_val :],
    }

    summary = {"partition_images": {}, "partition_robot_boxes": {}}
    for name, groups in partitions.items():
        rows = [row for group in groups for row in group]
        with open(args.output / f"{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        group_sizes: dict[int, int] = {}
        for g in groups:
            group_sizes[len(g)] = group_sizes.get(len(g), 0) + 1
        breakdown = ", ".join(f"{k}-robot={v}" for k, v in sorted(group_sizes.items()))
        print(f"  {name}: {len(groups)} images ({breakdown}), {len(rows)} robot boxes")
        summary["partition_images"][name] = len(groups)
        summary["partition_robot_boxes"][name] = len(rows)

    with open(args.output / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote manifests to {args.output}")


if __name__ == "__main__":
    main()
