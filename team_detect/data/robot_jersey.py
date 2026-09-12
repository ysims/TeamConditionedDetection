"""Multi-instance robot detection with per-role jersey-colour conditioning.

Unlike the ball dataset (one object per image, one fixed text description
per instance), each image here has one or more robots, each with its own
per-frame-random jersey colour (see meta yaml: `robots: [{id, jersey_colour,
bbox}, ...]` - "id" is just a scene-slot label, not a persistent identity:
the same slot gets a different random colour every frame it appears in, so
unlike ball descriptions there's no fixed colour vocabulary to hold out
across train/val/test - see scripts/prepare_robot_jersey_dataset.py, which
partitions images randomly instead). The renderer guarantees at most 2
distinct jersey colours per image regardless of robot count (a 3+-robot
image just means some colour is shared by more than one robot) - see
prepare_robot_jersey_dataset.py's MAX_DISTINCT_COLOURS check.

The conditioning signal is two RGB triplets - a "teammate colour" and an
"opponent colour" - concatenated into one 6-dim vector fed straight into
FiLM (whose own MLP, see film.py's FiLMGenerator, does the "colour pair ->
hidden -> gamma/beta" mapping; no separate embedder class needed the way
CLIP/BERT/E5 needed one for text).

Ground truth label for each robot in the image is whichever role's colour
matches that robot's actual jersey colour:
    - 2-distinct-colour image (any robot count): both roles are "real" -
      robots are grouped by their own colour, and which colour-group gets
      which role is decided per draw (see below). A 3-robot image with
      colours {A, A, B} labels both A-robots the same role.
    - 1-distinct-colour image (including single-robot images): the
      present colour is randomly assigned one role, and the *other* role
      gets a freshly-sampled random RGB distractor - nothing in the scene
      matches it, so the correct model behaviour is to predict zero boxes
      of that role. This doubles as a hard-negative test: conditioned on
      a colour nothing in the scene matches, does the model correctly
      detect nothing for that role.

Critically, this role<->colour assignment is *not* precomputed into the
manifest: on the training split it's re-drawn every access, in
__getitem__ - "when it sees image A and opponent is red, teammate is
blue, another time it sees image A the opponent might be blue and the
teammate red" (this is the whole point: prevent the model from learning
any position/robot-index shortcut and force it to actually key off the
conditioning colour). Validation/test use a *fixed* assignment instead,
drawn once at construction and cached - re-randomizing there too would
make epoch-to-epoch val metrics partly reflect which random assignment
got drawn rather than real model improvement, exactly the kind of noise
this project's multi-seed methodology has otherwise been built to avoid.
"""
from __future__ import annotations

import csv
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from team_conditioned_detection.data.transforms import letterbox, maybe_hflip_boxes, to_chw_float, transform_bbox_xyxy, xyxy_to_norm_cxcywh

ROLE_NAMES = ["teammate", "opponent"]  # index = class label
EMBED_DIM = 3 * len(ROLE_NAMES)  # RGB per role, concatenated


def hex_to_rgb01(hex_colour: str) -> tuple[float, float, float]:
    h = hex_colour.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def load_robot_manifest(path: str) -> list[list[dict]]:
    """Groups manifest rows by image_path - each group is one image's
    robot annotations (any count >= 1; row order within a group is
    manifest order, stable across loads).
    """
    groups: dict[str, list[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["image_path"], []).append(row)
    return list(groups.values())


def worker_init_fn(worker_id: int) -> None:
    """Reseeds Python's global `random` module per DataLoader worker.
    Without this, forked workers inherit identical RNG state and would
    draw *identical* teammate/opponent role assignments in lockstep across
    workers - torch.initial_seed() inside a worker is already unique per
    worker (PyTorch sets it up internally), so it's a good seed source;
    see https://pytorch.org/docs/stable/notes/randomness.html.
    """
    random.seed(torch.initial_seed() % 2**32)


class RobotJerseyDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        img_size: int = 640,
        augment: bool = False,
        hflip_prob: float = 0.5,
        randomize_roles: bool = True,
        seed: int = 0,
        wrong_conditioning: bool = False,
    ):
        self.images = load_robot_manifest(manifest_path)
        self.img_size = img_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.randomize_roles = randomize_roles
        # Ablation only - only takes effect together with randomize_roles
        # (i.e. only on the split actually being trained on). Deliberately
        # NOT applied when randomize_roles=False (val/test): the point is
        # to measure whether a model trained on uninformative conditioning
        # still ends up USING real conditioning when it's given real
        # conditioning at eval time, not to also corrupt the eval signal.
        self.wrong_conditioning = wrong_conditioning
        self._aug_rng = random.Random(seed)

        # Val/test: one fixed role assignment per image, drawn once here
        # (not per access) - see module docstring.
        self._fixed_first_is_teammate: list[bool] | None = None
        self._fixed_distractor: list[tuple[float, float, float]] | None = None
        if not randomize_roles:
            fixed_rng = random.Random(seed)
            self._fixed_first_is_teammate = [fixed_rng.random() < 0.5 for _ in self.images]
            self._fixed_distractor = [(fixed_rng.random(), fixed_rng.random(), fixed_rng.random()) for _ in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        robots = self.images[idx]
        image = Image.open(robots[0]["image_path"]).convert("RGB")
        canvas, scale, pad = letterbox(image, self.img_size)
        bboxes = [
            transform_bbox_xyxy((float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])), scale, pad)
            for r in robots
        ]

        if self.augment:
            canvas, bboxes = maybe_hflip_boxes(canvas, bboxes, self.img_size, self.hflip_prob, self._aug_rng)

        if self.randomize_roles:
            first_is_teammate = random.random() < 0.5
            distractor = (random.random(), random.random(), random.random())
        else:
            first_is_teammate = self._fixed_first_is_teammate[idx]
            distractor = self._fixed_distractor[idx]

        # Sorted so "first"/"second" means the same thing across repeated
        # accesses regardless of manifest row order - matters for the
        # fixed (val/test) assignment above to stay actually fixed.
        colours = sorted({r["jersey_colour"] for r in robots})
        if len(colours) == 2:
            first_rgb, second_rgb = hex_to_rgb01(colours[0]), hex_to_rgb01(colours[1])
            teammate_rgb = first_rgb if first_is_teammate else second_rgb
            opponent_rgb = second_rgb if first_is_teammate else first_rgb
            teammate_colour = colours[0] if first_is_teammate else colours[1]
            labels = [0 if r["jersey_colour"] == teammate_colour else 1 for r in robots]
        else:
            real_rgb = hex_to_rgb01(colours[0])
            teammate_rgb = real_rgb if first_is_teammate else distractor
            opponent_rgb = distractor if first_is_teammate else real_rgb
            labels = [0 if first_is_teammate else 1] * len(robots)

        if self.wrong_conditioning and self.randomize_roles:
            # Ground truth (labels) stays correct - only the injected
            # colours become pure noise, unrelated to what's actually
            # visible. See __init__'s docstring.
            teammate_rgb = (random.random(), random.random(), random.random())
            opponent_rgb = (random.random(), random.random(), random.random())

        bboxes_norm = [xyxy_to_norm_cxcywh(b, self.img_size) for b in bboxes]
        image_tensor = torch.from_numpy(to_chw_float(canvas))
        embedding = torch.tensor([*teammate_rgb, *opponent_rgb], dtype=torch.float32)

        return {
            "image": image_tensor,
            "bboxes": torch.tensor(bboxes_norm, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            # Single unconditioned "robot" class, same count as labels -
            # only team_conditioned_detection.models.robot_role's two-stage detector uses
            # this (its own box/objectness loss target); single-stage
            # detectors (FiLM/cross-attention/baseline) use "labels" above
            # directly and ignore this field.
            "detection_labels": torch.zeros(len(labels), dtype=torch.long),
            "embedding": embedding,
            "image_path": robots[0]["image_path"],
        }


def robot_collate_fn(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    batch_idx, cls, detection_cls, bboxes = [], [], [], []
    for i, b in enumerate(batch):
        n = b["bboxes"].shape[0]
        batch_idx.append(torch.full((n,), i, dtype=torch.float32))
        cls.append(b["labels"])
        detection_cls.append(b["detection_labels"])
        bboxes.append(b["bboxes"])
    return {
        "img": images,
        "cls": torch.cat(cls).float(),
        "detection_cls": torch.cat(detection_cls).float(),
        "bboxes": torch.cat(bboxes).float(),
        "batch_idx": torch.cat(batch_idx),
        "embedding": torch.stack([b["embedding"] for b in batch]),
        "image_paths": [b["image_path"] for b in batch],
    }
