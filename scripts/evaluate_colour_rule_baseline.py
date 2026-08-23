#!/usr/bin/env python3
"""
Pure post-processing baseline: no trained model at all. For each
ground-truth robot box, compute its colour signature directly from the
raw image and classify teammate/opponent by a fixed rule - whichever of
the two injected colours it's closer to - no learned classifier, no
network in the loop. This is the ceiling (or floor) of "can rules/maths
alone solve this" independent of any training dynamics.

Reports four variants of the colour signature:
    - whole-box mean RGB (the naive average - what dilutes into grey
      chassis, per the earlier visual diagnosis)
    - min-distance-over-cells (RoIAlign into a grid, take the closest-
      matching cell to each candidate colour - the same computation
      RoleHead's distance feature uses, minus the learned classifier on
      top of it)
    - centre-patch mean RGB (shrink the box to a `center_frac`-sized
      window around its centre, then average - betting the torso/jersey
      sits near the box centre rather than the edges)
    - centre-patch min-distance-over-cells (same shrunk window, but grid
      + min instead of a flat average)

Usage:
    python3 scripts/evaluate_colour_rule_baseline.py --manifest data/manifest_robots/test.csv
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team_conditioned_detection.data.robot_jersey import RobotJerseyDataset, robot_collate_fn
from team_conditioned_detection.engine.metrics import norm_cxcywh_to_xyxy


def _rule_predict(crop: torch.Tensor, teammate_rgb: torch.Tensor, opponent_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Given an (N,3,S,S) colour crop, return (mean-rule pred, min-cell-rule pred)."""
    mean_rgb = crop.mean(dim=(2, 3))
    dist_t_mean = (mean_rgb - teammate_rgb).norm(dim=-1)
    dist_o_mean = (mean_rgb - opponent_rgb).norm(dim=-1)
    pred_mean = (dist_o_mean < dist_t_mean).long()  # 0=teammate, 1=opponent

    dist_t_cells = (crop - teammate_rgb[:, :, None, None]).norm(dim=1).flatten(1).min(dim=1).values
    dist_o_cells = (crop - opponent_rgb[:, :, None, None]).norm(dim=1).flatten(1).min(dim=1).values
    pred_mincell = (dist_o_cells < dist_t_cells).long()
    return pred_mean, pred_mincell


def evaluate(manifest_path: str, img_size: int = 640, grid_size: int = 8, center_frac: float = 0.3, batch_size: int = 8) -> dict:
    ds = RobotJerseyDataset(manifest_path, randomize_roles=False, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=robot_collate_fn)

    n_total = 0
    correct = {"whole_box_mean": 0, "whole_box_mincell": 0, "center_mean": 0, "center_mincell": 0}

    with torch.no_grad():
        for batch in loader:
            boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], img_size)
            batch_idx = batch["batch_idx"].long()
            labels = batch["cls"].long()
            embeddings = batch["embedding"]
            per_box_embedding = embeddings[batch_idx]
            teammate_rgb, opponent_rgb = per_box_embedding[:, :3], per_box_embedding[:, 3:]

            # Whole box
            rois = torch.cat([batch_idx.unsqueeze(1).float(), boxes_xyxy], dim=1)
            crop = roi_align(batch["img"], rois, output_size=grid_size, spatial_scale=1.0, aligned=True)
            pred_mean, pred_mincell = _rule_predict(crop, teammate_rgb, opponent_rgb)
            correct["whole_box_mean"] += (pred_mean == labels).sum().item()
            correct["whole_box_mincell"] += (pred_mincell == labels).sum().item()

            # Shrunk, centred box
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
            cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
            half_w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / 2 * center_frac
            half_h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / 2 * center_frac
            center_boxes = torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)
            center_rois = torch.cat([batch_idx.unsqueeze(1).float(), center_boxes], dim=1)
            center_crop = roi_align(batch["img"], center_rois, output_size=grid_size, spatial_scale=1.0, aligned=True)
            pred_cmean, pred_cmincell = _rule_predict(center_crop, teammate_rgb, opponent_rgb)
            correct["center_mean"] += (pred_cmean == labels).sum().item()
            correct["center_mincell"] += (pred_cmincell == labels).sum().item()

            n_total += labels.shape[0]

    return {
        "n_boxes": n_total,
        "whole_box_mean_rule_acc": correct["whole_box_mean"] / max(n_total, 1),
        "min_distance_over_cells_rule_acc": correct["whole_box_mincell"] / max(n_total, 1),
        f"center_patch_frac{center_frac}_mean_rule_acc": correct["center_mean"] / max(n_total, 1),
        f"center_patch_frac{center_frac}_mincell_rule_acc": correct["center_mincell"] / max(n_total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/manifest_robots/test.csv")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--center-frac", type=float, default=0.3)
    args = parser.parse_args()

    result = evaluate(args.manifest, grid_size=args.grid_size, center_frac=args.center_frac)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
