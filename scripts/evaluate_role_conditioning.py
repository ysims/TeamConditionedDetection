#!/usr/bin/env python3
"""
Two post-hoc analyses of a trained robot-jersey two-stage detector,
against the same detected-and-IoU-matched boxes so results are directly
comparable:

1. Colour-extraction baseline: bypass the learned role head entirely -
   take each detected box's actual mean pixel RGB (RoIAlign on the raw
   image, no learning at all) and classify teammate/opponent by nearest
   distance to the two injected colours. Answers "how much is the learned,
   conditioned role head actually adding over a trivial nearest-colour
   heuristic?"

2. Correct vs incorrect conditioning: re-run prediction with
   teammate_rgb/opponent_rgb swapped, score against the *original*
   (unswapped) ground truth. A model genuinely using the conditioning
   causally should confidently predict the *wrong* role once told the
   wrong colour, so accuracy should crater; a model that's secretly
   ignoring conditioning (some other shortcut) would show little change.

Usage:
    python3 scripts/evaluate_role_conditioning.py \
        --config configs/robot_jersey_fasterrcnn_role_head.yaml \
        --checkpoint outputs/robot_jersey/robot_jersey_fasterrcnn_role_head/checkpoints/best.pt \
        --manifest data/manifest_robots/test.csv
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import team_conditioned_detection.models.robot_role  # noqa: F401 - registration side effect
from team_conditioned_detection.data.robot_jersey import EMBED_DIM, ROLE_NAMES, RobotJerseyDataset, robot_collate_fn
from team_conditioned_detection.engine.metrics import box_iou, batch_to_targets
from team_conditioned_detection.models.registry import build_model
from team_conditioned_detection.robot_config import load_robot_config


def evaluate(config_path: str, checkpoint_path: str, manifest_path: str, batch_size: int = 4) -> dict:
    config = load_robot_config(config_path)
    model = build_model(
        config.model.architecture,
        num_classes=len(ROLE_NAMES),
        class_names=ROLE_NAMES,
        variant=config.model.variant,
        img_size=config.data.img_size,
        pretrained=False,  # weights come from the checkpoint below, not re-downloaded
        embed_dim=EMBED_DIM,
        film_hidden_dim=config.model.film_hidden_dim,
        conditioning_method=config.model.conditioning_method,
        auxiliary_loss_weight=1.0,
        roi_output_size=config.model.roi_output_size,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = RobotJerseyDataset(manifest_path, randomize_roles=False, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=robot_collate_fn)

    total_gt = 0
    model_matched = model_correct = 0
    baseline_matched = baseline_correct = 0
    swap_matched = swap_correct_vs_true = 0

    with torch.no_grad():
        for batch in loader:
            preds = model.predict(batch["img"], batch["embedding"], conf_thres=0.25)
            swapped_embedding = torch.cat([batch["embedding"][:, 3:], batch["embedding"][:, :3]], dim=1)
            preds_swapped = model.predict(batch["img"], swapped_embedding, conf_thres=0.25)
            targets = batch_to_targets(batch, config.data.img_size)

            for i, (pred, pred_swap, target) in enumerate(zip(preds, preds_swapped, targets)):
                total_gt += target["boxes"].shape[0]
                teammate_rgb = batch["embedding"][i, :3]
                opponent_rgb = batch["embedding"][i, 3:]

                if pred["boxes"].shape[0] > 0 and target["boxes"].shape[0] > 0:
                    ious = box_iou(pred["boxes"], target["boxes"])
                    for g in range(target["boxes"].shape[0]):
                        best_p = ious[:, g].argmax().item()
                        if ious[best_p, g] < 0.5:
                            continue
                        model_matched += 1
                        if pred["labels"][best_p] == target["labels"][g]:
                            model_correct += 1

                        box = pred["boxes"][best_p : best_p + 1]
                        roi = torch.cat([torch.tensor([[float(i)]]), box], dim=1)
                        mean_rgb = roi_align(batch["img"], roi, output_size=8, spatial_scale=1.0, aligned=True).mean(dim=(2, 3))[0]
                        dist_teammate = (mean_rgb - teammate_rgb).norm()
                        dist_opponent = (mean_rgb - opponent_rgb).norm()
                        baseline_label = 0 if dist_teammate < dist_opponent else 1
                        baseline_matched += 1
                        if baseline_label == target["labels"][g].item():
                            baseline_correct += 1

                if pred_swap["boxes"].shape[0] > 0 and target["boxes"].shape[0] > 0:
                    ious_swap = box_iou(pred_swap["boxes"], target["boxes"])
                    for g in range(target["boxes"].shape[0]):
                        best_p = ious_swap[:, g].argmax().item()
                        if ious_swap[best_p, g] < 0.5:
                            continue
                        swap_matched += 1
                        if pred_swap["labels"][best_p] == target["labels"][g]:
                            swap_correct_vs_true += 1

    return {
        "localization_recall": model_matched / max(total_gt, 1),
        "model_role_acc": model_correct / max(model_matched, 1),
        "colour_baseline_role_acc": baseline_correct / max(baseline_matched, 1),
        "swapped_conditioning_acc_vs_true": swap_correct_vs_true / max(swap_matched, 1),
        "n_gt_boxes": total_gt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifest_robots/test.csv")
    args = parser.parse_args()

    result = evaluate(args.config, args.checkpoint, args.manifest)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
