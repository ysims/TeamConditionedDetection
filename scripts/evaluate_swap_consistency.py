#!/usr/bin/env python3
"""
Swap-consistency evaluation on GT boxes: for every ground-truth robot box,
run the role head twice on the SAME detector feature - once with the real
(teammate_rgb, opponent_rgb) embedding, once with it swapped - and compare
both predictions to the true label. Unlike scripts/evaluate_role_conditioning.py's
swap test (scored against the ORIGINAL label, on detected/IoU-matched
boxes), this scores the swapped prediction against the FLIPPED label (what
it should predict if conditioning is being used correctly) and reports it
conditioned on whether the normal prediction was also correct - i.e. "of
the boxes it gets right normally, does swapping the input also flip it to
the (correspondingly flipped) right answer, and how often does that hold
over the whole dataset."

Usage:
    python3 scripts/evaluate_swap_consistency.py \
        --config outputs/robot_jersey/<run>/config.yaml \
        --checkpoint outputs/robot_jersey/<run>/checkpoints/best.pt \
        --manifest data/manifest_robots/test.csv
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import team_conditioned_detection.models.robot_role  # noqa: F401
import team_conditioned_detection.models.rtdetr_film  # noqa: F401
import team_conditioned_detection.models.torchvision_film  # noqa: F401
import team_conditioned_detection.models.yolo_film  # noqa: F401
from team_conditioned_detection.data.robot_jersey import EMBED_DIM, ROLE_NAMES, RobotJerseyDataset, robot_collate_fn
from team_conditioned_detection.engine.metrics import norm_cxcywh_to_xyxy
from team_conditioned_detection.models.registry import build_model
from team_conditioned_detection.robot_config import load_robot_config


def _run_detector_and_capture(model, images: torch.Tensor) -> torch.Tensor:
    model._captured_feature = None
    if hasattr(model, "detector"):
        model.detector(images)
    else:
        # FasterRCNNRobotRoleDetector: torchvision's GeneralizedRCNN normally
        # runs images through self.model.transform (ImageNet mean/std
        # normalization + resize) before the backbone ever sees them - see
        # generalized_rcnn.py's forward(). Calling model.model.backbone(images)
        # directly skips that, feeding raw [0,1] pixels to a backbone trained
        # on normalized input, which silently craters accuracy without erroring.
        transformed, _ = model.model.transform(list(images.unbind(0)))
        model.model.backbone(transformed.tensors)
    return model._captured_feature


def evaluate(config_path: str, checkpoint_path: str, manifest_path: str, batch_size: int = 8, device: str = "cuda") -> dict:
    config = load_robot_config(config_path)
    model = build_model(
        config.model.architecture,
        num_classes=len(ROLE_NAMES),
        class_names=ROLE_NAMES,
        variant=config.model.variant,
        img_size=config.data.img_size,
        pretrained=False,  # weights come from the checkpoint below
        use_film=config.model.use_film,
        embed_dim=EMBED_DIM,
        film_hidden_dim=config.model.film_hidden_dim,
        film_layer_indices=config.model.film_layer_indices,
        film_early=config.model.film_early,
        conditioning_method=config.model.conditioning_method,
        auxiliary_loss_weight=config.model.auxiliary_loss_weight,
        roi_output_size=config.model.roi_output_size,
        role_head_use_deep_feature=config.model.role_head_use_deep_feature,
        role_head_use_distance_feature=config.model.role_head_use_distance_feature,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = RobotJerseyDataset(manifest_path, img_size=config.data.img_size, randomize_roles=False, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=robot_collate_fn)

    n_total = 0
    normal_correct = 0
    swap_correct = 0
    both_correct = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], config.data.img_size).to(device)
            batch_idx = batch["batch_idx"].long().to(device)
            labels = batch["cls"].long().to(device)
            embedding = batch["embedding"].to(device)
            swapped_embedding = torch.cat([embedding[:, 3:], embedding[:, :3]], dim=-1)

            feature = _run_detector_and_capture(model, images)
            normal_logits = model.role_head(feature, images, boxes_xyxy, batch_idx, embedding, model.feature_stride)
            swap_logits = model.role_head(feature, images, boxes_xyxy, batch_idx, swapped_embedding, model.feature_stride)

            normal_pred = normal_logits.argmax(dim=-1)
            swap_pred = swap_logits.argmax(dim=-1)
            flipped_labels = 1 - labels

            is_normal_correct = normal_pred == labels
            is_swap_correct = swap_pred == flipped_labels

            n_total += labels.shape[0]
            normal_correct += is_normal_correct.sum().item()
            swap_correct += is_swap_correct.sum().item()
            both_correct += (is_normal_correct & is_swap_correct).sum().item()

    return {
        "n_boxes": n_total,
        "normal_acc": normal_correct / n_total,
        "swap_acc_vs_flipped_label": swap_correct / n_total,
        "both_correct_count": both_correct,
        "both_correct_over_dataset": both_correct / n_total,
        "P(swap_correct_given_normal_correct)": both_correct / normal_correct if normal_correct else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifest_robots/test.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    result = evaluate(args.config, args.checkpoint, args.manifest, batch_size=args.batch_size)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
