#!/usr/bin/env python3
"""
Zero-shot eval of an open-vocabulary detector (Grounding DINO or
OWL-ViT) on a manifest, prompted with each image's ball description -
the reference point for "how does a giant off-the-shelf open-vocab
model do on this task, with no training at all, versus our small
trained-from-scratch approach." No fine-tuning happens here; pair
with scripts/params_report.py for the parameter-count side of that
comparison.

Images are letterboxed to --img-size exactly like
semdetect.data.dataset.DetectionDataset, so metrics land in the same
coordinate frame and are directly comparable to a trained run's
val/test numbers in outputs/*/metrics/test.json.

The prompt is "a {description} {class_name}" (e.g. "a black with yellow
patches ball"), not the bare description - OWL-ViT in particular barely
responds to an attribute-only phrase with no object noun (measured
max confidence ~0.004 for "black with yellow patches" alone on a frame
where the ball is clearly visible, vs ~0.5 once "ball" is added), and
Grounding DINO's phrase grounding does at least as well with the noun
included, so this template is used for both, for a fair comparison.

Usage:
    python3 scripts/zero_shot_eval.py --model grounding_dino \\
        --manifest data/manifest/test.csv
    python3 scripts/zero_shot_eval.py --model owlvit \\
        --manifest data/manifest/test.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from semdetect.data.manifest import load_descriptions, load_manifest
from semdetect.data.transforms import letterbox, transform_bbox_xyxy
from semdetect.engine.metrics import DetectionMetrics

MODEL_DEFAULTS = {
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
    "owlvit": "google/owlvit-base-patch32",
}


def build_grounding_dino(model_id: str, device: str):
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    processor = AutoProcessor.from_pretrained(model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def predict(image: Image.Image, text: str, conf_thres: float) -> dict:
        # Grounding DINO's phrase-grounding convention: each candidate
        # phrase ends with a period.
        prompt = text if text.endswith(".") else text + "."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=conf_thres,
            text_threshold=conf_thres,
            target_sizes=[image.size[::-1]],
        )[0]
        # One text query -> every returned region is a candidate for our
        # single class, regardless of which sub-phrase of the prompt matched.
        return {
            "boxes": result["boxes"].cpu(),
            "scores": result["scores"].cpu(),
            "labels": torch.zeros(len(result["scores"]), dtype=torch.long),
        }

    return predict


def build_owlvit(model_id: str, device: str):
    from transformers import AutoProcessor, OwlViTForObjectDetection

    processor = AutoProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device).eval()

    # OWL-ViT's text tower has a fixed 16-token context (its position
    # embedding table is sized for exactly that, unlike standard CLIP's
    # 77) - without truncation, a description long enough to tokenize past
    # 16 tokens crashes with a shape mismatch instead of being clipped.
    max_length = model.config.text_config.max_position_embeddings

    @torch.no_grad()
    def predict(image: Image.Image, text: str, conf_thres: float) -> dict:
        inputs = processor(
            images=image, text=[[text]], return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=device)
        result = processor.post_process_grounded_object_detection(
            outputs, threshold=conf_thres, target_sizes=target_sizes
        )[0]
        return {
            "boxes": result["boxes"].cpu(),
            "scores": result["scores"].cpu(),
            "labels": torch.zeros(len(result["scores"]), dtype=torch.long),
        }

    return predict


BUILDERS = {"grounding_dino": build_grounding_dino, "owlvit": build_owlvit}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--descriptions-csv", default="data/ball_descriptions.csv")
    parser.add_argument("--description-field", default="Description")
    parser.add_argument("--model", choices=list(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-id", default=None, help=f"Default per --model: {MODEL_DEFAULTS}")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images (debug)")
    args = parser.parse_args()

    model_id = args.model_id or MODEL_DEFAULTS[args.model]
    output_dir = Path(args.output_dir or f"outputs/zero_shot/{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    descriptions = load_descriptions(args.descriptions_csv)

    print(f"Loading {args.model} ({model_id}) on {args.device} ...")
    predict = BUILDERS[args.model](model_id, args.device)

    metrics = DetectionMetrics(["ball"], iou_thres=0.5, conf_thres=args.conf_thres)
    n_no_description = 0
    for i, row in enumerate(rows):
        image = Image.open(row.image_path).convert("RGB")
        canvas, scale, pad = letterbox(image, args.img_size)
        gt_box = transform_bbox_xyxy((row.x1, row.y1, row.x2, row.y2), scale, pad)

        desc_row = descriptions.get(row.instance_id)
        if desc_row is None:
            n_no_description += 1
        description = desc_row[args.description_field].strip().lower() if desc_row else None
        prompt = f"a {description} {row.class_name}" if description else f"a {row.class_name}"

        pred = predict(canvas, prompt, args.conf_thres)
        target = {"boxes": torch.tensor([gt_box]), "labels": torch.tensor([0])}
        metrics.update([pred], [target])

        if (i + 1) % 20 == 0 or (i + 1) == len(rows):
            print(f"  {i + 1}/{len(rows)}")

    if n_no_description:
        print(f"  {n_no_description} image(s) had no description row - prompted with the bare class name instead")

    result = metrics.compute()
    print(result)
    summary = {"model": args.model, "model_id": model_id, "conf_thres": args.conf_thres, "n_images": len(rows), **result}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
