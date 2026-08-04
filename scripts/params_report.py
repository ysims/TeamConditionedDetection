#!/usr/bin/env python3
"""
Parameter-count comparison across every detector this repo can train
(all of the semdetect.models registry, FiLM on/off) plus the zero-shot
open-vocabulary foundation models used as reference points
(Grounding DINO, OWL-ViT) - the evidence for "these foundation models
aren't a realistic fit for a low-resource robot" alongside the actual
accuracy numbers from training/zero-shot eval.

Foundation models are built from their HuggingFace config only (no
weight download - config.json is a few KB, the safetensors are
GB-scale), since we only need architecture-implied parameter counts
here; scripts/zero_shot_eval.py downloads real weights for actual
inference.

Output: <repo>/outputs/params_report.csv, columns:
    category, name, variant, use_film, detector_params_m,
    embedder_params_m, total_params_m, notes
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def semdetect_rows() -> list[dict]:
    import semdetect.models.rtdetr_film  # noqa: F401
    import semdetect.models.torchvision_film  # noqa: F401
    import semdetect.models.yolo_film  # noqa: F401
    from semdetect.data.embedders import HFMeanPoolingEmbedder, RandomTextEmbedder
    from semdetect.data.clip_embedder import ClipTextEmbedder
    from semdetect.models.registry import build_model

    embedders = {
        "clip (ViT-B-32)": lambda: ClipTextEmbedder("ViT-B-32-quickgelu", "openai", "cpu"),
        "bert-base-uncased": lambda: HFMeanPoolingEmbedder("bert-base-uncased", "cpu", prefix=""),
        "e5-base-v2": lambda: HFMeanPoolingEmbedder("intfloat/e5-base-v2", "cpu", prefix="query: "),
        "random": lambda: RandomTextEmbedder(512),
    }
    embedder_params = {}
    for name, factory in embedders.items():
        try:
            instance = factory()
            embedder_params[name] = count_params(instance.model) if hasattr(instance, "model") else 0.0
        except Exception as e:  # network/model-availability issues shouldn't kill the whole report
            print(f"  skipping embedder {name}: {e}")
    # A representative embedder size to pair with detectors below (CLIP:
    # the default in every example config) - swap providers add roughly
    # this order of magnitude, see the embedder rows themselves for exact figures.
    clip_params_m = embedder_params.get("clip (ViT-B-32)", 0.0)

    detector_specs = [
        # (architecture, variant, img_size)
        ("yolo", "yolo26n", 640),
        ("yolo", "yolo11n", 640),
        ("yolo", "yolov8n", 640),
        ("rtdetr", "rtdetr-l", 640),
        ("fasterrcnn", "resnet50", 640),
        ("fasterrcnn", "mobilenet_v3_large_320", 320),
        ("fcos", "resnet50", 640),
    ]

    rows = []
    for architecture, variant, img_size in detector_specs:
        for use_film in (False, True):
            try:
                model = build_model(
                    architecture,
                    num_classes=1,
                    class_names=["ball"],
                    variant=variant,
                    img_size=img_size,
                    pretrained=False,
                    use_film=use_film,
                    embed_dim=512,
                    film_hidden_dim=256,
                )
            except Exception as e:
                print(f"  skipping {architecture}/{variant} use_film={use_film}: {e}")
                continue
            detector_m = count_params(model)
            rows.append(
                {
                    "category": "ours (trainable)",
                    "name": architecture,
                    "variant": variant + (" +FiLM" if use_film else ""),
                    "use_film": use_film,
                    "detector_params_m": round(detector_m, 3),
                    "embedder_params_m": round(clip_params_m, 3) if use_film else 0.0,
                    "total_params_m": round(detector_m + (clip_params_m if use_film else 0.0), 3),
                    "notes": "FiLM MLP heads only add a few thousand params; the CLIP text tower is the real FiLM overhead, "
                    "and it only ever runs once per unique description (cacheable/offline), not per frame.",
                }
            )

    for name, params_m in embedder_params.items():
        rows.append(
            {
                "category": "text embedder",
                "name": name,
                "variant": "",
                "use_film": "",
                "detector_params_m": 0.0,
                "embedder_params_m": round(params_m, 3),
                "total_params_m": round(params_m, 3),
                "notes": "Encodes a description into the FiLM conditioning vector; not run per-frame at inference "
                "if descriptions are known in advance (cacheable).",
            }
        )
    return rows


def foundation_model_rows() -> list[dict]:
    rows = []

    try:
        from transformers import GroundingDinoConfig, GroundingDinoForObjectDetection

        for model_id in ["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"]:
            cfg = GroundingDinoConfig.from_pretrained(model_id)
            m = GroundingDinoForObjectDetection(cfg)
            rows.append(
                {
                    "category": "zero-shot open-vocab",
                    "name": "Grounding DINO",
                    "variant": model_id.split("/")[-1],
                    "use_film": "n/a (natively text-conditioned)",
                    "detector_params_m": round(count_params(m), 3),
                    "embedder_params_m": 0.0,
                    "total_params_m": round(count_params(m), 3),
                    "notes": "Text encoder (BERT) is inside this parameter count already - it's one fused "
                    "image+text model, not a detector + separate embedder.",
                }
            )
    except Exception as e:
        print(f"  skipping Grounding DINO: {e}")

    try:
        from transformers import OwlViTConfig, OwlViTForObjectDetection

        for model_id in ["google/owlvit-base-patch32", "google/owlvit-base-patch16", "google/owlvit-large-patch14"]:
            cfg = OwlViTConfig.from_pretrained(model_id)
            m = OwlViTForObjectDetection(cfg)
            rows.append(
                {
                    "category": "zero-shot open-vocab",
                    "name": "OWL-ViT",
                    "variant": model_id.split("/")[-1],
                    "use_film": "n/a (natively text-conditioned)",
                    "detector_params_m": round(count_params(m), 3),
                    "embedder_params_m": 0.0,
                    "total_params_m": round(count_params(m), 3),
                    "notes": "Text encoder (CLIP-style) is inside this parameter count already.",
                }
            )
    except Exception as e:
        print(f"  skipping OWL-ViT: {e}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "params_report.csv"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    rows = semdetect_rows() + foundation_model_rows()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "name", "variant", "use_film", "detector_params_m", "embedder_params_m", "total_params_m", "notes"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")
    width = max(len(r["name"] + str(r["variant"])) for r in rows)
    for r in sorted(rows, key=lambda r: r["total_params_m"]):
        label = f"{r['name']} {r['variant']}".ljust(width + 1)
        print(f"  {label} {r['total_params_m']:>9.2f}M  [{r['category']}]")


if __name__ == "__main__":
    main()
