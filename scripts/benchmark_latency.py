#!/usr/bin/env python3
"""Per-frame latency (batch 1, ms) of role classification on real test
frames with their ground-truth boxes:

    role_head   the integrated role head alone, given the detector feature
                map it taps (already computed by the detector's own forward)
    standalone  the standalone classifier, which crops and encodes the
                frame itself
    detector    the detector's own inference (forward + box decoding/NMS),
                for scale - role classification runs on top of it either way

Median over --frames frames after --warmup untimed ones. Run with nothing
else on the machine - a concurrent training job distorts both GPU and CPU
numbers.

    uv run python scripts/benchmark_latency.py            # cuda and cpu -> outputs/paper/latency.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from team_detect.data.robot_jersey import build_dataloader  # noqa: E402
from team_detect.engine.metrics import norm_cxcywh_to_xyxy  # noqa: E402
from team_detect.engine.trainer import move_batch  # noqa: E402
from team_detect.models.factory import STANDALONE, build_model  # noqa: E402
from team_detect.robot_config import load_robot_config  # noqa: E402

# key in latency.json -> experiment dir under --output-root
MODELS = {
    "fasterrcnn": "fasterrcnn_film",
    "rtdetr": "rtdetr_film",
    "yolo26n": "yolo26n_film",
    "standalone": "standalone_classifier",
}


def _time_ms(fn, device: str) -> float:
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0)


@torch.no_grad()
def benchmark(run_dir: Path, device: str, frames: int, warmup: int) -> dict[str, float]:
    config = load_robot_config(run_dir / "config.yaml")
    model = build_model(config, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)["model"])
    loader = build_dataloader(config, "test", batch_size=1)

    times: dict[str, list[float]] = {}
    n_boxes = []
    for i, batch in enumerate(loader):
        if i >= warmup + frames:
            break
        batch = move_batch(batch, device)
        img, emb = batch["img"], batch["embedding"]
        boxes = norm_cxcywh_to_xyxy(batch["bboxes"], config.data.img_size)
        idx = batch["batch_idx"].long()
        if config.model.architecture == STANDALONE:
            sample = {"standalone": _time_ms(lambda: model(img, boxes, idx, emb), device)}
        else:
            sample = {"detector": _time_ms(lambda: model._detect(img, config.train.conf_thres), device)}
            feature = model._capture_role_feature(img)
            sample["role_head"] = _time_ms(lambda: model.role_head(feature, img, boxes, idx, emb, model.feature_stride), device)
        if i >= warmup:
            n_boxes.append(len(boxes))
            for k, v in sample.items():
                times.setdefault(k, []).append(v)
    return {**{f"{k}_ms": statistics.median(v) for k, v in times.items()}, "frames": len(n_boxes), "mean_boxes_per_frame": statistics.mean(n_boxes)}


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs" / "paper")
    parser.add_argument("--seed", type=int, default=42, help="Which trained run to load (weights don't affect timing)")
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, help="Default: <output-root>/latency.json")
    args = parser.parse_args()

    result = {"cpu_name": _cpu_name(), "cpu_threads": torch.get_num_threads()}
    for device in args.devices:
        if device == "cuda":
            result["gpu_name"] = torch.cuda.get_device_name()
        result[device] = {}
        for key, experiment in MODELS.items():
            print(f"[{device}] {key}", flush=True)
            result[device][key] = benchmark(args.output_root / experiment / f"seed{args.seed}", device, args.frames, args.warmup)
    out = args.output or args.output_root / "latency.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
