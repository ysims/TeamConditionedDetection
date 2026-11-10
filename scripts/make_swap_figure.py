#!/usr/bin/env python3
"""Qualitative context-swap figure: test frames with the full pipeline's
predictions (detected boxes + roles) under the supplied teammate/opponent
colours (top row) and with the two colours exchanged (bottom row). Each box
is drawn in the supplied colour of its predicted role (T teammate, solid;
O opponent, dashed); the swatches top-left show the supplied context.

Frames are not hand-picked: by default, of the test frames that contain both
teams and at least --min-robots robots, --n frames evenly spaced through the
test set are shown. Pass --frames to choose specific test indices instead.

    uv run python scripts/make_swap_figure.py     # -> images/swap_qualitative.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from team_detect.data.robot_jersey import RobotJerseyDataset  # noqa: E402
from team_detect.evaluation import _swap_context  # noqa: E402
from team_detect.models.factory import build_model  # noqa: E402
from team_detect.robot_config import load_robot_config  # noqa: E402

ROLE_STYLE = {0: ("T", "-"), 1: ("O", "--")}  # teammate solid, opponent dashed


def candidate_frames(ds: RobotJerseyDataset, min_robots: int) -> list[int]:
    return [i for i, robots in enumerate(ds.images) if len(robots) >= min_robots and len({r["jersey_colour"] for r in robots}) == 2]


def content_window(ds: RobotJerseyDataset, idx: int) -> tuple[int, int, int, int]:
    """The letterboxed frame's non-padding region, (x0, y0, x1, y1)."""
    w, h = Image.open(ds.images[idx][0]["image_path"]).size
    scale = min(ds.img_size / w, ds.img_size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x, pad_y = (ds.img_size - new_w) // 2, (ds.img_size - new_h) // 2
    return pad_x, pad_y, pad_x + new_w, pad_y + new_h


def draw(ax, image: torch.Tensor, window, prediction: dict, embedding: torch.Tensor, title: str) -> None:
    x0, y0, x1, y1 = window
    ax.imshow(image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()[y0:y1, x0:x1])
    teammate_rgb, opponent_rgb = embedding[:3].tolist(), embedding[3:].tolist()
    for box, label in zip(prediction["boxes"].tolist(), prediction["labels"].tolist()):
        text, linestyle = ROLE_STYLE[label]
        colour = teammate_rgb if label == 0 else opponent_rgb
        bx0, by0, bx1, by1 = box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0
        ax.add_patch(patches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor=colour, linewidth=2, linestyle=linestyle))
        ax.text(bx0, by0 - 3, text, color="white", fontsize=7, fontweight="bold", va="bottom",
                bbox={"facecolor": colour, "edgecolor": "none", "pad": 1})
    for j, (name, rgb) in enumerate((("T", teammate_rgb), ("O", opponent_rgb))):  # the supplied context
        ax.add_patch(patches.Rectangle((8 + 62 * j, 8), 56, 34, facecolor=rgb, edgecolor="white", linewidth=1.5))
        ax.text(36 + 62 * j, 25, name, color="white", fontsize=8, fontweight="bold", ha="center", va="center")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=REPO / "outputs" / "paper" / "fasterrcnn_film" / "seed42")
    parser.add_argument("--frames", nargs="+", type=int, help="Test-set indices (default: evenly spaced candidates)")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--min-robots", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.5, help="Detection score threshold for drawn boxes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO / "images" / "swap_qualitative.pdf")
    args = parser.parse_args()

    config = load_robot_config(args.run_dir / "config.yaml")
    model = build_model(config, pretrained=False).to(args.device).eval()
    model.load_state_dict(torch.load(args.run_dir / "checkpoints" / "best.pt", map_location=args.device)["model"])
    ds = RobotJerseyDataset(config.data.test_manifest, img_size=config.data.img_size, randomize_roles=False, augment=False, seed=config.data.eval_seed)

    frames = args.frames
    if frames is None:
        candidates = candidate_frames(ds, args.min_robots)
        if len(candidates) < args.n:
            raise SystemExit(f"Only {len(candidates)} test frames with both teams and >= {args.min_robots} robots")
        frames = [candidates[round(k * (len(candidates) - 1) / max(args.n - 1, 1))] for k in range(args.n)]

    fig, axes = plt.subplots(2, len(frames), figsize=(2.4 * len(frames), 3.6), squeeze=False)
    for col, idx in enumerate(frames):
        item = ds[idx]
        image = item["image"].unsqueeze(0).to(args.device)
        embedding = item["embedding"].unsqueeze(0).to(args.device)
        window = content_window(ds, idx)
        for row, (context, title) in enumerate(((embedding, "Supplied"), (_swap_context(embedding), "Swapped"))):
            prediction = model.predict(image, context, conf_thres=args.conf)[0]
            draw(axes[row, col], image[0], window, prediction, context[0], f"{title} (test #{idx})")
    fig.tight_layout(pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output} (test frames {frames})")


if __name__ == "__main__":
    main()
