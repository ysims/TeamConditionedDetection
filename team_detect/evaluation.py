#!/usr/bin/env python3
"""Paper evaluation, on the test split with the fixed teammate/opponent
assignment drawn from config.data.eval_seed - so every training seed, and
the colour rules, are scored against the same labels.

    role_acc            role accuracy on ground-truth boxes, original context
    swap_acc            accuracy with teammate/opponent colours exchanged,
                        scored against the correspondingly flipped label
    both_correct        fraction of boxes correct under both contexts
    cond_swap_acc       P(swap correct | original correct)
    flip_rate           fraction of boxes whose predicted role changes when
                        the context is exchanged (correct or not)
    <role>_precision,   per-class precision/recall of the original-context
    <role>_recall       ground-truth-box predictions, role in ROLE_NAMES
    map_50              mAP50 over predict()'s boxes + roles (detectors only)
    role_params         parameters added for role classification
    per_box_correct     original-context correctness of each test box, in
                        loader order (matches colour_rule_accuracy's per_box)

    python -m team_detect.evaluation --run-dir outputs/paper/fasterrcnn_film/seed42
    python -m team_detect.evaluation --colour-rules --manifest data/manifest_robots/test.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

from team_detect.data.robot_jersey import ROLE_NAMES, RobotJerseyDataset, build_dataloader, robot_collate_fn
from team_detect.engine.metrics import DetectionMetrics, batch_to_targets, norm_cxcywh_to_xyxy
from team_detect.engine.trainer import move_batch
from team_detect.models.factory import STANDALONE, build_model, role_classification_params
from team_detect.models.robot_role import min_cell_distances
from team_detect.robot_config import load_robot_config


def _swap_context(embedding: torch.Tensor) -> torch.Tensor:
    return torch.cat([embedding[:, 3:], embedding[:, :3]], dim=-1)


@torch.no_grad()
def role_and_swap_accuracy(model, loader: DataLoader, device: str, img_size: int) -> dict[str, float]:
    """Runs model.classify_roles on every ground-truth box twice - with the
    supplied context and with it exchanged - and scores the exchanged
    prediction against the flipped label.
    """
    model.eval()
    labels, normal_pred, swapped_pred = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        boxes = norm_cxcywh_to_xyxy(batch["bboxes"], img_size)
        batch_idx = batch["batch_idx"].long()
        labels.append(batch["cls"].long().cpu())
        normal_pred.append(model.classify_roles(batch["img"], boxes, batch_idx, batch["embedding"]).argmax(dim=-1).cpu())
        swapped_pred.append(model.classify_roles(batch["img"], boxes, batch_idx, _swap_context(batch["embedding"])).argmax(dim=-1).cpu())
    labels, normal_pred, swapped_pred = torch.cat(labels), torch.cat(normal_pred), torch.cat(swapped_pred)
    normal, swapped = normal_pred == labels, swapped_pred == 1 - labels
    result = {
        "n_boxes": labels.numel(),
        "role_acc": normal.float().mean().item(),
        "swap_acc": swapped.float().mean().item(),
        "both_correct": (normal & swapped).float().mean().item(),
        "cond_swap_acc": (normal & swapped).sum().item() / normal.sum().item() if normal.any() else float("nan"),
        "flip_rate": (normal_pred != swapped_pred).float().mean().item(),
        "per_box_correct": normal.int().tolist(),
    }
    for cls, role in enumerate(ROLE_NAMES):
        tp = ((normal_pred == cls) & (labels == cls)).sum().item()
        result[f"{role}_precision"] = tp / max((normal_pred == cls).sum().item(), 1)
        result[f"{role}_recall"] = tp / max((labels == cls).sum().item(), 1)
    return result


@torch.no_grad()
def detection_map50(model, loader: DataLoader, device: str, img_size: int, conf_thres: float) -> float:
    model.eval()
    metrics = DetectionMetrics(ROLE_NAMES, iou_thres=0.5, conf_thres=conf_thres)
    for batch in loader:
        batch = move_batch(batch, device)
        metrics.update(model.predict(batch["img"], batch["embedding"], conf_thres=conf_thres), batch_to_targets(batch, img_size))
    return metrics.compute()["map_50"]


def evaluate_run(run_dir: str | Path, device: str = "cuda", batch_size: int = 8) -> dict:
    """Evaluates <run_dir>/checkpoints/best.pt using <run_dir>/config.yaml."""
    run_dir = Path(run_dir)
    config = load_robot_config(run_dir / "config.yaml")
    model = build_model(config, pretrained=False).to(device)  # weights come from the checkpoint
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)["model"])
    loader = build_dataloader(config, "test", batch_size=batch_size)

    result = role_and_swap_accuracy(model, loader, device, config.data.img_size)
    result["role_params"] = role_classification_params(model)
    if config.model.architecture != STANDALONE:
        result["map_50"] = detection_map50(model, loader, device, config.data.img_size, config.train.conf_thres)
    return result


TORSO_BAND = (1 / 3, 1 / 4, 2 / 3, 3 / 5)  # (x0, y0, x1, y1) as fractions of the box - covers the jersey on a crop sheet of test robots


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """(..., 3) RGB in [0,1] -> (..., 3) HSV, hue in [0,1)."""
    r, g, b = rgb.unbind(-1)
    v, _ = rgb.max(-1)
    delta = v - rgb.min(-1).values
    s = torch.where(v > 0, delta / v.clamp(min=1e-8), torch.zeros_like(v))
    d = delta.clamp(min=1e-8)
    h = torch.where(v == r, (g - b) / d, torch.where(v == g, 2 + (b - r) / d, 4 + (r - g) / d))
    h = torch.where(delta > 0, (h / 6) % 1.0, torch.zeros_like(h))
    return torch.stack([h, s, v], dim=-1)


def hue_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular distance between hues in [0,1), in [0, 0.5]."""
    d = (a - b).abs() % 1.0
    return torch.minimum(d, 1 - d)


def chroma_weighted_hue(pixels: torch.Tensor) -> torch.Tensor:
    """(N, P, 3) RGB -> (N,) circular mean hue, each pixel weighted by
    saturation x value so dark and grey pixels (chassis, shadow) barely count."""
    hsv = rgb_to_hsv(pixels)
    angle, weight = 2 * torch.pi * hsv[..., 0], hsv[..., 1] * hsv[..., 2]
    mean = torch.atan2((weight * angle.sin()).sum(-1), (weight * angle.cos()).sum(-1))
    return (mean / (2 * torch.pi)) % 1.0


def torso_colour(pixels: torch.Tensor) -> torch.Tensor:
    """(N, P, 3) RGB -> (N, 3) observed HSV of the torso band: the
    chroma-weighted circular mean hue, and saturation and value averaged
    with the same saturation x value weights."""
    hsv = rgb_to_hsv(pixels)
    weight = hsv[..., 1] * hsv[..., 2]
    sv = (weight[..., None] * hsv[..., 1:]).sum(1) / weight.sum(1, keepdim=True).clamp(min=1e-8)
    return torch.cat([chroma_weighted_hue(pixels)[:, None], sv], dim=1)


def _wrap(h: torch.Tensor) -> torch.Tensor:
    """Hue difference (fractions of a turn) wrapped to [-0.5, 0.5)."""
    return (h + 0.5) % 1.0 - 0.5


class HueCalibration:
    """Maps a supplied jersey colour to the colour its torso band is observed
    as after rendering, fitted on training robots. Supplied hue is split into
    `bins` equal bins; each holds the circular mean of the hue offset
    (observed - supplied) and the median observed/supplied saturation and
    value ratios. Between bin centres the offset is interpolated linearly,
    wrapping at 0/360 degrees; the ratios are taken from the nearest bin.
    Bins without training robots are skipped by the interpolation.
    """

    def __init__(self, bins: int):
        self.bins = bins

    def fit(self, supplied: torch.Tensor, observed: torch.Tensor) -> "HueCalibration":
        """supplied, observed: (N, 3) HSV, hue in [0, 1)."""
        idx = (supplied[:, 0] * self.bins).long() % self.bins
        angle = 2 * torch.pi * _wrap(observed[:, 0] - supplied[:, 0])
        self.offset = torch.full((self.bins,), float("nan"))
        self.ratio = torch.full((self.bins, 2), float("nan"))
        for b in range(self.bins):
            sel = idx == b
            if sel.any():
                self.offset[b] = torch.atan2(angle[sel].sin().mean(), angle[sel].cos().mean()) / (2 * torch.pi)
                self.ratio[b] = (observed[sel, 1:] / supplied[sel, 1:].clamp(min=1e-8)).median(dim=0).values
        valid = ~self.offset.isnan()
        self.centres, self.valid_offset = ((torch.arange(self.bins) + 0.5) / self.bins)[valid], self.offset[valid]
        self.valid_ratio = self.ratio[valid]
        return self

    def predict(self, supplied: torch.Tensor) -> torch.Tensor:
        """(N, 3) supplied HSV -> (N, 3) predicted observed HSV."""
        h = supplied[:, 0]
        c, o = self.centres, self.valid_offset
        right = torch.searchsorted(c, h.contiguous()) % len(c)  # first centre above h, wrapping
        left = (right - 1) % len(c)
        span = (c[right] - c[left]) % 1.0
        span = torch.where(span == 0, torch.ones_like(span), span)  # a single valid bin
        t = ((h - c[left]) % 1.0) / span
        offset = o[left] + t * _wrap(o[right] - o[left])
        nearest = torch.where(hue_distance(h, c[left]) <= hue_distance(h, c[right]), left, right)
        sv = (supplied[:, 1:] * self.valid_ratio[nearest]).clamp(max=1.0)
        return torch.cat([((h + offset) % 1.0)[:, None], sv], dim=1)


def hsv_cylinder_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between (..., 3) HSV colours embedded in the HSV
    cylinder as (s cos h, s sin h, v)."""
    def embed(x):
        angle = 2 * torch.pi * x[..., 0]
        return torch.stack([x[..., 1] * angle.cos(), x[..., 1] * angle.sin(), x[..., 2]], dim=-1)
    return (embed(a) - embed(b)).norm(dim=-1)


def _torso_band_pixels(images: torch.Tensor, boxes: torch.Tensor, batch_idx: torch.Tensor, size: int = 16) -> torch.Tensor:
    """(N, size*size, 3) RGB pixels of each box's TORSO_BAND."""
    wh = boxes[:, 2:] - boxes[:, :2]
    band = torch.cat([boxes[:, :2] + wh * torch.tensor(TORSO_BAND[:2]), boxes[:, :2] + wh * torch.tensor(TORSO_BAND[2:])], dim=1)
    rois = torch.cat([batch_idx.unsqueeze(1).float(), band], dim=1)
    return roi_align(images, rois, output_size=size, spatial_scale=1.0, aligned=True).flatten(2).transpose(1, 2)


@torch.no_grad()
def torso_observations(manifest: str, img_size: int = 640, eval_seed: int = 0, batch_size: int = 8) -> dict[str, torch.Tensor]:
    """Per ground-truth box, in loader order: the observed torso-band HSV,
    the HSV of the supplied teammate and opponent colours, and the label."""
    ds = RobotJerseyDataset(manifest, img_size=img_size, randomize_roles=False, augment=False, seed=eval_seed)
    out = {"observed": [], "teammate": [], "opponent": [], "labels": []}
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=robot_collate_fn):
        batch_idx = batch["batch_idx"].long()
        per_box_embedding = batch["embedding"][batch_idx]
        out["observed"].append(torso_colour(_torso_band_pixels(batch["img"], norm_cxcywh_to_xyxy(batch["bboxes"], img_size), batch_idx)))
        out["teammate"].append(rgb_to_hsv(per_box_embedding[:, :3]))
        out["opponent"].append(rgb_to_hsv(per_box_embedding[:, 3:]))
        out["labels"].append(batch["cls"].long().view(-1))
    return {k: torch.cat(v) for k, v in out.items()}


def _calibrated_correct(cal: HueCalibration, obs: dict[str, torch.Tensor], cylinder: bool) -> torch.Tensor:
    """Per-box correctness of assigning each box the role whose calibrated
    (predicted observed) colour is closer to its observed torso colour."""
    pred_t, pred_o = cal.predict(obs["teammate"]), cal.predict(obs["opponent"])
    if cylinder:
        dist_t, dist_o = hsv_cylinder_distance(obs["observed"], pred_t), hsv_cylinder_distance(obs["observed"], pred_o)
    else:
        dist_t, dist_o = hue_distance(obs["observed"][:, 0], pred_t[:, 0]), hue_distance(obs["observed"][:, 0], pred_o[:, 0])
    return (dist_o < dist_t).long() == obs["labels"]


def calibrated_colour_rules(train_manifest: str, val_manifest: str, test_obs: dict[str, torch.Tensor], img_size: int, eval_seed: int,
                            bin_choices: tuple[int, ...] = (12, 24, 36)) -> dict:
    """Fits HueCalibration on the training robots (each paired with its own
    supplied colour) for every bin count, picks the count with the best
    validation accuracy (the smallest on ties) separately for the hue-only
    and HSV-cylinder rules, and scores the chosen ones on the test boxes."""
    train, val = (torso_observations(m, img_size, eval_seed) for m in (train_manifest, val_manifest))
    own = torch.where(train["labels"][:, None] == 0, train["teammate"], train["opponent"])
    fits = {k: HueCalibration(k).fit(own, train["observed"]) for k in bin_choices}
    result = {}
    for name, cylinder in (("calibrated_hue", False), ("calibrated_hsv", True)):
        val_acc = {k: _calibrated_correct(cal, val, cylinder).float().mean().item() for k, cal in fits.items()}
        best = max(bin_choices, key=lambda k: (val_acc[k], -k))
        correct = _calibrated_correct(fits[best], test_obs, cylinder)
        result[name] = {"bins": best, "val_acc": val_acc, "test_acc": correct.float().mean().item(), "correct": correct}
    return result


def dominant_colour(pixels: torch.Tensor, k: int = 3, iters: int = 20) -> torch.Tensor:
    """(N, P, 3) -> (N, 3): centre of the largest of k k-means clusters per box,
    initialised by k-means++ with a fixed seed."""
    gen = torch.Generator().manual_seed(0)
    out = []
    for px in pixels:
        centres = px[torch.randint(len(px), (1,), generator=gen)]
        for _ in range(k - 1):
            d2 = torch.cdist(px, centres).min(dim=1).values ** 2
            centres = torch.cat([centres, px[torch.multinomial(d2 + 1e-12, 1, generator=gen)]])
        for _ in range(iters):
            assign = torch.cdist(px, centres).argmin(dim=1)
            centres = torch.stack([px[assign == j].mean(0) if (assign == j).any() else centres[j] for j in range(k)])
        out.append(centres[torch.bincount(assign, minlength=k).argmax()])
    return torch.stack(out)


@torch.no_grad()
def colour_rule_accuracy(manifest: str, train_manifest: str, val_manifest: str, img_size: int = 640, eval_seed: int = 0, grid_size: int = 8,
                         centre_frac: float = 0.15, batch_size: int = 8) -> dict:
    """Non-learned baselines on ground-truth boxes: assign the box the role
    of whichever supplied colour its colour statistic is closer to.

        whole_/centre_mean  mean RGB of the whole box / a centred window of
                            centre_frac of its width and height
        whole_/centre_min   minimum RGB distance over a grid_size x grid_size
                            grid of cells (the same computation as RoleHead's
                            distance feature)
        torso_hue           saturation x value weighted circular mean hue of
                            the TORSO_BAND, vs the supplied colours' hues
        torso_kmeans        RGB of the largest k-means (k=3) cluster in the
                            TORSO_BAND, vs the supplied colours
        calibrated_hue      torso_hue's observed hue vs each supplied colour's
                            hue after HueCalibration (fitted on train_manifest,
                            bin count chosen on val_manifest)
        calibrated_hsv      the same in the HSV cylinder, with calibrated
                            saturation and value as well; *_bins and *_val_acc
                            record the bin selection

    per_box holds, per test box in loader order, the hue separation of the
    two supplied colours (delta_hue, degrees) and torso_hue's and the
    calibrated rules' correctness.

    Also reports how far the rendered jersey is from the colour it was
    supplied as, measured at the TORSO_BAND pixel closest in hue to the
    box's own colour (among pixels with saturation x value >= 0.1):
    hue_error_median/_p75 (degrees), and the rendered/supplied
    saturation_ratio_median and value_ratio_median.
    """
    ds = RobotJerseyDataset(manifest, img_size=img_size, randomize_roles=False, augment=False, seed=eval_seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=robot_collate_fn)
    correct = {"whole_mean": 0, "whole_min": 0, "centre_mean": 0, "centre_min": 0, "torso_hue": 0, "torso_kmeans": 0}
    shift = {"hue_error": [], "saturation_ratio": [], "value_ratio": []}
    test_obs = {"observed": [], "teammate": [], "opponent": [], "labels": []}
    per_box = {"delta_hue": [], "torso_hue": []}
    n = 0
    for batch in loader:
        boxes = norm_cxcywh_to_xyxy(batch["bboxes"], img_size)
        batch_idx = batch["batch_idx"].long()
        labels = batch["cls"].long()
        per_box_embedding = batch["embedding"][batch_idx]
        teammate, opponent = per_box_embedding[:, :3], per_box_embedding[:, 3:]
        own_colour = torch.where(labels[:, None] == 0, teammate, opponent)

        def predict(dist_t, dist_o):  # 0=teammate, 1=opponent
            return ((dist_o < dist_t).long().view(-1) == labels).sum().item()

        def crop(region_boxes, size):
            rois = torch.cat([batch_idx.unsqueeze(1).float(), region_boxes], dim=1)
            return roi_align(batch["img"], rois, output_size=size, spatial_scale=1.0, aligned=True)

        centre = (boxes[:, :2] + boxes[:, 2:]) / 2
        half = (boxes[:, 2:] - boxes[:, :2]) / 2 * centre_frac
        for region, region_boxes in {"whole": boxes, "centre": torch.cat([centre - half, centre + half], dim=1)}.items():
            grid = crop(region_boxes, grid_size)
            mean_rgb = grid.mean(dim=(2, 3))
            correct[f"{region}_mean"] += predict((mean_rgb - teammate).norm(dim=-1), (mean_rgb - opponent).norm(dim=-1))
            correct[f"{region}_min"] += predict(*min_cell_distances(grid, per_box_embedding))

        pixels = _torso_band_pixels(batch["img"], boxes, batch_idx)  # (N, 256, 3)
        observed, teammate_hsv, opponent_hsv = torso_colour(pixels), rgb_to_hsv(teammate), rgb_to_hsv(opponent)
        for key, value in (("observed", observed), ("teammate", teammate_hsv), ("opponent", opponent_hsv), ("labels", labels)):
            test_obs[key].append(value)
        per_box["delta_hue"].append(360 * hue_distance(teammate_hsv[:, 0], opponent_hsv[:, 0]))

        hue_correct = (hue_distance(observed[:, 0], opponent_hsv[:, 0]) < hue_distance(observed[:, 0], teammate_hsv[:, 0])).long() == labels
        correct["torso_hue"] += hue_correct.sum().item()
        per_box["torso_hue"].append(hue_correct)

        hsv, own_hsv = rgb_to_hsv(pixels), rgb_to_hsv(own_colour)
        d = hue_distance(hsv[..., 0], own_hsv[:, None, 0])
        d[hsv[..., 1] * hsv[..., 2] < 0.1] = 1.0
        closest = hsv[torch.arange(len(d)), d.argmin(dim=1)]
        shift["hue_error"].append(360 * d.min(dim=1).values)
        shift["saturation_ratio"].append(closest[:, 1] / own_hsv[:, 1].clamp(min=1e-8))
        shift["value_ratio"].append(closest[:, 2] / own_hsv[:, 2].clamp(min=1e-8))

        dominant = dominant_colour(pixels)
        correct["torso_kmeans"] += predict((dominant - teammate).norm(dim=-1), (dominant - opponent).norm(dim=-1))
        n += labels.shape[0]

    calibrated = calibrated_colour_rules(train_manifest, val_manifest, {k: torch.cat(v) for k, v in test_obs.items()}, img_size, eval_seed)
    per_box = {k: torch.cat(v) for k, v in per_box.items()}
    for name, fit in calibrated.items():
        correct[name] = fit["test_acc"] * n
        per_box[name] = fit["correct"]
    return {
        "n_boxes": n,
        **{k: v / n for k, v in correct.items()},
        **{f"{name}_{field}": fit[field] for name, fit in calibrated.items() for field in ("bins", "val_acc")},
        "hue_error_median": torch.cat(shift["hue_error"]).median().item(),
        "hue_error_p75": torch.cat(shift["hue_error"]).quantile(0.75).item(),
        "saturation_ratio_median": torch.cat(shift["saturation_ratio"]).median().item(),
        "value_ratio_median": torch.cat(shift["value_ratio"]).median().item(),
        "per_box": {"delta_hue": [round(d, 2) for d in per_box.pop("delta_hue").tolist()], **{k: v.int().tolist() for k, v in per_box.items()}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", help="A training output dir containing config.yaml and checkpoints/best.pt")
    parser.add_argument("--colour-rules", action="store_true", help="Evaluate the non-learned colour rules instead")
    parser.add_argument("--manifest", default="data/manifest_robots/test.csv")
    parser.add_argument("--train-manifest", default="data/manifest_robots/train.csv", help="--colour-rules only: fits the calibrated rules")
    parser.add_argument("--val-manifest", default="data/manifest_robots/val.csv", help="--colour-rules only: selects the calibration bin count")
    parser.add_argument("--img-size", type=int, default=640, help="--colour-rules only")
    parser.add_argument("--eval-seed", type=int, default=0, help="--colour-rules only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", help="Also write the result as JSON to this path")
    args = parser.parse_args()

    if args.colour_rules:
        result = colour_rule_accuracy(args.manifest, args.train_manifest, args.val_manifest, img_size=args.img_size, eval_seed=args.eval_seed)
    elif args.run_dir:
        result = evaluate_run(args.run_dir, args.device)
    else:
        parser.error("pass --run-dir or --colour-rules")
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
