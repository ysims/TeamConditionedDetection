"""Detection metrics: COCO-style mAP via torchmetrics, plus a simple
precision/recall/F1 at a fixed confidence + IoU threshold (mAP integrates
over confidence, which is the right metric for comparing models, but a
single P/R/F1 number at the operating point you'd actually deploy at is
easier to eyeball in a report).

All boxes are expected in the same coordinate frame (pixels of the
letterboxed img_size x img_size input the model was run on) - both the
predictions coming out of Detector.predict() and the targets built by
`batch_to_targets` below satisfy this.
"""
from __future__ import annotations

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def norm_cxcywh_to_xyxy(bboxes: torch.Tensor, img_size: int) -> torch.Tensor:
    cx, cy, w, h = bboxes.unbind(-1)
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return torch.stack([x1, y1, x2, y2], dim=-1)


def batch_to_targets(batch: dict, img_size: int) -> list[dict[str, torch.Tensor]]:
    """Split the collated (cls, bboxes, batch_idx) tensors back into one
    dict per image, matching Detector.predict()'s output format.
    """
    batch_size = batch["img"].shape[0]
    boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], img_size)
    targets = []
    for i in range(batch_size):
        mask = batch["batch_idx"] == i
        targets.append({"boxes": boxes_xyxy[mask], "labels": batch["cls"][mask].long()})
    return targets


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-9)


class DetectionMetrics:
    """Accumulates predictions/targets across a whole eval pass, mirroring
    the usual `metric.update(...)` per batch, `metric.compute()` once loop.
    """

    def __init__(self, class_names: list[str], iou_thres: float = 0.5, conf_thres: float = 0.25):
        self.class_names = class_names
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=len(class_names) > 1)
        self._tp = self._fp = self._fn = 0

    def update(self, preds: list[dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]]) -> None:
        self.map_metric.update(preds, targets)
        for pred, target in zip(preds, targets):
            keep = pred["scores"] >= self.conf_thres
            tp, fp, fn = self._match_one_image(pred["boxes"][keep], pred["labels"][keep], target["boxes"], target["labels"])
            self._tp += tp
            self._fp += fp
            self._fn += fn

    def _match_one_image(self, pred_boxes, pred_labels, gt_boxes, gt_labels) -> tuple[int, int, int]:
        if gt_boxes.shape[0] == 0:
            return 0, pred_boxes.shape[0], 0
        if pred_boxes.shape[0] == 0:
            return 0, 0, gt_boxes.shape[0]
        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        tp = 0
        for p in range(pred_boxes.shape[0]):
            best_iou, best_g = -1.0, -1
            for g in range(gt_boxes.shape[0]):
                if g in matched_gt or pred_labels[p] != gt_labels[g]:
                    continue
                if ious[p, g] > best_iou:
                    best_iou, best_g = ious[p, g].item(), g
            if best_iou >= self.iou_thres:
                matched_gt.add(best_g)
                tp += 1
        fp = pred_boxes.shape[0] - tp
        fn = gt_boxes.shape[0] - len(matched_gt)
        return tp, fp, fn

    def compute(self) -> dict[str, float]:
        map_result = self.map_metric.compute()
        out = {}
        for key in ("map", "map_50", "map_75", "mar_1", "mar_10", "mar_100"):
            if key in map_result:
                value = map_result[key]
                out[key] = float(value) if value.numel() == 1 else value.tolist()
        precision = self._tp / max(self._tp + self._fp, 1)
        recall = self._tp / max(self._tp + self._fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        out.update(
            {
                f"precision@conf{self.conf_thres}_iou{self.iou_thres}": precision,
                f"recall@conf{self.conf_thres}_iou{self.iou_thres}": recall,
                f"f1@conf{self.conf_thres}_iou{self.iou_thres}": f1,
            }
        )
        return out

    def reset(self) -> None:
        self.map_metric.reset()
        self._tp = self._fp = self._fn = 0
