"""RT-DETR (Ultralytics) detector with an optional FiLM conditioning path.

Shares all construction/FiLM-hook/loss machinery with YOLOFiLMDetector
(see ultralytics_base.py) - RT-DETR's `RTDETRDetectionModel` is built from
the same layer-indexed `nn.Sequential` structure and the same
`detector.loss(batch)` convention as YOLO's `DetectionModel`, just with a
transformer hybrid-encoder/decoder instead of a conv head, so the FiLM
hooks (forward hooks on chosen layer indices) apply identically.

Only `predict()` differs: the decoder head's eval-mode output is
`[cx, cy, w, h, conf, cls]` in *normalized* (0-1) coordinates (DETR-style
query regression), not the pixel-scale xyxy YOLO's anchor-based head
produces, so it needs its own decode.

For rtdetr-l the layer graph (`model.detector.model`) is an HGNetv2
backbone (HGStem/HGBlock stages) feeding a hybrid encoder (AIFI attention
+ RepC3 fusion) before the RTDETRDecoder head:

    0  HGStem                -\
    1  HGBlock                |  backbone stem, downsampling
    2  DWConv                 |
    3  HGBlock  (P3-ish)     -/  <- early film_layer_indices target:
    4  DWConv                      plain backbone features, referenced
    5  HGBlock                     directly by layer 19 below as a skip
    6  HGBlock  (P4-ish)
    7  HGBlock                -\  <- also an early target: referenced
    8  DWConv                     directly by layer 14 as a skip
    9  HGBlock  (P5-ish)      -/  <- also an early target: feeds straight
                                      into the encoder (10-11) below
    10 Conv
    11 AIFI                       <- attention, end of backbone/start of encoder
    12 Conv
    13 Upsample  -\
    14 [7] Conv    |  hybrid encoder: fuses backbone stages
    15 Concat       |  top-down + bottom-up (RepC3 blocks)
    16 RepC3         |
    17 Conv           |
    18 Upsample        |
    19 [3] Conv        |
    20 Concat          |
    21 RepC3   (P3) ---+  <- default film_layer_indices target: the
    22 Conv            |     exact inputs to RTDETRDecoder - "late"
    23 Concat          |     conditioning, after backbone AND encoder
    24 RepC3   (P4) ---+     fusion, right before the decoder's own
    25 Conv            |     cross-attention over these features.
    26 Concat          |
    27 RepC3   (P5) ---+
    28 RTDETRDecoder  <- takes [21, 24, 27] as input

so `[3, 7, 9]` is the backbone-level equivalent of `[21, 24, 27]`, same
relationship as YOLO's `[4, 6, 8]` vs `[16, 19, 22]` - same resolutions,
before the hybrid encoder mixes information across scales. Print
`model.detector.model` before picking film_layer_indices for a different
rtdetr variant; the stage numbering is architecture-specific.
"""
from __future__ import annotations

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from team_conditioned_detection.models.registry import register_model
from team_conditioned_detection.models.ultralytics_base import UltralyticsFiLMDetector


@register_model("rtdetr")
class RTDETRFiLMDetector(UltralyticsFiLMDetector):
    model_cls = RTDETRDetectionModel

    def _pretrained_loader(self, variant: str):
        from ultralytics import RTDETR

        return RTDETR(f"{variant}.pt")

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, embeddings: torch.Tensor | None = None, conf_thres: float = 0.25
    ) -> list[dict[str, torch.Tensor]]:
        was_training = self.training
        self.eval()
        try:
            y, _ = self._raw_forward(images, embeddings)
        finally:
            self.train(was_training)

        # y: (B, num_queries, 6) = [cx, cy, w, h, conf, cls], normalized (0-1).
        cx, cy, w, h = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        x1 = (cx - w / 2) * self.img_size
        y1 = (cy - h / 2) * self.img_size
        x2 = (cx + w / 2) * self.img_size
        y2 = (cy + h / 2) * self.img_size
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        results = []
        for i in range(y.shape[0]):
            keep = y[i, :, 4] >= conf_thres
            results.append({"boxes": boxes[i][keep], "scores": y[i, keep, 4], "labels": y[i, keep, 5].long()})
        return results
