"""YOLO (Ultralytics) detector with an optional FiLM conditioning path.

This builds a real Ultralytics `DetectionModel` (whatever `variant`
resolves to, e.g. yolo26n/yolo11n/yolov8n - backbone, neck, and detection
head all come straight from Ultralytics) and reuses its official loss
(`v8DetectionLoss` / `E2ELoss`) and decode logic, rather than
reimplementing YOLO's label assignment. See ultralytics_base.py for the
shared construction/FiLM-hook/loss machinery (identical for RT-DETR) -
this file only adds YOLO's own eval-time decode in `predict()`.

    One or more feature maps are each modulated by a per-image
    (gamma, beta) predicted from the embedding of that image's object
    description, via a small MLP per modulated layer (see film.py). This
    is implemented as forward hooks, so it works for any Ultralytics
    single-Detect-head yaml without needing to hand-edit the architecture.

    Which layers get hooked is `film_layer_indices` (None = auto: the
    layers feeding the Detect head). For yolo26n the layer graph is:

        0  Conv                    -\
        1  Conv                     |  backbone stem, downsampling
        2  C3k2                     |
        3  Conv                     |
        4  C3k2  (P3/8)            -/  <- plain backbone features,
        5  Conv                     |     still spatially/texturally
        6  C3k2  (P4/16)            |     detailed - "early" conditioning
        7  Conv                     |
        8  C3k2  (P5/32)           -/
        9  SPPF                     |
        10 C2PSA                   -/  <- end of backbone (attention)
        11 Upsample  -\
        12 Concat      |
        13 C3k2        |  neck (PAFPN): fuses backbone stages
        14 Upsample     |  top-down + bottom-up
        15 Concat        |
        16 C3k2  (P3)   -+  <- default film_layer_indices target: the
        17 Conv          |     exact P3/P4/P5 inputs to Detect - "late"
        18 Concat        |     conditioning, after backbone AND neck
        19 C3k2  (P4)   -+     fusion, right before the head's own
        20 Conv          |     (unmodulated) cv2/cv3 conv towers.
        21 Concat        |
        22 C3k2  (P5)   -+
        23 Detect  <- takes [16, 19, 22] as input

    [4, 6, 8] is the backbone-level equivalent of [16, 19, 22]: same
    P3/P4/P5 resolutions, but before the neck mixes information across
    scales - worth comparing against the default when the thing FiLM is
    meant to inform (fine-grained colour/texture) is more available
    early than late. (yolov8n's graph numbers backbone/neck stages
    differently - print `model.detector.model` to check before reusing
    these indices with a different variant.)

Set `use_film=False` to disable: no hooks are registered at all, so this
collapses to the plain Ultralytics architecture with zero overhead,
which is the intended baseline to compare FiLM against.
"""
from __future__ import annotations

import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression

from semdetect.models.registry import register_model
from semdetect.models.ultralytics_base import UltralyticsFiLMDetector


@register_model("yolo")
class YOLOFiLMDetector(UltralyticsFiLMDetector):
    model_cls = DetectionModel

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

        results = []
        if self.detector.end2end:
            # y: (B, max_det, 6) = [x1, y1, x2, y2, conf, cls], already
            # top-k'd/NMS-free per Ultralytics' end-to-end head.
            for dets in y:
                keep = dets[:, 4] >= conf_thres
                kept = dets[keep]
                results.append({"boxes": kept[:, :4], "scores": kept[:, 4], "labels": kept[:, 5].long()})
        else:
            # y: (B, 4 + nc, num_anchors) raw decoded boxes/scores, needs NMS.
            nms_out = non_max_suppression(y, conf_thres=conf_thres, iou_thres=0.5, max_det=300)
            for dets in nms_out:
                results.append({"boxes": dets[:, :4], "scores": dets[:, 4], "labels": dets[:, 5].long()})
        return results
