"""Faster R-CNN / FCOS (torchvision) detectors with an optional FiLM
conditioning path.

Both share torchvision's `BackboneWithFPN`: `backbone.body` (per-stage
ResNet/MobileNet features, before the FPN, varying channels per stage -
"early") feeds `backbone.fpn` (4-5 pyramid levels, uniform 256ch - "late",
exactly what feeds the detection head). FiLM is a single forward hook on
whichever of the two `film_early` selects, modulating every level in its
output dict with its own per-level generator.

Unlike the Ultralytics-backed detectors (yolo_film.py, rtdetr_film.py),
torchvision's own forward pass *requires* targets during training - loss
is computed inside forward, not as a separate step - so
Detector.compute_loss calls `self.model(images, targets)` directly, and
predict() calls `self.model(images)` in eval mode, which torchvision
already returns in our {"boxes", "scores", "labels"}-per-image format
(with predicted boxes already rescaled back to the input image's own
coordinate frame, even though the model internally resizes to its own
default processing size - min_size/max_size are pinned to img_size below
so that internal resize is a no-op and every architecture in this repo
sees images at the same resolution).
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

from team_conditioned_detection.models.base import Detector
from team_conditioned_detection.models.film import FiLMGenerator, apply_film
from team_conditioned_detection.models.registry import register_model

# variant name -> torchvision.models.detection constructor function name.
_FASTERRCNN_CONSTRUCTORS = {
    "resnet50": "fasterrcnn_resnet50_fpn_v2",
    "resnet50_v1": "fasterrcnn_resnet50_fpn",
    "mobilenet_v3_large": "fasterrcnn_mobilenet_v3_large_fpn",
    "mobilenet_v3_large_320": "fasterrcnn_mobilenet_v3_large_320_fpn",
}
_FCOS_CONSTRUCTORS = {
    "resnet50": "fcos_resnet50_fpn",
}


class TorchvisionFPNFiLMDetector(Detector):
    constructors: dict[str, str] = {}  # set by subclass
    #: Faster R-CNN reserves class 0 for background (so num_classes+1 in
    #: the constructor, and +1/-1 shifting labels in/out); FCOS is
    #: anchor-free/focal-loss and uses classes 0..N-1 directly.
    label_offset: int = 0
    #: Each torchvision detector applies its own internal confidence
    #: threshold before NMS (FCOS defaults to 0.2, Faster R-CNN to 0.05,
    #: under different kwarg names). Dropped low (not to exactly 0.0 - that
    #: forces NMS to run over the *entire* unfiltered anchor grid every
    #: eval call, tens of thousands of boxes/image for FCOS, which was
    #: observed to occasionally pathologically stall a whole training run)
    #: so our own conf_thres is still effectively the only filter that
    #: matters, same intent as the Ultralytics-backed detectors (which do
    #: no internal filtering), without the pre-NMS candidate set being
    #: unbounded.
    score_thresh_kwarg: str = "score_thresh"
    score_thresh_value: float = 1e-3

    def __init__(
        self,
        num_classes: int = 1,
        class_names: list[str] | None = None,
        variant: str = "resnet50",
        img_size: int = 640,
        pretrained: bool = False,
        use_film: bool = True,
        embed_dim: int = 512,
        film_hidden_dim: int = 256,
        film_early: bool = False,
        **kwargs,
    ):
        super().__init__()
        if variant not in self.constructors:
            raise ValueError(f"Unknown variant {variant!r} for {type(self).__name__}, expected one of {sorted(self.constructors)}")
        self.use_film = use_film
        self.img_size = img_size
        self.film_early = film_early

        constructor = getattr(torchvision.models.detection, self.constructors[variant])
        self.model = constructor(
            weights=None,
            weights_backbone="DEFAULT" if pretrained else None,
            num_classes=num_classes + self.label_offset,
            min_size=img_size,
            max_size=img_size,
            **{self.score_thresh_kwarg: self.score_thresh_value},
        )

        self._current_embedding: torch.Tensor | None = None
        self._hook_handles: list = []
        if self.use_film:
            channels = self._discover_channels()
            self.film_generators = nn.ModuleDict(
                {key: FiLMGenerator(embed_dim, c, hidden_dim=film_hidden_dim) for key, c in channels.items()}
            )
            self._register_hooks()
        else:
            self.film_generators = nn.ModuleDict()

    def _film_target_module(self) -> nn.Module:
        return self.model.backbone.body if self.film_early else self.model.backbone.fpn

    @torch.no_grad()
    def _discover_channels(self) -> dict[str, int]:
        was_training = self.model.training
        self.model.eval()
        captured: dict[str, int] = {}

        def hook(module, inputs, output):
            for key, feat in output.items():
                captured[key] = feat.shape[1]

        handle = self._film_target_module().register_forward_hook(hook)
        self.model(torch.zeros(1, 3, self.img_size, self.img_size))
        handle.remove()
        self.model.train(was_training)
        return captured

    def _register_hooks(self) -> None:
        def hook(module, inputs, output):
            if self._current_embedding is None:
                return output
            modulated = OrderedDict()
            for key, feat in output.items():
                gamma, beta = self.film_generators[key](self._current_embedding.to(feat.dtype))
                modulated[key] = apply_film(feat, gamma, beta)
            return modulated

        handle = self._film_target_module().register_forward_hook(hook)
        self._hook_handles.append(handle)

    def _to_targets(self, batch: dict) -> tuple[list[torch.Tensor], list[dict]]:
        images = list(batch["img"].unbind(0))
        targets = []
        for i in range(batch["img"].shape[0]):
            mask = batch["batch_idx"] == i
            cx, cy, w, h = batch["bboxes"][mask].unbind(-1)
            size = self.img_size
            xyxy = torch.stack(
                [(cx - w / 2) * size, (cy - h / 2) * size, (cx + w / 2) * size, (cy + h / 2) * size], dim=-1
            )
            labels = batch["cls"][mask].long() + self.label_offset
            targets.append({"boxes": xyxy, "labels": labels})
        return images, targets

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        self._current_embedding = batch.get("embedding") if self.use_film else None
        images, targets = self._to_targets(batch)
        was_training = self.model.training
        self.model.train()
        if not self.training:
            # Called for a *validation* loss (self.training is the wrapper
            # Detector's own flag - False here means the caller is
            # engine/trainer.py's evaluate(), not a real training step)
            # while torchvision's forward only returns a loss dict in train
            # mode - a torchvision API quirk, not a request to actually
            # train. Forcing train() would otherwise update BatchNorm
            # running stats from this eval batch every time val/test loss
            # is computed, silently contaminating the trained model with
            # validation-set statistics over the course of a run. Freeze
            # BN specifically (still "trained enough" to emit losses, but
            # not accumulating stats) rather than skip val loss entirely.
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        try:
            loss_dict = self.model(images, targets)
        finally:
            self._current_embedding = None
            self.model.train(was_training)
        total = sum(loss_dict.values())
        out = {k: float(v.detach()) for k, v in loss_dict.items()}
        out["total_loss"] = float(total.detach())
        return total, out

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, embeddings: torch.Tensor | None = None, conf_thres: float = 0.25
    ) -> list[dict[str, torch.Tensor]]:
        was_training = self.training
        self.eval()
        self._current_embedding = embeddings if self.use_film else None
        try:
            outputs = self.model(list(images.unbind(0)))
        finally:
            self._current_embedding = None
            self.train(was_training)

        results = []
        for out in outputs:
            keep = out["scores"] >= conf_thres
            results.append(
                {
                    "boxes": out["boxes"][keep],
                    "scores": out["scores"][keep],
                    "labels": out["labels"][keep] - self.label_offset,
                }
            )
        return results


@register_model("fasterrcnn")
class FasterRCNNFiLMDetector(TorchvisionFPNFiLMDetector):
    constructors = _FASTERRCNN_CONSTRUCTORS
    label_offset = 1
    score_thresh_kwarg = "box_score_thresh"


@register_model("fcos")
class FCOSFiLMDetector(TorchvisionFPNFiLMDetector):
    constructors = _FCOS_CONSTRUCTORS
    label_offset = 0
