"""Common interface every registered detector must implement, so
engine/trainer.py and engine/metrics.py never need to know which
architecture is underneath.

compute_loss owns its own forward pass rather than taking precomputed
preds: some architectures (RT-DETR's denoising decoder, torchvision's
RPN/ROI heads) need ground truth available *during* their forward call,
not just after, so "forward" and "compute loss" aren't cleanly separable
across every architecture family - only YOLO's TAL-based assignment
happens to allow it. Folding them into one call keeps every subclass
honest about what its underlying library actually requires instead of
faking a forward/loss split that only some of them have.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Detector(nn.Module, ABC):
    @abstractmethod
    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Runs its own forward pass and returns (total_loss, {loss_name:
        value}) for logging. `batch` is the collated dict from
        team_conditioned_detection.data.dataset.collate_fn (at least "img", "cls",
        "bboxes", "batch_idx", and "embedding" if the model uses FiLM),
        already moved to the model's device.
        """

    @abstractmethod
    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, embeddings: torch.Tensor | None = None, conf_thres: float = 0.25
    ) -> list[dict[str, torch.Tensor]]:
        """Returns one dict per image: {"boxes": (N,4) xyxy pixel coords in
        the input image's own coordinate frame, "scores": (N,), "labels":
        (N,)}. This is the torchmetrics MeanAveragePrecision input format.
        """
