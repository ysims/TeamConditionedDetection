"""A standalone role classifier: no shared backbone/gradient with any
detector at all, unlike the two-stage detectors' RoleHead
(team_detect/models/robot_role.py), which taps a hook into the detector's
own internal feature map. This crops each box directly out of
the raw image (RoIAlign on pixels, not features), runs the crop through
its own small dedicated CNN, and classifies from that - deployable as a
second, independent model behind any box-producing detector, trained
completely separately.

Deliberately mirrors RoleHead's conditioning mechanism as closely as
possible (same build_conditioner call, same optional min-distance-over-
cells colour feature) so the comparison between "shared-backbone role
head" and "standalone classifier" isolates architecture (shared feature
vs. dedicated lightweight encoder), not conditioning method.

See team_detect/train.py for its training loop - it has no detection
loss/map50 concept, so it doesn't implement Detector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from team_detect.models.conditioners import build_conditioner
from team_detect.models.robot_role import min_cell_distances


class LightweightCropEncoder(nn.Module):
    """Tiny dedicated CNN stem for a crop_size x crop_size RGB crop -
    deliberately small (3 conv blocks, no residual/attention machinery) so
    this stays "lightweight" relative to the detector backbones the shared-
    feature role heads reuse (ResNet-50 FPN / YOLO26n / RT-DETR HGNetv2).
    """

    def __init__(self, out_channels: int = 128, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # /2
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # /4
            nn.Conv2d(64, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),  # /8
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StandaloneRoleClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int = 6,
        hidden_dim: int = 256,
        conditioning_method: str = "film",
        crop_size: int = 64,
        encoder_channels: int = 128,
        use_distance_feature: bool = False,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.use_distance_feature = use_distance_feature
        self.encoder = LightweightCropEncoder(encoder_channels)
        self.conditioner = build_conditioner(conditioning_method, embed_dim, encoder_channels, hidden_dim)
        classifier_in = encoder_channels + (2 if use_distance_feature else 0)
        self.classifier = nn.Linear(classifier_in, 2)

    def forward(
        self,
        images: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        box_batch_idx: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """images: (B,3,img_size,img_size) raw pixels. boxes_xyxy: (N,4)
        absolute pixel coords. box_batch_idx: (N,) which image each box
        belongs to. embeddings: (B,6) = (teammate_rgb, opponent_rgb).
        Returns (N,2) role logits.
        """
        rois = torch.cat([box_batch_idx.unsqueeze(1).to(boxes_xyxy.dtype), boxes_xyxy], dim=1)
        crops = roi_align(images, rois, output_size=self.crop_size, spatial_scale=1.0, aligned=True)  # (N,3,S,S)

        feat = self.encoder(crops)  # (N,C,S/8,S/8), own dedicated feature, not from any detector
        per_box_embedding = embeddings[box_batch_idx.long()].to(feat.dtype)
        modulated = self.conditioner(feat, per_box_embedding)
        features = [modulated.mean(dim=(2, 3))]

        if self.use_distance_feature:
            features.extend(min_cell_distances(crops, per_box_embedding))

        return self.classifier(torch.cat(features, dim=-1))

    @torch.no_grad()
    def classify_roles(
        self, images: torch.Tensor, boxes_xyxy: torch.Tensor, box_batch_idx: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Same signature as Detector.classify_roles, so team_detect.evaluation
        treats this and the two-stage detectors uniformly."""
        return self(images, boxes_xyxy, box_batch_idx, embeddings)
