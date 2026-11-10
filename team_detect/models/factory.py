"""Builds a model from config. `model.architecture` picks the class:

    fasterrcnn_robot_role       two-stage: unconditioned Faster R-CNN + conditioned role head
    rtdetr_robot_role           two-stage, RT-DETR detector
    yolo_robot_role             two-stage, YOLO detector
    standalone_role_classifier  role classification only, on supplied boxes, no detector
"""
from __future__ import annotations

import torch.nn as nn

from team_detect.data.robot_jersey import EMBED_DIM
from team_detect.models.robot_role import FasterRCNNRobotRoleDetector, RTDETRRobotRoleDetector, YOLORobotRoleDetector
from team_detect.models.standalone_role_classifier import StandaloneRoleClassifier
from team_detect.robot_config import RobotConfig

ROLE_DETECTORS = {
    "fasterrcnn_robot_role": FasterRCNNRobotRoleDetector,
    "rtdetr_robot_role": RTDETRRobotRoleDetector,
    "yolo_robot_role": YOLORobotRoleDetector,
}
STANDALONE = "standalone_role_classifier"


def build_model(config: RobotConfig, pretrained: bool | None = None) -> nn.Module:
    """pretrained=None uses config.model.pretrained; pass False when weights
    come from a checkpoint anyway (avoids downloading pretrained weights).
    """
    m = config.model
    if m.architecture == STANDALONE:
        return StandaloneRoleClassifier(
            embed_dim=EMBED_DIM,
            hidden_dim=m.hidden_dim,
            conditioning_method=m.conditioning_method,
            crop_size=m.crop_size,
            encoder_channels=m.encoder_channels,
            use_distance_feature=m.use_distance_feature,
        )
    if m.architecture not in ROLE_DETECTORS:
        raise ValueError(f"Unknown architecture {m.architecture!r}, expected one of {sorted([*ROLE_DETECTORS, STANDALONE])}")
    return ROLE_DETECTORS[m.architecture](
        variant=m.variant,
        img_size=config.data.img_size,
        pretrained=m.pretrained if pretrained is None else pretrained,
        embed_dim=EMBED_DIM,
        hidden_dim=m.hidden_dim,
        conditioning_method=m.conditioning_method,
        role_loss_weight=m.role_loss_weight,
        roi_output_size=m.roi_output_size,
        use_distance_feature=m.use_distance_feature,
    )


def role_classification_params(model: nn.Module) -> int:
    """Parameters added for role classification, excluding any shared
    detection backbone: the role head for two-stage detectors, the whole
    model for the standalone classifier.
    """
    module = model.role_head if hasattr(model, "role_head") else model
    return sum(p.numel() for p in module.parameters())
