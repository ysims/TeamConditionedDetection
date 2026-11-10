"""Two-stage robot detection + jersey-colour role classification.

    1. Detection: a standard, UNCONDITIONED single-class ("robot")
       detector. Box regression and objectness never see the colour
       context.
    2. Role classification: a small head that RoIAligns an intermediate
       detector feature map at each box (GT boxes during training,
       predicted boxes at inference) and predicts teammate/opponent from
       the box-local feature, conditioned on (teammate_rgb, opponent_rgb).

Conditioning the whole detector instead (an earlier single-stage design,
see git history) exposed box regression to the same hard-to-learn
conditioning signal as the role decision, and a swap test showed its
predictions barely changing when the colours were exchanged.

predict() returns boxes/scores/labels in teammate/opponent space, so mAP50
measures detection and role assignment jointly. classify_roles() runs only
the role head on supplied boxes - the ground-truth-box role accuracy and
context-swap evaluations in team_detect.evaluation use it.

Requires team_detect.data.robot_jersey's dataset to supply two label sets
per box: "cls" (role: teammate=0/opponent=1, for the role loss) and
"detection_cls" (always 0, "robot", for the detector's own loss).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import roi_align
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.nms import non_max_suppression

from team_detect.engine.metrics import norm_cxcywh_to_xyxy
from team_detect.models.base import Detector
from team_detect.models.conditioners import build_conditioner

DETECTION_CLASSES = 1  # single unconditioned "robot" class for the box/objectness pathway


class RoleHead(nn.Module):
    """RoIAlign a feature map at each box, condition the crop on the
    injected colour pair (see team_detect.models.conditioners), spatially
    average, and classify teammate/opponent - optionally with an explicit
    colour-distance feature appended before the classifier.

    conditioning_method "concat" is the unconditioned-feature baseline:
    no conditioner, the context is concatenated to the pooled feature and
    a two-layer MLP (hidden_dim wide) classifies.
    """

    def __init__(
        self,
        feature_channels: int,
        embed_dim: int,
        hidden_dim: int = 256,
        roi_output_size: int = 8,
        conditioning_method: str = "film",
        use_distance_feature: bool = False,
    ):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.use_distance_feature = use_distance_feature
        self.concat_context = conditioning_method == "concat"
        classifier_in = feature_channels + (2 if use_distance_feature else 0)
        if self.concat_context:
            self.conditioner = None
            self.classifier = nn.Sequential(nn.Linear(classifier_in + embed_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 2))
        else:
            self.conditioner = build_conditioner(conditioning_method, embed_dim, feature_channels, hidden_dim)
            self.classifier = nn.Linear(classifier_in, 2)

    def forward(
        self,
        feature_map: torch.Tensor,
        images: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        box_batch_idx: torch.Tensor,
        embeddings: torch.Tensor,
        feature_stride: int,
    ) -> torch.Tensor:
        """feature_map: (B,C,H,W), the captured (unmodulated) detection
        feature. images: (B,3,img_size,img_size) raw input in the same
        pixel frame as boxes_xyxy - used only for the colour-distance
        feature. boxes_xyxy: (N,4) absolute pixel coords. box_batch_idx:
        (N,) which image each box belongs to. embeddings: (B,6) =
        (teammate_rgb, opponent_rgb). Returns (N,2) role logits.
        """
        rois = torch.cat([box_batch_idx.unsqueeze(1).to(boxes_xyxy.dtype), boxes_xyxy], dim=1)
        per_box_embedding = embeddings[box_batch_idx.long()].to(feature_map.dtype)

        pooled = roi_align(feature_map, rois, output_size=self.roi_output_size, spatial_scale=1.0 / feature_stride, aligned=True)
        if self.concat_context:
            features = [pooled.mean(dim=(2, 3)), per_box_embedding]
        else:
            features = [self.conditioner(pooled, per_box_embedding).mean(dim=(2, 3))]

        if self.use_distance_feature:
            # A jersey covers only a small torso patch of the box, so a
            # box-wide average RGB dilutes it into the surrounding chassis.
            # Instead take the best-matching cell of a raw-pixel grid crop:
            # the minimum RGB distance over cells to each supplied colour.
            rgb_crop = roi_align(images, rois, output_size=self.roi_output_size, spatial_scale=1.0, aligned=True)  # (N,3,S,S)
            features.extend(min_cell_distances(rgb_crop, per_box_embedding))

        return self.classifier(torch.cat(features, dim=-1))


def min_cell_distances(rgb_crop: torch.Tensor, per_box_embedding: torch.Tensor) -> list[torch.Tensor]:
    """(N,3,S,S) crop -> [(N,1) teammate distance, (N,1) opponent distance],
    each the minimum over the S*S cells of the RGB distance to that colour.
    """
    teammate_rgb, opponent_rgb = per_box_embedding[:, :3], per_box_embedding[:, 3:]
    return [
        (rgb_crop - rgb[:, :, None, None]).norm(dim=1).flatten(1).min(dim=1, keepdim=True).values
        for rgb in (teammate_rgb, opponent_rgb)
    ]


class _RoleDetector(Detector):
    """Shared role-head plumbing. Subclasses build the detector, register a
    forward hook that stores the tapped feature map in self._captured_feature,
    and implement _detection_loss, _detect, and _capture_role_feature.
    """

    role_head: RoleHead
    feature_stride: int
    img_size: int
    role_loss_weight: float

    def _build_role_head(self, feature_channels: int, embed_dim: int, hidden_dim: int, roi_output_size: int, conditioning_method: str, use_distance_feature: bool) -> None:
        self.role_head = RoleHead(
            feature_channels,
            embed_dim,
            hidden_dim=hidden_dim,
            roi_output_size=roi_output_size,
            conditioning_method=conditioning_method,
            use_distance_feature=use_distance_feature,
        )

    def _detection_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Runs the detector's own loss, capturing the role feature as a side effect."""
        raise NotImplementedError

    def _detect(self, images: torch.Tensor, conf_thres: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Eval-mode detection, capturing the role feature as a side effect.
        Returns per-image (boxes_xyxy, scores) with scores >= conf_thres.
        """
        raise NotImplementedError

    def _capture_role_feature(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        self._captured_feature = None
        detection_loss, loss_dict = self._detection_loss(batch)

        boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], self.img_size)
        role_logits = self.role_head(
            self._captured_feature, batch["img"], boxes_xyxy, batch["batch_idx"].long(), batch["embedding"], self.feature_stride
        )
        role_labels = batch["cls"].long()
        role_loss = F.cross_entropy(role_logits, role_labels)
        role_acc = (role_logits.argmax(dim=-1) == role_labels).float().mean()

        total = detection_loss + self.role_loss_weight * role_loss
        loss_dict["detection_loss"] = float(detection_loss.detach())
        loss_dict["role_loss"] = float(role_loss.detach())
        loss_dict["role_acc"] = float(role_acc.detach())
        loss_dict["total_loss"] = float(total.detach())
        return total, loss_dict

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, embeddings: torch.Tensor | None = None, conf_thres: float = 0.25
    ) -> list[dict[str, torch.Tensor]]:
        was_training = self.training
        self.eval()
        self._captured_feature = None
        try:
            detections = self._detect(images, conf_thres)
        finally:
            self.train(was_training)
        feature_map = self._captured_feature

        results = []
        for b, (boxes, scores) in enumerate(detections):
            if boxes.shape[0] == 0:
                results.append({"boxes": boxes, "scores": scores, "labels": torch.zeros(0, dtype=torch.long, device=boxes.device)})
                continue
            box_batch_idx = torch.full((boxes.shape[0],), b, device=boxes.device, dtype=torch.long)
            role_logits = self.role_head(feature_map, images, boxes, box_batch_idx, embeddings, self.feature_stride)
            results.append({"boxes": boxes, "scores": scores, "labels": role_logits.argmax(dim=-1)})
        return results

    @torch.no_grad()
    def classify_roles(
        self, images: torch.Tensor, boxes_xyxy: torch.Tensor, box_batch_idx: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        self._captured_feature = None
        feature_map = self._capture_role_feature(images)
        return self.role_head(feature_map, images, boxes_xyxy, box_batch_idx, embeddings, self.feature_stride)


class _UltralyticsRoleDetector(_RoleDetector):
    """YOLO and RT-DETR share Ultralytics' DetectionModel-family construction
    and `detector.loss(batch)` convention; subclasses supply the model class,
    pretrained-weight loader, and output decoding.
    """

    model_cls: type
    default_role_layer_index: int

    def __init__(
        self,
        variant: str,
        img_size: int = 640,
        pretrained: bool = False,
        embed_dim: int = 6,
        hidden_dim: int = 256,
        conditioning_method: str = "film",
        role_loss_weight: float = 1.0,
        roi_output_size: int = 8,
        use_distance_feature: bool = False,
        role_layer_index: int | None = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.role_loss_weight = role_loss_weight
        self.role_layer_index = self.default_role_layer_index if role_layer_index is None else role_layer_index

        self.detector = self.model_cls(cfg=f"{variant}.yaml", ch=3, nc=DETECTION_CLASSES, verbose=False)
        self.detector.args = copy.copy(DEFAULT_CFG)
        self.detector.names = {0: "robot"}
        self.detector.nc = DETECTION_CLASSES

        if pretrained:
            self._load_pretrained_backbone(variant)

        feature_channels, self.feature_stride = self._discover_channels_and_stride(self.role_layer_index)
        self._build_role_head(feature_channels, embed_dim, hidden_dim, roi_output_size, conditioning_method, use_distance_feature)

        self._captured_feature: torch.Tensor | None = None
        self.detector.model[self.role_layer_index].register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inputs, output):
        self._captured_feature = output
        return output  # pass-through: the detection pathway must stay unconditioned/undisturbed

    def _pretrained_model(self, variant: str) -> nn.Module:
        raise NotImplementedError

    def _load_pretrained_backbone(self, variant: str) -> None:
        """Partial load, matched by tensor name+shape: the COCO-pretrained
        checkpoint's detection-head tensors (shape-dependent on nc=80) don't
        match our nc=1 head and are skipped; backbone tensors load.
        """
        own_state = self.detector.state_dict()
        pretrained_state = self._pretrained_model(variant).state_dict()
        matched = {k: v for k, v in pretrained_state.items() if k in own_state and own_state[k].shape == v.shape}
        own_state.update(matched)
        self.detector.load_state_dict(own_state)
        print(f"[{type(self).__name__}] loaded {len(matched)}/{len(own_state)} pretrained tensors from {variant}.pt")

    @torch.no_grad()
    def _discover_channels_and_stride(self, layer_index: int) -> tuple[int, int]:
        was_training = self.detector.training
        self.detector.eval()
        captured = {}

        def hook(module, inputs, output):
            captured["shape"] = output.shape

        handle = self.detector.model[layer_index].register_forward_hook(hook)
        self.detector(torch.zeros(1, 3, self.img_size, self.img_size))
        handle.remove()
        self.detector.train(was_training)
        _, channels, h, _ = captured["shape"]
        return channels, self.img_size // h

    def configure(self, epochs: int) -> None:
        self.detector.args.epochs = epochs

    def _detection_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        loss_vec, loss_items = self.detector.loss({**batch, "cls": batch["detection_cls"]})
        return loss_vec.sum(), {k: float(v) for k, v in loss_items.items()}

    def _capture_role_feature(self, images: torch.Tensor) -> torch.Tensor:
        self.detector(images)
        return self._captured_feature


class YOLORobotRoleDetector(_UltralyticsRoleDetector):
    """variant: an Ultralytics yolo*.yaml stem (e.g. "yolo26n"). The role head
    taps layer 4, the P3/8 backbone feature before the neck's cross-scale fusion.
    """

    model_cls = DetectionModel
    default_role_layer_index = 4

    def _pretrained_model(self, variant: str) -> nn.Module:
        from ultralytics import YOLO

        return YOLO(f"{variant}.pt").model

    def _detect(self, images: torch.Tensor, conf_thres: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
        y, _ = self.detector(images)
        if self.detector.end2end:
            raw_dets = [dets[dets[:, 4] >= conf_thres] for dets in y]
        else:
            raw_dets = non_max_suppression(y, conf_thres=conf_thres, iou_thres=0.5, max_det=300)
        return [(dets[:, :4], dets[:, 4]) for dets in raw_dets]


class RTDETRRobotRoleDetector(_UltralyticsRoleDetector):
    """variant: an Ultralytics rtdetr-*.yaml stem (e.g. "rtdetr-l"). The role
    head taps layer 3, the backbone-level P3 feature before the hybrid
    encoder's AIFI fusion - RTDETRDecoder's own output is query embeddings,
    not a spatial map, so it can't be a RoIAlign tap.
    """

    model_cls = RTDETRDetectionModel
    default_role_layer_index = 3

    def _pretrained_model(self, variant: str) -> nn.Module:
        from ultralytics import RTDETR

        return RTDETR(f"{variant}.pt").model

    def _detect(self, images: torch.Tensor, conf_thres: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # y: (B, num_queries, 6) = [cx, cy, w, h, conf, cls], normalized (0-1).
        # The cls column is always 0 ("robot") - role comes from role_head.
        y, _ = self.detector(images)
        boxes = norm_cxcywh_to_xyxy(y[..., :4], self.img_size)
        detections = []
        for b in range(y.shape[0]):
            keep = y[b, :, 4] >= conf_thres
            detections.append((boxes[b][keep], y[b, keep, 4]))
        return detections


_FASTERRCNN_CONSTRUCTORS = {"resnet50": "fasterrcnn_resnet50_fpn_v2"}


class FasterRCNNRobotRoleDetector(_RoleDetector):
    """torchvision Faster R-CNN (ResNet-50-FPN v2, ImageNet-pretrained
    backbone). RPN, box regression, and foreground/background classification
    are standard and unconditioned (num_classes=1 "robot" + torchvision's
    reserved background class 0). The role head taps the finest FPN level
    read-only.
    """

    def __init__(
        self,
        variant: str = "resnet50",
        img_size: int = 640,
        pretrained: bool = False,
        embed_dim: int = 6,
        hidden_dim: int = 256,
        conditioning_method: str = "film",
        role_loss_weight: float = 1.0,
        roi_output_size: int = 8,
        use_distance_feature: bool = False,
        role_fpn_level: str = "0",  # finest FPN level - torchvision's BackboneWithFPN output dict key
    ):
        super().__init__()
        if variant not in _FASTERRCNN_CONSTRUCTORS:
            raise ValueError(f"Unknown variant {variant!r} for fasterrcnn_robot_role, expected one of {sorted(_FASTERRCNN_CONSTRUCTORS)}")
        self.img_size = img_size
        self.role_loss_weight = role_loss_weight
        self.role_fpn_level = role_fpn_level

        constructor = getattr(torchvision.models.detection, _FASTERRCNN_CONSTRUCTORS[variant])
        self.model = constructor(
            weights=None,  # never the full detection head (COCO classes don't match ours)
            weights_backbone="DEFAULT" if pretrained else None,
            num_classes=DETECTION_CLASSES + 1,  # +1: torchvision reserves class 0 for background
            min_size=img_size,
            max_size=img_size,
            # Not the class's own 0.05 default and deliberately not 0.0 either:
            # an unfiltered NMS candidate set has made torchvision detectors
            # hang silently on runs that otherwise train normally.
            box_score_thresh=1e-3,
        )

        feature_channels, self.feature_stride = self._discover_fpn_channels_and_stride(role_fpn_level)
        self._build_role_head(feature_channels, embed_dim, hidden_dim, roi_output_size, conditioning_method, use_distance_feature)
        self._captured_feature: torch.Tensor | None = None
        self.model.backbone.fpn.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inputs, output):
        self._captured_feature = output[self.role_fpn_level]  # output is an OrderedDict of level name -> feature map
        return output  # pass-through: RPN/box regression must stay unconditioned/undisturbed

    @torch.no_grad()
    def _discover_fpn_channels_and_stride(self, level: str) -> tuple[int, int]:
        was_training = self.model.training
        self.model.eval()
        out = self.model.backbone(torch.zeros(1, 3, self.img_size, self.img_size))
        self.model.train(was_training)
        _, channels, h, _ = out[level].shape
        return channels, self.img_size // h

    def configure(self, epochs: int) -> None:
        pass  # no epoch-dependent loss scheduling for this architecture

    def _detection_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        images = list(batch["img"].unbind(0))
        boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], self.img_size)
        targets = []
        for i in range(len(images)):
            boxes = boxes_xyxy[batch["batch_idx"] == i]
            labels = torch.ones(boxes.shape[0], dtype=torch.long, device=boxes.device)  # class 1 = "robot" (0 reserved for background)
            targets.append({"boxes": boxes, "labels": labels})

        was_training = self.model.training
        self.model.train()  # torchvision only returns a loss dict in train mode
        if not self.training:
            # Validation loss: keep BatchNorm running stats frozen rather than
            # updating them from validation batches.
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        try:
            loss_dict = self.model(images, targets)
        finally:
            self.model.train(was_training)
        return sum(loss_dict.values()), {k: float(v.detach()) for k, v in loss_dict.items()}

    def _detect(self, images: torch.Tensor, conf_thres: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
        detections = []
        for out in self.model(list(images.unbind(0))):
            keep = out["scores"] >= conf_thres
            detections.append((out["boxes"][keep], out["scores"][keep]))
        return detections

    def _capture_role_feature(self, images: torch.Tensor) -> torch.Tensor:
        # GeneralizedRCNN normally runs images through self.model.transform
        # (ImageNet mean/std normalization) before the backbone - skipping it
        # feeds raw [0,1] pixels to a backbone trained on normalized input.
        transformed, _ = self.model.transform(list(images.unbind(0)))
        self.model.backbone(transformed.tensors)
        return self._captured_feature
