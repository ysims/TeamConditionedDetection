"""Two-stage robot detection + jersey-colour role classification.

Motivated by a clean failure mode found in the single-stage FiLM/cross-
attention experiments (team_conditioned_detection/data/robot_jersey.py, configs/
robot_jersey_film.yaml and friends): conditioning the *whole detection
head* on (teammate_rgb, opponent_rgb) meant the box-regression/objectness
pathway was exposed to the same hard-to-learn conditioning signal as the
classification decision, with no structural way to tell whether poor
results were bad localization or bad role classification - a swap test
(same image, same two colours, roles reversed) showed predictions barely
changed at all, for both FiLM and cross-attention alike.

This splits the two concerns structurally instead of diagnosing after the
fact:

    1. Detection: a completely standard, UNCONDITIONED single-class
       ("robot") YOLO detector - identical in spirit to the ball baseline,
       which reliably hit ~0.85 map50. Box regression and objectness never
       see the colour conditioning at all.
    2. Role classification: a small head that RoIAligns a feature map at
       each box's location (GT boxes during training, predicted boxes at
       inference) and predicts teammate/opponent from the *cropped,
       box-local* feature, FiLM-conditioned on (teammate_rgb,
       opponent_rgb) - a much more direct "does this crop match this
       colour" signal than modulating the whole backbone ever gave it.

predict()'s output contract is unchanged (boxes/scores/labels in
teammate/opponent space, same as the single-stage detectors in
yolo_film.py) so it drops into the exact same evaluate()/DetectionMetrics/
run_training_loop used everywhere else - only compute_loss and predict's
internals differ.

Requires team_conditioned_detection.data.robot_jersey's dataset to supply *two* label sets
per box: "cls" (role: teammate=0/opponent=1, used for the role loss - same
field the single-stage detectors use directly) and "detection_cls" (always
0, "robot" - used for the internal box/objectness loss only).
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

from team_conditioned_detection.engine.metrics import norm_cxcywh_to_xyxy
from team_conditioned_detection.models.base import Detector
from team_conditioned_detection.models.conditioners import build_conditioner
from team_conditioned_detection.models.registry import register_model

DETECTION_CLASSES = 1  # single unconditioned "robot" class for the box/objectness pathway


class RoleHead(nn.Module):
    """RoIAlign a feature map at each box, condition the crop on the
    injected colour pair (FiLM or cross-attention - see
    team_conditioned_detection.models.conditioners, whichever build_conditioner is told to
    build), classify teammate/opponent from the result plus an explicit
    colour-distance feature.

    use_deep_feature/use_distance_feature exist for ablating which of the
    two actually earns its keep (each defaults True, i.e. unchanged
    behaviour) - at least one must stay on.
    """

    def __init__(
        self,
        feature_channels: int,
        embed_dim: int,
        hidden_dim: int = 256,
        roi_output_size: int = 4,
        conditioning_method: str = "film",
        use_deep_feature: bool = True,
        use_distance_feature: bool = True,
    ):
        super().__init__()
        if not use_deep_feature and not use_distance_feature:
            raise ValueError("RoleHead needs at least one of use_deep_feature/use_distance_feature")
        self.roi_output_size = roi_output_size
        self.use_deep_feature = use_deep_feature
        self.use_distance_feature = use_distance_feature
        self.conditioner = build_conditioner(conditioning_method, embed_dim, feature_channels, hidden_dim) if use_deep_feature else None
        classifier_in = (feature_channels if use_deep_feature else 0) + (2 if use_distance_feature else 0)
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
        pixel frame as boxes_xyxy - used only for the explicit mean-RGB
        distance feature below, not the conditioned pathway. boxes_xyxy:
        (N,4) absolute pixel coords. box_batch_idx: (N,) which image each
        box belongs to. embeddings: (B,6) = (teammate_rgb, opponent_rgb).
        Returns (N,2) role logits.
        """
        rois = torch.cat([box_batch_idx.unsqueeze(1).to(boxes_xyxy.dtype), boxes_xyxy], dim=1)
        per_box_embedding = embeddings[box_batch_idx.long()].to(feature_map.dtype)
        features = []

        if self.use_deep_feature:
            pooled = roi_align(feature_map, rois, output_size=self.roi_output_size, spatial_scale=1.0 / feature_stride, aligned=True)
            modulated = self.conditioner(pooled, per_box_embedding)
            features.append(modulated.mean(dim=(2, 3)))

        if self.use_distance_feature:
            # Explicit colour-distance feature: a jersey typically covers
            # only a small torso patch of a robot's full bounding box
            # (confirmed visually - most of the box is grey/dark chassis,
            # limbs, and background between them), so a box-wide *average*
            # RGB dilutes the jersey colour into whatever's more plentiful.
            # Instead, look at each cell of the raw-pixel crop individually
            # and take the BEST-matching cell's distance to each candidate
            # colour - one genuinely jersey-coloured cell can win outright
            # regardless of how much non-jersey material surrounds it,
            # where an average never could.
            rgb_crop = roi_align(images, rois, output_size=self.roi_output_size, spatial_scale=1.0, aligned=True)  # (N,3,S,S)
            teammate_rgb, opponent_rgb = per_box_embedding[:, :3], per_box_embedding[:, 3:]
            dist_teammate = (rgb_crop - teammate_rgb[:, :, None, None]).norm(dim=1).flatten(1).min(dim=1, keepdim=True).values
            dist_opponent = (rgb_crop - opponent_rgb[:, :, None, None]).norm(dim=1).flatten(1).min(dim=1, keepdim=True).values
            features.append(dist_teammate)
            features.append(dist_opponent)

        return self.classifier(torch.cat(features, dim=-1))


@register_model("yolo_robot_role")
class RobotRoleDetector(Detector):
    """variant: an Ultralytics yolo*.yaml stem (e.g. "yolo26n"), same as
    YOLOFiLMDetector. num_classes/class_names describe the *role* labels
    predict() outputs (["teammate","opponent"]) - the internal detector is
    always built with DETECTION_CLASSES=1 ("robot"), independent of
    num_classes, which this class otherwise ignores.
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_names: list[str] | None = None,
        variant: str = "yolo26n",
        img_size: int = 640,
        pretrained: bool = False,
        embed_dim: int = 6,
        film_hidden_dim: int = 256,
        role_layer_index: int = 4,  # P3 for yolo26n/yolov8n - see yolo_film.py's layer map
        conditioning_method: str = "film",  # role head's conditioner - "film" or "cross_attention"
        auxiliary_loss_weight: float = 1.0,  # role loss weight; 0.0 = detector-only warm-start, see freeze_detector
        roi_output_size: int = 4,
        role_head_use_deep_feature: bool = True,
        role_head_use_distance_feature: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.role_loss_weight = auxiliary_loss_weight
        self.role_layer_index = role_layer_index

        self.detector = DetectionModel(cfg=f"{variant}.yaml", ch=3, nc=DETECTION_CLASSES, verbose=False)
        self.detector.args = copy.copy(DEFAULT_CFG)
        self.detector.names = {0: "robot"}
        self.detector.nc = DETECTION_CLASSES

        if pretrained:
            self._load_pretrained_backbone(variant)

        feature_channels, self.feature_stride = self._discover_channels_and_stride(role_layer_index)
        self.role_head = RoleHead(
            feature_channels,
            embed_dim,
            hidden_dim=film_hidden_dim,
            roi_output_size=roi_output_size,
            conditioning_method=conditioning_method,
            use_deep_feature=role_head_use_deep_feature,
            use_distance_feature=role_head_use_distance_feature,
        )

        self._captured_feature: torch.Tensor | None = None
        self.detector.model[role_layer_index].register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inputs, output):
        self._captured_feature = output
        return output  # pass-through: the detection pathway must stay unconditioned/undisturbed

    def _load_pretrained_backbone(self, variant: str) -> None:
        """Partial load, matched by tensor name+shape - same pattern as
        UltralyticsFiLMDetector._load_pretrained_backbone (yolo_film.py's
        base class). self.detector has nc=DETECTION_CLASSES=1, so the
        COCO-pretrained checkpoint's detection-head tensors (shape-
        dependent on nc=80) simply won't match and get skipped; the
        backbone tensors (nc-independent) do match and load.
        """
        from ultralytics import YOLO

        pretrained_model = YOLO(f"{variant}.pt").model
        own_state = self.detector.state_dict()
        pretrained_state = pretrained_model.state_dict()
        matched = {k: v for k, v in pretrained_state.items() if k in own_state and own_state[k].shape == v.shape}
        own_state.update(matched)
        self.detector.load_state_dict(own_state)
        print(f"[RobotRoleDetector] loaded {len(matched)}/{len(own_state)} pretrained tensors from {variant}.pt")

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

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        detection_batch = {**batch, "cls": batch["detection_cls"]}
        self._captured_feature = None
        loss_vec, loss_items = self.detector.loss(detection_batch)
        detection_loss = loss_vec.sum()

        boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], self.img_size)
        role_logits = self.role_head(
            self._captured_feature, batch["img"], boxes_xyxy, batch["batch_idx"].long(), batch["embedding"], self.feature_stride
        )
        role_labels = batch["cls"].long()
        role_loss = F.cross_entropy(role_logits, role_labels)
        role_acc = (role_logits.argmax(dim=-1) == role_labels).float().mean()

        total = detection_loss + self.role_loss_weight * role_loss
        loss_dict = {k: float(v) for k, v in loss_items.items()}
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
            y, _ = self.detector(images)
        finally:
            self.train(was_training)
        feature_map = self._captured_feature

        if self.detector.end2end:
            raw_dets = []
            for dets in y:
                keep = dets[:, 4] >= conf_thres
                raw_dets.append(dets[keep])
        else:
            raw_dets = non_max_suppression(y, conf_thres=conf_thres, iou_thres=0.5, max_det=300)

        results = []
        for b, dets in enumerate(raw_dets):
            boxes, scores = dets[:, :4], dets[:, 4]
            if boxes.shape[0] == 0:
                results.append({"boxes": boxes, "scores": scores, "labels": torch.zeros(0, dtype=torch.long, device=boxes.device)})
                continue
            box_batch_idx = torch.full((boxes.shape[0],), b, device=boxes.device, dtype=torch.long)
            role_logits = self.role_head(feature_map, images, boxes, box_batch_idx, embeddings, self.feature_stride)
            results.append({"boxes": boxes, "scores": scores, "labels": role_logits.argmax(dim=-1)})
        return results


@register_model("rtdetr_robot_role")
class RTDETRRobotRoleDetector(Detector):
    """Same two-stage idea as RobotRoleDetector, on RT-DETR instead of
    YOLO - both share Ultralytics' DetectionModel-family construction and
    `detector.loss(batch)` convention (see ultralytics_base.py), so this
    class is a near-identical copy of RobotRoleDetector with RTDETRDetectionModel
    swapped in and RT-DETR's own query-based predict() decode (see
    rtdetr_film.py's RTDETRFiLMDetector.predict, reused verbatim here).

    role_layer_index defaults to 3, RT-DETR's backbone-level P3-equivalent
    stage (before the hybrid encoder's cross-scale AIFI attention fusion) -
    a plain (B,C,H,W) conv feature map, same requirement as YOLO's
    role_layer_index=4 "P3" tap. See rtdetr_film.py's module docstring for
    the full layer graph and why [3, 7, 9] (backbone) is the equivalent of
    [21, 24, 27] (post-encoder): RTDETRDecoder's own output at the end of
    the graph is query embeddings, not a spatial feature map, so it can
    never be a valid RoIAlign tap for this role head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_names: list[str] | None = None,
        variant: str = "rtdetr-l",
        img_size: int = 640,
        pretrained: bool = False,
        embed_dim: int = 6,
        film_hidden_dim: int = 256,
        role_layer_index: int = 3,  # backbone-level P3-equivalent, pre-encoder-fusion - see rtdetr_film.py's layer map
        conditioning_method: str = "film",
        auxiliary_loss_weight: float = 1.0,
        roi_output_size: int = 4,
        role_head_use_deep_feature: bool = True,
        role_head_use_distance_feature: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.role_loss_weight = auxiliary_loss_weight
        self.role_layer_index = role_layer_index

        self.detector = RTDETRDetectionModel(cfg=f"{variant}.yaml", ch=3, nc=DETECTION_CLASSES, verbose=False)
        self.detector.args = copy.copy(DEFAULT_CFG)
        self.detector.names = {0: "robot"}
        self.detector.nc = DETECTION_CLASSES

        if pretrained:
            self._load_pretrained_backbone(variant)

        feature_channels, self.feature_stride = self._discover_channels_and_stride(role_layer_index)
        self.role_head = RoleHead(
            feature_channels,
            embed_dim,
            hidden_dim=film_hidden_dim,
            roi_output_size=roi_output_size,
            conditioning_method=conditioning_method,
            use_deep_feature=role_head_use_deep_feature,
            use_distance_feature=role_head_use_distance_feature,
        )

        self._captured_feature: torch.Tensor | None = None
        self.detector.model[role_layer_index].register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inputs, output):
        self._captured_feature = output
        return output  # pass-through: the detection pathway must stay unconditioned/undisturbed

    def _load_pretrained_backbone(self, variant: str) -> None:
        """Partial load, matched by tensor name+shape - same pattern as
        RobotRoleDetector's, using RTDETR's own high-level loader (nc=80
        COCO head tensors won't match our nc=1 head, backbone tensors do).
        """
        from ultralytics import RTDETR

        pretrained_model = RTDETR(f"{variant}.pt").model
        own_state = self.detector.state_dict()
        pretrained_state = pretrained_model.state_dict()
        matched = {k: v for k, v in pretrained_state.items() if k in own_state and own_state[k].shape == v.shape}
        own_state.update(matched)
        self.detector.load_state_dict(own_state)
        print(f"[RTDETRRobotRoleDetector] loaded {len(matched)}/{len(own_state)} pretrained tensors from {variant}.pt")

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

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        detection_batch = {**batch, "cls": batch["detection_cls"]}
        self._captured_feature = None
        loss_vec, loss_items = self.detector.loss(detection_batch)
        detection_loss = loss_vec.sum()

        boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], self.img_size)
        role_logits = self.role_head(
            self._captured_feature, batch["img"], boxes_xyxy, batch["batch_idx"].long(), batch["embedding"], self.feature_stride
        )
        role_labels = batch["cls"].long()
        role_loss = F.cross_entropy(role_logits, role_labels)
        role_acc = (role_logits.argmax(dim=-1) == role_labels).float().mean()

        total = detection_loss + self.role_loss_weight * role_loss
        loss_dict = {k: float(v) for k, v in loss_items.items()}
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
            y, _ = self.detector(images)
        finally:
            self.train(was_training)
        feature_map = self._captured_feature

        # y: (B, num_queries, 6) = [cx, cy, w, h, conf, cls], normalized (0-1) -
        # DETR-style query regression, not YOLO's anchor/NMS output - same
        # decode as rtdetr_film.py's RTDETRFiLMDetector.predict. The "cls"
        # column is discarded (always 0, "robot") - role comes from
        # role_head below, exactly as in RobotRoleDetector.
        cx, cy, w, h = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        x1 = (cx - w / 2) * self.img_size
        y1 = (cy - h / 2) * self.img_size
        x2 = (cx + w / 2) * self.img_size
        y2 = (cy + h / 2) * self.img_size
        all_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        results = []
        for b in range(y.shape[0]):
            keep = y[b, :, 4] >= conf_thres
            boxes, scores = all_boxes[b][keep], y[b, keep, 4]
            if boxes.shape[0] == 0:
                results.append({"boxes": boxes, "scores": scores, "labels": torch.zeros(0, dtype=torch.long, device=boxes.device)})
                continue
            box_batch_idx = torch.full((boxes.shape[0],), b, device=boxes.device, dtype=torch.long)
            role_logits = self.role_head(feature_map, images, boxes, box_batch_idx, embeddings, self.feature_stride)
            results.append({"boxes": boxes, "scores": scores, "labels": role_logits.argmax(dim=-1)})
        return results


_FASTERRCNN_CONSTRUCTORS = {"resnet50": "fasterrcnn_resnet50_fpn_v2"}


@register_model("fasterrcnn_robot_role")
class FasterRCNNRobotRoleDetector(Detector):
    """Same two-stage idea as RobotRoleDetector, on a torchvision Faster
    R-CNN backbone instead of YOLO. RPN, box regression, and Faster R-CNN's
    own foreground/background classification are all completely standard
    and unconditioned (num_classes=1 "robot" + torchvision's reserved
    background class 0); only the role head - the exact same RoleHead
    class as the YOLO version, same RoIAlign+conditioner+mean-RGB-distance
    mechanism - is conditioned. Reuses a raw FPN level (finest by default)
    as a read-only tap for the role head, not the FiLM-on-backbone hook
    pattern in torchvision_film.py (that one modulates the pathway; this
    one only observes it).
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_names: list[str] | None = None,
        variant: str = "resnet50",
        img_size: int = 640,
        pretrained: bool = False,
        embed_dim: int = 6,
        film_hidden_dim: int = 256,
        conditioning_method: str = "film",
        auxiliary_loss_weight: float = 1.0,
        roi_output_size: int = 4,
        role_fpn_level: str = "0",  # finest FPN level - torchvision's BackboneWithFPN output dict key
        role_head_use_deep_feature: bool = True,
        role_head_use_distance_feature: bool = True,
        **kwargs,
    ):
        super().__init__()
        if variant not in _FASTERRCNN_CONSTRUCTORS:
            raise ValueError(f"Unknown variant {variant!r} for fasterrcnn_robot_role, expected one of {sorted(_FASTERRCNN_CONSTRUCTORS)}")
        self.img_size = img_size
        self.role_loss_weight = auxiliary_loss_weight
        self.role_fpn_level = role_fpn_level

        constructor = getattr(torchvision.models.detection, _FASTERRCNN_CONSTRUCTORS[variant])
        self.model = constructor(
            weights=None,  # never the full detection head (COCO classes don't match ours)
            weights_backbone="DEFAULT" if pretrained else None,  # ImageNet-pretrained ResNet50, same convention as torchvision_film.py
            num_classes=DETECTION_CLASSES + 1,  # +1: torchvision reserves class 0 for background
            min_size=img_size,
            max_size=img_size,
            # Same mitigation as torchvision_film.py's TorchvisionFPNFiLMDetector
            # (score_thresh_value=1e-3, not the class's own 0.05 default, and
            # deliberately not 0.0 either) - that repo's own FCOS path once
            # hung for 25h with an unfiltered NMS candidate set; this class
            # never got the same treatment and has now hung twice (silently,
            # no error, only caught by run_all's job-level timeout) on runs
            # that otherwise train normally, which is consistent with the
            # same failure mode.
            box_score_thresh=1e-3,
        )

        feature_channels, self.feature_stride = self._discover_fpn_channels_and_stride(role_fpn_level)
        self.role_head = RoleHead(
            feature_channels,
            embed_dim,
            hidden_dim=film_hidden_dim,
            roi_output_size=roi_output_size,
            conditioning_method=conditioning_method,
            use_deep_feature=role_head_use_deep_feature,
            use_distance_feature=role_head_use_distance_feature,
        )
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

    def _to_targets(self, batch: dict) -> tuple[list[torch.Tensor], list[dict]]:
        images = list(batch["img"].unbind(0))
        targets = []
        for i in range(batch["img"].shape[0]):
            mask = batch["batch_idx"] == i
            cx, cy, w, h = batch["bboxes"][mask].unbind(-1)
            size = self.img_size
            xyxy = torch.stack([(cx - w / 2) * size, (cy - h / 2) * size, (cx + w / 2) * size, (cy + h / 2) * size], dim=-1)
            labels = torch.ones(xyxy.shape[0], dtype=torch.long, device=xyxy.device)  # class 1 = "robot" (0 reserved for background)
            targets.append({"boxes": xyxy, "labels": labels})
        return images, targets

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        images, targets = self._to_targets(batch)
        was_training = self.model.training
        self.model.train()
        if not self.training:
            # Called for a *validation* loss (see the identical guard and
            # rationale in torchvision_film.py's compute_loss) - torchvision
            # only returns a loss dict in train mode, but forcing it
            # unconditionally would let every val-loss computation update
            # BatchNorm running stats from validation batches.
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        self._captured_feature = None
        try:
            loss_dict = self.model(images, targets)
        finally:
            self.model.train(was_training)
        detection_loss = sum(loss_dict.values())

        boxes_xyxy = norm_cxcywh_to_xyxy(batch["bboxes"], self.img_size)
        role_logits = self.role_head(
            self._captured_feature, batch["img"], boxes_xyxy, batch["batch_idx"].long(), batch["embedding"], self.feature_stride
        )
        role_labels = batch["cls"].long()
        role_loss = F.cross_entropy(role_logits, role_labels)
        role_acc = (role_logits.argmax(dim=-1) == role_labels).float().mean()

        total = detection_loss + self.role_loss_weight * role_loss
        out = {k: float(v.detach()) for k, v in loss_dict.items()}
        out["detection_loss"] = float(detection_loss.detach())
        out["role_loss"] = float(role_loss.detach())
        out["role_acc"] = float(role_acc.detach())
        out["total_loss"] = float(total.detach())
        return total, out

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, embeddings: torch.Tensor | None = None, conf_thres: float = 0.25
    ) -> list[dict[str, torch.Tensor]]:
        was_training = self.training
        self.eval()
        self._captured_feature = None
        try:
            outputs = self.model(list(images.unbind(0)))
        finally:
            self.train(was_training)
        feature_map = self._captured_feature

        results = []
        for b, out in enumerate(outputs):
            keep = out["scores"] >= conf_thres
            boxes, scores = out["boxes"][keep], out["scores"][keep]
            if boxes.shape[0] == 0:
                results.append({"boxes": boxes, "scores": scores, "labels": torch.zeros(0, dtype=torch.long, device=boxes.device)})
                continue
            box_batch_idx = torch.full((boxes.shape[0],), b, device=boxes.device, dtype=torch.long)
            role_logits = self.role_head(feature_map, images, boxes, box_batch_idx, embeddings, self.feature_stride)
            results.append({"boxes": boxes, "scores": scores, "labels": role_logits.argmax(dim=-1)})
        return results
