"""Shared base for the two Ultralytics-backed detectors (YOLO, RT-DETR):
model construction, FiLM hook registration/channel discovery, and loss
are identical between them (`Detector.compute_loss` on both is just
"forward-hooked detector.loss(batch)" - Ultralytics' own
`BaseModel.loss(batch, preds=None)` already runs the forward pass
internally when no precomputed preds are given, which is what lets RT-DETR's
denoising decoder see ground truth during its own forward call). Only
`predict()` (decoding raw outputs to boxes/scores/labels) and pretrained-
weight loading differ per architecture, so subclasses override those two.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from ultralytics.utils import DEFAULT_CFG

from semdetect.models.base import Detector
from semdetect.models.film import FiLMGenerator, apply_film


class UltralyticsFiLMDetector(Detector):
    #: set by subclass: ultralytics.nn.tasks.DetectionModel or RTDETRDetectionModel
    model_cls = None

    def __init__(
        self,
        num_classes: int = 1,
        class_names: list[str] | None = None,
        variant: str = "yolo26n",
        img_size: int = 640,
        pretrained: bool = False,
        use_film: bool = True,
        embed_dim: int = 512,
        film_hidden_dim: int = 256,
        film_layer_indices: list[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.use_film = use_film
        self.img_size = img_size
        self.variant = variant

        self.detector = self.model_cls(cfg=f"{variant}.yaml", ch=3, nc=num_classes, verbose=False)
        self.detector.args = copy.copy(DEFAULT_CFG)
        self.detector.names = {i: n for i, n in enumerate(class_names or [f"class_{i}" for i in range(num_classes)])}
        # Normally set by Ultralytics' own Trainer from the dataset yaml,
        # outside model construction; RTDETRDetectionLoss reads model.nc
        # directly (unlike v8DetectionLoss, which reads it off the head).
        self.detector.nc = num_classes

        if pretrained:
            self._load_pretrained_backbone(variant)

        if film_layer_indices is not None:
            n_layers = len(self.detector.model)
            invalid = [i for i in film_layer_indices if not (0 <= i < n_layers - 1)]
            if invalid:
                raise ValueError(
                    f"film_layer_indices {invalid} out of range for a {n_layers}-layer "
                    f"{variant} (valid: 0..{n_layers - 2}, the head itself at "
                    f"{n_layers - 1} can't be hooked as its own input)"
                )
            self.film_layer_indices = list(film_layer_indices)
        else:
            # Auto: the layers whose output feeds the detection head,
            # discovered from the head's own `.f` (from-index) attribute
            # rather than hardcoded, so this works for any single-head
            # Ultralytics yaml.
            head = self.detector.model[-1]
            self.film_layer_indices = list(head.f)

        self._current_embedding: torch.Tensor | None = None
        self._hook_handles: list = []
        if self.use_film:
            channels = self._discover_channels(self.film_layer_indices)
            self.film_generators = nn.ModuleList(
                [FiLMGenerator(embed_dim, c, hidden_dim=film_hidden_dim) for c in channels]
            )
            self._register_hooks()
        else:
            self.film_generators = nn.ModuleList()

    def _pretrained_loader(self, variant: str):
        """Returns a loaded ultralytics high-level model whose .model is a
        state_dict-compatible network for this architecture family.
        """
        from ultralytics import YOLO

        return YOLO(f"{variant}.pt")

    def _load_pretrained_backbone(self, variant: str) -> None:
        pretrained_model = self._pretrained_loader(variant).model
        own_state = self.detector.state_dict()
        pretrained_state = pretrained_model.state_dict()
        matched = {k: v for k, v in pretrained_state.items() if k in own_state and own_state[k].shape == v.shape}
        own_state.update(matched)
        self.detector.load_state_dict(own_state)
        print(f"[{type(self).__name__}] loaded {len(matched)}/{len(own_state)} pretrained tensors from {variant}.pt")

    @torch.no_grad()
    def _discover_channels(self, layer_indices: list[int]) -> list[int]:
        was_training = self.detector.training
        self.detector.eval()
        captured = {}

        def make_hook(idx):
            def hook(module, inputs, output):
                captured[idx] = output.shape[1]

            return hook

        handles = [self.detector.model[i].register_forward_hook(make_hook(i)) for i in layer_indices]
        self.detector(torch.zeros(1, 3, self.img_size, self.img_size))
        for h in handles:
            h.remove()
        self.detector.train(was_training)
        return [captured[i] for i in layer_indices]

    def _register_hooks(self) -> None:
        def make_hook(generator: FiLMGenerator):
            def hook(module, inputs, output):
                if self._current_embedding is None:
                    return output
                gamma, beta = generator(self._current_embedding.to(output.dtype))
                return apply_film(output, gamma, beta)

            return hook

        for idx, generator in zip(self.film_layer_indices, self.film_generators):
            handle = self.detector.model[idx].register_forward_hook(make_hook(generator))
            self._hook_handles.append(handle)

    def configure(self, epochs: int) -> None:
        """Must be called with the run's total epoch count before training:
        Ultralytics' end-to-end loss decays its one-to-many/one-to-one
        weighting over `args.epochs`.
        """
        self.detector.args.epochs = epochs

    def _raw_forward(self, images: torch.Tensor, embeddings: torch.Tensor | None = None):
        """Forward pass with no ground truth - fine for eval/predict, but
        NOT what compute_loss uses (see class docstring: training needs GT
        available during forward for some architectures, so compute_loss
        goes through detector.loss(batch) instead, which runs its own
        forward internally).
        """
        self._current_embedding = embeddings if self.use_film else None
        try:
            return self.detector(images)
        finally:
            self._current_embedding = None

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        self._current_embedding = batch.get("embedding") if self.use_film else None
        try:
            loss_vec, loss_items = self.detector.loss(batch)
        finally:
            self._current_embedding = None
        total = loss_vec.sum()
        loss_dict = {k: float(v) for k, v in loss_items.items()}
        loss_dict["total_loss"] = float(total.detach())
        return total, loss_dict
