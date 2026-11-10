"""Mechanisms for injecting the (teammate_rgb, opponent_rgb) context vector
into a role head's RoI feature. Both share one interface so RoleHead and
StandaloneRoleClassifier don't need to know which one they're driving -
only `model.conditioning_method` in config changes.

    Conditioner.forward(feature_map, embedding) -> feature_map'

    film            - FiLMGenerator + apply_film (film.py): channel-wise
                      scale and shift predicted from the context.
    film_shift      - FiLM with the scale fixed at 1: a context-predicted
                      channel-wise shift only. Ablates what FiLM's scale
                      contributes - see FiLMShiftConditioner's docstring.
    cross_attention - spatial positions attend to num_tokens context-derived
                      key/value tokens (nn.MultiheadAttention), added back
                      as a zero-initialised residual. Additive only - see
                      CrossAttentionConditioner's docstring.

All start as the identity at initialization (zero-initialised final
layers, CrossAttentionConditioner's zero residual scale).

RoleHead also accepts conditioning_method "concat", which has no
conditioner: it concatenates the pooled feature with the context and
classifies with an MLP (see RoleHead).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from team_detect.models.film import FiLMGenerator, apply_film


class Conditioner(nn.Module, ABC):
    @abstractmethod
    def forward(self, feature_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """embedding: (B, embed_dim). Returns a feature map of the same shape."""


class FiLMConditioner(Conditioner):
    def __init__(self, embed_dim: int, channels: int, hidden_dim: int = 256):
        super().__init__()
        self.generator = FiLMGenerator(embed_dim, channels, hidden_dim=hidden_dim)

    def forward(self, feature_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.generator(embedding.to(feature_map.dtype))
        return apply_film(feature_map, gamma, beta)


class FiLMShiftConditioner(Conditioner):
    """x + beta(c), no scale. RoleHead spatially averages and then applies a
    linear classifier, so this makes the role logits W mean(x) + W beta(c):
    a visual term plus a context term, with no interaction between them.
    It can't represent "the jersey matches the teammate colour" - the
    ablation shows what FiLM's multiplicative term contributes.
    """

    def __init__(self, embed_dim: int, channels: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return feature_map + self.net(embedding.to(feature_map.dtype))[:, :, None, None]


class CrossAttentionConditioner(Conditioner):
    """The context embedding is mapped by a two-layer MLP to `num_tokens`
    key/value tokens; every spatial position in the feature map attends to
    them as a query. The attention output is added back as a residual
    (scaled by a zero-initialized learnable scalar, so training starts
    identical to no conditioning).

    The tokens depend only on the context, so each position receives an
    additive update drawn from a context-dependent set of vectors, with the
    visual content choosing only the softmax mixing weights - the feature
    map itself is never multiplicatively modulated, unlike FiLM.

    num_tokens must be > 1: with a single K/V token, softmax over one
    element is always exactly 1.0 regardless of the query, so the
    "attention" degenerates to a constant broadcast of the value token to
    every spatial position - after spatial pooling and a linear classifier,
    a context-dependent bias with no interaction with the visual feature
    at all.
    """

    def __init__(self, embed_dim: int, channels: int, hidden_dim: int = 256, num_heads: int = 4, num_tokens: int = 8):
        super().__init__()
        if num_tokens < 2:
            raise ValueError(f"num_tokens={num_tokens} degenerates attention to a constant broadcast - see class docstring")
        self.num_tokens = num_tokens
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, num_tokens * channels)
        )
        heads = min(num_heads, channels)
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, feature_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature_map.shape
        kv = self.text_proj(embedding.to(feature_map.dtype)).view(b, self.num_tokens, c)
        q = feature_map.flatten(2).transpose(1, 2)  # (B, HW, C)
        attn_out, _ = self.mha(q, kv, kv)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return feature_map + self.residual_scale * attn_out


_BUILDERS = {
    "film": FiLMConditioner,
    "film_shift": FiLMShiftConditioner,
    "cross_attention": CrossAttentionConditioner,
}


def build_conditioner(method: str, embed_dim: int, channels: int, hidden_dim: int = 256) -> Conditioner:
    if method not in _BUILDERS:
        raise ValueError(f"Unknown conditioning_method {method!r}, expected one of {sorted(_BUILDERS)}")
    return _BUILDERS[method](embed_dim=embed_dim, channels=channels, hidden_dim=hidden_dim)
