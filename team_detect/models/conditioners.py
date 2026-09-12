"""Alternative mechanisms for injecting a semantic embedding into a
feature map, alongside FiLM (film.py). All share one interface so the
hook infrastructure in ultralytics_base.py doesn't need to know which
mechanism it's driving - only `conditioning_method` in config changes.

    Conditioner.forward(feature_map, embedding) -> feature_map'

    film                    - FiLMGenerator + apply_film (film.py), the
                              baseline mechanism this repo started with.
    cross_attention         - spatial positions attend to a single text
                              token (nn.MultiheadAttention), added back
                              as a residual.
    conditional_batchnorm,
    conditional_layernorm,
    adain                   - ConditionalNormConditioner parameterized by
                              which statistics normalize the feature map
                              (batch / channel-wise-per-position / per-
                              instance-per-channel) before a FiLM-style
                              predicted affine is applied - the textbook
                              distinction between these three techniques
                              and plain FiLM is exactly "normalize first,
                              using these statistics, THEN modulate" vs.
                              FiLM's "modulate the raw feature directly."
    gated                   - a single learned per-channel gate in [0, 1]
                              (sigmoid), multiplicative only, no additive
                              shift - tests whether FiLM's beta term (an
                              unrestricted shift) is doing anything beyond
                              what a bounded multiplicative gate could.

Every mechanism here defaults to a near-identity start at initialization
(matching FiLMGenerator's own zero-init trick, see film.py) EXCEPT the
conditional-norm family: inserting a normalization layer that wasn't
there before necessarily perturbs feature statistics from the first
forward pass, regardless of how its affine is initialized - that's an
inherent property of retrofitting normalization into an already-designed
architecture, not a bug to engineer around.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from team_conditioned_detection.models.film import FiLMGenerator, apply_film


class Conditioner(nn.Module, ABC):
    @abstractmethod
    def forward(self, feature_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """embedding: (B, embed_dim). Returns a feature map of the same shape."""

    def pop_auxiliary_loss(self) -> torch.Tensor | None:
        """Returns and clears any loss term accumulated by forward() calls
        since the last pop - None for mechanisms with no auxiliary loss
        (all current mechanisms; this is an extension point for a future
        one that trains via a side loss rather than forward-pass
        modulation, as shared_embedding briefly did before it was dropped -
        see git history if reviving that idea).
        """
        return None


class FiLMConditioner(Conditioner):
    def __init__(self, embed_dim: int, channels: int, hidden_dim: int = 256, **_):
        super().__init__()
        self.generator = FiLMGenerator(embed_dim, channels, hidden_dim=hidden_dim)

    def forward(self, feature_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.generator(embedding.to(feature_map.dtype))
        return apply_film(feature_map, gamma, beta)


class CrossAttentionConditioner(Conditioner):
    """The text embedding is projected into `num_tokens` learned key/value
    tokens; every spatial position in the feature map attends to them as a
    query. The attention output is added back as a residual (scaled by a
    zero-initialized learnable scalar, so training starts identical to no
    conditioning).

    num_tokens must be > 1: with a single K/V token, softmax over one
    element is always exactly 1.0 regardless of the query, so the "attention"
    degenerates to a constant broadcast of the value token to every spatial
    position - content-independent, strictly less expressive than FiLM (no
    multiplicative term at all), while carrying more parameters whose query
    path is entirely wasted. Confirmed empirically: with num_tokens=1, 50
    wildly different query vectors produced bit-identical output. Multiple
    tokens (all still derived from the same text embedding, via one shared
    projection reshaped into num_tokens vectors) give the softmax something
    real to discriminate between, so different spatial positions can attend
    differently depending on their own content.
    """

    def __init__(self, embed_dim: int, channels: int, hidden_dim: int = 256, num_heads: int = 4, num_tokens: int = 8, **_):
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


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W): normalizes over the
    channel dim independently at each spatial position (the ConvNeXt-style
    "LayerNorm2d", distinct from GroupNorm(1, C) which pools spatial dims
    in too). No affine - ConditionalNormConditioner supplies that.
    """

    def __init__(self, channels: int, eps: float = 1e-5, **_):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


class ConditionalNormConditioner(Conditioner):
    """Normalize (no affine) using norm_type's statistics, then apply a
    FiLM-style predicted affine on the *normalized* feature - this is the
    actual technical distinction between plain FiLM (modulates the raw
    feature) and conditional Batch/Layer/Instance(=AdaIN) norm.

        norm_type="batch"    -> nn.BatchNorm2d(affine=False): per-channel
                                 statistics over (batch, H, W).
        norm_type="layer"    -> LayerNorm2d above: per-position statistics
                                 over channels only.
        norm_type="instance" -> nn.InstanceNorm2d(affine=False): AdaIN's
                                 own normalization step - per-sample,
                                 per-channel statistics over (H, W).
    """

    def __init__(self, embed_dim: int, channels: int, hidden_dim: int = 256, norm_type: str = "batch", **_):
        super().__init__()
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(channels, affine=False)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == "layer":
            self.norm = LayerNorm2d(channels)
        else:
            raise ValueError(f"Unknown norm_type {norm_type!r} (expected batch/instance/layer)")
        self.norm_type = norm_type
        self.generator = FiLMGenerator(embed_dim, channels, hidden_dim=hidden_dim)

    def forward(self, feature_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(feature_map)
        gamma, beta = self.generator(embedding.to(feature_map.dtype))
        return apply_film(normalized, gamma, beta)


class GatedConditioner(Conditioner):
    """A single learned per-channel gate in (0, 1) (sigmoid), multiplicative
    only - no additive shift. The bias is initialized high (not zero) so
    the gate starts near 1 (identity-ish); sigmoid can't reach exactly 1.0
    from a zero-init logit the way FiLM's unrestricted gamma can reach
    exactly 1, so this needed a different init to match the "start near
    identity" convention the rest of this module follows.
    """

    def __init__(self, embed_dim: int, channels: int, hidden_dim: int = 256, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, channels)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)  # sigmoid(4) ~= 0.982

    def forward(self, feature_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.net(embedding.to(feature_map.dtype)))
        return feature_map * gate[:, :, None, None]


_BUILDERS = {
    "film": lambda **kw: FiLMConditioner(**kw),
    "cross_attention": lambda **kw: CrossAttentionConditioner(**kw),
    "conditional_batchnorm": lambda **kw: ConditionalNormConditioner(norm_type="batch", **kw),
    "conditional_layernorm": lambda **kw: ConditionalNormConditioner(norm_type="layer", **kw),
    "adain": lambda **kw: ConditionalNormConditioner(norm_type="instance", **kw),
    "gated": lambda **kw: GatedConditioner(**kw),
}


def build_conditioner(method: str, embed_dim: int, channels: int, hidden_dim: int = 256) -> Conditioner:
    if method not in _BUILDERS:
        raise ValueError(f"Unknown conditioning_method {method!r}, expected one of {sorted(_BUILDERS)}")
    return _BUILDERS[method](embed_dim=embed_dim, channels=channels, hidden_dim=hidden_dim)
