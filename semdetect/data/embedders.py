"""Text embedders beyond CLIP, plus the two ablation embedders and a
factory that builds whichever one a config asks for.

Every embedder here exposes the same duck-typed interface used by
DetectionDataset: `.encode(text) -> (embed_dim,) CPU tensor` (cached per
exact string) and `.embed_dim`. Swapping providers is purely a config
change (embedding.provider), never a dataset/model code change.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F


class HFMeanPoolingEmbedder:
    """Frozen HuggingFace encoder -> attention-masked mean-pooled,
    L2-normalized sentence embedding. Used for both BERT and E5 - they
    differ only in model_name and whether E5's "query: " prefix convention
    applies (E5 was trained with it; plain BERT wasn't trained for
    sentence embeddings at all, so mean pooling is a reasonable generic
    choice rather than a model-prescribed one).
    """

    def __init__(self, model_name: str, device: str = "cpu", prefix: str = ""):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.prefix = prefix
        self.embed_dim = self.model.config.hidden_size
        self._cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        batch = self.tokenizer([self.prefix + text], padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )
        out = self.model(**batch)
        mask = batch["attention_mask"][..., None].bool()
        summed = out.last_hidden_state.masked_fill(~mask, 0.0).sum(dim=1)
        pooled = summed / batch["attention_mask"].sum(dim=1, keepdim=True)
        embedding = F.normalize(pooled, p=2, dim=1).squeeze(0).cpu()
        self._cache[text] = embedding
        return embedding


class RandomTextEmbedder:
    """Ablation: a fixed per-text random unit vector - same shape and the
    same "consistent per instance" property a real embedding has, but with
    no semantic relationship to the text at all. If FiLM performs about
    the same with this as with a real embedding, the gains are coming from
    extra model capacity/a per-instance identifier, not from semantic
    content; see also description_mode="shuffled" in DataConfig, which
    ablates the opposite way (real content, wrong instance).
    """

    def __init__(self, embed_dim: int = 512, seed: int = 0):
        self.embed_dim = embed_dim
        self.seed = seed
        self._cache: dict[str, torch.Tensor] = {}

    def encode(self, text: str) -> torch.Tensor:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        # Deterministic per-text seed (not per-call) so the "random"
        # embedding is stable across epochs/workers, same as a real one.
        digest = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
        text_seed = int.from_bytes(digest[:8], "big")
        generator = torch.Generator().manual_seed(text_seed)
        vec = torch.randn(self.embed_dim, generator=generator)
        embedding = F.normalize(vec, p=2, dim=0)
        self._cache[text] = embedding
        return embedding


def build_text_embedder(config) -> "HFMeanPoolingEmbedder | RandomTextEmbedder":
    """config: semdetect.config.EmbeddingConfig."""
    if config.provider == "clip":
        from semdetect.data.clip_embedder import ClipTextEmbedder

        return ClipTextEmbedder(config.model_name, config.pretrained, config.device)
    if config.provider == "bert":
        return HFMeanPoolingEmbedder(config.model_name, config.device, prefix="")
    if config.provider == "e5":
        return HFMeanPoolingEmbedder(config.model_name, config.device, prefix="query: ")
    if config.provider == "random":
        return RandomTextEmbedder(config.random_dim)
    raise ValueError(f"Unknown embedding.provider '{config.provider}' (expected clip/bert/e5/random)")
