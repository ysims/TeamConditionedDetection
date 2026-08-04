"""Frozen CLIP text encoder used to turn a ball/object description into a
semantic embedding vector for FiLM conditioning.

Encodes are cached by exact text string: a dataset like this one has a
couple hundred unique descriptions shared across hundreds/thousands of
images, so the cache turns "one CLIP forward pass per __getitem__" into
"one CLIP forward pass per unique description".
"""
from __future__ import annotations

import open_clip
import torch


class ClipTextEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.embed_dim = self.model.text_projection.shape[-1]
        self._cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Returns an L2-normalized (embed_dim,) CPU tensor for `text`."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        tokens = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embedding = feats.squeeze(0).cpu()
        self._cache[text] = embedding
        return embedding

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            tokens = self.tokenizer(uncached).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for text, embedding in zip(uncached, feats.cpu()):
                self._cache[text] = embedding
        return torch.stack([self._cache[t] for t in texts])
