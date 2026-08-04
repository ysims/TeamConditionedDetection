"""Generic bounding-box + semantic-embedding detection dataset.

Consumes a manifest CSV (image_path,
width, height, x1, y1, x2, y2, class_name, instance_id) joined against a
descriptions CSV keyed by instance_id. Swapping "ball" for some other
object with intra-class appearance variance (e.g. jerseys, signs, fruit)
only requires a different manifest + descriptions CSV with the same shape.

Each item also carries a text embedding of that instance's description,
computed via whichever embedder semdetect.data.embedders.build_text_embedder
constructed for the configured provider (CLIP/BERT/E5/random).
"""
from __future__ import annotations

import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from semdetect.data.manifest import ManifestRow, load_descriptions, load_manifest
from semdetect.data.transforms import letterbox, maybe_hflip, to_chw_float, transform_bbox_xyxy, xyxy_to_norm_cxcywh


def build_shuffled_lookup(instance_ids: set[str]) -> dict[str, str]:
    """Deterministic derangement (cyclic shift by 1 over sorted ids) used
    by description_mode="shuffled": every instance maps to a *different*
    instance's description, consistently across the whole dataset/run.
    """
    ids = sorted(instance_ids)
    if len(ids) < 2:
        return {i: i for i in ids}
    return dict(zip(ids, ids[1:] + ids[:1]))


class DetectionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        descriptions_csv: str,
        class_names: list[str],
        text_embedder,
        description_field: str = "Description",
        description_mode: str = "correct",
        img_size: int = 640,
        augment: bool = False,
        hflip_prob: float = 0.5,
        seed: int = 0,
    ):
        self.rows: list[ManifestRow] = load_manifest(manifest_path)
        self.descriptions = load_descriptions(descriptions_csv)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.text_embedder = text_embedder
        self.description_field = description_field
        if description_mode not in ("correct", "shuffled"):
            raise ValueError(f"Unknown description_mode {description_mode!r} (expected 'correct' or 'shuffled')")
        self.description_mode = description_mode
        self.img_size = img_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.embed_dim = text_embedder.embed_dim if text_embedder is not None else 0
        self._rng = random.Random(seed)
        self._shuffled_lookup = (
            build_shuffled_lookup({r.instance_id for r in self.rows}) if description_mode == "shuffled" else {}
        )

        missing = {r.class_name for r in self.rows} - set(self.class_to_idx)
        if missing:
            raise ValueError(f"Manifest contains class(es) not in class_names: {missing}")

    def __len__(self) -> int:
        return len(self.rows)

    def _description_source_id(self, instance_id: str) -> str:
        return self._shuffled_lookup[instance_id] if self.description_mode == "shuffled" else instance_id

    def _description_text(self, instance_id: str) -> str:
        source_id = self._description_source_id(instance_id)
        row = self.descriptions.get(source_id)
        if row is None:
            raise KeyError(f"No description row for instance_id={source_id!r}")
        text = row.get(self.description_field)
        if not text:
            raise KeyError(f"Description field {self.description_field!r} empty for {source_id!r}")
        return text

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(row.image_path).convert("RGB")
        canvas, scale, pad = letterbox(image, self.img_size)
        bbox = transform_bbox_xyxy((row.x1, row.y1, row.x2, row.y2), scale, pad)

        if self.augment:
            canvas, bbox = maybe_hflip(canvas, bbox, self.img_size, self.hflip_prob, self._rng)

        bbox_norm = xyxy_to_norm_cxcywh(bbox, self.img_size)
        image_tensor = torch.from_numpy(to_chw_float(canvas))

        text = self._description_text(row.instance_id)
        if self.text_embedder is not None:
            embedding = self.text_embedder.encode(text)
        else:
            embedding = torch.zeros(0)

        return {
            "image": image_tensor,
            "bboxes": torch.tensor([bbox_norm], dtype=torch.float32),
            "labels": torch.tensor([self.class_to_idx[row.class_name]], dtype=torch.long),
            "embedding": embedding,
            "instance_id": row.instance_id,
            "description_source_id": self._description_source_id(row.instance_id),
            "description": text,
            "image_path": row.image_path,
        }


def collate_fn(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])

    batch_idx, cls, bboxes = [], [], []
    for i, b in enumerate(batch):
        n = b["bboxes"].shape[0]
        batch_idx.append(torch.full((n,), i, dtype=torch.float32))
        cls.append(b["labels"])
        bboxes.append(b["bboxes"])

    out = {
        "img": images,
        "cls": torch.cat(cls).float(),
        "bboxes": torch.cat(bboxes).float(),
        "batch_idx": torch.cat(batch_idx),
        "instance_ids": [b["instance_id"] for b in batch],
        "description_source_ids": [b["description_source_id"] for b in batch],
        "descriptions": [b["description"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }
    if batch[0]["embedding"].numel() > 0:
        out["embedding"] = torch.stack([b["embedding"] for b in batch])
    else:
        out["embedding"] = None
    return out
