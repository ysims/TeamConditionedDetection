"""Minimal, dependency-free image/box preprocessing: letterbox resize (the
standard YOLO preprocessing - resize keeping aspect ratio, pad to square)
plus an optional horizontal flip. Boxes are carried through in absolute
pixel xyxy coordinates of the *output* (letterboxed) image throughout, so
callers never have to re-derive the resize/pad transform themselves.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image


def letterbox(image: Image.Image, size: int) -> tuple[Image.Image, float, tuple[int, int]]:
    """Resize `image` to fit in an (size, size) canvas, preserving aspect
    ratio, padded with mid-grey. Returns (canvas, scale, (pad_x, pad_y)).
    """
    w, h = image.size
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, (pad_x, pad_y)


def transform_bbox_xyxy(bbox: tuple[float, float, float, float], scale: float, pad: tuple[int, int]) -> list[float]:
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = pad
    return [x1 * scale + pad_x, y1 * scale + pad_y, x2 * scale + pad_x, y2 * scale + pad_y]


def xyxy_to_norm_cxcywh(bbox: list[float], size: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2 / size, (y1 + y2) / 2 / size
    w, h = (x2 - x1) / size, (y2 - y1) / size
    return [cx, cy, w, h]


def to_chw_float(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)


def maybe_hflip(image: Image.Image, bbox: list[float], size: int, p: float, rng: random.Random) -> tuple[Image.Image, list[float]]:
    """Horizontal flip applied post-letterbox, so it only needs to know the
    canvas size, not the original image geometry.
    """
    if rng.random() >= p:
        return image, bbox
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    x1, y1, x2, y2 = bbox
    flipped_bbox = [size - x2, y1, size - x1, y2]
    return flipped, flipped_bbox


def maybe_hflip_boxes(
    image: Image.Image, bboxes: list[list[float]], size: int, p: float, rng: random.Random
) -> tuple[Image.Image, list[list[float]]]:
    """Multi-box variant of maybe_hflip, for images with more than one
    annotated object: one flip decision for the whole canvas, applied to
    every box the same way.
    """
    if rng.random() >= p:
        return image, bboxes
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_bboxes = [[size - x2, y1, size - x1, y2] for x1, y1, x2, y2 in bboxes]
    return flipped, flipped_bboxes
