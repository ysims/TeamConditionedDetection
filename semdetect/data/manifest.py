"""Loading of the manifest CSVs produced by scripts/prepare_nupbr_dataset.py
and the description CSV they're joined against.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass
class ManifestRow:
    image_path: str
    width: int
    height: int
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    instance_id: str


def load_manifest(path: str) -> list[ManifestRow]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                ManifestRow(
                    image_path=r["image_path"],
                    width=int(r["width"]),
                    height=int(r["height"]),
                    x1=float(r["x1"]),
                    y1=float(r["y1"]),
                    x2=float(r["x2"]),
                    y2=float(r["y2"]),
                    class_name=r["class_name"],
                    instance_id=r["instance_id"],
                )
            )
    return rows


def load_descriptions(path: str) -> dict[str, dict[str, str]]:
    """instance_id -> {column_name: value}, e.g. {"Description": ..., "Colours": ...}."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        name_col = reader.fieldnames[0]
        for row in reader:
            out[row[name_col]] = row
    return out
