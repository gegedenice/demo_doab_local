from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class DatasetPreview:
    name: str
    split: str = "train"
    source: str = "hf"


_LOADED_DATASETS: dict[str, list[dict[str, Any]]] = {}


def list_loaded_datasets() -> list[DatasetPreview]:
    return [DatasetPreview(name=name, split="train", source="cached") for name in _LOADED_DATASETS]


def load_dataset(request: DatasetPreview) -> DatasetPreview:
    if request.source == "hf":
        dataset = load_dataset(request.name, split=request.split)
        records = []
        for idx, item in enumerate(dataset):
            records.append({
                "id": str(item.get("id", idx)),
                "images": item.get("images") or item.get("image") or [],
                "metadata_text": item.get("metadata", ""),
                "unimarc_record": item.get("unimarc", ""),
            })
        _LOADED_DATASETS[request.name] = records
    else:
        raise ValueError("Only Hugging Face datasets are supported in this mock loader.")
    return request
