from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass
class DatasetPreview:
    name: str
    split: str = "train"
    source: str = "hf"
    limit: int | None = None
    sample: int | None = None
    path: str | None = None
    id_column: str | None = None
    image_column: str | None = None
    metadata_column: str | None = None
    unimarc_column: str | None = None


_LOADED_DATASETS: dict[str, list[dict[str, Any]]] = {}


def list_loaded_datasets() -> list[DatasetPreview]:
    return [DatasetPreview(name=name, split="train", source="cached") for name in _LOADED_DATASETS]


def load_dataset(request: DatasetPreview) -> DatasetPreview:
    if request.source == "hf":
        dataset = load_dataset(request.name, split=request.split)
        if request.sample:
            dataset = dataset.shuffle(seed=42).select(range(request.sample))
        elif request.limit:
            dataset = dataset.select(range(request.limit))
        id_column = request.id_column or "id"
        image_column = request.image_column or "images"
        metadata_column = request.metadata_column or "metadata"
        unimarc_column = request.unimarc_column or "unimarc"
        records = []
        for idx, item in enumerate(dataset):
            images_value = item.get(image_column) or item.get("image")
            images = images_value if isinstance(images_value, list) else [images_value] if images_value else []
            records.append({
                "id": str(item.get(id_column, idx)),
                "images": images,
                "metadata_text": item.get(metadata_column, ""),
                "unimarc_record": item.get(unimarc_column, ""),
            })
        _LOADED_DATASETS[request.name] = records
    elif request.source == "local":
        if not request.path:
            raise ValueError("Local dataset path is required.")
        dataset_name = request.name or Path(request.path).name
        records = _load_local_dataset(Path(request.path))
        _LOADED_DATASETS[dataset_name] = records
        request = DatasetPreview(
            name=dataset_name,
            split=request.split,
            source=request.source,
            limit=request.limit,
            sample=request.sample,
            path=request.path,
        )
    else:
        raise ValueError("Only Hugging Face or local datasets are supported in this mock loader.")
    return request


def _load_local_dataset(root_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for doc_path in sorted(root_path.glob("doc_*")):
        page_text_path = doc_path / "page_texts.json"
        if not page_text_path.exists():
            continue
        with page_text_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        page_texts = payload.get("page_texts", [])
        images = sorted(str(path) for path in doc_path.glob("*.png"))
        combined_text = "\n".join(text for text in page_texts if text)
        records.append({
            "id": str(payload.get("doc_index", doc_path.name)),
            "images": images,
            "metadata_text": combined_text,
            "unimarc_record": combined_text,
        })
    return records
