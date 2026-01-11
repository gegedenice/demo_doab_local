from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelEntry:
    name: str
    model_type: str
    endpoint: str
    description: str


_MODEL_REGISTRY: list[ModelEntry] = []


def register_default_models() -> None:
    if _MODEL_REGISTRY:
        return
    _MODEL_REGISTRY.extend([
        ModelEntry(
            name="mock-vlm",
            model_type="vlm",
            endpoint="http://localhost:8001/v1",
            description="Mock VLM via OpenAI-compatible endpoint",
        ),
        ModelEntry(
            name="mock-slm",
            model_type="slm",
            endpoint="http://localhost:8002/v1",
            description="Mock SLM via OpenAI-compatible endpoint",
        ),
    ])


def list_models() -> list[dict]:
    return [entry.__dict__ for entry in _MODEL_REGISTRY]
