from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.data.loader import _LOADED_DATASETS
from app.db.repository import save_run
from app.eval.metrics.mock import MockMetric


@dataclass
class EvaluationRequest:
    dataset_name: str
    model_name: str
    eval_type: str


def run_evaluation(request: EvaluationRequest) -> dict:
    dataset = _LOADED_DATASETS.get(request.dataset_name, [])
    metric = MockMetric()
    results = []
    for item in dataset:
        prediction = _mock_prediction(request.eval_type, item)
        score = metric.compute(prediction, item.get("unimarc_record", ""))
        results.append({
            "item_id": item.get("id"),
            "prediction": prediction,
            "score": score,
        })
    run_id = save_run(request, results)
    return {
        "run_id": run_id,
        "count": len(results),
        "average_score": _average_score(results),
    }


def _mock_prediction(eval_type: str, item: dict[str, Any]) -> str:
    if eval_type == "vlm":
        return f"mock-metadata:{item.get('metadata_text', '')}"
    return f"mock-unimarc:{item.get('metadata_text', '')}"


def _average_score(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(result["score"] for result in results) / len(results)
