from fastapi import APIRouter

from app.data.loader import DatasetPreview, list_loaded_datasets, load_dataset
from app.eval.pipeline import EvaluationRequest, run_evaluation
from app.models.registry import list_models, register_default_models
from app.db.repository import list_runs

api_router = APIRouter()


@api_router.on_event("startup")
async def startup_event() -> None:
    register_default_models()


@api_router.get("/datasets")
def datasets() -> list[DatasetPreview]:
    return list_loaded_datasets()


@api_router.post("/datasets/load")
def load_dataset_endpoint(request: DatasetPreview) -> DatasetPreview:
    return load_dataset(request)


@api_router.get("/models")
def models() -> list[dict]:
    return list_models()


@api_router.post("/eval/run")
def eval_run(request: EvaluationRequest) -> dict:
    return run_evaluation(request)


@api_router.get("/runs")
def runs() -> list[dict]:
    return list_runs()
