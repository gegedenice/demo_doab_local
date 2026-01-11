import gradio as gr

from app.data.loader import DatasetPreview, list_loaded_datasets, load_dataset
from app.eval.pipeline import EvaluationRequest, run_evaluation
from app.models.registry import list_models, register_default_models
from app.db.repository import list_runs


def build_dashboard() -> gr.Blocks:
    register_default_models()
    with gr.Blocks(title="VLM/SLM Evaluation") as demo:
        gr.Markdown("# VLM/SLM Evaluation Dashboard")
        with gr.Tab("Datasets"):
            dataset_name = gr.Textbox(label="Hugging Face dataset name")
            dataset_split = gr.Textbox(label="Split", value="train")
            dataset_limit = gr.Number(label="Limit rows", value=None, precision=0)
            dataset_sample = gr.Number(label="Sample rows (random)", value=None, precision=0)
            load_button = gr.Button("Load dataset")
            dataset_status = gr.JSON(label="Loaded dataset")
            dataset_list = gr.JSON(label="Cached datasets")

            def load_dataset_ui(name: str, split: str, limit: float | None, sample: float | None) -> dict:
                preview = DatasetPreview(
                    name=name,
                    split=split,
                    source="hf",
                    limit=int(limit) if limit else None,
                    sample=int(sample) if sample else None,
                )
                load_dataset(preview)
                return preview.__dict__

            load_button.click(
                load_dataset_ui,
                inputs=[dataset_name, dataset_split, dataset_limit, dataset_sample],
                outputs=dataset_status,
            )
            dataset_list.value = [preview.__dict__ for preview in list_loaded_datasets()]
            gr.Markdown("## Local dataset upload")
            local_dataset = gr.File(label="Upload dataset folder", file_count="directory")
            local_name = gr.Textbox(label="Dataset name (optional)")
            local_button = gr.Button("Load local dataset")

            def load_local_dataset_ui(uploaded, name: str) -> dict:
                path = _extract_upload_path(uploaded)
                preview = DatasetPreview(name=name or path, split="train", source="local", path=path)
                load_dataset(preview)
                return preview.__dict__

            local_button.click(load_local_dataset_ui, inputs=[local_dataset, local_name], outputs=dataset_status)

        with gr.Tab("Evaluate"):
            model_choices = gr.Dropdown(
                label="Model",
                choices=[model["name"] for model in list_models()],
            )
            dataset_choices = gr.Dropdown(
                label="Dataset",
                choices=[preview.name for preview in list_loaded_datasets()],
            )
            eval_type = gr.Radio(["vlm", "slm"], label="Evaluation type", value="vlm")
            run_button = gr.Button("Run evaluation")
            run_output = gr.JSON(label="Run summary")

            def run_eval_ui(dataset: str, model: str, eval_kind: str) -> dict:
                request = EvaluationRequest(dataset_name=dataset, model_name=model, eval_type=eval_kind)
                return run_evaluation(request)

            run_button.click(run_eval_ui, inputs=[dataset_choices, model_choices, eval_type], outputs=run_output)

        with gr.Tab("Runs"):
            refresh_runs = gr.Button("Refresh runs")
            run_table = gr.JSON(label="Run history")
            refresh_runs.click(lambda: list_runs(), outputs=run_table)

    return demo


def _extract_upload_path(uploaded) -> str:
    if isinstance(uploaded, list) and uploaded:
        uploaded = uploaded[0]
    if hasattr(uploaded, "name"):
        return uploaded.name
    return str(uploaded)
