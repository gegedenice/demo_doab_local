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
            load_button = gr.Button("Load dataset")
            dataset_status = gr.JSON(label="Loaded dataset")
            dataset_list = gr.JSON(label="Cached datasets")

            def load_dataset_ui(name: str, split: str) -> dict:
                preview = DatasetPreview(name=name, split=split, source="hf")
                load_dataset(preview)
                return preview.__dict__

            load_button.click(load_dataset_ui, inputs=[dataset_name, dataset_split], outputs=dataset_status)
            dataset_list.value = [preview.__dict__ for preview in list_loaded_datasets()]

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
