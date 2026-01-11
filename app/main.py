from fastapi import FastAPI
import gradio as gr

from app.api.routes import api_router
from app.ui.dashboard import build_dashboard


def create_app() -> FastAPI:
    app = FastAPI(title="VLM/SLM Evaluation App")
    app.include_router(api_router, prefix="/api")

    dashboard = build_dashboard()
    app = gr.mount_gradio_app(app, dashboard, path="/")
    return app


app = create_app()
