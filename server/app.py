from fastapi import FastAPI

from .api import router

app = FastAPI(
    title="Ring LLM Pipeline API",
    version="1.0.0",
    summary="HTTP wrapper for the ring generation and HITL pipeline.",
)

app.include_router(router)
