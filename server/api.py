from fastapi import APIRouter

from src.llm_pipeline.core.config import config
from src.llm_pipeline.core.schemas import PipelineRequest, PipelineResponse
from src.llm_pipeline.pipelines import process_generation_request

router = APIRouter()


@router.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "ring-llm-pipeline",
        "status": "ok",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
    }


@router.get("/healthz", tags=["meta"])
def healthz() -> dict:
    return {
        "status": "ok",
        "service": "ring-llm-pipeline",
        "chat_model": config.VLLM_CHAT_MODEL,
        "embed_model": config.VLLM_EMBED_MODEL,
        "checkpoint_path": config.LANGGRAPH_CHECKPOINT_DB_PATH,
    }


@router.post(
    "/pipeline",
    response_model=PipelineResponse,
    response_model_exclude_none=True,
    tags=["pipeline"],
    summary="Execute or resume the ring generation pipeline.",
)
def run_pipeline(request: PipelineRequest) -> PipelineResponse:
    return process_generation_request(request)
