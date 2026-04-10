from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # vLLM chat / multimodal inference
    VLLM_CHAT_BASE_URL: str = "http://127.0.0.1:8000/v1"
    VLLM_CHAT_MODEL: str = "gemma4-26b"
    VLLM_CHAT_API_KEY: str = "EMPTY"

    # vLLM embedding inference
    VLLM_EMBED_BASE_URL: str = "http://127.0.0.1:8001/v1"
    VLLM_EMBED_MODEL: str = "BAAI/bge-m3"
    VLLM_EMBED_API_KEY: str = "EMPTY"

    # Short responses keep validator/prompt latency stable on repeated calls.
    VLLM_VALIDATOR_MAX_TOKENS: int = 120
    VLLM_PROMPT_MAX_TOKENS: int = 256

    # Backend Webhook
    WEBHOOK_URL: str = "https://graduation-work-backend.onrender.com/api/model-result"

    # Local ComfyUI Endpoint
    COMFYUI_URL: str = "http://127.0.0.1:8188"
    COMFYUI_HISTORY_TIMEOUT_SECONDS: int = 300

    # TRELLIS-style prompt additions kept neutral to avoid conflicting with
    # the complementary-background rule used before rembg.
    TRELLIS_REQUIRED_PROMPT: str = (
        ", isolated product render, high resolution, clean silhouette, "
        "studio lighting, orthographic view, highly detailed"
    )

    # Vector DB Settings
    VECTOR_DB_PATH: str = "./data/chroma_db"

    # Validation policy
    ALLOW_VALIDATION_BYPASS: bool = False
    MULTI_VIEW_VALIDATION_SAMPLE_COUNT: int = 2


config = Settings()
