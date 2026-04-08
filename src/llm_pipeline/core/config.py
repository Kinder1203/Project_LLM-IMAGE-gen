import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM Settings (단일 Gemma 4 모델 통합)
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:26b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Backend Webhook 연동 (결과 전송)
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "https://graduation-work-backend.onrender.com/api/model-result")
    
    # ComfyUI Endpoint
    COMFYUI_URL: str = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
    
    # TRELLIS Prompt Additions
    TRELLIS_REQUIRED_PROMPT: str = ", solid white background, high resolution, isolated on white, studio lighting, orthographic view, highly detailed"
    
    # DB Settings (기존 SpeakNode KuzuDB 연동용)
    VECTOR_DB_PATH: str = "./data/chroma_db"
    GRAPH_DB_PATH: str = "./data/kuzu_db"
    
    class Config:
        env_file = ".env"

config = Settings()
