import logging
from functools import lru_cache
from typing import Iterable, Optional

from openai import OpenAI

from .config import config

logger = logging.getLogger(__name__)

CHAT_BASE_URL = (config.VLLM_CHAT_BASE_URL or "").rstrip("/")
CHAT_MODEL = config.VLLM_CHAT_MODEL
CHAT_API_KEY = config.VLLM_CHAT_API_KEY

EMBED_BASE_URL = (config.VLLM_EMBED_BASE_URL or "").rstrip("/")
EMBED_MODEL = config.VLLM_EMBED_MODEL
EMBED_API_KEY = config.VLLM_EMBED_API_KEY


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").rstrip("/")


@lru_cache(maxsize=4)
def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    normalized_base_url = _normalize_base_url(base_url)
    if not normalized_base_url:
        raise ValueError("vLLM base_url must not be blank.")
    return OpenAI(base_url=normalized_base_url, api_key=api_key)


def _chat_client() -> OpenAI:
    return _get_openai_client(CHAT_BASE_URL, CHAT_API_KEY)


def _embed_client() -> OpenAI:
    return _get_openai_client(EMBED_BASE_URL, EMBED_API_KEY)


def _extract_text_content(message_content) -> str:
    if message_content is None:
        return ""
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(message_content)


def invoke_text_prompt(prompt: str, temperature: float = 0.3, max_tokens: Optional[int] = None) -> str:
    response = _chat_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens or config.VLLM_PROMPT_MAX_TOKENS,
    )
    return _extract_text_content(response.choices[0].message.content).strip()


def invoke_multimodal_json(prompt: str, image_data_url: str, max_tokens: Optional[int] = None) -> str:
    response = _chat_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=max_tokens or config.VLLM_VALIDATOR_MAX_TOKENS,
    )
    return _extract_text_content(response.choices[0].message.content).strip()


class VLLMEmbeddingFunction:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or EMBED_MODEL
        self.base_url = _normalize_base_url(base_url or EMBED_BASE_URL)
        self.api_key = api_key or EMBED_API_KEY

    def _client(self) -> OpenAI:
        return _get_openai_client(self.base_url, self.api_key)

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        response = self._client().embeddings.create(model=self.model, input=text_list)
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []
