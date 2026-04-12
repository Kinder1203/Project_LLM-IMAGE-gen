import logging
import os
from functools import lru_cache

from langchain_chroma import Chroma

from ..core.config import config
from ..core.schemas import AgentState
from ..core.vector_db_runtime import resolve_active_collection_name
from ..core.vllm_client import VLLMEmbeddingFunction

logger = logging.getLogger(__name__)


def _format_context_piece(page_content: str, category: str) -> str:
    normalized_content = (page_content or "").strip()
    prefix = f"[{category}]"
    if normalized_content.startswith(prefix):
        return normalized_content
    return f"{prefix} {normalized_content}".strip()


class RingVectorRAG:
    """Chroma를 활용하여 사용자 질문에 맞는 Gemma 통제 가이드 및 반지 전문 지식을 찾아온다."""

    def __init__(
        self,
        vector_db_path: str | None = None,
        embed_model: str | None = None,
        collection_name: str | None = None,
    ):
        self.vector_db_path = vector_db_path or config.VECTOR_DB_PATH
        self.embed_model = embed_model or config.VLLM_EMBED_MODEL
        self.collection_name = collection_name or resolve_active_collection_name()
        self.embeddings = VLLMEmbeddingFunction(model=self.embed_model)

        # db_feeder.py가 먼저 실행되었다는 가정 하에 로드
        if os.path.exists(self.vector_db_path):
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_path,
            )
        else:
            self.vector_store = None
            logger.warning("Vector DB not found! Please run `python -m src.scripts.db_feeder` first.")

    def search_ring_rules(self, query: str, top_k: int | None = None) -> str:
        """가장 연관성이 높은 반지 소재 지식이나 프롬프트 팁을 문자열로 반환"""
        if not self.vector_store:
            return "No specific instructions found. Follow standard jewelry prompting."

        logger.info(f"RingVectorRAG searching for exact logic matching: '{query}'")

        try:
            search_top_k = top_k if top_k is not None else config.RAG_DEFAULT_TOP_K
            results = self.vector_store.similarity_search(query, k=search_top_k)

            context_parts = []
            for doc in results:
                cat = doc.metadata.get("category", "General")
                context_parts.append(_format_context_piece(doc.page_content, cat))

            return "\n\n".join(context_parts)
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            return "Failure during context retrieval."


@lru_cache(maxsize=8)
def _get_rag_engine(vector_db_path: str, embed_model: str, collection_name: str) -> RingVectorRAG:
    return RingVectorRAG(
        vector_db_path=vector_db_path,
        embed_model=embed_model,
        collection_name=collection_name,
    )


def retrieve_rules_for_query(query: str, top_k: int | None = None) -> str:
    if not query.strip():
        return "No specific instructions found. Follow standard jewelry prompting."
    collection_name = resolve_active_collection_name()
    rag_engine = _get_rag_engine(config.VECTOR_DB_PATH, config.VLLM_EMBED_MODEL, collection_name)
    return rag_engine.search_ring_rules(query, top_k=top_k)


def retrieve_ring_context(state: AgentState) -> dict:
    """(Step 2 - Text Branch) RingVectorRAG를 이용해 커플링 디자인용 맥락 확보."""
    prompt = state.get("user_prompt", "")
    logger.info("Executing Lightweight Vector RAG for Generative Rules...")

    collection_name = resolve_active_collection_name()
    rag_engine = _get_rag_engine(config.VECTOR_DB_PATH, config.VLLM_EMBED_MODEL, collection_name)
    real_context = rag_engine.search_ring_rules(prompt, top_k=config.RAG_DEFAULT_TOP_K)

    logger.info(f"Retrieved multi-view control rules length: {len(real_context)}")
    return {"rag_context": real_context}
