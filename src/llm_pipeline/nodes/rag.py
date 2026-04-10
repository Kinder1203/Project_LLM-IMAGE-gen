import logging
import os
from langchain_chroma import Chroma
from ..core.schemas import AgentState
from ..core.config import config
from ..core.vllm_client import VLLMEmbeddingFunction

logger = logging.getLogger(__name__)

# 단일 구조의 RAG 검색기 (유사도 검색 특화)
class RingVectorRAG:
    """
    Chroma를 활용하여 사용자 질문에 맞는 Gemma 통제 가이드 및 반지 전문 지식을 찾아옴.
    """
    def __init__(self):
        self.embeddings = VLLMEmbeddingFunction(model=config.VLLM_EMBED_MODEL)
        
        # db_feeder.py가 먼저 실행되었다는 가정 하에 로드
        if os.path.exists(config.VECTOR_DB_PATH):
            self.vector_store = Chroma(
                collection_name="ring_gemma_rules",
                embedding_function=self.embeddings,
                persist_directory=config.VECTOR_DB_PATH
            )
        else:
            self.vector_store = None
            logger.warning("Vector DB not found! Please run `scripts/db_feeder.py` first.")

    def search_ring_rules(self, query: str, top_k: int = 3) -> str:
        """가장 연관성이 높은 반지 소재 지식이나 프롬프트 팁을 문자열로 반환"""
        if not self.vector_store:
            return "No specific instructions found. Follow standard jewelry prompting."
            
        logger.info(f"RingVectorRAG searching for exact logic matching: '{query}'")
        
        try:
            # 단순 Vector Similarity Search 수행
            results = self.vector_store.similarity_search(query, k=top_k)
            
            context_parts = []
            for doc in results:
                cat = doc.metadata.get("category", "General")
                context_parts.append(f"[{cat}] {doc.page_content}")
                
            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return "Failure during context retrieval."


def retrieve_rules_for_query(query: str, top_k: int = 3) -> str:
    if not query.strip():
        return "No specific instructions found. Follow standard jewelry prompting."
    rag_engine = RingVectorRAG()
    return rag_engine.search_ring_rules(query, top_k=top_k)

# 노드 래퍼 함수 (State 반영용)
def retrieve_ring_context(state: AgentState) -> dict:
    """
    (Step 2 - Text Branch) RingVectorRAG를 이용해 커플링 디자인용 맥락 확보.
    """
    prompt = state.get("user_prompt", "")
    logger.info("Executing Lightweight Vector RAG for Generative Rules...")
    
    rag_engine = RingVectorRAG()
    real_context = rag_engine.search_ring_rules(prompt)
    
    logger.info(f"Retrieved 3D Control Rules Length: {len(real_context)}")
    return {"rag_context": real_context}
