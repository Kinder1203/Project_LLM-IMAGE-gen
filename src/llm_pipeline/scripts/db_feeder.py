import os
import json
from loguru import logger
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ..core.config import config

# [B] 커플링/반지 공방 전문 지식 (Jewelry Terminology)
RING_KNOWLEDGE_BASE = [
    {
        "category": "Jewelry_Design",
        "title": "Band Materials",
        "content": "18k White Gold offers a sleek, modern look. Platinum is highly durable and hypoallergenic. Rose Gold provides a warm, romantic hue. Silver is affordable but requires maintenance against tarnishing."
    },
    {
        "category": "Jewelry_Design",
        "title": "Gemstones and Engravings",
        "content": "A 'Princess Cut' cubic gives a modern square silhouette, while a 'Round Brilliant' maximizes sparkle. Engravings should be specified explicitly, e.g., 'Engraved with the word Forever on the inner band'."
    }
]

# [A] Gemma 커플링 3D 렌더링 프롬프트 가이드 (템플릿 제어 규칙)
GEMMA_PROMPT_GUIDES = [
    {
        "category": "Gemma_Multi_Angle_Prompting",
        "title": "Azimuth (Horizontal Angle) Control",
        "content": "To generate multiple views of a ring, specify camera angles clearly: 'front view' (showing the main gem), 'side view' (profile of the band), and 'top-down view' (looking down at the ring)."
    },
    {
        "category": "Gemma_Multi_Angle_Prompting",
        "title": "Modification and Inpainting Control",
        "content": "When adding an engraving or a cubic to an existing ring, focus the prompt strictly on the added element: e.g., 'Add a small princess-cut diamond cubic to the center', 'Engrave initials inside the band'."
    },
    {
        "category": "Gemma_Multi_Angle_Prompting",
        "title": "Distance and Background",
        "content": "Always append 'macro shot, highly detailed jewelry photography, solid white background' for consistency."
    }
]

def init_vector_db():
    """
    위의 딕셔너리 지식들을 Chroma Vector DB에 임베딩하여 로컬에 밀어 넣는 초기화 함수입니다.
    기존 Kuzu(GraphDB) 오버엔지니어링 코드를 제거하고 순수 Vector RAG로 다이어트했습니다.
    """
    db_path = config.VECTOR_DB_PATH
    logger.info(f"Initializing Lightweight Chroma DB at {db_path}...")
    
    # 1. 임베딩 모델 준비 (로컬 Ollama 임베딩)
    embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=config.OLLAMA_BASE_URL)
    
    # 임시 디렉토리 생성
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    # 2. Chroma DB 객체 연결
    vector_store = Chroma(
        collection_name="ring_gemma_rules",
        embedding_function=embedder,
        persist_directory=db_path
    )
    
    # 3. 문서화 및 임베딩 
    all_docs = GEMMA_PROMPT_GUIDES + RING_KNOWLEDGE_BASE
    texts = []
    metadatas = []
    
    for doc in all_docs:
        text = f"[{doc['category']}] {doc['title']}: {doc['content']}"
        texts.append(text)
        metadatas.append({"category": doc["category"], "title": doc["title"]})
        
    logger.debug(f"Ingesting {len(texts)} logic entries into Vector DB. This may take a moment...")
    
    # 4. DB에 일괄 삽입
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    logger.success(f"Successfully ingested {len(all_docs)} foundational rules into ChromaDB.")
    logger.info("Now the LLM Synthesizer will quickly fetch and exactly know how to prompt the Gemma 4 model for Rings!")

if __name__ == "__main__":
    init_vector_db()
