import json
import logging
from ..core.schemas import AgentState

logger = logging.getLogger(__name__)

def multimodal_intent_router(state: AgentState) -> dict:
    """
    (Step 1) 입력된 이미지 URL과 Prompt 유무를 체크하여 초기 의도를 판별합니다.
    """
    input_type = state.get("input_type", "text")
    prompt = state.get("user_prompt", "")
    base_image = state.get("base_ring_image_url", "")
    
    logger.info(f"Router received input type: {input_type}, base_image: {'yes' if base_image else 'no'}, prompt: {'yes' if prompt else 'no'}")
    
    # 1. 이미지만 있고 텍스트가 없는 경우 (단순 다각도 분리)
    if not prompt and base_image:
        return {"intent": "multi_view_only"}
    
    # 2. 이미지도 있고 프롬프트(각인/수정 등)도 있는 경우
    if base_image and prompt:
        return {"intent": "partial_modification"}
        
    # 3. 그 외 (텍스트만 있음)
    return {"intent": "full_custom"}

def intent_router_condition(state: AgentState) -> str:
    """ LangGraph 초기 분기 """
    intent = state.get("intent", "")
    
    if intent == "multi_view_only":
        return "generate_multi_view" # 1단계를 모두 스킵하고 2단계 직행
    elif intent == "partial_modification":
        # 공방 시안 이미지 + 프롬프트
        return "edit_image" 
    elif intent == "full_custom":
        # 백지 상태 텍스트
        return "rag_retriever"
    else:
        # User가 중단 후 나중에 보낸 승인/커스텀 리포트
        if intent == "approved_base_only":
            return "generate_multi_view"
        elif intent == "user_requested_customization":
            return "edit_image"
        
        return "generate_multi_view"
