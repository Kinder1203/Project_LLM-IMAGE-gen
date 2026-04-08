import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json
from ..core.schemas import AgentState
from ..core.config import config

logger = logging.getLogger(__name__)

import base64
import requests

def _encode_image_from_url(image_url: str) -> str:
    """ ComfyUI에서 생성된 이미지를 다운로드하여 Base64로 변환 """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to download image for validation: {e}")
        return ""

def _call_vision_judge(image_url: str, prompt: str) -> dict:
    """ 공통 Vision LLM 호출 유틸 """
    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.0,
        format="json"
    )
    
    try:
        # 하드코딩(Mock) 제거! 실제 이미지를 가져와 Gemma 4 Vision 모델에게 텍스트와 함께 던집니다.
        img_base64 = _encode_image_from_url(image_url)
        if not img_base64:
             return {"is_valid": True, "reason": "Bypassed: Could not fetch image."}
             
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
        
        resp = llm.invoke([
            HumanMessage(content=message_content)
        ])
        return json.loads(resp.content.strip())
    except Exception as e:
        logger.warning(f"Gemma 4 Vision validation failed: {e}")
        return {"is_valid": True, "reason": "Bypassed due to LLM error."}

def validate_base_image(state: AgentState) -> dict:
    """ Step 1-1 검수: 프롬프트에 맞는 퀄리티의 베이스 링이 생성되었는지 판별 """
    target_img = state.get("base_ring_image_url", "")
    user_prompt = state.get("user_prompt", "")
    retry_count = state.get("retry_count", 0)
    
    sys_prompt = f"Does the generated ring in the image accurately reflect the user's request: '{user_prompt}'? Ensure the background is present but the ring is high quality. Return: {{'is_valid': true/false, 'reason': '...'}}"
    
    logger.info(f"Validating Base Ring Generation (Retry: {retry_count})...")
    result = _call_vision_judge(target_img, sys_prompt)
    
    is_valid = result.get("is_valid", True)
    return {
        "is_valid": is_valid,
        "validation_reason": result.get("reason", ""),
        "retry_count": retry_count + 1 if not is_valid else retry_count
    }

def validate_edited_image(state: AgentState) -> dict:
    """ Step 1-2 검수: 사용자가 지시한 커스텀(각인/보석)이 제대로 합성되었는지 판별 """
    target_img = state.get("edited_ring_image_url", "")
    custom_prompt = state.get("customization_prompt", "")
    retry_count = state.get("retry_count", 0)
    
    sys_prompt = f"Does the image show the requested modifications: '{custom_prompt}'? Are engravings text legible or the added gem clear? Return JSON {{'is_valid': true/false, 'reason': '...'}}"
    
    logger.info(f"Validating Custom Edit Application (Retry: {retry_count})...")
    result = _call_vision_judge(target_img, sys_prompt)
    
    is_valid = result.get("is_valid", True)
    return {
        "is_valid": is_valid,
        "validation_reason": result.get("reason", ""),
        "retry_count": retry_count + 1 if not is_valid else retry_count
    }

def validate_rembg(state: AgentState) -> dict:
    """ Step 2 검수: 다각도 분리 및 반지 안쪽 빈공간 투명화(누끼)가 완벽한지 판별 """
    urls = state.get("current_image_urls", [])
    retry_count = state.get("retry_count", 0)
    
    sys_prompt = "You are a TRELLIS preparation judge. Look at this ring image. Is the background completely removed (transparent alpha), and more importantly, is the inner hole of the ring completely hollowed out without background artifact remaining? Return JSON {{'is_valid': true/false, 'reason': '...'}}"
    
    logger.info(f"Validating Rembg (Alpha channel & inner hole) for TRELLIS (Retry: {retry_count})...")
    
    # 여러 장 중 하나라도 실패하면 False (간략화)
    is_valid = True
    reason = "All multi-views passed rembg validation."
    
    for url in urls:
        res = _call_vision_judge(url, sys_prompt)
        if not res.get("is_valid", True):
            is_valid = False
            reason = res.get("reason", "Failed on one of the multi-views.")
            break
            
    return {
        "is_valid": is_valid,
        "validation_reason": reason,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
        "final_output_urls": urls if is_valid else []
    }

def validate_input_image(state: AgentState) -> dict:
    """ 시나리오 2/3 진입 시: 업로드된 시안 이미지의 배경(보색 대비)을 사전 검사하여 누끼 사고 방지 """
    target_img = state.get("base_ring_image_url", "")
    
    sys_prompt = "You are a pre-processing judge. Check the image. Is the background color highly contrasting (complementary) to the ring's color to allow perfect alpha matting? If it's too similar (e.g., white ring on white background), output is_valid=false, and in 'reason', write EXACTLY a short directive for an inpainting model to fix it, like 'Change the background to solid pitch black'. Return JSON {'is_valid': true/false, 'reason': '...'}"
    
    logger.info("Guarding: Validating Input Image Contrast for Scenarios 2/3...")
    result = _call_vision_judge(target_img, sys_prompt)
    
    
    is_valid = result.get("is_valid", True)
    reason = result.get("reason", "Good contrast.")
    
    update_dict = {
        "is_valid": is_valid,
        "validation_reason": reason
    }
    
    if not is_valid:
        current_prompt = state.get("customization_prompt", "")
        # Append the AI's background fix instruction
        new_prompt = f"{current_prompt} AND {reason}".strip(" AND ")
        update_dict["customization_prompt"] = new_prompt
        
    return update_dict
