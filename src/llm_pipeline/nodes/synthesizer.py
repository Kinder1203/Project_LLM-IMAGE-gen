import time
import json
import logging
import requests
import re
import random
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from ..core.schemas import AgentState
from ..core.config import config

logger = logging.getLogger(__name__)

COMFY_URL = config.COMFYUI_URL

def _sync_call_comfyui(payload: dict) -> list:
    """
    ComfyUI에 프롬프트를 쏘고, 완료될 때까지 기다렸다가 진짜 저장된 파일 URL들을 긁어오는 함수.
    이 함수 덕분에 하드코딩된 mock_url이 필요 없어집니다.
    """
    try:
        # 1. 큐에 워크플로우 전송
        response = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=10)
        prompt_id = response.json().get("prompt_id")
        
        if not prompt_id:
            logger.error("Failed to get prompt_id from ComfyUI.")
            return []
            
        logger.info(f"Sent workflow to ComfyUI. Waiting for generation... (Prompt ID: {prompt_id})")
        
        # 2. 로딩 대기 (100초 타임아웃 방지 등 실제론 더 정교화 필요)
        while True:
            hist_res = requests.get(f"{COMFY_URL}/history/{prompt_id}")
            hist_data = hist_res.json()
            
            # 생성 완료 시 history 객체에 prompt_id 키가 생성됨
            if prompt_id in hist_data:
                # 3. 완료됨! 산출물 파일명 전부 추출
                image_urls = []
                outputs = hist_data[prompt_id].get("outputs", {})
                
                # 저장된 모든 노드의 outputs 순회
                for node_id, out_data in outputs.items():
                    if "images" in out_data:
                        for img in out_data["images"]:
                            filename = img["filename"]
                            # ComfyUI의 View API 템플릿 반환
                            image_urls.append(f"{COMFY_URL}/view?filename={filename}")
                return image_urls
                
            time.sleep(2.0) # 2초마다 폴링
            
    except Exception as e:
        logger.error(f"ComfyUI Polling Error: {e}")
        return []

def generate_base_image(state: AgentState) -> dict:
    """ 
    (Step 1-1) RAG 컨텍스트를 활용해 프롬프트를 짜고 완전 신규 생성 (배경 유지)
    - birefnet 노드는 제거한 상태 (z-image-turbo 템플릿 사용)
    """
    user_prompt = state.get("user_prompt", "")
    rag_context = state.get("rag_context", "")

    logger.info("Enhancing Prompt using Gemma 4 & RAG rules...")
    
    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.3
    )
    
    sys_prompt = f"""
You are an expert jewelry prompt engineer for Stable Diffusion. 
User requested: '{user_prompt}'
RAG Rules to follow: '{rag_context}'

Your task:
1. Identify the ring's design and color/material from the user's request.
2. Based on the RAG rules, calculate the EXACT complementary background color.
3. Output ONLY a comma-separated keywords prompt. It MUST end with 'solid [COLOR] background'.
NO conversational text, NO quotes. Just the prompt string.
"""
    try:
        resp = llm.invoke([HumanMessage(content=sys_prompt)])
        enhanced_prompt = resp.content.strip()
        logger.info(f"Gemma 4 Enhanced Prompt: {enhanced_prompt}")
    except Exception as e:
        logger.warning(f"Prompt enhancement failed: {e}. Using original prompt.")
        enhanced_prompt = user_prompt + ", highly detailed, solid dark background"
        
    try:
        # 바탕화면 혹은 프로젝트 경로에 있는 JSON 파일을 직접 로드합니다.
        # 실제 경로와 노드 번호("57" 등)는 파일에 맞게 수정해두면 됩니다!
        with open("image_z_image_turbo.json", "r", encoding="utf-8") as f:
            workflow = json.load(f)
            
        # 노드 번호 하드코딩 대신 마법의 키워드(Magic String) 치환 방식 사용!
        workflow_str = json.dumps(workflow)
        
        # 큰따옴표(") 인코딩 이슈를 막기 위해 순수 문자열만 안전하게 추출해서 치환
        safe_user_prompt = json.dumps(enhanced_prompt)[1:-1]
        
        # ComfyUI에서 프롬프트 칸에 ___USER_PROMPT___ 라고 적어두면 파이썬이 알아서 찾아서 바꿉니다.
        workflow_str = workflow_str.replace("___USER_PROMPT___", safe_user_prompt)
        
        # 재시도나 매 생성 시마다 똑같은 이미지가 나오지 않게 하드코딩된 seed 파괴 및 랜덤값 주입
        workflow_str = re.sub(r'"seed":\s*\d+', f'"seed": {random.randint(1, 2147483647)}', workflow_str)
        workflow_str = re.sub(r'"noise_seed":\s*\d+', f'"noise_seed": {random.randint(1, 2147483647)}', workflow_str)
        
        workflow = json.loads(workflow_str)
        
        comfyUI_payload = {
            "client_id": "llm_backend",
            "prompt": workflow
        }
    except Exception as e:
        logger.error(f"Failed to load ComfyUI JSON template: {e}")
        comfyUI_payload = {}
        
    # 실제 생성 루프 대기 후 첫번째 이미지 반환
    result_urls = _sync_call_comfyui(comfyUI_payload)
    final_url = result_urls[0] if result_urls else ""
    return {"base_ring_image_url": final_url}


def edit_image(state: AgentState) -> dict:
    """
    (Step 1-2) 기존 이미지(또는 방금 생성된베이스 이미지)를 기반으로 각인/큐빅 등 커스텀
    - qwen image edit 템플릿 사용
    """
    base_image = state.get("base_ring_image_url", "")
    custom_prompt = state.get("customization_prompt", state.get("user_prompt", ""))
    
    logger.info(f"Applying Customization (Engraving/Cubic) to image: {base_image}...")
    
    try:
        # 커스텀(Inpainting)용 JSON 파일 로드
        with open("image_qwen_image_edit_2509.json", "r", encoding="utf-8") as f:
            workflow = json.load(f)
            
        # 노드 번호 하드코딩 완전 제거 (치환 마법)
        workflow_str = json.dumps(workflow)
        
        safe_custom_prompt = json.dumps(custom_prompt)[1:-1]
        
        # ComfyUI에서 이미지 경로 값에 ___BASE_IMAGE___, 각인 텍스트 값에 ___CUSTOM_PROMPT___ 기입
        workflow_str = workflow_str.replace("___BASE_IMAGE___", base_image)
        workflow_str = workflow_str.replace("___CUSTOM_PROMPT___", safe_custom_prompt)
        
        # 재시도 루프 시 동일 결과 방지
        workflow_str = re.sub(r'"seed":\s*\d+', f'"seed": {random.randint(1, 2147483647)}', workflow_str)
        workflow_str = re.sub(r'"noise_seed":\s*\d+', f'"noise_seed": {random.randint(1, 2147483647)}', workflow_str)
        
        workflow = json.loads(workflow_str)
        
        comfyUI_payload = {
            "client_id": "llm_backend",
            "prompt": workflow
        }
    except Exception as e:
        logger.error(f"Failed to load ComfyUI JSON template: {e}")
        comfyUI_payload = {}
        
    result_urls = _sync_call_comfyui(comfyUI_payload)
    final_url = result_urls[0] if result_urls else ""
    return {"edited_ring_image_url": final_url}


def generate_multi_view(state: AgentState) -> dict:
    """
    (Step 2) 최종 채택된 이미지로 다각도 생성 + Birefnet (Rembg) 투명 누끼 적용
    """
    # 편집된 이미지가 있으면 우선 사용, 없으면 베이스 이미지 사용
    target_image = state.get("edited_ring_image_url", "") or state.get("base_ring_image_url", "")
    
    logger.info(f"Extracting Multi-views and Applying Birefnet Rembg for Trellis. Target: {target_image}")
    
    try:
        # 다각도 + Birefnet 템플릿 로드
        with open("templates-1_click_multiple_character_angles-v1.0 (3).json", "r", encoding="utf-8") as f:
            workflow = json.load(f)
            
        # 노드 번호에 의존하지 않고 텍스트 파일 단위로 교체
        workflow_str = json.dumps(workflow)
        
        # ComfyUI 안에서 LoadImage 노드의 이미지 이름 란을 ___TARGET_IMAGE___ 로 설정
        workflow_str = workflow_str.replace("___TARGET_IMAGE___", target_image)
        
        workflow = json.loads(workflow_str)
        
        comfyUI_payload = {
            "client_id": "llm_backend",
            "prompt": workflow
        }
    except Exception as e:
        logger.error(f"Failed to load ComfyUI JSON template: {e}")
        comfyUI_payload = {}
        
    # 다각도 추출의 경우 저장된 이미지가 여러 장이므로 리스트 통째로 반환
    result_urls = _sync_call_comfyui(comfyUI_payload)
    
    return {"current_image_urls": result_urls}
