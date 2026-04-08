from typing import Optional, Dict, Any, Literal, List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# 외부에서 파이프라인으로 들어오는 Request
class PipelineRequest(BaseModel):
    input_type: Literal["text", "image", "modification"] = Field(..., description="입력 타입: 텍스트 단독, 이미지 단독, 또는 기존 이미지+프롬프트 수정")
    prompt: Optional[str] = Field(None, description="커플링 디자인 설명 본문")
    image_url: Optional[str] = Field(None, description="기존 반지 시안 또는 이미지 입력 URL")
    
    # Human-in-the-loop 대응 파라미터
    thread_id: str = Field("default_thread", description="LangGraph 세션 스레드 ID")
    action: Literal["start", "accept_base", "request_customization"] = Field("start", description="실행 액션 (처음 시작, 승인, 또는 커스텀 요청)")
    customization_prompt: Optional[str] = Field(None, description="수정 요청 시 추가 각인/보석 디테일 설명")

# 파이프라인이 외부로 내보내는 Response (다각도 배열 반환)
class PipelineResponse(BaseModel):
    status: Literal["success", "failed", "waiting_for_user"]
    optimized_image_urls: List[str] = Field(default_factory=list, description="TRELLIS Multi-view용으로 변환된 다각도 이미지 배열")
    message: str = Field(..., description="사용자 안내 메시지")
    base_image_url: Optional[str] = None

# LangGraph 내부 State (Current Image URL이 List[str]로 진화)
class AgentState(TypedDict):
    input_type: str
    user_prompt: str
    user_image: str
    intent: str
    rag_context: str
    synthesized_prompt: str
    
    # 1/2단계 산출물 및 커스텀 요청 필드
    base_ring_image_url: str # z-image-turbo 생성 결과 또는 사용자가 업로드한 시안
    customization_prompt: str # 유저의 추가 수정 요청 사항 (각인명 등)
    edited_ring_image_url: str # qwen image edit 결과를 저장할 필드
    
    # 자가 검열 루프 (Self-correction) 속성
    validation_reason: str
    retry_count: int
    
    # 처리 중인 다각도 이미지 리스트 (1장~N장 수용)
    current_image_urls: List[str]
    
    # Validation (검증)
    is_valid: bool
    validation_feedback: str
    
    # 최종 결과물 배열
    final_output_urls: List[str]
    status_message: str
