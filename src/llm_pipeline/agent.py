from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .core.schemas import AgentState

from .nodes.router import multimodal_intent_router, intent_router_condition
from .nodes.rag import retrieve_ring_context

from .nodes.synthesizer import generate_base_image, edit_image, generate_multi_view
from .nodes.validator import validate_base_image, validate_edited_image, validate_rembg

def check_base_validation(state: AgentState) -> str:
    is_valid = state.get("is_valid", False)
    retries = state.get("retry_count", 0)
    
    if is_valid:
        # 합격하면 사용자 승인 대기 전용 빈 노드로 이동
        return "wait_for_user_approval"
    
    if retries >= 3:
        return "end"
    return "generate_base_image"

def check_edit_validation(state: AgentState) -> str:
    is_valid = state.get("is_valid", False)
    retries = state.get("retry_count", 0)
    
    if is_valid:
        return "generate_multi_view"
    
    if retries >= 3:
        return "end"
    return "edit_image"

def check_rembg_validation(state: AgentState) -> str:
    is_valid = state.get("is_valid", False)
    retries = state.get("retry_count", 0)
    
    if is_valid:
        return "end"
    
    if retries >= 3:
        return "end"
    return "generate_multi_view"

def wait_for_user_approval(state: AgentState) -> dict:
    """ 
    사용자 승인을 받기 위한 정지(Interrupt) 지점을 형성하는 빈 노드. 
    로직 상 generate_multi_view로 강제로 넘어가는 이전 설계의 치명적 결함을 방지합니다. 
    """
    return {}

def route_after_approval(state: AgentState) -> str:
    """ 사용자의 응답(action) 결과에 맞춰서 최종적으로 분기를 나눠줍니다. """
    intent = state.get("intent", "")
    if intent == "user_requested_customization":
        return "edit_image"
    return "generate_multi_view"

def build_ring_generation_graph():
    """ LangGraph 빌드 (조건부 분기 및 Human-in-the-loop 적용) """
    workflow = StateGraph(AgentState)
    
    # 1. 메모리 세이버 (Interrupt 지원)
    checkpointer = MemorySaver()
    
    # 2. 노드 등록
    workflow.add_node("intent_router", multimodal_intent_router)
    workflow.add_node("rag_retriever", retrieve_ring_context)
    
    workflow.add_node("generate_base_image", generate_base_image)
    workflow.add_node("validate_base_image", validate_base_image)
    
    workflow.add_node("wait_for_user_approval", wait_for_user_approval) # 추가된 휴게소 노드
    
    workflow.add_node("edit_image", edit_image)
    workflow.add_node("validate_edited_image", validate_edited_image)
    
    workflow.add_node("generate_multi_view", generate_multi_view)
    workflow.add_node("validate_rembg", validate_rembg)
    
    # 3. 엣지 연결 (흐름 제어)
    workflow.set_entry_point("intent_router")
    
    # 라우터 조건부 분기
    workflow.add_conditional_edges(
        "intent_router",
        intent_router_condition,
        {
            "rag_retriever": "rag_retriever",
            "edit_image": "edit_image",
            "generate_multi_view": "generate_multi_view"
        }
    )
    
    # Text-to-Image 생성 분기
    workflow.add_edge("rag_retriever", "generate_base_image")
    workflow.add_edge("generate_base_image", "validate_base_image")
    workflow.add_conditional_edges(
        "validate_base_image",
        check_base_validation,
        {
            "wait_for_user_approval": "wait_for_user_approval",
            "generate_base_image": "generate_base_image",
            "end": END
        }
    )
    
    # 사용자 승인 후의 분기 (휴게소에서 출발)
    workflow.add_conditional_edges(
        "wait_for_user_approval",
        route_after_approval,
        {
            "edit_image": "edit_image",
            "generate_multi_view": "generate_multi_view"
        }
    )
    
    # Image Edit 생성 분기
    workflow.add_edge("edit_image", "validate_edited_image")
    workflow.add_conditional_edges(
        "validate_edited_image",
        check_edit_validation,
        {
            "generate_multi_view": "generate_multi_view",
            "edit_image": "edit_image",
            "end": END
        }
    )
    
    # 다각도 + Rembg 분기
    workflow.add_edge("generate_multi_view", "validate_rembg")
    workflow.add_conditional_edges(
        "validate_rembg",
        check_rembg_validation,
        {
            "generate_multi_view": "generate_multi_view",
            "end": END
        }
    )
    
    # 4. 컴파일 (사용자 피드백을 받기 전 일시정지할 지점 선언)
    # validate_* 노드가 끝나고 사용자가 대답할 휴게소(wait_for_user_approval) 진입 전에 파이프라인 정지.
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["wait_for_user_approval"]
    )
    
    return app
