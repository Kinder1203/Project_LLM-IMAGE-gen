from .core.schemas import PipelineRequest, PipelineResponse
from .agent import build_ring_generation_graph
from .core.config import config
from loguru import logger
import requests

# 싱글톤 형태로 그래프 초기화
app_graph = build_ring_generation_graph()

def process_generation_request(request: PipelineRequest) -> PipelineResponse:
    """
    (Entrypoint) 외부 연동 API
    - Action에 따라 LangGraph 모델을 초기 시작(start) 하거나 다시 진행시킵니다.
    """
    logger.info(f"User Request Intent Type: {request.input_type}")
    logger.info(f"Action: {request.action} on Thread: {request.thread_id}")

    thread_config = {"configurable": {"thread_id": request.thread_id}}

    try:
        if request.action == "start":
            initial_state = {
                "user_prompt": request.prompt or "",
                "input_type": request.input_type,
                "base_ring_image_url": request.image_url or "",
                "retry_count": 0,
                "intent": "",
                "customization_prompt": ""
            }
            # 초기 실행
            app_graph.invoke(initial_state, config=thread_config)
            
        elif request.action == "accept_base":
            # 인간 승인. 변경사항 없이 다각도 생성으로 속행
            app_graph.update_state(thread_config, {"intent": "approved_base_only"})
            app_graph.invoke(None, config=thread_config)
            
        elif request.action == "request_customization":
            # 커스텀(각인/큐빅) 요청 속행
            app_graph.update_state(thread_config, {
                "intent": "user_requested_customization",
                "customization_prompt": request.customization_prompt or "",
                "retry_count": 0 
            })
            app_graph.invoke(None, config=thread_config)

        # 현재 Langgraph 상태 체크 (중단/종료 구분)
        current_state_obj = app_graph.get_state(thread_config)
        final_state = current_state_obj.values
        next_nodes = current_state_obj.next
        
        # 만약 다음 노드가 잡혀있다면 == interrupt 발생 (사용자 대기)
        if next_nodes:
            logger.info(f"Pipeline paused. Waiting for human. Next nodes: {next_nodes}")
            return PipelineResponse(
                status="waiting_for_user",
                optimized_image_urls=[],
                base_image_url=final_state.get("base_ring_image_url", ""),
                message="기본 반지가 준비되었습니다. 승인하시겠습니까, 아니면 커스텀(각인/보석)을 진행하시겠습니까?"
            )
            
        # 끝났다면
        is_valid = final_state.get("is_valid", False)
        
        if is_valid:
            output_urls = final_state.get("final_output_urls", [])
            log_msg = f"렌더링 최적화 성공! (이미지 수: {len(output_urls)}장)"
            
            # 메인 서버 Webhook 전송 (하드코딩 주석 해제)
            try:
                payload = {
                    "status": "success",
                    "images": output_urls,
                    "prompt_used": final_state.get("synthesized_prompt", "")
                }
                logger.info(f"Sending Webhook to backend: {config.WEBHOOK_URL}")
                requests.post(config.WEBHOOK_URL, json=payload, timeout=10)
            except Exception as e:
                logger.error(f"Webhook 발송 실패: {e}")
                
            return PipelineResponse(
                status="success",
                optimized_image_urls=output_urls,
                message=log_msg,
                base_image_url=final_state.get("base_ring_image_url", "")
            )
        else:
            return PipelineResponse(
                status="failed",
                optimized_image_urls=[],
                message=final_state.get("status_message", "검수 과정에서 최종 불합격 처리되었습니다."),
                base_image_url=final_state.get("base_ring_image_url", "")
            )
            
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        return PipelineResponse(
            status="failed",
            optimized_image_urls=[],
            message="서버 오류로 인해 파이프라인이 중단되었습니다.",
            base_image_url=""
        )
