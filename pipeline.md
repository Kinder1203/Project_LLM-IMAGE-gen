# Couple Ring Customization Pipeline Architecture
해당 문서는 커플링 커스텀 서비스의 핵심인 LangGraph 기반 오케스트레이션 및 ComfyUI 렌더링 파이프라인의 설계 사상을 명세합니다.

## Architecture Highlights
본 아키텍처는 **LangGraph의 상태 머신 (StateGraph)** 과 **MemorySaver Checkpointer** 를 활용한 Human-in-the-loop 패턴을 기반으로 설계되었습니다. 단순 스크립트 기반 연결이 아닌, 조건과 상황(사용자 승인, 재시도 루프)에 유연하게 대처할 수 있는 견고한 백엔드 구조입니다.

### 1. Human-in-the-loop (인간 개입 대기)
LangGraph에서 특정 노드에 진입하기 직전에 파이프라인을 의도적으로 일시 정지(Interrupt)시킵니다.
* **1단계 완료 시점**: 배경이 있는 단일 앞면 반지 이미지를 생성한 직후 파이프라인은 멈춥니다. 결과를 프론트엔드로 보내 사용자가 수락(Accept)할지, 수정(Customization)할지 선택하기를 기다립니다.
* **선택 시 Resume**: 선택 결과가 API로 들어오면, 멈췄던 StateGraph가 깨어나 선택된 노드(`edit_image` 또는 `generate_multi_view`)로 진입합니다.

### 2. Gemma 4 통합 자가 검열 루프 (Self-correction)
LangGraph의 Conditional Edge를 이용하여 **최대 3회**의 자가 복구 로직을 구현했습니다.
* 각각의 생성 노드 바로 뒤에 전담 검수 노드(`validate_base_image`, `validate_edited_image`, `validate_rembg`)가 배치되어 있습니다.
* 퀄리티 미달로 판정(is_valid: False)되면, LangGraph는 즉시 직전 생성 모델로 회귀시킵니다. 3회가 넘어갈 경우 무한 루프 방지를 위해 종료(Failed) 처리합니다.

---

## 파이프라인 진행 플로우 (The Workflow)

### Step 1: Input & Routing (의도 파악)
- **`router.py`**: 입력 데이터(텍스트 단독, 이미지 단독, 이미지+텍스트)를 판단하여 `full_custom`, `multi_view_only`, `partial_modification` 3가지 분기 중 하나로 연결시킵니다.

### Step 2: RAG & Base Generation (초기 시안 제작)
*(텍스트만 입력되었을 경우)*
- **`rag.py`**: Chroma DB에서 18k 프래티넘, 각인 등의 전문 지식과 프롬프트 가이드를 검색합니다.
- **`generate_base_image`**: `z-image-turbo` 형식을 사용하여, 배경이 존재하는 고품질의 반지 이미지를 **1장** 만듭니다. (Rembg 미적용)
- **`validate_base_image`**: Gemma 4 가 프롬프트 요구사항 반영 여부를 확인합니다.
- **[정지 지점]**: 여기서 멈춰 사용자에게 반지를 보여주고 응답을 기다립니다.

### Step 3: Customization (커스텀 합성)
*(사용자가 추가 각인이나 큐빅을 요구하거나, 초기부터 공방 시안을 입력한 경우)*
- **`edit_image`**: `qwen_image_edit` 형식을 사용하여 기존 반지의 기하학적 형태는 유지한 채 사용자가 입력한 프롬프트를 바탕으로 합성합니다.
- **`validate_edited_image`**: 수정된 사항이 잘 들어갔는지 다시 Gemma 4가 검수합니다.

### Step 4: Multi-view & Rembg (다각도 분해 및 투명화)
*(사용자 승인이 떨어졌거나, 커스텀이 완료된 최종 시안을 대상으로)*
- **`generate_multi_view`**: ComfyUI 다중 각도 렌더링을 지시하고, 핵심적으로 **birefnet 노드**를 태워 배경과 **반지 안쪽 공간을 완전히 투명화** 시킵니다.
- **`validate_rembg`**: Gemma 4 가 누끼가 완벽히 따졌는지(반지 링 안쪽까지) 까다롭게 검수합니다.

### Step 5: Webhook Dispatch
모든 검수와 다각도 추출이 끝났다면, 메인 렌더링 서버(onrender.com) 의 Webhook으로 투명화된 다각도 이미지들을 조용히 밀어넣습니다.

---

## 실행 및 테스트 방법

### 1. 지식 DB 가동
처음 1회는 반지 도메인 지식을 DB화해야 합니다.
```bash
python -m src.llm_pipeline.scripts.db_feeder
```

### 2. 로컬 API 테스트 가동
해당 파이프라인의 접점은 `pipelines.py` 의 `process_generation_request` 입니다.
Mocking된 FastAPI 환경이나 스트립트를 통해 다음과 같은 페이로드를 날리면 구동됩니다:

**처음 시작할 때:**
```json
{
    "thread_id": "user_id_123",
    "action": "start",
    "input_type": "text",
    "prompt": "18k 로즈골드에 우아한 곡선이 들어간 커플링"
}
```
결과는 `waiting_for_user` 상태가 반환됩니다.

**사용자가 커스텀 추가를 클릭했을 때:**
```json
{
    "thread_id": "user_id_123",
    "action": "request_customization",
    "customization_prompt": "내부에 Forever 라고 각인 새겨줘"
}
```
LangGraph는 MemorySaver에 저장해두었던 `user_id_123` 스레드를 깨워서, 멈춰있던 커스텀 편집 노드부터 알아서 이어 나갑니다!