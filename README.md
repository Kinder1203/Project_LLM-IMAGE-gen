# 반지 커스텀 다각도 2D 이미지 세트용 LLM 파이프라인 서비스

LangGraph, `vLLM`, ComfyUI, Vision 검수를 조합해 반지 시안 생성, 커스텀 수정, 최종 다각도 2D 이미지 세트 생성을 오케스트레이션하는 내부 서비스입니다. 이 저장소는 프론트엔드가 직접 호출하는 앱 서버가 아니라, 상위 애플리케이션 백엔드가 호출하는 LLM 파이프라인 서비스 구현을 담습니다. 현재 도메인은 `반지`이며, 추론 백엔드는 `vLLM OpenAI-compatible endpoint`를 기준으로 합니다.

## 프로젝트 개요
- `text`: 텍스트만 받아 베이스 반지 시안을 생성한 뒤, 사용자 승인 후 다각도 2D 이미지 세트를 만듭니다.
- `image_and_text`: 기존 반지 이미지를 입력받아 텍스트 수정 요청을 반영한 뒤, 사용자 승인 후 다각도 2D 이미지 세트를 만듭니다.
- `image_only`: 완성된 반지 이미지를 입력받아 다각도 2D 이미지 세트와 배경 제거 결과를 바로 만듭니다.
- 현재 공개 계약의 최종 산출물은 `optimized_image_urls`에 담기는 다각도 2D 이미지 세트입니다. 별도 mesh 파일, model 파일, CAD 산출물은 이 문서 범위에 포함하지 않습니다.

## 시스템 경계
```text
프론트엔드
  -> 상위 애플리케이션 백엔드
  -> LLM 파이프라인 서비스
  -> vLLM / ComfyUI / Chroma
```

LLM 파이프라인 서비스는 입력 정규화, LangGraph 기반 HITL 흐름, ComfyUI 실행, Vision 검수, RAG 조회를 담당합니다. 사용자 세션 보관, 화면 상태 관리, 외부 제품 API 결합은 상위 애플리케이션 백엔드 책임으로 둡니다.

## 문서 맵
- `README.md`: 사람용 온보딩, 설치, 실행 순서, 폴더 구조
- `AGENTS.md`: Codex 작업 규칙과 문서/테스트 우선순위
- `src/README.md`: 액션, 상태, 검수 정책, HITL, webhook, RAG 재색인 계약의 단일 기준 문서
- `server/README.md`: 상위 애플리케이션 백엔드가 호출하는 HTTP wrapper 계약
- `src/scripts/README.md`: `db_feeder.py` 운영 메모

## 아키텍처 요약
- `src/pipelines.py`: 외부 엔트리포인트 `process_generation_request()`를 제공합니다.
- `src/agent.py`: LangGraph 상태 그래프와 pause/resume 흐름을 정의합니다.
- `src/nodes/`: 라우팅, RAG, ComfyUI 생성, Vision 검수를 수행합니다.
- `src/core/`: 설정, Pydantic 스키마, vLLM 클라이언트, 벡터 DB 런타임 유틸을 둡니다.
- `server/`: FastAPI HTTP wrapper를 둡니다.
- `src/scripts/`: 벡터 DB 적재 같은 운영 스크립트를 둡니다.

## 설치
```bash
pip install -r requirements.txt
```

## 환경 준비
아래는 로컬 온보딩용 최소 예시입니다. 세부 계약과 내부 튜닝값 설명은 `src/README.md`와 `src/core/config.py`를 기준으로 봅니다.

```env
VLLM_CHAT_BASE_URL=http://127.0.0.1:8000/v1
VLLM_CHAT_MODEL=gemma4-e4b
VLLM_CHAT_API_KEY=EMPTY
VLLM_EMBED_BASE_URL=http://127.0.0.1:8002/v1
VLLM_EMBED_MODEL=BAAI/bge-m3
VLLM_EMBED_API_KEY=EMPTY
COMFYUI_URL=http://127.0.0.1:8188
VECTOR_DB_PATH=./data/chroma_db
LANGGRAPH_CHECKPOINT_DB_PATH=./data/langgraph_checkpoints.sqlite
WEBHOOK_URL=NONE
ALLOW_VALIDATION_BYPASS=false
```

## 실행 순서
1. 채팅·비전 추론용 `vLLM` 인스턴스를 실행합니다.
2. 임베딩 전용 `vLLM` 인스턴스를 실행합니다.
3. ComfyUI를 실행합니다.
4. 벡터 DB를 적재합니다.

```bash
python -m src.scripts.db_feeder
```

5. 아래 둘 중 하나로 서비스를 사용합니다.

직접 호출:
```python
from src.core.schemas import PipelineRequest
from src.pipelines import process_generation_request

response = process_generation_request(
    PipelineRequest(
        thread_id="demo-thread",
        action="start",
        prompt="18k 화이트골드에 얇은 곡선 각인이 들어간 커플링",
    )
)
print(response.status)
```

HTTP wrapper:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8080
```

로컬 수동 시연:
```bash
python test_run.py
```

## 하네스 엔지니어링 기준
- 이 저장소는 무거운 통합 테스트보다 계약과 경계면을 먼저 고정하는 방식으로 관리합니다.
- 계약을 바꿀 때는 `계약 문서 -> 계약 테스트 -> 코드 -> 데모 스크립트` 순서를 기준으로 정리합니다.
- 기본 계약 검증은 아래 명령을 사용합니다.

```bash
python -m unittest tests.test_pipeline_contract
```

## 폴더 구조
```text
.
|-- README.md
|-- AGENTS.md
|-- requirements.txt
|-- test_run.py
|-- comfyui_workflow/
|-- data/
|-- input_images/
|-- output_images/
|-- server/
|-- tests/
`-- src/
    |-- README.md
    |-- __init__.py
    |-- agent.py
    |-- pipelines.py
    |-- core/
    |-- nodes/
    `-- scripts/
```

`input_images/`와 `output_images/`는 로컬 데모용 폴더입니다. `data/`는 Chroma와 LangGraph checkpoint 같은 런타임 산출물 위치로 사용됩니다.
