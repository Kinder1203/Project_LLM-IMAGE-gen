# 🧠 Nodes Module (Business Logic Layer)

## [핵심 변경: LangGraph 분할 정복 & 상태 루프]
아키텍처가 고도화됨에 따라 기존의 비대했던 2개의 노드가, 목적이 뚜렷한 6개의 마이크로 노드 구조로 분할되었습니다.

### 노드 목록
* `router.py`: 의도 분류 및 분기점 (Condition Edge 연결용)
* `rag.py`: Chroma DB 지식 검색
* `synthesizer.py`: ComfyUI JSON 조합 및 호출
    - `generate_base_image`: 초기 생성 (Text2Img)
    - `edit_image`: 유저 커스텀 수정 (Inpainting)
    - `generate_multi_view`: 다각도 추출 및 Birefnet (Rembg)
* `validator.py`: Gemma 4 Vision 봇
    - `validate_base_image`: 초기 퀄리티 체크
    - `validate_edited_image`: 각인/보석 적용 여부 체크
    - `validate_rembg`: 누끼(알파채널) 및 빈 공간 타공 체크

## 제약 사항 ⚠️ (하네스 원칙)
1. **노드 간 직접 호출 금지**: 노드는 서로를 `import` 해서 함수처럼 부르면 안 됩니다. 모든 연결은 `agent.py` 의 LangGraph가 통제합니다.
2. **State 변형 금지**: 노드는 오직 자신에게 허용된 필드(예: `is_valid`, `retry_count`)만 업데이트해야 합니다.
3. **무한 루프 방지**: 모든 Validator 노드는 `retry_count`를 읽고 카운팅을 올려 반환해야 합니다. `agent.py`가 3회 이상일 때 루프를 강제 종료합니다.
