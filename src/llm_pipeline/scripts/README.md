# scripts 운영 메모

`src/llm_pipeline/scripts/`는 파이프라인 본체와 분리된 보조 실행 스크립트를 둡니다. 현재는 `db_feeder.py`가 유일한 운영 스크립트입니다.

## db_feeder.py
- 반지 재질, 보색 배경, 각인, 수정, 다각도, rembg 검수 규칙을 Chroma DB에 적재합니다.
- 전용 컬렉션을 새로 채우는 방식이라 동일 규칙이 중복 누적되지 않습니다.
- 초기 세팅 직후, 임베딩 모델 변경 시, 규칙 지식이 바뀌었을 때 수동으로 다시 실행합니다.

## 실행 명령
```bash
python -m src.llm_pipeline.scripts.db_feeder
```

## 운영 메모
- 파이프라인 계약과 액션/상태 규칙은 상위 문서인 `src/llm_pipeline/README.md`를 기준으로 봅니다.
- 이 폴더에는 운영 스크립트 설명만 두고, 파이프라인 본문 계약은 중복 작성하지 않습니다.
- 현재 적재 경로는 `vLLM` 임베딩 endpoint를 사용하므로, 임베딩 모델이 바뀌면 기존 Chroma DB를 재색인해야 합니다.
