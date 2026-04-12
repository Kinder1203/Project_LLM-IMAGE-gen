# scripts 운영 메모

`src/scripts/`는 파이프라인 본체와 분리된 보조 실행 스크립트를 둡니다. 현재는 `db_feeder.py`가 유일한 운영 스크립트입니다.

## db_feeder.py
- 반지 재질, 보색 배경, 각인, 수정, 다각도, rembg 검수 규칙을 Chroma DB에 적재합니다.
- 전용 컬렉션을 새로 채우는 방식이라 동일 규칙이 중복 누적되지 않습니다.
- 초기 세팅 직후, 임베딩 모델 변경 시, 규칙 지식이 바뀌었을 때 수동으로 다시 실행합니다.
- 전환 가능한 컬렉션 슬롯은 `primary collection`과 `staging collection` 두 개입니다.
- 현재 `active collection`은 `VECTOR_DB_PATH/active_collection.txt` 포인터 파일이 가리키는 슬롯입니다.
- 새 규칙은 현재 active가 아닌 슬롯에 먼저 적재하고, 적재 수 검증이 끝난 뒤에만 `active_collection.txt` 포인터를 전환합니다.
- 이전 active 컬렉션 내용은 마지막 성공 시점 기준으로 `backup collection`에 보존합니다.

## 실행 명령
```bash
python -m src.scripts.db_feeder
```

## 운영 메모
- 파이프라인 계약과 액션/상태 규칙은 상위 문서인 `src/README.md`를 기준으로 봅니다.
- 이 폴더에는 운영 스크립트 설명만 두고, 파이프라인 본문 계약은 중복 작성하지 않습니다.
- 현재 적재 경로는 `vLLM` 임베딩 endpoint를 사용하므로, 임베딩 모델이 바뀌면 기존 Chroma DB를 재색인해야 합니다.
- 문서에서 `inactive`라는 표현이 나오더라도 독립 용어로 쓰지 않고, 단순히 현재 active가 아닌 `primary` 또는 `staging` 슬롯을 가리키는 설명으로만 사용합니다.
