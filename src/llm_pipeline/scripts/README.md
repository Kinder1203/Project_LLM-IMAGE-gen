# Scripts 디렉토리 명세 (Harness Engineering)

파이프라인 백엔드 웹 서버와 독립적으로 실행되는 스크립트 도구들입니다.
주로 데이터베이스 초기화 및 배치성 작업용으로 쓰입니다.

## 책임 (Responsibilities)
1. **`db_feeder.py`**: RAG용 Chroma Vector DB에 커플링 전문 지식(재질, 보석 세공, 프롬프팅 가이드 등)을 영구 적재(Ingest)하는 독립 실행형 파이썬 스크립트.

## 제약 사항 (Constraints)
* 파이프라인의 핵심 라우팅 과정 중에는 이 스크립트 파일이 호출되지 않습니다.
* DB 구조가 변경되면 이 스크립트만 단독 실행(`python db_feeder.py`)하여 지식을 초기화합니다.
