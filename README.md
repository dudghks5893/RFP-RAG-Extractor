# 🚀 RFP-RAG-Extractor: 지능형 입찰 제안서(RFP) 분석 및 요약 RAG 시스템

> **B2G 입찰지원 전문 컨설팅 스타트업 – ‘입찰메이트’ 사내 프로젝트**  
> "최적의 RAG 아키텍처를 찾기 위한 시나리오별 비교 실험: 자체 구축 vs API 활용"

---

## 📑 프로젝트 개요 (Project Overview)

본 프로젝트는 공공입찰 컨설팅 스타트업 **‘입찰메이트’**의 엔지니어링 팀으로서, 산적해 있는 **RFP(제안요청서)** 분석 업무를 자동화하는 RAG 시스템을 구축합니다.

특히, 실무 환경의 제약 사항을 고려하여 **[시나리오 A: 자체 구축형]**과 **[시나리오 B: API 기반형]** 두 가지 경로를 모두 구현하고, 각 방식의 성능과 비용, 효율성을 비교 분석하는 데 중점을 둡니다.

---

## 📅 프로젝트 일정 (Timeline)

- **프로젝트 시작**: 2026-05-13
- **프로젝트 발표**: 2026-06-02
- **프로젝트 종료**: 2026-06-04

일정표: https://docs.google.com/spreadsheets/d/1DTFdOJc9QJ7a0s2nsPM6GOhIQz8b61wI/edit?usp=drive_link&ouid=112188858007110237324&rtpof=true&sd=true
---

## 🎯 주요 미션 및 실험 목표 (Key Missions & Goals)

1. **Dual-Scenario 구현**: 
   - **Scenario A (On-premise Style)**: 클라우드 환경에서 오픈소스 LLM을 활용한 A to Z 시스템 구축 (Unique 포인트)
   - **Scenario B (Cloud API Style)**: LLM API를 활용한 고성능/고효율 시스템 신속 구축
2. **비교 분석**: 두 시나리오 간의 답변 정확도, 응답 속도, 구축 난이도 등을 정량적/정성적으로 비교
3. **핵심 정보 추출**: 100여 개의 실제 RFP 문서에서 예산, 마감 기한, 제출 방식 등 주요 메타데이터 추출
4. **팀별 평가 체계**: 자체 선정 지표를 통한 성능 평가 및 의사 결정 과정 기록

---

## 🛠 기술 스택 (Tech Stack Options)

팀별 논의에 따라 선택적으로 활용하는 유연한 스택을 지향합니다.

| 구분 | 시나리오 A (Self-hosted) | 시나리오 B (API-based) |
| :--- | :--- | :--- |
| **LLM** | Open-source LLM (Llama 3, Mistral 등) | OpenAI GPT, Anthropic Claude, etc. |
| **Embedding** | Local Embedding Models (Hugging Face) | OpenAI / Google Embedding API |
| **Framework** | LangChain / LlamaIndex / Haystack | LangChain / LlamaIndex |
| **Vector DB** | ChromaDB / FAISS / Qdrant | Pinecone / Weaviate / ChromaDB |
| **Infra** | GCP Compute Engine (GPU) | Cloud Serverless / API |

---

## 🏗 시스템 아키텍처 (Architecture)

1. **Data Ingestion**: RFP PDF/HWP 문서 텍스트 추출 및 구조화
2. **Experimental RAG Pipeline**:
   - **Scenario A**: 자체 GPU 인스턴스 기반 모델 서빙 및 검색 엔진 구축
   - **Scenario B**: 상용 LLM API 인터페이스를 통한 신속한 RAG 파이프라인 연결
3. **Evaluation Phase**: 시나리오별 결과물 비교 및 최적의 솔루션 제안

---

## 📊 평가 및 지표 (Evaluation)

- **RAG Metrics**: Context Precision, Faithfulness, Answer Relevancy (RAGAS 등 활용)
- **Comparison Points**:
  - 인프라 구축 난이도 및 유지보수성
  - 질의응답의 정확도 및 할루시네이션 발생 빈도
  - 실무 적용 가능성 및 비용 효율성

---

## 프로젝트 구조
```
RFP-RAG-Extractor/                         # 기업/정부 RFP 문서 요약·질의응답 RAG 시스템 프로젝트 루트
│
├── configs/                               # 실험/실행 설정 파일 모음
│   └── baseline_rag.yaml                  # 베이스라인 RAG 실행 설정: 경로, 모델명, top_k, batch_size, 평가 샘플 등
│
├── data/                                  # 데이터 저장소, Git 관리 제외 권장
│   ├── raw/                               # 원본 데이터 저장 위치
│   │   ├── data_list.csv                  # 메타데이터 CSV: 공고번호, 사업명, 발주기관, 파일명, 예산 등
│   │   ├── *.hwp                          # 원본 HWP RFP 문서
│   │   ├── *.pdf                          # 원본 PDF RFP 문서
│   │   └── *.docx                         # 원본 DOCX RFP 문서
│   │
│   ├── processed/                         # 전처리 및 평가 관련 중간 산출물
│   │   ├── extracted/                     # 원본 파일에서 직접 추출한 raw text 저장 위치
│   │   │   └── <doc_id>.txt               # 문서별 추출 원문 텍스트
│   │   │
│   │   ├── cleaned/                       # 정제 완료 텍스트 저장 위치
│   │   │   └── <doc_id>.txt               # 문서별 정제 텍스트
│   │   │
│   │   └── eval/                          # 평가 데이터셋 및 RAG 실행 결과 저장 위치
│   │       ├── eval_dataset.json       # 최종 평가 데이터셋
│   │       ├── eval_dataset_sample_20.json # 실제 빠른 평가용 샘플 평가셋
│   │       ├── rag_outputs_baseline_section_sample_20.json        # RAG 응답 원본 결과
│   │       └── rag_outputs_baseline_section_sample_20_scored.json # keyword score 부착 결과
│   │
│   ├── chunks/                            # 청킹 결과 저장 위치
│   │   └── section/                       # 목차/섹션 기반 청킹 결과
│   │       └── section_chunks.jsonl       # RAG 인덱싱에 사용할 최종 청크 파일
│   │
│   └── vector_db/                         # 로컬 벡터 DB 저장 위치
│       └── baseline_section_kure_faiss/   # section chunk + embedding +  Vector DB 인덱스
│           ├── index.faiss                # 벡터 인덱스 파일
│           ├── chunks.pkl                 # 벡터 DB row와 매칭되는 청크 메타데이터/본문
│           └── config.json                # 인덱스 생성 당시의 config snapshot
│
├── logs/                                  # 실행 로그 저장 위치
│   ├── extract_clean_chunk_log.csv        # 원본 파일 추출/정제/청킹 처리 로그
│   └── run_rag_eval.log                   # 선택: RAG 평가 실행 로그 저장용
│
├── notebooks/                             # 실험용 Jupyter Notebook 공간
│   ├── 01_extract_clean_chunk.ipynb       # 원본 파일 추출, 정제, 청킹 실험용 노트북
│   └── 02_baseline_rag_eval.ipynb         # 벡터 DB 구축, RAG 실행, 평가 실험용 노트북
│
├── reports/                               # 평가 리포트 및 분석 결과 저장 위치
│   └── evaluation/                        # RAG 평가 결과 모음
│       ├── baseline_section...sample20_metrics.json              # 전체 평가 지표
│       ├── baseline_section...sample20_by_question_type.json     # question_type별 평가 결과
│       ├── baseline_section...sample20_by_source_type.json       # source_type별 평가 결과
│       ├── baseline_section...sample20_by_answer_format.json     # answer_format별 평가 결과
│       ├── baseline_section...sample20_by_file_type.json         # file_type별 평가 결과
│       ├── baseline_section...sample20_retrieval_failures.csv    # 검색 실패 케이스
│       ├── baseline_section...sample20_keyword_failures.csv      # 키워드 정답요소 누락 케이스
│       ├── baseline_section...ample20_summary.csv                # 사람이 보기 쉬운 평가 요약표
│       └── baseline_section...sample20_experiment_summary.json   # 실험 설정+결과 요약
│
├── scripts/                               # 터미널에서 실행하는 진입점 스크립트
│   ├── check_project_modules.py           # 전체 모듈 import, config, 주요 경로 존재 여부 점검
│   ├── run_extract_chunk.py               # 원본 PDF/HWP/DOCX 추출·정제·청킹 파이프라인 실행
│   └── run_rag_eval.py                    # YAML config 기반 RAG 평가 파이프라인 실행
│
├── src/                                   # 재사용 가능한 핵심 Python 모듈
│   ├── __init__.py                        # src 패키지 인식용 파일
│   │
│   ├── extractors/                        # 원본 문서 포맷별 텍스트 추출 모듈
│   │   ├── __init__.py                    # 확장자별 추출 함수 import 및 분기 제공
│   │   ├── pdf_extractor.py               # PDF 텍스트 추출, PyMuPDF 기반
│   │   ├── hwp_extractor.py               # HWP 텍스트 추출, olefile 기반
│   │   └── docx_extractor.py              # DOCX 텍스트 추출, python-docx 기반
│   │
│   ├── chunking/                          # 청킹 전략 관련 모듈
│   │   ├── __init__.py                    # chunking 패키지 인식용
│   │   ├── outline_detector.py            # 제목/목차 후보 라인 탐지 로직
│   │   └── section_chunker.py             # 제목/섹션 기반 청크 생성 로직
│   │
│   ├── utils/                             # 공통 유틸리티 함수 모음
│   │   ├── __init__.py                    # utils 패키지 인식용
│   │   ├── config_utils.py                # YAML config 로드, 경로 해석, config snapshot 저장/비교
│   │   ├── file_utils.py                  # JSONL 저장/로드 등 파일 입출력 유틸
│   │   ├── path_utils.py                  # 프로젝트 루트 탐색 함수
│   │   ├── progress_utils.py              # tqdm 대체용 print 기반 진행률 로거
│   │   ├── text_cleaner.py                # 유니코드, 제어문자, 공백, 줄바꿈 등 텍스트 정제 함수
│   │   ├── eval_dataset_utils.py          # 평가 데이터셋 로드, 저장, 균형 샘플링 함수
│   │   ├── seed.py                        # random, numpy, torch seed 고정 함수
│   │   └── device.py                      # CUDA, MPS, CPU device 탐지 함수
│   │
│   ├── embeddings/                        # 임베딩 모델 관련 모듈
│   │   ├── __init__.py                    # EmbeddingModel import 제공
│   │   └── embedding_model.py             # SentenceTransformer 로드, 문서/쿼리 임베딩 생성
│   │
│   ├── vectorstores/                      # Vector DB 저장/로드/검색 모듈
│   │   ├── __init__.py                    # FAISSVectorStore import 제공
│   │   └── faiss_store.py                 # FAISS 인덱스 생성, 저장, 로드, 검색, config 호환성 확인
│   │
│   ├── retrieval/                         # 검색 단계 모듈
│   │   ├── __init__.py                    # RAGRetriever import 제공
│   │   └── retriever.py                   # query embedding + FAISS search + retrieved_ids/context 추출
│   │
│   ├── generation/                        # LLM 프롬프트 및 답변 생성 모듈
│   │   ├── __init__.py                    # prompt 함수와 LLMGenerator import 제공
│   │   ├── prompts.py                     # RFP 전용 system prompt, context formatting, chat messages 생성
│   │   └── llm_generator.py               # llm tokenizer/model 로드, 답변 생성, token/latency 기록
│   │
│   ├── evaluation/                        # RAG 성능 평가 모듈
│   │   ├── __init__.py                    # evaluation 패키지 인식용
│   │   └── rag_evaluator.py               # retrieval, generation, keyword, latency/cost 평가 및 실패 케이스 저장
│   │
│   └── pipeline/                          # 전체 실행 파이프라인 모듈
│       ├── __init__.py                    # ExtractChunkPipeline, RAGEvalPipeline import 제공
│       ├── extract_chunk_pipeline.py      # 원본 파일 직접 추출→정제→목차 기반 청킹→section_chunks 저장
│       └── rag_eval_pipeline.py           # 청크 로드→FAISS 구축/로드→RAG 실행→평가→결과 저장
│
├── .env                                   # API 키, 환경변수 등 민감 정보, Git 제외
├── .gitignore                             # data, vector_db, .env, __pycache__ 등 Git 제외 설정
├── requirements-mac.txt                   # 프로젝트 의존성 패키지 목록
├── requirements-win.txt                   # 프로젝트 의존성 패키지 목록
├── requirements-jupiter.txt               # 프로젝트 의존성 패키지 목록
├── requirements-common.txt                # 프로젝트 의존성 패키지 목록
└── README.md                              # 프로젝트 설명, 설치 방법, 실행 방법, 실험 관리 문서
```




## 👥 팀원 (Team Members)

| 이름 | 역할 | GitHub |
| :--- | :--- | :--- |
| **채영환** | 역할1 | [@dudghks5893](https://github.com/dudghks5893) |
| **원숙현** | 역할2 | [@sooqhyunwon](https://github.com/id) |
| **한성택** | 역할3 | [@zoyhanee](https://github.com/id) |
| **양기우** | 역할4 | [@yang12-1](https://github.com/id) |

---

## ⚙️ 설치 및 실행 방법 (Installation)

```bash
# 저장소 복제
git clone [https://github.com/dudghks5893/RFP-RAG-Extractor.git](https://github.com/dudghks5893/RFP-RAG-Extractor.git)
cd RFP-RAG-Extractor

# GCP 관리자 전체 가상환경 활성화
source /opt/jhub-venv/bin/activate

# 개인 주피터 가상환경 활성화
source .venv/bin/activate


# 필수 패키지 설치

# 현재 설치된 모든 패키지를 리스트업해서 한꺼번에 삭제 (필요 시 사용)
pip freeze | xargs pip uninstall -y

# Mac 전용 설정으로 설치 (MPS(Metal) 가속 또는 CPU 사용)
pip install -r requirements-mac.txt

# Windows 전용 설정으로 설치 (윈도우용 CUDA 또는 CPU 사용)
pip install -r requirements-win.txt

# 주피터 전용 설정으로 설치 (리눅스용 CUDA 커널 사용)
pip install -r requirements-jupyter.txt

# 시나리오 선택 실행 (예시)
python main.py --scenario A
python main.py --scenario B
