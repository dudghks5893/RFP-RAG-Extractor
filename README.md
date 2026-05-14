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


```
bidding-mate-rag/
│
├── data/                      # 데이터 저장소 (Git에 올라가지 않도록 .gitignore 처리 필수)
│   ├── raw/                   # 원본 데이터 (100개의 HWP, PDF 파일 및 data_list.csv)
│   ├── processed/             # 전처리된 텍스트, 분할된 청크 데이터 등
│   └── vector_db/             # 로컬에 저장되는 Vector DB 파일 (ChromaDB, FAISS 등 사용 시)
│
├── notebooks/                 # 팀원별/기능별 실험용 Jupyter Notebook
│   ├── member1/               # 팀원 각자의 실험 공간 (Git 충돌 방지용)
│   ├── member2/
│   ├── 01_data_eda.ipynb      # 메타데이터 분석 및 데이터 탐색
│   ├── 02_pdf_hwp_parsing.ipynb # 문서 파싱 실험
│   ├── 03_chunking_exp.ipynb  # 청킹 전략 실험
│   ├── 04_scenario_b_api.ipynb# 시나리오 B (LLM API 기반) 실험
│   └── 05_scenario_a_loc.ipynb# 시나리오 A (로컬/클라우드 직접 구축) 실험
│
├── src/                       # 핵심 파이썬 모듈 (Notebook에서 import 하여 사용)
│   ├── __init__.py
│   ├── data_loader/           # 데이터 불러오기 및 파싱
│   │   ├── __init__.py
│   │   ├── pdf_parser.py      # PDF 추출 로직 (PyMuPDF, pdfplumber 등)
│   │   └── hwp_parser.py      # HWP 추출 로직 (hwp5txt, OLE 파싱 등)
│   │
│   ├── chunking/              # 문서 청킹 로직
│   │   ├── __init__.py
│   │   └── splitter.py        # RecursiveCharacter, Semantic 분할, 중첩(Overlap) 등
│   │
│   ├── retrieval/             # 검색 및 Vector DB 관련 로직
│   │   ├── __init__.py
│   │   ├── vector_store.py    # DB 연결, 임베딩, 데이터 적재
│   │   └── searcher.py        # 메타데이터 필터링, MMR, Hybrid Search, Re-ranking 로직
│   │
│   ├── generation/            # 답변 생성 및 LLM 연결
│   │   ├── __init__.py
│   │   ├── prompts.py         # 프롬프트 템플릿 관리 (RAG 컨텍스트 반영 로직)
│   │   └── llm_client.py      # 시나리오 A/B에 따른 LLM 호출 래퍼 함수
│   │
│   └── evaluation/            # 성능 평가 로직
│       ├── __init__.py
│       ├── metrics.py         # 정량적/정성적 평가지표 계산 로직
│       └── test_cases.py      # 제공된 질문 세트 테스트 자동화 로직
│
├── configs/                   # 환경 설정 파일 (청크 사이즈, k값, 모델 파라미터 등)
│   ├── config.yaml            # 하이퍼파라미터 및 설정값 (chunk_size, top_k, temperature 등)
│   └── prompt_templates.json  # 시스템 프롬프트 등
│
├── app/                       # (선택) 최종 결과물 데모용 폴더
│   └── main.py                # Streamlit 또는 Gradio를 이용한 챗봇 UI
│
├── .env                       # API 키, DB 비밀번호 등 (절대 Git에 올리지 않음!)
├── .gitignore                 # 무시할 파일 목록 (data폴더, .env, __pycache__ 등)
├── requirements.txt           # 프로젝트 의존성 패키지 목록
└── README.md                  # 프로젝트 설명, 팀원 역할, 실행 방법 등
```




## 👥 팀원 (Team Members)

| 이름 | 역할 | GitHub |
| :--- | :--- | :--- |
| **채영환** | 역할1 | [@dudghks5893](https://github.com/dudghks5893) |
| **원숙현** | 역할2 | [@sooq.won@gmail.com](https://github.com/id) |
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
source ~/myenv/bin/activate


# 필수 패키지 설치
pip install -r requirements.txt

# 시나리오 선택 실행 (예시)
python main.py --scenario A
python main.py --scenario B
