# OpenAI Multi VectorDB RAG 실행 가이드

이 문서는 기존 RFP 추출/정제/청킹 흐름은 그대로 사용하고, 마지막 RAG 구축 및 평가 단계만 OpenAI 모델과 여러 VectorDB 조합으로 실행하는 방법을 정리합니다.

## 실행방법 요약
1.open api key 입력 
 - .env를 통해서
 - read -s를 통해서

2. 필요한 import 확인 
 python -c "import torch; import faiss; import chromadb; import transformers; import langchain; import openai; print('ALL IMPORT SUCCESS')"
정상일 경우 : ALL IMPORT SUCCESS

2. 로드 확인
  - ls data/raw

3. 모듈 점검
   - python scripts/check_project_modules.py --config configs/baseline_rag.yaml
   정상일 경우 : OK 나와야 함

4. 추출, 정제, 청킹
 - python scripts/run_extract_chunk.py --config configs/baseline_rag.yaml

5. python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-mini --vector-db faiss
위와 같이 실험 하고자 하는 모델과 벡터 선택하여 실행

6. Qdrant, Supabase의 경우 별도의 설정 필요함


## 추가된 파일

- `configs/baseline_rag.yaml`
  - OpenAI embedding/LLM과 FAISS, Chroma, Qdrant, Supabase 설정을 포함합니다.
- `src/pipeline/openai_rag_eval_pipeline.py`
  - OpenAI embedding/LLM, 4개 VectorDB, 기존 `RAGEvaluator`를 연결한 평가 파이프라인입니다.
- `scripts/run_rag_eval.py`
  - OpenAI 설정이면 `OpenAIRAGEvalPipeline`을 실행합니다.
  - `--matrix` 옵션으로 8개 실험을 한 번에 실행할 수 있습니다.
- `scripts/run_extract_chunk.py`
  - 기존 2단계 추출/정제/청킹 실행 스크립트입니다.
- `scripts/check_openai_rag_modules.py`
  - OpenAI와 VectorDB 의존성, API key, 주요 경로를 점검합니다.
- `requirements-openai-rag.txt`
  - OpenAI RAG 실험에 필요한 추가 패키지 목록입니다.
- 'src/pipeline/__init__.py'
  - from src.pipeline.openai_rag_eval_pipeline import OpenAIRAGEvalPipeline 추가 하였습니다.

## 0. 설치

프로젝트 루트에서 실행합니다.

```bash
pip install -r requirements-openai-rag.txt
```


## 1. OpenAI API Key 설정

# 간단한 방법(대신 터미널 끝나면 사라짐)
주피터랩에서 다음과 같이 실행하시면 됩니다.
1단계 
read -s OPENAI_API_KEY
실행 후 엔터

2단계
커서만 깜빡이고 아무것도 안 보일 것입니다.  
여기에 sk-xxxx(open api key) 입력
입력해도 화면에 안 보이는 게 정상임.

3단계
export OPENAI_API_KEY 
실행

echo ${OPENAI_API_KEY:0:10} : 입력되는 확인하는 명령어
sk-proj-... 처럼 나와야 정상

# .env를 이용한 방식
터미널에 입력
cat .env

OPENAI_API_KEY 있는지 확인
grep "OPENAI_API_KEY" .env

없을 경우
nano .env

맨 아래에 추가:
OPENAI_API_KEY=sk-xxxx

저장:
Ctrl + O
Enter
Ctrl + X

gitignore 확인
grep ".env" .gitignore
echo ".env" >> .gitignore(없으면 추가)

## 3. 모듈 및 경로 점검

기존 프로젝트 전체 점검:

```bash
python scripts/check_project_modules.py --config configs/baseline_rag.yaml
```

OpenAI RAG 의존성 점검:

```bash
python scripts/check_openai_rag_modules.py --config configs/baseline_rag.yaml --vector-db faiss
python scripts/check_openai_rag_modules.py --config configs/baseline_rag.yaml --vector-db chroma
python scripts/check_openai_rag_modules.py --config configs/baseline_rag.yaml --vector-db qdrant
python scripts/check_openai_rag_modules.py --config configs/baseline_rag.yaml --vector-db supabase
```

## 4. 원본 파일 추출, 정제, 청킹

기존 1, 2단계 경로와 방식을 그대로 사용합니다.

```bash
python scripts/run_extract_chunk.py --config configs/baseline_rag.yaml
```

생성되는 주요 파일:

```text
data/chunks/section/section_chunks.jsonl
logs/extract_clean_chunk_log.csv
```

## 5. 단일 OpenAI RAG 실험 실행

각 실험은 `evaluation.sample_size: 20` 설정에 따라 기본적으로 20개 평가 문항만 실행합니다.

```bash
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-mini --vector-db qdrant
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-mini --vector-db supabase
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-mini --vector-db faiss
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-mini --vector-db chroma

python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-nano --vector-db qdrant
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-nano --vector-db supabase
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-nano --vector-db faiss
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-nano --vector-db chroma
```

## 6. 8개 실험 한 번에 실행

```bash
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --matrix
```

실행 조합:

- `gpt-5-mini + Qdrant`
- `gpt-5-mini + Supabase`
- `gpt-5-mini + FAISS`
- `gpt-5-mini + Chroma`
- `gpt-5-nano + Qdrant`
- `gpt-5-nano + Supabase`
- `gpt-5-nano + FAISS`
- `gpt-5-nano + Chroma`

## 7. 평가 결과 파일

기존 `RAGEvaluator`를 그대로 사용하므로 기존 평가 지표와 결과 파일 구조를 유지합니다.

예시 경로:

```text
RFP-RAG-Extractor/reports/evaluation/baseline_multi_rag/baseline_multi_rag_sample20_metrics.json
RFP-RAG-Extractor/reports/evaluation/baseline_multi_rag/baseline_multi_rag_sample20_by_question_type.json
RFP-RAG-Extractor/reports/evaluation/baseline_multi_rag/baseline_multi_rag_sample20_by_source_type.json
RFP-RAG-Extractor/reports/evaluation/baseline_multi_rag/baseline_multi_rag_sample20_by_answer_format.json
RFP-RAG-Extractor/reports/evaluation/baseline_multi_rag/baseline_multi_rag_sample20_retrieval_failures.csv
```


RAG 원본 출력:

```text
data/processed/eval/<experiment_name>_rag_outputs.json
data/processed/eval/<experiment_name>_rag_outputs_scored.json
```

## 8. VectorDB별 준비 사항

### FAISS

로컬 파일 기반이라 별도 서버가 필요 없습니다.

```bash
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-mini --vector-db faiss
```

### Chroma

로컬 persistent Chroma DB를 사용합니다. 별도 서버는 필요 없습니다.

```bash
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-mini --vector-db chroma
```

### Qdrant

`configs/baseline_rag.yaml` 기본값은 다음 주소를 사용합니다.

```yaml
url: http://localhost:6333
```

따라서 Qdrant 서버가 먼저 실행되어 있어야 합니다.

```bash
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-mini --vector-db qdrant
```

### Supabase

Supabase pgvector 테이블과 match RPC가 준비되어 있어야 합니다. 환경변수도 필요합니다.

```bash
export SUPABASE_URL="https://..."
export SUPABASE_SERVICE_ROLE_KEY="..."
```

Windows PowerShell:

```powershell
$env:SUPABASE_URL="https://..."
$env:SUPABASE_SERVICE_ROLE_KEY="..."
```

실행:

```bash
python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --llm-model gpt-5-mini --vector-db supabase
```

## 9. 평가 개수 변경

기본은 20개입니다.

```yaml
evaluation:
  sample_size: 20
```

더 많이 평가하려면 `configs/baseline_rag.yaml`에서 `sample_size`를 변경하고, 기존 샘플 파일을 삭제하거나 새 샘플 경로를 지정합니다.

```text
data/processed/eval/eval_dataset_sample_20.json