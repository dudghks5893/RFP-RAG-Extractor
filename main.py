"""
Streamlit 기반 간단 RAG 챗봇 UI입니다.
실행:
streamlit run scripts/run_rag_chat_ui.py -- --config configs/baseline_rag.yaml

실행 스크립트

1. 모듈/경로 점검
   python scripts/check_project_modules.py --config configs/baseline_rag.yaml

2. 원본 파일 추출·정제·청킹
   python scripts/run_extract_chunk.py --config configs/baseline_rag.yaml

3. RAG 구축 및 평가
   python scripts/run_rag_eval.py --config configs/baseline_rag.yaml

4. Vector-DB 수동제거
    python scripts/run_rag_eval.py --config configs/baseline_rag.yaml --delete-vector-db

디스크 정리 스크립트
1. 삭제 대상만 확인
python scripts/run_disk_cleanup.py

2. 삭제 대상 삭제 진행
python scripts/run_disk_cleanup.py --apply

3. apt 캐시까지 같이 정리
python scripts/run_disk_cleanup.py --apply --apt

4. 임베딩/벡터DB까지 삭제하고 싶을 때
python scripts/run_disk_cleanup.py --apply --remove-embeddings --remove-vector-db

5. 평가 산출물까지 삭제하고 싶을 때
python scripts/run_disk_cleanup.py --apply --remove-eval-results

디스크 정리 가이드
# 1. 프로젝트 루트로 이동
cd /home/user1/RFP-RAG-Extractor

# 2. 삭제 대상 먼저 확인
python scripts/run_disk_cleanup.py

# 3. 문제 없으면 실제 삭제
python scripts/run_disk_cleanup.py --apply

# 4. 디스크 확인
df -h

"""

"""
평가 지표
1. 전체 성능 요약: *_metrics.json
reports/evaluation/<experiment_name>_sample20_metrics.json

hits@k                         # 상위 k개 검색 결과 안에 정답 문서가 포함된 비율
precision@k                    # 상위 k개 검색 결과 중 정답 문서 비율
recall@k                       # 정답 문서 중 상위 k개 검색 결과에 포함된 비율
mrr@k                          # 정답 문서가 얼마나 높은 순위에 있는지 보는 지표
retrieval_eval_count           # 검색 평가에 실제 사용된 문항 수
retrieval_eval_skipped_count   # 검색 평가에서 제외된 문항 수

avg_bleu                       # 생성 답변과 정답 문장의 n-gram 유사도 평균
avg_rougeL                     # 생성 답변과 정답 문장의 긴 공통 부분 문자열 기반 유사도 평균
avg_token_f1                   # 생성 답변과 정답의 토큰 단위 F1 평균

avg_keyword_group_recall       # 필수 키워드 그룹을 얼마나 포함했는지 평균
exact_keyword_group_match_rate # 모든 필수 키워드 그룹을 전부 맞춘 문항 비율
avg_matched_keyword_groups     # 문항당 평균으로 맞춘 키워드 그룹 수
avg_total_keyword_groups       # 문항당 평균 전체 키워드 그룹 수

avg_retrieval_latency_sec      # 질문 1개당 평균 검색 시간
avg_generation_latency_sec     # 질문 1개당 평균 LLM 답변 생성 시간
avg_total_latency_sec          # 질문 1개당 평균 전체 처리 시간
p50_total_latency_sec          # 전체 처리 시간의 중앙값
p95_total_latency_sec          # 느린 케이스 기준 상위 95% 지연 시간

avg_input_tokens               # 질문 1개당 평균 입력 토큰 수
avg_output_tokens              # 질문 1개당 평균 출력 토큰 수
avg_total_tokens               # 질문 1개당 평균 전체 토큰 수
total_tokens                   # 전체 평가에 사용된 총 토큰 수

avg_cost_per_query             # 질문 1개당 평균 API 비용
total_cost                     # 전체 평가 총 API 비용



2. 문항별 결과 종합표: *_summary.csv
qid                            # 평가 문항 ID
doc_id                         # 정답 문서 ID
question_type                  # 질문 유형, 예: fact_budget, llm_1, llm_2
source_type                    # 정답 근거 유형
answer_format                  # 기대 답변 형식, 예: short, list, summary 등
file_type                      # 원본 문서 형식, 예: hwp, pdf, docx
project_name                   # 사업명
organization                   # 발주 기관

question                       # 평가 질문
reference                      # 기준 정답 또는 참고 정답
response                       # RAG 시스템이 생성한 답변

retrieved_ids                  # 검색된 문서 doc_id 목록
retrieval_hit                  # 정답 doc_id가 top-k 검색 결과에 포함됐는지 여부
keyword_group_recall           # 해당 문항에서 필수 키워드 그룹을 맞춘 비율
matched_keyword_group_count    # 맞춘 키워드 그룹 수
total_keyword_group_count      # 전체 필수 키워드 그룹 수
missed_groups                  # 놓친 키워드 그룹 목록

retrieval_latency_sec          # 해당 문항의 검색 소요 시간
generation_latency_sec         # 해당 문항의 답변 생성 소요 시간
total_latency_sec              # 해당 문항의 전체 처리 시간
input_tokens                   # 해당 문항의 입력 토큰 수
output_tokens                  # 해당 문항의 출력 토큰 수

error                          # 해당 문항 처리 중 발생한 오류 메시지


3. 검색 실패 케이스 파일: *_retrieval_failures.csv
qid                            # 평가 문항 ID
doc_id                         # 정답 문서 ID
question_type                  # 질문 유형
question                       # 평가 질문
reference                      # 기준 정답
retrieved_ids                  # 실제 검색된 doc_id 목록
response                       # 검색 실패 상태에서 생성된 답변
retrieval_hit                  # 보통 False
rank                           # 정답 문서가 검색 결과에 있으면 해당 순위, 없으면 비어 있을 수 있음

이 파일은 주로 아래를 볼 때 씁니다.
청킹 문제인지
임베딩 모델 문제인지
top_k가 작은 문제인지
질문 표현과 문서 표현 차이 때문인지


4. 키워드 실패 케이스 파일: *_keyword_failures.csv
qid                            # 평가 문항 ID
doc_id                         # 정답 문서 ID
question_type                  # 질문 유형
question                       # 평가 질문
reference                      # 기준 정답
response                       # RAG 시스템 답변

required_keyword_groups        # 반드시 포함되어야 하는 키워드 그룹 목록
matched_groups                 # 답변에서 맞춘 키워드 그룹 목록
missed_groups                  # 답변에서 놓친 키워드 그룹 목록
keyword_group_recall           # 필수 키워드 그룹 recall
matched_keyword_group_count    # 맞춘 키워드 그룹 수
total_keyword_group_count      # 전체 키워드 그룹 수

retrieved_ids                  # 검색된 doc_id 목록
retrieval_hit                  # 정답 문서가 검색됐는지 여부

이 파일은 주로 아래를 구분할 때 봅니다.

검색은 됐는데 LLM 답변이 틀린 경우
검색부터 실패해서 답변도 틀린 경우
프롬프트가 약한 경우
청크에 정답 근거가 잘려 있는 경우


5. 실험 전체 기록 파일: *_experiment_summary.json
config                         # 해당 실험에 사용한 YAML 설정 전체
paths                          # 해당 실험에서 사용한 주요 파일 경로
metrics                        # 전체 성능 지표
num_rag_outputs                # 생성된 RAG 답변 개수
num_retrieval_failures         # 검색 실패 문항 수
num_keyword_failures           # 키워드 실패 문항 수

여러 실험을 비교할 때 가장 중요한 기록 파일입니다.

"""