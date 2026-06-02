import os
import re
import json
import ssl
import hashlib
import collections
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from ranx import Qrels, Run, evaluate as ranx_evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ---------------------------------------------------------
# SSL 인증서 검증 우회
# ---------------------------------------------------------
# 일부 개발 환경에서 nltk.download()가 SSL 인증서 문제로 실패하는 경우가 있습니다.
# 해당 문제를 우회하기 위한 설정입니다.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class RAGEvaluator:
    """
    RAG 시스템 평가를 위한 통합 평가지표 모듈입니다.

    eval_dataset 필드:
    - qid
    - doc_id
    - question
    - reference
    - required_keyword_groups
    - question_type
    - source_type
    - answer_format
    - project_name
    - organization
    - file_name
    - file_type

    RAG 실행 결과 eval_rows에 추가되어야 하는 필드:
    - retrieved_ids
    - response

    선택적으로 추가되면 효율성 평가에 사용되는 필드:
    - retrieval_latency_sec
    - generation_latency_sec
    - total_latency_sec
    - input_tokens
    - output_tokens
    - total_tokens
    - estimated_cost

    주요 평가 지표:
    1. Retrieval
       - hits@K
       - precision@K
       - recall@K
       - mrr@K

    2. Generation
       - BLEU
       - ROUGE-L
       - Token F1

    3. Keyword Group
       - avg_keyword_group_recall
       - exact_keyword_group_match_rate

    4. Efficiency
       - avg_total_latency_sec
       - p95_total_latency_sec
       - avg_cost_per_query
       - total_cost
    """

    def __init__(
        self,
        auto_download_nltk: bool = True,
        use_nltk_tokenizer: bool = False
    ):
        """
        Parameters
        ----------
        auto_download_nltk:
            True이면 NLTK 리소스가 없을 때 자동 다운로드합니다.

        use_nltk_tokenizer:
            True이면 BLEU 계산 시 nltk.word_tokenize를 사용합니다.
            False이면 단순 split() 기반 토큰화를 사용합니다.

            한국어 RFP 평가에서는 BLEU가 핵심 지표가 아니므로,
            NLTK 의존도를 줄이려면 False를 권장합니다.
        """
        self.use_nltk_tokenizer = use_nltk_tokenizer

        if auto_download_nltk and self.use_nltk_tokenizer:
            self._ensure_nltk_resource("tokenizers/punkt", "punkt")

            # 일부 NLTK 버전/환경에서 punkt_tab이 필요할 수 있습니다.
            try:
                self._ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
            except Exception:
                pass

        self.smoothie = SmoothingFunction().method1
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # ---------------------------------------------------------
    # NLTK 리소스 관리
    # ---------------------------------------------------------
    def _ensure_nltk_resource(self, resource_name: str, download_name: str):
        """
        NLTK 리소스가 이미 설치되어 있는지 확인하고,
        없을 때만 다운로드합니다.

        매번 nltk.download()를 호출하는 것을 방지하기 위한 함수입니다.
        """
        try:
            nltk.data.find(resource_name)
        except LookupError:
            nltk.download(download_name, quiet=True)

    # ---------------------------------------------------------
    # 고유 ID 추출 헬퍼
    # ---------------------------------------------------------
    def get_doc_id(self, doc: Any) -> str:
        """
        LangChain Document 객체나 dict에서 doc_id를 안전하게 추출합니다.

        Retrieval 평가는 chunk_id가 아니라 doc_id 기준으로 수행합니다.
        따라서 검색된 청크에서 반드시 doc_id를 추출해 retrieved_ids에 넣어야 합니다.

        우선순위:
        1. dict["doc_id"]
        2. dict["metadata"]["doc_id"]
        3. 객체.metadata["doc_id"]
        4. dict["chunk_id"]
        5. 객체.metadata["chunk_id"]
        6. 본문 기반 hash
        """
        # dict 형태 처리
        if isinstance(doc, dict):
            if doc.get("doc_id") is not None:
                return str(doc["doc_id"])

            metadata = doc.get("metadata", {}) or {}
            if metadata.get("doc_id") is not None:
                return str(metadata["doc_id"])

            if doc.get("chunk_id") is not None:
                return str(doc["chunk_id"])

            if metadata.get("chunk_id") is not None:
                return str(metadata["chunk_id"])

            content = doc.get("text") or doc.get("page_content") or str(doc)
            return hashlib.md5(str(content).encode("utf-8")).hexdigest()[:16]

        # LangChain Document 등 객체 형태 처리
        metadata = getattr(doc, "metadata", {}) or {}

        if metadata.get("doc_id") is not None:
            return str(metadata["doc_id"])

        if metadata.get("chunk_id") is not None:
            return str(metadata["chunk_id"])

        content = getattr(doc, "page_content", str(doc))
        return hashlib.md5(str(content).encode("utf-8")).hexdigest()[:16]

    # ---------------------------------------------------------
    # 평가용 정규화
    # ---------------------------------------------------------
    def normalize_for_eval(self, text: Any) -> str:
        """
        키워드 매칭 평가를 위한 정규화 함수입니다.

        목적:
        - RAG 답변과 required_keyword_groups를 유연하게 비교하기 위함입니다.

        처리 내용:
        - Unicode NFKC 정규화
        - 소문자화
        - 공백 제거
        - 쉼표 제거
        - 하이픈/대시류 제거
        - 전화번호 비교를 위한 '-' 제거
        - 한국어 문서에서 자주 나오는 중점/구분자 제거
        - 괄호/따옴표 제거

        예:
        - "130,000,000원" -> "130000000원"
        - "063-716-2787" -> "0637162787"
        - "제한 경쟁 입찰" -> "제한경쟁입찰"
        - "개인정보·처리" -> "개인정보처리"
        """
        if text is None:
            return ""

        text = str(text)
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()

        # 공백류 제거
        text = re.sub(r"\s+", "", text)

        # 금액/숫자 비교를 위해 쉼표 제거
        text = text.replace(",", "")

        # 하이픈/대시류 제거
        for ch in ["-", "–", "—", "−", "‐", "-", "‒"]:
            text = text.replace(ch, "")

        # 한국어 문서에서 자주 나오는 구분자 제거
        for ch in ["·", "ㆍ", "･", "・", "/", "\\", ":", "：", ";", "；", "~", "～", "|"]:
            text = text.replace(ch, "")

        # 괄호류 제거
        text = re.sub(r"[(){}\[\]〈〉<>「」『』《》]", "", text)

        # 따옴표류 제거
        text = re.sub(r"[\"'“”‘’`]", "", text)

        return text.strip()

    # ---------------------------------------------------------
    # 토큰화
    # ---------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        """
        BLEU 및 Token F1 계산용 토큰화 함수입니다.

        use_nltk_tokenizer=True:
            nltk.word_tokenize 사용

        use_nltk_tokenizer=False:
            단순 split() 사용

        한국어 평가에서는 BLEU/F1은 보조 지표이므로,
        NLTK 의존도를 줄이고 싶다면 split() 기반을 권장합니다.
        """
        text = str(text or "")

        if self.use_nltk_tokenizer:
            return nltk.word_tokenize(text)

        return text.split()

    # ---------------------------------------------------------
    # 정답 doc_id 추출
    # ---------------------------------------------------------
    def get_gold_doc_ids(self, row: Dict[str, Any]) -> List[str]:
        """
        정답 문서 ID 목록을 추출합니다.

        eval_dataset_v6는 doc_id만 사용합니다.
        단, 과거 gold_ids가 있는 데이터도 호환되도록 처리합니다.
        """
        if row.get("gold_ids"):
            gold_ids = row.get("gold_ids", [])

            if isinstance(gold_ids, list):
                return [
                    str(x).strip()
                    for x in gold_ids
                    if x is not None and str(x).strip()
                ]

            return [str(gold_ids).strip()]

        doc_id = row.get("doc_id")

        if doc_id is None or not str(doc_id).strip():
            return []

        return [str(doc_id).strip()]

    # ---------------------------------------------------------
    # 1. Retrieval 평가
    # ---------------------------------------------------------
    def evaluate_retrieval(
        self,
        eval_rows: List[Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Retrieval 검색 성능을 평가합니다.

        평가 지표:
        - hits@K:
            Top-K 안에 정답 문서가 1개라도 있는지 평가합니다.

        - precision@K:
            검색된 K개 문서 중 정답 문서의 비율입니다.

        - recall@K:
            실제 정답 문서 중 K개 안에 찾아온 문서의 비율입니다.

        - mrr@K:
            첫 번째 정답 문서가 몇 번째 순위에 등장했는지 평가합니다.

        eval_dataset_v6 기준:
        - 정답 문서: row["doc_id"]
        - 검색 결과: row["retrieved_ids"]

        주의:
        retrieved_ids에는 chunk_id가 아니라 doc_id가 들어가야 합니다.

        예:
        retrieved_ids = [
            "20241001798",
            "20241001798",
            "20241002912"
        ]
        """
        qrels_dict = {}
        run_dict = {}
        skipped_count = 0

        for idx, row in enumerate(eval_rows):
            qid = str(row.get("qid", f"q{idx}"))

            gold_doc_ids = self.get_gold_doc_ids(row)

            # doc_id가 없는 row는 retrieval 평가에서 제외합니다.
            if not gold_doc_ids:
                skipped_count += 1
                continue

            retrieved_ids = row.get("retrieved_ids", []) or []

            qrels_dict[qid] = {
                str(doc_id): 1.0
                for doc_id in gold_doc_ids
                if doc_id is not None and str(doc_id).strip()
            }

            # 같은 doc_id가 여러 번 검색될 수 있으므로 첫 등장 순위만 유지합니다.
            ranked_scores = {}

            for rank, doc_id in enumerate(retrieved_ids):
                if doc_id is None:
                    continue

                doc_id = str(doc_id).strip()

                if not doc_id:
                    continue

                if doc_id not in ranked_scores:
                    ranked_scores[doc_id] = 1.0 / (rank + 1)

            run_dict[qid] = ranked_scores

        metrics = [
            f"hits@{k}",
            f"precision@{k}",
            f"recall@{k}",
            f"mrr@{k}"
        ]

        if not qrels_dict:
            result = {metric: 0.0 for metric in metrics}
        else:
            result = ranx_evaluate(
                Qrels(qrels_dict),
                Run(run_dict),
                metrics=metrics
            )

        result["retrieval_eval_count"] = len(qrels_dict)
        result["retrieval_eval_skipped_count"] = skipped_count

        return result

    # ---------------------------------------------------------
    # 2. Generation 평가
    # ---------------------------------------------------------
    def evaluate_generation(
        self,
        eval_rows: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        RAG 생성 답변과 reference를 비교해 생성 품질을 평가합니다.

        평가 지표:
        - BLEU:
            생성 답변이 reference와 문장 구조/단어 순서 측면에서 얼마나 유사한지 봅니다.

        - ROUGE-L:
            reference의 핵심 문맥을 얼마나 포함했는지 봅니다.

        - Token F1:
            생성 답변과 reference의 토큰 교집합 기반 조화평균입니다.

        주의:
        한국어 RFP 답변에서는 BLEU/ROUGE/F1은 보조 지표입니다.
        실제 핵심 평가는 required_keyword_groups 기반 keyword recall입니다.
        """
        bleu_scores = []
        rouge_scores = []
        f1_scores = []

        for row in eval_rows:
            pred = str(row.get("response", "") or "")
            gold = str(row.get("reference", "") or "")

            if gold and pred:
                ref_tokens = [self.tokenize(gold)]
                pred_tokens = self.tokenize(pred)

                bleu = sentence_bleu(
                    ref_tokens,
                    pred_tokens,
                    smoothing_function=self.smoothie
                )

                rouge = self.scorer.score(gold, pred)["rougeL"].fmeasure
            else:
                bleu = 0.0
                rouge = 0.0

            bleu_scores.append(bleu)
            rouge_scores.append(rouge)

            gold_tokens = self.tokenize(gold)
            pred_tokens = self.tokenize(pred)

            common = sum(
                (
                    collections.Counter(gold_tokens)
                    & collections.Counter(pred_tokens)
                ).values()
            )

            if len(gold_tokens) == 0 or len(pred_tokens) == 0:
                f1_scores.append(1.0 if gold == pred else 0.0)
            elif common == 0:
                f1_scores.append(0.0)
            else:
                precision = common / len(pred_tokens)
                recall = common / len(gold_tokens)
                f1_scores.append((2 * precision * recall) / (precision + recall))

        return {
            "avg_bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            "avg_rougeL": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
            "avg_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        }

    # ---------------------------------------------------------
    # 3. Required Keyword Group Recall 평가
    # ---------------------------------------------------------
    def evaluate_keyword_group_recall(
        self,
        eval_rows: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        required_keyword_groups를 기준으로 RAG 답변이 핵심 정답 요소를
        얼마나 포함했는지 평가합니다.

        평가 규칙:
        - 그룹 내부는 OR 조건입니다.
          예:
          ["130,000,000원", "130000000원", "1억3천만원"]
          중 하나만 맞으면 해당 그룹은 통과입니다.

        - 그룹 간은 coverage 방식입니다.
          예:
          총 3개 그룹 중 2개 그룹 통과 → 2 / 3 = 0.667

        반환 지표:
        - avg_keyword_group_recall:
            전체 문항 평균 keyword group recall

        - exact_keyword_group_match_rate:
            모든 keyword group을 맞춘 문항 비율

        - avg_matched_keyword_groups:
            문항당 평균 맞춘 keyword group 개수

        - avg_total_keyword_groups:
            문항당 평균 전체 keyword group 개수
        """
        recalls = []
        exact_matches = []
        matched_counts = []
        total_counts = []

        for row in eval_rows:
            pred = row.get("response", "") or ""
            required_groups = row.get("required_keyword_groups", []) or []

            normalized_pred = self.normalize_for_eval(pred)

            if not required_groups:
                recalls.append(0.0)
                exact_matches.append(0.0)
                matched_counts.append(0)
                total_counts.append(0)
                continue

            matched = 0
            total = len(required_groups)

            for group in required_groups:
                if not isinstance(group, list):
                    group = [group]

                group_hit = False

                for keyword in group:
                    normalized_keyword = self.normalize_for_eval(keyword)

                    if not normalized_keyword:
                        continue

                    if normalized_keyword in normalized_pred:
                        group_hit = True
                        break

                if group_hit:
                    matched += 1

            recall = matched / total if total else 0.0

            recalls.append(recall)
            exact_matches.append(1.0 if matched == total and total > 0 else 0.0)
            matched_counts.append(matched)
            total_counts.append(total)

        return {
            "avg_keyword_group_recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "exact_keyword_group_match_rate": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
            "avg_matched_keyword_groups": sum(matched_counts) / len(matched_counts) if matched_counts else 0.0,
            "avg_total_keyword_groups": sum(total_counts) / len(total_counts) if total_counts else 0.0
        }

    # ---------------------------------------------------------
    # 4. Row별 Keyword Score 부착
    # ---------------------------------------------------------
    def attach_keyword_scores(
        self,
        eval_rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        각 row에 keyword group 평가 결과를 추가합니다.

        추가되는 필드:
        - keyword_group_recall
        - matched_keyword_group_count
        - total_keyword_group_count
        - matched_groups
        - missed_groups

        실패 케이스 분석이나 CSV 저장 전에 사용하면 좋습니다.
        """
        scored_rows = []

        for row in eval_rows:
            pred = row.get("response", "") or ""
            required_groups = row.get("required_keyword_groups", []) or []
            normalized_pred = self.normalize_for_eval(pred)

            matched_groups = []
            missed_groups = []

            for group in required_groups:
                if not isinstance(group, list):
                    group = [group]

                group_hit = False

                for keyword in group:
                    normalized_keyword = self.normalize_for_eval(keyword)

                    if not normalized_keyword:
                        continue

                    if normalized_keyword in normalized_pred:
                        group_hit = True
                        break

                if group_hit:
                    matched_groups.append(group)
                else:
                    missed_groups.append(group)

            total = len(required_groups)
            matched = len(matched_groups)
            recall = matched / total if total else 0.0

            new_row = dict(row)
            new_row["keyword_group_recall"] = recall
            new_row["matched_keyword_group_count"] = matched
            new_row["total_keyword_group_count"] = total
            new_row["matched_groups"] = matched_groups
            new_row["missed_groups"] = missed_groups

            scored_rows.append(new_row)

        return scored_rows

    # ---------------------------------------------------------
    # 5. 효율성 평가: 속도, 토큰, 비용
    # ---------------------------------------------------------
    def evaluate_efficiency(
        self,
        eval_rows: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        RAG 시스템의 효율성을 평가합니다.

        평가 지표:
        - avg_retrieval_latency_sec:
            검색 평균 소요 시간

        - avg_generation_latency_sec:
            답변 생성 평균 소요 시간

        - avg_total_latency_sec:
            전체 RAG 평균 응답 시간

        - p50_total_latency_sec:
            전체 응답 시간 중앙값

        - p95_total_latency_sec:
            느린 요청 기준 95퍼센타일 응답 시간

        - avg_input_tokens:
            질문당 평균 입력 토큰 수

        - avg_output_tokens:
            질문당 평균 출력 토큰 수

        - avg_total_tokens:
            질문당 평균 전체 토큰 수

        - total_tokens:
            전체 평가에 사용된 총 토큰 수

        - avg_cost_per_query:
            질문당 평균 비용

        - total_cost:
            전체 평가 총 비용

        주의:
        이 함수가 의미 있으려면 RAG 실행 결과 row에 latency/token/cost 정보가
        저장되어 있어야 합니다.
        """
        def get_values(key: str) -> List[float]:
            values = []

            for row in eval_rows:
                value = row.get(key)

                if value is None or value == "":
                    continue

                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue

            return values

        def avg(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def percentile(values: List[float], p: float) -> float:
            if not values:
                return 0.0

            values = sorted(values)
            idx = int(round((len(values) - 1) * p))
            return values[idx]

        retrieval_latencies = get_values("retrieval_latency_sec")
        generation_latencies = get_values("generation_latency_sec")
        total_latencies = get_values("total_latency_sec")

        input_tokens = get_values("input_tokens")
        output_tokens = get_values("output_tokens")
        total_tokens = get_values("total_tokens")
        costs = get_values("estimated_cost")

        return {
            "avg_retrieval_latency_sec": avg(retrieval_latencies),
            "avg_generation_latency_sec": avg(generation_latencies),
            "avg_total_latency_sec": avg(total_latencies),
            "p50_total_latency_sec": percentile(total_latencies, 0.50),
            "p95_total_latency_sec": percentile(total_latencies, 0.95),

            "avg_input_tokens": avg(input_tokens),
            "avg_output_tokens": avg(output_tokens),
            "avg_total_tokens": avg(total_tokens),
            "total_tokens": sum(total_tokens),

            "avg_cost_per_query": avg(costs),
            "total_cost": sum(costs)
        }

    # ---------------------------------------------------------
    # 6. 비용 계산
    # ---------------------------------------------------------
    def estimate_llm_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        input_price_per_1m: float,
        output_price_per_1m: float
    ) -> float:
        """
        입력/출력 토큰 수와 모델 단가를 기준으로 예상 비용을 계산합니다.

        Parameters
        ----------
        input_tokens:
            입력 토큰 수

        output_tokens:
            출력 토큰 수

        input_price_per_1m:
            입력 100만 토큰당 가격입니다.
            단위는 USD 기준으로 두는 것을 추천합니다.

        output_price_per_1m:
            출력 100만 토큰당 가격입니다.
            단위는 USD 기준으로 두는 것을 추천합니다.

        Returns
        -------
        float:
            예상 비용
        """
        input_cost = (input_tokens / 1_000_000) * input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * output_price_per_1m

        return input_cost + output_cost

    def attach_costs(
        self,
        eval_rows: List[Dict[str, Any]],
        input_price_per_1m: float,
        output_price_per_1m: float
    ) -> List[Dict[str, Any]]:
        """
        input_tokens/output_tokens가 있는 row에 total_tokens와 estimated_cost를 추가합니다.

        이미 estimated_cost가 있는 경우에는 기존 값을 유지합니다.
        """
        updated_rows = []

        for row in eval_rows:
            new_row = dict(row)

            input_tokens = int(new_row.get("input_tokens") or 0)
            output_tokens = int(new_row.get("output_tokens") or 0)

            if not new_row.get("total_tokens"):
                new_row["total_tokens"] = input_tokens + output_tokens

            if not new_row.get("estimated_cost"):
                new_row["estimated_cost"] = self.estimate_llm_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    input_price_per_1m=input_price_per_1m,
                    output_price_per_1m=output_price_per_1m
                )

            updated_rows.append(new_row)

        return updated_rows

    # ---------------------------------------------------------
    # 7. 통합 평가
    # ---------------------------------------------------------
    def evaluate_all(
        self,
        eval_rows: List[Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Retrieval, Generation, Keyword Group, Efficiency 평가를 한 번에 수행합니다.
        """
        retrieval_metrics = self.evaluate_retrieval(eval_rows, k=k)
        generation_metrics = self.evaluate_generation(eval_rows)
        keyword_metrics = self.evaluate_keyword_group_recall(eval_rows)
        efficiency_metrics = self.evaluate_efficiency(eval_rows)

        return {
            **retrieval_metrics,
            **generation_metrics,
            **keyword_metrics,
            **efficiency_metrics
        }

    # ---------------------------------------------------------
    # 8. 그룹별 평가
    # ---------------------------------------------------------
    def evaluate_by_group(
        self,
        eval_rows: List[Dict[str, Any]],
        group_key: str = "question_type",
        k: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        특정 필드 기준으로 그룹별 평가를 수행합니다.

        예:
        - question_type별 평가
        - source_type별 평가
        - answer_format별 평가
        - file_type별 평가

        결과에는 num_rows도 함께 포함됩니다.
        """
        grouped = {}

        for row in eval_rows:
            group_value = str(row.get(group_key, "unknown"))
            grouped.setdefault(group_value, []).append(row)

        results = {}

        for group_value, rows in grouped.items():
            group_metrics = self.evaluate_all(rows, k=k)
            group_metrics["num_rows"] = len(rows)
            results[group_value] = group_metrics

        return results

    # ---------------------------------------------------------
    # 9. Retrieval 실패 케이스 추출
    # ---------------------------------------------------------
    def get_retrieval_failure_cases(
        self,
        eval_rows: List[Dict[str, Any]],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Top-K 검색 결과 안에 정답 doc_id가 없는 문항을 추출합니다.

        이 함수는 청킹 전략, 임베딩 모델, retriever 품질을 분석할 때 중요합니다.

        검색 실패:
        - 정답 문서가 Top-K 검색 결과에 없음
        - 원인 후보: 청킹 문제, 임베딩 문제, 검색 방식 문제

        검색 성공 + 답변 실패:
        - 검색은 맞았지만 답변이 틀림
        - 원인 후보: 프롬프트 문제, LLM 문제, context 압축 문제
        """
        failures = []

        for row in eval_rows:
            gold_doc_ids = set(self.get_gold_doc_ids(row))

            retrieved_ids = [
                str(doc_id).strip()
                for doc_id in (row.get("retrieved_ids", []) or [])[:k]
                if doc_id is not None and str(doc_id).strip()
            ]

            hit = any(doc_id in gold_doc_ids for doc_id in retrieved_ids)

            if not hit:
                failures.append({
                    "qid": row.get("qid"),
                    "doc_id": row.get("doc_id"),
                    "question_type": row.get("question_type"),
                    "source_type": row.get("source_type"),
                    "answer_format": row.get("answer_format"),
                    "project_name": row.get("project_name"),
                    "organization": row.get("organization"),
                    "question": row.get("question"),
                    "reference": row.get("reference"),
                    "retrieved_ids": retrieved_ids,
                    "response": row.get("response")
                })

        return failures

    # ---------------------------------------------------------
    # 10. Keyword 실패 케이스 추출
    # ---------------------------------------------------------
    def get_keyword_failure_cases(
        self,
        eval_rows: List[Dict[str, Any]],
        threshold: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        keyword_group_recall이 threshold 미만인 문항을 추출합니다.

        기본값 threshold=1.0은 required keyword group을 하나라도 놓친 문항을
        모두 실패 케이스로 반환합니다.
        """
        scored_rows = self.attach_keyword_scores(eval_rows)
        failures = []

        for row in scored_rows:
            recall = row.get("keyword_group_recall", 0.0)

            if recall < threshold:
                failures.append({
                    "qid": row.get("qid"),
                    "doc_id": row.get("doc_id"),
                    "question_type": row.get("question_type"),
                    "source_type": row.get("source_type"),
                    "answer_format": row.get("answer_format"),
                    "project_name": row.get("project_name"),
                    "organization": row.get("organization"),
                    "question": row.get("question"),
                    "reference": row.get("reference"),
                    "response": row.get("response"),
                    "keyword_group_recall": recall,
                    "matched_keyword_group_count": row.get("matched_keyword_group_count"),
                    "total_keyword_group_count": row.get("total_keyword_group_count"),
                    "matched_groups": row.get("matched_groups"),
                    "missed_groups": row.get("missed_groups")
                })

        return failures

    # ---------------------------------------------------------
    # 11. 평가 결과 저장
    # ---------------------------------------------------------
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        output_path: str
    ):
        """
        평가 지표 dict를 JSON 파일로 저장합니다.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"평가 결과 저장 완료: {output_path}")

    def save_rows_as_json(
        self,
        rows: List[Dict[str, Any]],
        output_path: str
    ):
        """
        eval_rows 또는 scoring이 부착된 row 목록을 JSON 파일로 저장합니다.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        print(f"JSON 저장 완료: {output_path} / {len(rows)}건")

    def save_rows_as_csv(
        self,
        rows: List[Dict[str, Any]],
        output_path: str
    ):
        """
        row 목록을 CSV 파일로 저장합니다.
        리스트/딕셔너리 값은 JSON 문자열로 변환해 저장합니다.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        normalized_rows = []

        for row in rows:
            new_row = {}

            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    new_row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    new_row[key] = value

            normalized_rows.append(new_row)

        df = pd.DataFrame(normalized_rows)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"CSV 저장 완료: {output_path} / {len(df)}건")

    # ---------------------------------------------------------
    # 12. LLM-as-a-Judge 뼈대
    # ---------------------------------------------------------
    def evaluate_llm_as_a_judge(
        self,
        question: str,
        response: str,
        reference: str
    ) -> str:
        """
        LLM-as-a-Judge 평가 함수의 뼈대입니다.

        목적:
        BLEU/ROUGE/F1이나 keyword matching으로 판단하기 어려운 의미적 정답 여부를
        LLM이 직접 평가하게 하기 위한 함수입니다.

        현재는 비용 절약을 위해 구현하지 않았습니다.
        추후 OpenAI API 또는 로컬 LLM Judge를 연결하면 됩니다.
        """
        return "LLM-as-a-Judge는 추후 API 또는 로컬 Judge 모델 연결 시 구현 예정입니다."

    # ---------------------------------------------------------
    # 13. 팀원 평가 로그 저장
    # ---------------------------------------------------------
    def log_for_human_eval(
        self,
        question: str,
        rag_result: Dict[str, Any],
        output_csv: str = "real_user_eval_sheet.csv"
    ):
        """
        실제 사용자의 질문과 RAG 답변을 팀원 평가용 CSV에 누적 저장합니다.

        자동 평가셋이 아닌 실제 사용 질의에 대한 수동 평가용입니다.
        """

        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        contexts = rag_result.get("retrieved_contexts", ["(참고 문서 없음)"])
        context_text = "\n\n".join(contexts) if isinstance(contexts, list) else str(contexts)

        new_row = {
            "1. 실제 사용자 질문 (Question)": question,
            "2. RAG 참고 문서 (Context)": context_text,
            "3. RAG 최종 답변 (Response)": rag_result.get("response", ""),

            "질문 적합성 점수 (1~5)": "",
            "핵심 정보 포함 점수 (1~5)": "",
            "근거 일치성 점수 (1~5)": "",
            "환각 여부 (Y/N)": "",
            "실무 활용 가능성 점수 (1~5)": "",
            "코멘트": ""
        }

        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        print(
            f"새로운 사용자 질문이 '{output_csv}' 파일에 추가되었습니다. "
            f"현재 총 {len(df)}건 누적."
        )