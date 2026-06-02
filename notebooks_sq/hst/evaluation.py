import re
import math
import hashlib
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import pandas as pd
from tqdm.auto import tqdm

try:
    from ranx import Qrels, Run, evaluate as ranx_evaluate
    RANX_AVAILABLE = True
except Exception:
    RANX_AVAILABLE = False
    
try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    RAGAS_AVAILABLE = True
except Exception:
    RAGAS_AVAILABLE = False
    
# Basic Utility

def normalize_text(text: Any) -> str:
    text = str(text)
    text = text.lower()
    text = text.replace(",", "")
    text = re.sub(r'\s+', '', text)
    return text.strip()

def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()[:length]

def get_doc_id_from_doc(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}

    if metadata.get("doc_id") is not None:
        return str(metadata["doc_id"])

    if metadata.get("chunk_id") is not None:
        return str(metadata["chunk_id"])

    source = metadata.get("source")
    page = metadata.get("page")
    section = metadata.get("section")

    if source is not None or page is not None or section is not None:
        return stable_hash(f"{source}-{page}-{section}-{getattr(doc, 'page_content', '')[:80]}")

    return stable_hash(getattr(doc, "page_content", str(doc))[:300])

def get_page_content(doc: Any) -> str:
    """
    LangChain Document 또는 dict에서 텍스트 추출.
    """
    if hasattr(doc, "page_content"):
        return str(doc.page_content)

    if isinstance(doc, dict):
        for key in ["page_content", "text", "content", "context"]:
            if key in doc:
                return str(doc[key])

    return str(doc)

# Retrieval Metrics

def reciprocal_rank_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int = 5,
) -> float:
    """
    하나의 질문에 대한 RR@K.
    """
    gold_set = set(map(str, gold_ids))

    for rank, doc_id in enumerate(list(map(str, retrieved_ids))[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / rank

    return 0.0

def hit_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int = 5,
) -> float:
    """
    top-k 안에 정답 문서가 하나라도 있으면 1.
    """
    retrieved_top_k = set(map(str, retrieved_ids[:k]))
    gold_set = set(map(str, gold_ids))

    return 1.0 if len(retrieved_top_k & gold_set) > 0 else 0.0


def recall_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int = 5,
) -> float:
    """
    gold_ids 중 top-k 안에 들어온 비율.
    """
    gold_set = set(map(str, gold_ids))

    if not gold_set:
        return 0.0

    retrieved_top_k = set(map(str, retrieved_ids[:k]))
    return len(retrieved_top_k & gold_set) / len(gold_set)


def precision_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int = 5,
) -> float:
    """
    top-k 중 정답 문서 비율.
    """
    if k <= 0:
        return 0.0

    retrieved_top_k = list(map(str, retrieved_ids[:k]))

    if not retrieved_top_k:
        return 0.0

    gold_set = set(map(str, gold_ids))
    return sum(1 for doc_id in retrieved_top_k if doc_id in gold_set) / k


def f1_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int = 5,
) -> float:
    """
    Retrieval Precision@K와 Recall@K의 조화평균.
    """
    p = precision_at_k(retrieved_ids, gold_ids, k)
    r = recall_at_k(retrieved_ids, gold_ids, k)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def top1_hit(
    retrieved_ids: List[str],
    gold_ids: List[str],
) -> float:
    """
    1위 문서가 정답이면 1.
    """
    if not retrieved_ids:
        return 0.0

    return 1.0 if str(retrieved_ids[0]) in set(map(str, gold_ids)) else 0.0


def first_relevant_rank(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int = 5,
) -> Optional[int]:
    """
    top-k 안에서 첫 정답 문서 순위 반환.
    없으면 None.
    """
    gold_set = set(map(str, gold_ids))

    for rank, doc_id in enumerate(list(map(str, retrieved_ids))[:k], start=1):
        if doc_id in gold_set:
            return rank

    return None


def evaluate_retrieval_basic(
    eval_rows: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    기본 retrieval metric 계산.

    eval_rows 형식:
    [
        {
            "qid": "q1",
            "question": "...",
            "retrieved_ids": ["doc1", "doc2", ...],
            "gold_ids": ["doc2", ...]
        }
    ]

    return:
    - summary_df
    - detail_df
    """
    detail_rows = []

    for row in eval_rows:
        qid = row.get("qid", "")
        question = row.get("question", "")
        retrieved_ids = [str(x) for x in row.get("retrieved_ids", [])]
        gold_ids = [str(x) for x in row.get("gold_ids", [])]

        result = {
            "qid": qid,
            "question": question,
            "num_retrieved": len(retrieved_ids),
            "num_gold": len(gold_ids),
            "retrieved_ids": retrieved_ids,
            "gold_ids": gold_ids,
        }

        for k in k_values:
            result[f"hit@{k}"] = hit_at_k(retrieved_ids, gold_ids, k)
            result[f"recall@{k}"] = recall_at_k(retrieved_ids, gold_ids, k)
            result[f"precision@{k}"] = precision_at_k(retrieved_ids, gold_ids, k)
            result[f"f1@{k}"] = f1_at_k(retrieved_ids, gold_ids, k)
            result[f"mrr@{k}"] = reciprocal_rank_at_k(retrieved_ids, gold_ids, k)
            result[f"first_rank@{k}"] = first_relevant_rank(retrieved_ids, gold_ids, k)

        result["top1_hit"] = top1_hit(retrieved_ids, gold_ids)

        detail_rows.append(result)

    detail_df = pd.DataFrame(detail_rows)

    summary = {
        "num_questions": len(detail_df),
    }

    metric_cols = [
        col for col in detail_df.columns
        if any(col.startswith(prefix) for prefix in ["hit@", "recall@", "precision@", "f1@", "mrr@", "top1_hit"])
    ]

    for col in metric_cols:
        summary[col] = detail_df[col].mean() if len(detail_df) else 0.0

    summary_df = pd.DataFrame([summary])

    return summary_df, detail_df

# ranx Retrieval Evaluation

def build_ranx_qrels_run(
    eval_rows: List[Dict[str, Any]],
    score_key: str = "retrieved_scores",
) -> Tuple[Any, Any]:
    """
    eval_rows를 ranx Qrels / Run으로 변환.

    eval_rows 형식:
    [
        {
            "qid": "q1",
            "retrieved_ids": ["doc1", "doc2"],
            "retrieved_scores": [0.9, 0.7],  # optional
            "gold_ids": ["doc2"],
            "gold_scores": {"doc2": 1}       # optional
        }
    ]
    """
    if not RANX_AVAILABLE:
        raise ImportError("ranx가 설치되어 있지 않습니다. `pip install ranx` 후 다시 실행하세요.")

    qrels_dict = {}
    run_dict = {}

    for idx, row in enumerate(eval_rows):
        qid = str(row.get("qid", f"q{idx+1}"))

        gold_ids = [str(x) for x in row.get("gold_ids", [])]
        gold_scores = row.get("gold_scores", None)

        if gold_scores is not None:
            qrels_dict[qid] = {
                str(doc_id): float(score)
                for doc_id, score in gold_scores.items()
            }
        else:
            qrels_dict[qid] = {
                doc_id: 1.0
                for doc_id in gold_ids
            }

        retrieved_ids = [str(x) for x in row.get("retrieved_ids", [])]
        retrieved_scores = row.get(score_key, None)

        if retrieved_scores is None:
            # score가 없으면 rank 기반 점수 부여
            retrieved_scores = [1.0 / (rank + 1) for rank in range(len(retrieved_ids))]

        run_dict[qid] = {
            doc_id: float(score)
            for doc_id, score in zip(retrieved_ids, retrieved_scores)
        }

    return Qrels(qrels_dict), Run(run_dict)


def evaluate_retrieval_ranx(
    eval_rows: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    ranx 기반 retrieval 평가.
    """
    if not RANX_AVAILABLE:
        raise ImportError("ranx가 설치되어 있지 않습니다. `pip install ranx` 후 다시 실행하세요.")

    if metrics is None:
        metrics = [
            "hit_rate@5",
            "precision@5",
            "recall@5",
            "f1@5",
            "mrr@5",
            "ndcg@5",
            "map@5",
        ]

    qrels, run = build_ranx_qrels_run(eval_rows)
    scores = ranx_evaluate(qrels, run, metrics=metrics)

    return scores

# Keyword-based Retrieval Evaluation
def is_relevant_by_keyword_groups(
    text: str,
    keyword_groups: List[List[str]],
) -> bool:
    """
    각 keyword group에서 하나 이상 포함되면 통과.
    모든 group을 통과해야 relevant로 판단.

    예:
    [
        ["월세액"],
        ["8천만원", "8000만원"],
        ["1,000만원", "1000만원"],
        ["15%", "17%"]
    ]
    """
    text_norm = normalize_text(text)

    for group in keyword_groups:
        group_hit = any(normalize_text(keyword) in text_norm for keyword in group)

        if not group_hit:
            return False

    return True


def evaluate_retrieval_by_keywords(
    eval_rows: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    doc_id gold가 없을 때 키워드 기반으로 retrieval 평가.

    eval_rows 형식:
    [
        {
            "qid": "q1",
            "question": "...",
            "retrieved_contexts": ["chunk1", "chunk2", ...],
            "gold_keyword_groups": [
                ["월세액"],
                ["8천만원", "8000만원"]
            ]
        }
    ]
    """
    detail_rows = []

    for row in eval_rows:
        qid = row.get("qid", "")
        question = row.get("question", "")
        contexts = row.get("retrieved_contexts", [])
        keyword_groups = row.get("gold_keyword_groups", [])

        result = {
            "qid": qid,
            "question": question,
            "num_retrieved": len(contexts),
            "gold_keyword_groups": keyword_groups,
        }

        relevant_flags = [
            is_relevant_by_keyword_groups(ctx, keyword_groups)
            for ctx in contexts
        ]

        for k in k_values:
            flags_k = relevant_flags[:k]
            relevant_ranks = [
                rank for rank, is_rel in enumerate(flags_k, start=1)
                if is_rel
            ]

            result[f"hit@{k}"] = 1.0 if relevant_ranks else 0.0
            result[f"mrr@{k}"] = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
            result[f"top1_hit@{k}"] = 1.0 if relevant_ranks and relevant_ranks[0] == 1 else 0.0
            result[f"first_rank@{k}"] = relevant_ranks[0] if relevant_ranks else None

            # keyword 기반은 gold 문서 수를 정확히 모르므로 recall은 hit와 동일하게 해석 가능
            result[f"pseudo_recall@{k}"] = result[f"hit@{k}"]

        detail_rows.append(result)

    detail_df = pd.DataFrame(detail_rows)

    summary = {
        "num_questions": len(detail_df),
    }

    metric_cols = [
        col for col in detail_df.columns
        if any(col.startswith(prefix) for prefix in ["hit@", "mrr@", "top1_hit@", "pseudo_recall@"])
    ]

    for col in metric_cols:
        summary[col] = detail_df[col].mean() if len(detail_df) else 0.0

    summary_df = pd.DataFrame([summary])

    return summary_df, detail_df

# Rule-based Generation Check
def keyword_coverage(
    response: str,
    required_keyword_groups: List[List[str]],
) -> Dict[str, Any]:
    """
    생성 답변이 필수 키워드/숫자를 포함했는지 확인.

    required_keyword_groups:
    [
        ["8천만원", "8000만원"],
        ["1,000만원", "1000만원"],
        ["15%", "17%"]
    ]

    각 그룹에서 하나 이상 포함되면 통과.
    """
    response_norm = normalize_text(response)

    group_results = []

    for group in required_keyword_groups:
        hit_keywords = [
            keyword for keyword in group
            if normalize_text(keyword) in response_norm
        ]

        group_results.append({
            "group": group,
            "hit": len(hit_keywords) > 0,
            "hit_keywords": hit_keywords,
        })

    num_groups = len(group_results)
    num_hit = sum(1 for x in group_results if x["hit"])

    coverage = num_hit / num_groups if num_groups else 1.0
    all_pass = num_hit == num_groups

    return {
        "coverage": coverage,
        "all_pass": all_pass,
        "num_groups": num_groups,
        "num_hit": num_hit,
        "details": group_results,
    }


def evaluate_generation_keywords(
    eval_rows: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    생성 답변 필수 키워드 포함 여부 평가.

    eval_rows 형식:
    [
        {
            "qid": "q1",
            "question": "...",
            "response": "...",
            "required_keyword_groups": [
                ["월세액"],
                ["8천만원", "8000만원"]
            ]
        }
    ]
    """
    details = []

    for row in eval_rows:
        check = keyword_coverage(
            response=row.get("response", ""),
            required_keyword_groups=row.get("required_keyword_groups", []),
        )

        details.append({
            "qid": row.get("qid", ""),
            "question": row.get("question", ""),
            "keyword_coverage": check["coverage"],
            "keyword_all_pass": 1.0 if check["all_pass"] else 0.0,
            "num_keyword_groups": check["num_groups"],
            "num_keyword_hit": check["num_hit"],
            "details": check["details"],
        })

    detail_df = pd.DataFrame(details)

    summary_df = pd.DataFrame([{
        "num_questions": len(detail_df),
        "keyword_coverage": detail_df["keyword_coverage"].mean() if len(detail_df) else 0.0,
        "keyword_all_pass_rate": detail_df["keyword_all_pass"].mean() if len(detail_df) else 0.0,
    }])

    return summary_df, detail_df

# RAGAS Evaluation
def build_ragas_dataset(
    eval_rows: List[Dict[str, Any]],
) -> Any:
    """
    RAGAS Dataset 생성.

    eval_rows 형식:
    [
        {
            "question": "...",
            "response": "...",
            "retrieved_contexts": ["...", "..."],
            "reference": "..."
        }
    ]

    RAGAS 최신 컬럼명:
    - user_input
    - response
    - retrieved_contexts
    - reference
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("ragas 또는 datasets가 설치되어 있지 않습니다. `pip install ragas datasets` 후 다시 실행하세요.")

    rows = []

    for row in eval_rows:
        rows.append({
            "user_input": row.get("user_input", row.get("question", "")),
            "response": row.get("response", row.get("answer", "")),
            "retrieved_contexts": row.get("retrieved_contexts", row.get("contexts", [])),
            "reference": row.get("reference", row.get("ground_truth", "")),
        })

    return Dataset.from_list(rows)


def evaluate_generation_ragas(
    eval_rows: List[Dict[str, Any]],
    metrics: List[Any],
    llm: Any = None,
    embeddings: Any = None,
) -> pd.DataFrame:
    """
    RAGAS 기반 generation/context 평가.

    metrics 예:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("ragas 또는 datasets가 설치되어 있지 않습니다. `pip install ragas datasets` 후 다시 실행하세요.")

    dataset = build_ragas_dataset(eval_rows)

    kwargs = {
        "dataset": dataset,
        "metrics": metrics,
    }

    if llm is not None:
        kwargs["llm"] = llm

    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    result = ragas_evaluate(**kwargs)

    return result.to_pandas()

# RAG Output Adapter
def extract_retrieval_from_rag_result(
    rag_result: Dict[str, Any],
    retrieved_key: str = "retrieved",
    score_keys: List[str] = ["rerank_score", "fusion_score", "score"],
) -> Dict[str, Any]:
    """
    사용자의 RAG 결과를 평가 가능한 형태로 변환.

    rag_result 예상 형태:
    {
        "answer": "...",
        "retrieved": [
            {"doc": Document(...), "rerank_score": 3.2},
            ...
        ]
    }
    """
    retrieved_items = rag_result.get(retrieved_key, [])

    retrieved_ids = []
    retrieved_scores = []
    retrieved_contexts = []

    for rank, item in enumerate(retrieved_items, start=1):
        doc = item.get("doc", item) if isinstance(item, dict) else item

        doc_id = get_doc_id_from_doc(doc)
        context = get_page_content(doc)

        score = None

        if isinstance(item, dict):
            for key in score_keys:
                if item.get(key) is not None:
                    score = item[key]
                    break

        if score is None:
            score = 1.0 / rank

        retrieved_ids.append(str(doc_id))
        retrieved_scores.append(float(score))
        retrieved_contexts.append(context)

    return {
        "retrieved_ids": retrieved_ids,
        "retrieved_scores": retrieved_scores,
        "retrieved_contexts": retrieved_contexts,
    }


def run_rag_eval_items(
    eval_items: List[Dict[str, Any]],
    rag_fn: Callable[..., Dict[str, Any]],
    rag_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    eval_items의 질문을 rag_fn에 넣고 평가용 row 생성.

    eval_items 형식:
    [
        {
            "qid": "q1",
            "question": "...",
            "gold_ids": ["..."],
            "reference": "...",
            "required_keyword_groups": [...],
            "gold_keyword_groups": [...]
        }
    ]

    rag_fn:
        question을 받아 dict 반환.
        예: answer_with_rag(question, verbose=False)
    """
    rag_fn_kwargs = rag_fn_kwargs or {}
    rows = []

    for idx, item in enumerate(tqdm(eval_items, desc="Running RAG for evaluation"), start=1):
        qid = str(item.get("qid", f"q{idx}"))
        question = item["question"]

        rag_result = rag_fn(question, **rag_fn_kwargs)

        extracted = extract_retrieval_from_rag_result(rag_result)

        response = (
            rag_result.get("answer")
            or rag_result.get("response")
            or rag_result.get("result")
            or ""
        )

        row = {
            "qid": qid,
            "question": question,
            "user_input": question,
            "response": response,
            "answer": response,

            "retrieved_ids": extracted["retrieved_ids"],
            "retrieved_scores": extracted["retrieved_scores"],
            "retrieved_contexts": extracted["retrieved_contexts"],

            "gold_ids": [str(x) for x in item.get("gold_ids", [])],
            "reference": item.get("reference", item.get("ground_truth", "")),
            "ground_truth": item.get("ground_truth", item.get("reference", "")),

            "required_keyword_groups": item.get("required_keyword_groups", []),
            "gold_keyword_groups": item.get("gold_keyword_groups", []),
        }

        rows.append(row)

    return rows

# Full Evaluation
def evaluate_rag_system(
    eval_items: List[Dict[str, Any]],
    rag_fn: Callable[..., Dict[str, Any]],
    rag_fn_kwargs: Optional[Dict[str, Any]] = None,
    k_values: List[int] = [1, 3, 5, 10],
    use_ranx: bool = True,
    ranx_metrics: Optional[List[str]] = None,
    use_keyword_retrieval_eval: bool = False,
    use_generation_keyword_eval: bool = True,
    ragas_metrics: Optional[List[Any]] = None,
    ragas_llm: Any = None,
    ragas_embeddings: Any = None,
) -> Dict[str, Any]:
    """
    RAG 시스템 통합 평가 함수.

    반환:
    {
        "eval_rows": ...,
        "retrieval_basic_summary": ...,
        "retrieval_basic_detail": ...,
        "retrieval_ranx_scores": ...,
        "retrieval_keyword_summary": ...,
        "generation_keyword_summary": ...,
        "ragas_df": ...
    }
    """
    eval_rows = run_rag_eval_items(
        eval_items=eval_items,
        rag_fn=rag_fn,
        rag_fn_kwargs=rag_fn_kwargs,
    )

    results = {
        "eval_rows": eval_rows,
    }

    # 1. ID 기반 기본 retrieval 평가
    retrieval_basic_summary, retrieval_basic_detail = evaluate_retrieval_basic(
        eval_rows=eval_rows,
        k_values=k_values,
    )

    results["retrieval_basic_summary"] = retrieval_basic_summary
    results["retrieval_basic_detail"] = retrieval_basic_detail

    # 2. ranx retrieval 평가
    if use_ranx:
        try:
            ranx_scores = evaluate_retrieval_ranx(
                eval_rows=eval_rows,
                metrics=ranx_metrics,
            )
            results["retrieval_ranx_scores"] = ranx_scores
        except Exception as e:
            results["retrieval_ranx_error"] = str(e)

    # 3. keyword 기반 retrieval 평가
    if use_keyword_retrieval_eval:
        retrieval_keyword_rows = [
            row for row in eval_rows
            if row.get("gold_keyword_groups")
        ]

        if retrieval_keyword_rows:
            retrieval_keyword_summary, retrieval_keyword_detail = evaluate_retrieval_by_keywords(
                eval_rows=retrieval_keyword_rows,
                k_values=k_values,
            )
            results["retrieval_keyword_summary"] = retrieval_keyword_summary
            results["retrieval_keyword_detail"] = retrieval_keyword_detail

    # 4. 생성 답변 keyword check
    if use_generation_keyword_eval:
        generation_keyword_rows = [
            row for row in eval_rows
            if row.get("required_keyword_groups")
        ]

        if generation_keyword_rows:
            generation_keyword_summary, generation_keyword_detail = evaluate_generation_keywords(
                generation_keyword_rows
            )
            results["generation_keyword_summary"] = generation_keyword_summary
            results["generation_keyword_detail"] = generation_keyword_detail

    # 5. RAGAS 평가
    if ragas_metrics is not None:
        try:
            ragas_df = evaluate_generation_ragas(
                eval_rows=eval_rows,
                metrics=ragas_metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
            results["ragas_df"] = ragas_df
        except Exception as e:
            results["ragas_error"] = str(e)

    return results