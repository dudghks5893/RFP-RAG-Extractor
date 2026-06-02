import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_json(path: str | Path) -> Any:
    """
    JSON 파일을 로드합니다.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """
    JSON 파일을 저장합니다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sample_balanced_eval_dataset(
    data: List[Dict[str, Any]],
    sample_size: int = 20,
    random_seed: int = 42,
    target_types: Optional[List[str]] = None,
    quotas: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    평가 데이터셋에서 question_type 기준으로 균형 있게 샘플링합니다.

    기본 대상 question_type:
    - fact_budget
    - llm_1
    - llm_2

    sample_size=20일 때 기본 비율:
    - fact_budget: 5
    - llm_1: 7
    - llm_2: 8

    sample_size=30일 때 기본 비율:
    - fact_budget: 8
    - llm_1: 11
    - llm_2: 11

    Parameters
    ----------
    data:
        전체 평가 데이터셋 리스트입니다.

    sample_size:
        샘플링할 문항 수입니다.

    random_seed:
        재현 가능한 샘플링을 위한 seed입니다.

    target_types:
        샘플링 대상 question_type 목록입니다.
        None이면 ["fact_budget", "llm_1", "llm_2"]를 사용합니다.

    quotas:
        question_type별 샘플 개수를 직접 지정할 때 사용합니다.
        예: {"fact_budget": 5, "llm_1": 7, "llm_2": 8}

    Returns
    -------
    List[Dict[str, Any]]
        샘플링된 평가 데이터셋입니다.
    """
    random.seed(random_seed)

    if target_types is None:
        target_types = ["fact_budget", "llm_1", "llm_2"]

    grouped = {question_type: [] for question_type in target_types}

    for item in data:
        question_type = item.get("question_type", "unknown")

        if question_type in grouped:
            grouped[question_type].append(item)

    if quotas is None:
        if sample_size == 20:
            quotas = {
                "fact_budget": 5,
                "llm_1": 7,
                "llm_2": 8,
            }
        elif sample_size == 30:
            quotas = {
                "fact_budget": 8,
                "llm_1": 11,
                "llm_2": 11,
            }
        else:
            base_quota = sample_size // len(target_types)
            remainder = sample_size % len(target_types)

            quotas = {
                question_type: base_quota + (1 if idx < remainder else 0)
                for idx, question_type in enumerate(target_types)
            }

    sampled = []

    for question_type, quota in quotas.items():
        candidates = grouped.get(question_type, [])

        if not candidates:
            continue

        quota = min(quota, len(candidates))
        sampled.extend(random.sample(candidates, quota))

    # quota 기준으로 뽑은 수가 sample_size보다 부족하면 전체 후보에서 추가 샘플링합니다.
    if len(sampled) < sample_size:
        sampled_qids = {item.get("qid") for item in sampled}

        remaining = [
            item for item in data
            if item.get("qid") not in sampled_qids
            and item.get("question_type") in target_types
        ]

        additional_count = min(sample_size - len(sampled), len(remaining))

        if additional_count > 0:
            sampled.extend(random.sample(remaining, additional_count))

    # sample_size보다 많이 뽑혔을 경우 자릅니다.
    if len(sampled) > sample_size:
        sampled = sampled[:sample_size]

    # 결과 비교가 쉽도록 qid 기준 정렬합니다.
    sampled = sorted(sampled, key=lambda x: str(x.get("qid", "")))

    return sampled


def create_and_save_eval_sample(
    input_path: str | Path,
    output_path: str | Path,
    sample_size: int = 20,
    random_seed: int = 42,
    target_types: Optional[List[str]] = None,
    quotas: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    전체 평가 데이터셋을 로드한 뒤 균형 샘플을 생성하고 저장합니다.

    사용 예:
    sample_eval_dataset = create_and_save_eval_sample(
        input_path="data/processed/eval/eval_dataset_v6.json",
        output_path="data/processed/eval/eval_dataset_sample_20.json",
        sample_size=20,
        random_seed=42
    )
    """
    data = load_json(input_path)

    sampled = sample_balanced_eval_dataset(
        data=data,
        sample_size=sample_size,
        random_seed=random_seed,
        target_types=target_types,
        quotas=quotas,
    )

    save_json(sampled, output_path)

    return sampled