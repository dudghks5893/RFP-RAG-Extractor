# src/utils/config_utils.py
#
# YAML config 로드, 프로젝트 경로 해석, config snapshot 저장/비교를 담당하는 유틸 모듈입니다.
#
# 사용 목적:
# - configs/baseline_rag.yaml을 읽어서 pipeline에서 사용
# - YAML 안의 상대 경로를 PROJECT_ROOT 기준 절대 경로로 변환
# - FAISS 인덱스 생성 당시의 config를 저장
# - 현재 config와 저장된 config가 호환되는지 확인

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    YAML config 파일을 로드합니다.

    Parameters
    ----------
    config_path:
        YAML config 파일 경로입니다.

    Returns
    -------
    Dict[str, Any]
        YAML 내용을 Python dict로 변환한 객체입니다.

    Raises
    ------
    FileNotFoundError
        config 파일이 존재하지 않는 경우 발생합니다.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 빈 YAML 파일이면 None이 반환될 수 있으므로 dict로 보정합니다.
    if config is None:
        config = {}

    return config


def resolve_project_path(project_root: str | Path, path_value: str | Path) -> Path:
    """
    config 안의 경로 값을 프로젝트 루트 기준 절대 경로로 변환합니다.

    YAML에는 보통 아래처럼 상대 경로를 저장합니다.

    예:
    data/chunks/section/section_chunks.jsonl

    이 함수는 위 경로를 PROJECT_ROOT 기준 절대 경로로 바꿉니다.

    Parameters
    ----------
    project_root:
        프로젝트 루트 경로입니다.
        예: /home/user1/RFP-RAG-Extractor

    path_value:
        YAML에 적힌 경로입니다.
        상대 경로 또는 절대 경로 모두 가능합니다.

    Returns
    -------
    Path
        절대 경로입니다.
    """
    project_root = Path(project_root)
    path_value = Path(path_value)

    if path_value.is_absolute():
        return path_value

    return project_root / path_value


def get_nested_value(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    점 표기법으로 중첩 dict 값을 가져옵니다.

    예:
    dotted_key = "embedding.model_name"

    config["embedding"]["model_name"] 값을 반환합니다.

    Parameters
    ----------
    config:
        config dict입니다.

    dotted_key:
        점으로 연결된 key입니다.
        예: "embedding.model_name"

    default:
        key가 없을 때 반환할 기본값입니다.

    Returns
    -------
    Any
        찾은 값 또는 default입니다.
    """
    value = config

    for key in dotted_key.split("."):
        if not isinstance(value, dict):
            return default

        if key not in value:
            return default

        value = value[key]

    return value


def save_config_snapshot(config: Dict[str, Any], output_path: str | Path) -> None:
    """
    실험에 사용한 config를 JSON 파일로 저장합니다.

    주 사용처:
    - FAISS 인덱스를 생성할 때 해당 인덱스가 어떤 설정으로 만들어졌는지 저장
    - 이후 현재 config와 비교해서 재생성이 필요한지 판단

    Parameters
    ----------
    config:
        저장할 config dict입니다.

    output_path:
        저장할 JSON 파일 경로입니다.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_config_snapshot(snapshot_path: str | Path) -> Dict[str, Any]:
    """
    저장된 config snapshot JSON을 로드합니다.

    파일이 없으면 빈 dict를 반환합니다.
    이러면 최초 실행 여부를 쉽게 판단할 수 있습니다.

    Parameters
    ----------
    snapshot_path:
        config snapshot JSON 경로입니다.

    Returns
    -------
    Dict[str, Any]
        저장된 config dict입니다. 파일이 없으면 {}를 반환합니다.
    """
    snapshot_path = Path(snapshot_path)

    if not snapshot_path.exists():
        return {}

    with open(snapshot_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_vector_config_compatibility(
    current_config: Dict[str, Any],
    saved_config: Dict[str, Any],
    keys_to_check: Optional[List[str]] = None,
) -> List[str]:
    """
    현재 config와 저장된 vector DB config가 호환되는지 확인합니다.

    FAISS 인덱스는 아래 값이 바뀌면 다시 만들어야 합니다.

    대표적으로:
    - embedding.model_name
    - chunking.strategy
    - paths.chunk_path

    Parameters
    ----------
    current_config:
        현재 실행에 사용할 config입니다.

    saved_config:
        기존 FAISS 인덱스 생성 당시 저장된 config입니다.

    keys_to_check:
        비교할 key 목록입니다.
        점 표기법을 사용합니다.
        예: ["embedding.model_name", "paths.chunk_path"]

    Returns
    -------
    List[str]
        값이 달라진 key 목록입니다.
        빈 리스트면 호환된다고 볼 수 있습니다.
    """
    if keys_to_check is None:
        keys_to_check = [
            "embedding.model_name",
            "chunking.strategy",
            "paths.chunk_path",
        ]

    changed_keys = []

    # 저장된 config가 없으면 비교할 수 없으므로 모든 key가 변경된 것으로 간주합니다.
    if not saved_config:
        return keys_to_check

    for key in keys_to_check:
        current_value = get_nested_value(current_config, key)
        saved_value = get_nested_value(saved_config, key)

        if current_value != saved_value:
            changed_keys.append(key)

    return changed_keys


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    현재 config의 핵심 설정을 보기 좋게 출력합니다.

    Notebook이나 script 실행 초기에 사용하면 좋습니다.
    """
    print("===== Config Summary =====")
    print("Experiment:", get_nested_value(config, "experiment.name"))
    print("Description:", get_nested_value(config, "experiment.description"))
    print("Random seed:", get_nested_value(config, "experiment.random_seed"))

    print("\n--- Models ---")
    print("Embedding:", get_nested_value(config, "embedding.model_name"))
    print("LLM:", get_nested_value(config, "llm.model_name"))

    print("\n--- Retrieval ---")
    print("Vector DB:", get_nested_value(config, "vector_db.type"))
    print("Top-K:", get_nested_value(config, "retrieval.top_k"))

    print("\n--- Chunking ---")
    print("Strategy:", get_nested_value(config, "chunking.strategy"))
    print("Chunk path:", get_nested_value(config, "paths.chunk_path"))

    print("\n--- Evaluation ---")
    print("Sample size:", get_nested_value(config, "evaluation.sample_size"))
    print("==========================")