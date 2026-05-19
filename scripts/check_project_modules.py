# scripts/check_project_modules.py
#
# 프로젝트 주요 모듈 import 및 기본 경로/파일 존재 여부를 확인하는 체크 스크립트입니다.
#
# 사용 예:
# python scripts/check_project_modules.py --config configs/baseline_rag.yaml
#
# 목적:
# - src 모듈 import 오류 조기 발견
# - config 로드 확인
# - 주요 파일 존재 여부 확인
# - extractor/chunker/evaluator/pipeline class 접근 가능 여부 확인

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def find_project_root_from_script(
    project_name: str = "RFP-RAG-Extractor",
) -> Path:
    current = Path(__file__).resolve()

    for path in [current, *current.parents]:
        if path.name == project_name:
            return path

    raise FileNotFoundError(
        f"프로젝트 루트 폴더 '{project_name}'를 찾을 수 없습니다. 현재 위치: {current}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check project modules and config."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_rag.yaml",
        help="YAML config 파일 경로입니다.",
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="RFP-RAG-Extractor",
        help="프로젝트 루트 폴더 이름입니다.",
    )

    return parser.parse_args()


def check_path(label: str, path: Path, must_exist: bool = True) -> None:
    """
    경로 존재 여부를 출력합니다.
    """
    exists = path.exists()

    status = "OK" if exists else "MISSING"

    if not must_exist and not exists:
        status = "NOT_FOUND_ALLOWED"

    print(f"[{status}] {label}: {path}")

    if must_exist and not exists:
        raise FileNotFoundError(f"{label} 경로가 없습니다: {path}")


def main() -> int:
    args = parse_args()

    project_root = find_project_root_from_script(args.project_name)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        print("===== Project Module Check =====")
        print("project_root:", project_root)
        print("config:", args.config)

        # -----------------------------------------------------
        # utils
        # -----------------------------------------------------
        from src.utils.config_utils import (
            load_yaml_config,
            resolve_project_path,
            print_config_summary,
        )
        from src.utils.path_utils import find_project_root
        from src.utils.file_utils import load_jsonl, save_jsonl
        from src.utils.eval_dataset_utils import load_json, save_json, create_and_save_eval_sample
        from src.utils.progress_utils import ProgressLogger, progress_iter, log_step
        from src.utils.seed import set_seed
        from src.utils.device import get_device

        print("[OK] utils imports")

        # -----------------------------------------------------
        # extractors / chunking
        # -----------------------------------------------------
        from src.extractors import extract_text_by_file_type
        from src.extractors.pdf_extractor import extract_pdf_text
        from src.extractors.hwp_extractor import extract_hwp_text
        from src.extractors.docx_extractor import extract_docx_text

        from src.utils.text_cleaner import clean_extracted_text

        print("[OK] extractors and text_cleaner imports")

        # -----------------------------------------------------
        # RAG modules
        # -----------------------------------------------------
        from src.embeddings import EmbeddingModel, load_embedding_model
        from src.vectorstores import FAISSVectorStore, build_or_load_faiss_store
        from src.retrieval import RAGRetriever
        from src.generation import (
            build_rfp_rag_messages,
            LLMGenerator,
            load_llm_generator,
        )
        from src.evaluation.evaluator import RAGEvaluator

        print("[OK] RAG module imports")

        # -----------------------------------------------------
        # pipelines
        # -----------------------------------------------------
        from src.pipeline import ExtractChunkPipeline, RAGEvalPipeline

        print("[OK] pipeline imports")

        # -----------------------------------------------------
        # config
        # -----------------------------------------------------
        config_path = resolve_project_path(project_root, args.config)
        check_path("config_path", config_path, must_exist=True)

        config = load_yaml_config(config_path)
        print_config_summary(config)

        # -----------------------------------------------------
        # required paths
        # -----------------------------------------------------
        extract_cfg = config.get("extract", {})
        paths_cfg = config.get("paths", {})

        if extract_cfg:
            check_path(
                "extract.raw_dir",
                resolve_project_path(project_root, extract_cfg["raw_dir"]),
                must_exist=True,
            )
            check_path(
                "extract.data_list_path",
                resolve_project_path(project_root, extract_cfg["data_list_path"]),
                must_exist=True,
            )
            check_path(
                "extract.output_chunk_path",
                resolve_project_path(project_root, extract_cfg["output_chunk_path"]),
                must_exist=False,
            )

        if paths_cfg:
            check_path(
                "paths.chunk_path",
                resolve_project_path(project_root, paths_cfg["chunk_path"]),
                must_exist=False,
            )
            check_path(
                "paths.eval_dataset_path",
                resolve_project_path(project_root, paths_cfg["eval_dataset_path"]),
                must_exist=True,
            )
            check_path(
                "paths.eval_sample_path",
                resolve_project_path(project_root, paths_cfg["eval_sample_path"]),
                must_exist=False,
            )
            check_path(
                "paths.vector_db_dir",
                resolve_project_path(project_root, paths_cfg["vector_db_dir"]),
                must_exist=False,
            )

        print("\n===== Module Check Passed =====")

        return 0

    except Exception as e:
        print("\n===== Module Check Failed =====")
        print("error:", repr(e))
        print("\ntraceback:")
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    raise SystemExit(main())