# YAML config 기반 RAG 평가 파이프라인 실행 스크립트입니다.
#
# 사용 예:
# 프로젝트 루트에서 실행합니다.
# python scripts/run_rag_eval.py --config configs/baseline_rag.yaml
#
# 주요 동작:
# 1. 프로젝트 루트 찾기
# 2. src import path 설정
# 3. RAGEvalPipeline 생성
# 4. 전체 RAG 평가 실행
# 5. 실행 완료 후 cleanup
#
# 주의:
# - 이 스크립트는 notebooks/02_baseline_rag_eval.ipynb에서 실험한 내용을
#   재사용 가능한 배치 실행 형태로 만든 것입니다.
# - config의 embedding.force_rebuild_index 값에 따라 FAISS를 새로 만들거나 기존 DB를 로드합니다.

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
        description="Run RAG evaluation pipeline with YAML config."
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

    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="pipeline.cleanup()을 생략합니다. 디버깅 목적일 때만 사용하세요.",
    )

    parser.add_argument(
        "--skip-cache-cleanup",
        action="store_true",
        help="실행 캐시 폴더 삭제를 생략합니다. 디버깅 목적일 때만 사용하세요.",
    )

    parser.add_argument(
        "--cache-root",
        type=str,
        default=".runtime_cache",
        help="이번 실행 캐시를 저장할 루트 폴더입니다.",
    )

    parser.add_argument(
        "--delete-vector-db",
        action="store_true",
        help="실행 후 FAISS vector_db_dir도 삭제합니다. 결과 파일만 남기고 싶을 때 사용하세요.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = find_project_root_from_script(args.project_name)

    # src import를 위해 프로젝트 루트를 sys.path에 추가합니다.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 중요:
    # RAGEvalPipeline import 전에 디스크 캐시 가드를 먼저 import하고 환경변수를 설정해야 합니다.
    from src.utils.run_disk_guard import RunDiskGuard

    disk_guard = RunDiskGuard(
        project_root=project_root,
        cache_root=args.cache_root,
        verbose=True,
    )

    # HuggingFace / Torch / pip / temp 캐시 위치를 이번 실행 전용 폴더로 고정
    disk_guard.setup_env()

    # 중요:
    # 캐시 환경변수 설정 후에 pipeline을 import합니다.
    from src.pipeline import RAGEvalPipeline

    pipeline = None

    try:
        print("===== Run RAG Evaluation Pipeline =====")
        print("project_root:", project_root)
        print("config:", args.config)
        print("cache_root:", args.cache_root)

        pipeline = RAGEvalPipeline(
            config_path=args.config,
            project_root=project_root,
            project_name=args.project_name,
        )

        # 결과 파일만 남기고 싶다면 vector DB도 삭제 대상으로 등록
        # pipeline 생성 후 paths가 생기므로 여기에서 등록합니다.
        if args.delete_vector_db:
            disk_guard.add_delete_path(pipeline.paths["vector_db_dir"])

        results = pipeline.run()

        print("\n===== Pipeline Finished Successfully =====")
        print("metrics:")
        print(results.get("metrics"))

        print("\n===== Saved Result Files =====")
        print("rag_output_path:", pipeline.paths.get("rag_output_path"))
        print("rag_output_scored_path:", pipeline.paths.get("rag_output_scored_path"))
        print("summary_csv_path:", pipeline.paths.get("summary_csv_path"))
        print("metrics_path:", pipeline.paths.get("metrics_path"))
        print("experiment_summary_path:", pipeline.paths.get("experiment_summary_path"))

        return 0

    except Exception as e:
        print("\n===== Pipeline Failed =====")
        print("error:", repr(e))
        print("\ntraceback:")
        traceback.print_exc()

        return 1

    finally:
        if pipeline is not None and not args.skip_cleanup:
            print("\n===== Pipeline Cleanup =====")
            try:
                pipeline.cleanup()
            except Exception as cleanup_error:
                print("pipeline cleanup 중 오류 발생:", repr(cleanup_error))

        if not args.skip_cache_cleanup:
            print("\n===== Disk Cache Cleanup =====")
            try:
                disk_guard.cleanup()
            except Exception as cache_cleanup_error:
                print("disk cache cleanup 중 오류 발생:", repr(cache_cleanup_error))


if __name__ == "__main__":
    raise SystemExit(main())