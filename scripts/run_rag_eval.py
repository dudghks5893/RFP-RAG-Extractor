# scripts/run_rag_eval.py
#
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
    """
    현재 스크립트 위치에서 상위 폴더를 탐색하며 프로젝트 루트를 찾습니다.

    예:
    현재 파일 위치:
    RFP-RAG-Extractor/scripts/run_rag_eval.py

    반환:
    RFP-RAG-Extractor/
    """
    current = Path(__file__).resolve()

    for path in [current, *current.parents]:
        if path.name == project_name:
            return path

    raise FileNotFoundError(
        f"프로젝트 루트 폴더 '{project_name}'를 찾을 수 없습니다. 현재 위치: {current}"
    )


def parse_args() -> argparse.Namespace:
    """
    커맨드라인 인자를 파싱합니다.

    Returns
    -------
    argparse.Namespace
        실행 인자 객체입니다.
    """
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation pipeline with YAML config."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_rag.yaml",
        help="YAML config 파일 경로입니다. 프로젝트 루트 기준 상대 경로 또는 절대 경로를 사용할 수 있습니다.",
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
        help="실행 후 cleanup을 생략합니다. 디버깅 목적일 때만 사용하세요.",
    )

    return parser.parse_args()


def main() -> int:
    """
    스크립트 메인 실행 함수입니다.

    Returns
    -------
    int
        정상 종료 시 0, 실패 시 1을 반환합니다.
    """
    args = parse_args()

    project_root = find_project_root_from_script(args.project_name)

    # src import를 위해 프로젝트 루트를 sys.path에 추가합니다.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.pipeline import RAGEvalPipeline

    pipeline = None

    try:
        print("===== Run RAG Evaluation Pipeline =====")
        print("project_root:", project_root)
        print("config:", args.config)

        pipeline = RAGEvalPipeline(
            config_path=args.config,
            project_root=project_root,
            project_name=args.project_name,
        )

        results = pipeline.run()

        print("\n===== Pipeline Finished Successfully =====")
        print("metrics:")
        print(results.get("metrics"))

        return 0

    except Exception as e:
        print("\n===== Pipeline Failed =====")
        print("error:", repr(e))
        print("\ntraceback:")
        traceback.print_exc()

        return 1

    finally:
        if pipeline is not None and not args.skip_cleanup:
            print("\n===== Cleanup =====")
            try:
                pipeline.cleanup()
            except Exception as cleanup_error:
                print("cleanup 중 오류 발생:", repr(cleanup_error))


if __name__ == "__main__":
    raise SystemExit(main())