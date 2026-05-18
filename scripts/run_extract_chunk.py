# scripts/run_extract_chunk.py
#
# YAML config 기반 원본 파일 추출/정제/청킹 파이프라인 실행 스크립트입니다.
#
# 사용 예:
# python scripts/run_extract_chunk.py --config configs/baseline_rag.yaml

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
    """
    current = Path(__file__).resolve()

    for path in [current, *current.parents]:
        if path.name == project_name:
            return path

    raise FileNotFoundError(
        f"프로젝트 루트 폴더 '{project_name}'를 찾을 수 없습니다. 현재 위치: {current}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extract/clean/chunk pipeline with YAML config."
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


def main() -> int:
    args = parse_args()

    project_root = find_project_root_from_script(args.project_name)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.pipeline import ExtractChunkPipeline

    try:
        print("===== Run Extract/Clean/Chunk Pipeline =====")
        print("project_root:", project_root)
        print("config:", args.config)

        pipeline = ExtractChunkPipeline(
            config_path=args.config,
            project_root=project_root,
            project_name=args.project_name,
        )

        results = pipeline.run()

        print("\n===== Extract/Chunk Finished Successfully =====")
        print("chunk_path:", results.get("chunk_path"))
        print("num_chunks:", results.get("num_chunks"))
        print("skipped:", results.get("skipped"))

        return 0

    except Exception as e:
        print("\n===== Extract/Chunk Pipeline Failed =====")
        print("error:", repr(e))
        print("\ntraceback:")
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    raise SystemExit(main())