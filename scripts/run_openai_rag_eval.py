from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path


def find_project_root(project_name: str = "RFP-RAG-Extractor") -> Path:
    current = Path(__file__).resolve()
    for path in [current, *current.parents]:
        if path.name == project_name:
            return path
    raise FileNotFoundError(f"Project root '{project_name}' not found from {current}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation from YAML config.")
    parser.add_argument("--config", default="configs/baseline_rag.yaml")
    parser.add_argument("--project-name", default="RFP-RAG-Extractor")
    parser.add_argument("--llm-model", choices=["gpt-5-mini", "gpt-5-nano"])
    parser.add_argument("--vector-db", choices=["faiss", "chroma", "qdrant", "supabase"])
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run gpt-5-mini/nano x Qdrant/Supabase/FAISS/Chroma OpenAI experiments.",
    )
    return parser.parse_args()


def is_openai_config(config: dict) -> bool:
    embedding_provider = config.get("embedding", {}).get("provider")
    llm_provider = config.get("llm", {}).get("provider")
    return embedding_provider == "openai" or llm_provider == "openai" or "openai" in config


def run_single(project_root: Path, args: argparse.Namespace, overrides: dict | None = None) -> dict:
    from src.pipeline import OpenAIRAGEvalPipeline, RAGEvalPipeline
    from src.utils.config_utils import load_yaml_config, resolve_project_path

    config_path = resolve_project_path(project_root, args.config)
    config = load_yaml_config(config_path)

    effective_overrides = dict(overrides or {})
    if args.llm_model:
        effective_overrides["llm_model"] = args.llm_model
    if args.vector_db:
        effective_overrides["vector_db_type"] = args.vector_db
    if args.experiment_name:
        effective_overrides["experiment_name"] = args.experiment_name

    if is_openai_config(config) or effective_overrides:
        pipeline = OpenAIRAGEvalPipeline(
            config_path=args.config,
            project_root=project_root,
            overrides=effective_overrides,
        )
    else:
        pipeline = RAGEvalPipeline(
            config_path=args.config,
            project_root=project_root,
            project_name=args.project_name,
        )

    try:
        return pipeline.run()
    finally:
        if hasattr(pipeline, "cleanup"):
            pipeline.cleanup()


def main() -> int:
    args = parse_args()
    project_root = find_project_root(args.project_name)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        if args.matrix:
            from src.pipeline import build_openai_experiment_matrix

            failures = []
            for overrides in build_openai_experiment_matrix():
                name = overrides["experiment_name"]
                print(f"\n===== OpenAI RAG Experiment: {name} =====")
                matrix_args = copy.copy(args)
                matrix_args.llm_model = None
                matrix_args.vector_db = None
                matrix_args.experiment_name = None
                try:
                    result = run_single(project_root, matrix_args, overrides)
                    print("metrics:", result.get("metrics"))
                except Exception as exc:
                    failures.append((name, repr(exc)))
                    traceback.print_exc()
            if failures:
                print("\n===== Matrix Failures =====")
                for name, error in failures:
                    print(f"- {name}: {error}")
                return 1
            return 0

        result = run_single(project_root, args)
        print("\n===== RAG Evaluation Finished =====")
        print("metrics:", result.get("metrics"))
        return 0
    except Exception as exc:
        print("\n===== RAG Evaluation Failed =====")
        print("error:", repr(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
