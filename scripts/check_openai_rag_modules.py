from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


BACKEND_PACKAGES = {
    "faiss": ["faiss", "numpy"],
    "chroma": ["chromadb"],
    "qdrant": ["qdrant_client"],
    "supabase": ["supabase"],
}


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenAI RAG modules and paths.")
    parser.add_argument("--config", default="configs/baseline_rag.yaml")
    parser.add_argument("--llm-model", choices=["gpt-5-mini", "gpt-5-nano"])
    parser.add_argument("--vector-db", choices=["faiss", "chroma", "qdrant", "supabase"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from src.utils.config_utils import load_yaml_config, resolve_project_path

    config = load_yaml_config(resolve_project_path(root, args.config))
    vector_db = args.vector_db or config.get("vector_db", {}).get("type", "faiss")
    llm_model = args.llm_model or config.get("llm", {}).get("openai_model_name")

    print(f"[project_root] {root}")
    print(f"[config] {resolve_project_path(root, args.config)}")
    print(f"[llm_model] {llm_model}")
    print(f"[vector_db] {vector_db}")

    required = ["openai", *BACKEND_PACKAGES.get(vector_db, [])]
    missing = []
    for package in required:
        ok = has_module(package)
        print(f"[module] {package}: {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(package)

    api_key_env = config.get("openai", {}).get("api_key_env", "OPENAI_API_KEY")
    print(f"[env] {api_key_env}: {'OK' if os.getenv(api_key_env) else 'MISSING'}")

    chunk_path = resolve_project_path(root, config["paths"]["chunk_path"])
    eval_path = resolve_project_path(root, config["paths"]["eval_dataset_path"])
    print(f"[chunk_path] {chunk_path} ({'exists' if chunk_path.exists() else 'missing'})")
    print(f"[eval_dataset_path] {eval_path} ({'exists' if eval_path.exists() else 'missing'})")

    if missing:
        print("\nInstall missing packages with: python -m pip install -r requirements-openai-rag.txt")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
