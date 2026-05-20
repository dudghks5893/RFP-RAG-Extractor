from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfp_rag_pipeline.config import load_config
from rfp_rag_pipeline.documents import discover_documents


RAW_DATA_EXTENSIONS = {".hwp", ".pdf", ".docx", ".txt", ".md", ".markdown"}
VECTOR_BACKEND_MODULES = {
    "faiss": ["faiss"],
    "chroma": ["chromadb"],
    "qdrant": ["qdrant_client"],
    "supabase": ["supabase"],
}


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def status(path: Path) -> str:
    return "exists" if path.exists() else "missing"


def main() -> int:
    config = load_config("configs/openai_rag.yaml")
    raw_files = [
        path
        for path in config.raw_dir.rglob("*")
        if path.is_file() and path.name.lower() != "readme.md" and path.suffix.lower() in RAW_DATA_EXTENSIONS
    ] if config.raw_dir.exists() else []
    directly_supported_files = discover_documents(config.raw_dir)
    data_list = config.root / "data" / "raw" / "data_list.csv"
    meta_data = config.root / "data" / "raw" / "meta_data.csv"
    main_py = config.root / "main.py"
    active_store = config.vector_store_name
    print(f"[python] {sys.version.split()[0]}")
    print(f"[root] {config.root}")
    print(f"[raw_documents] {len(raw_files)}")
    print(f"[raw_documents_expected_101] {'OK' if len(raw_files) == 101 else 'CHECK'}")
    print(f"[direct_openai_loader_supported_documents] {len(directly_supported_files)}")
    print(f"[metadata:data_list_csv] {data_list} ({status(data_list)})")
    print(f"[metadata:meta_data_csv] {meta_data} ({status(meta_data)}; optional alias)")
    print(f"[main.py] {main_py} ({status(main_py)})")
    print(f"[chunks] {config.chunks_file} ({status(config.chunks_file)})")
    if not config.chunks_file.exists():
        print("[next_step] Run: python scripts/run_extract_chunk.py --config configs/baseline_rag.yaml")
    print(f"[OPENAI_API_KEY] {'set' if os.getenv('OPENAI_API_KEY') else 'missing'}")
    for module in ["openai", "numpy"]:
        print(f"[module:core] {module}: {'OK' if has_module(module) else 'MISSING'}")
    for store, modules in VECTOR_BACKEND_MODULES.items():
        label = "active" if store == active_store else "optional"
        for module in modules:
            print(f"[module:{label}:{store}] {module}: {'OK' if has_module(module) else 'MISSING'}")
    for module in ["pypdf", "docx"]:
        note = "optional; only needed for direct OpenAI loader, not for existing baseline chunk step"
        print(f"[module:optional_loader] {module}: {'OK' if has_module(module) else 'MISSING'} ({note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
