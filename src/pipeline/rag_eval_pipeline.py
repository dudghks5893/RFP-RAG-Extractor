# src/pipeline/rag_eval_pipeline.py
#
# YAML config 기반 RAG 자동 평가 파이프라인입니다.
#
# 이 파일은 notebooks/02_baseline_rag_eval.ipynb에서 검증한 흐름을
# 재사용 가능한 Python 모듈로 옮긴 것입니다.
#
# 주요 흐름:
# 1. config 로드
# 2. 경로 해석
# 3. 평가 샘플 로드/생성
# 4. 청크 로드
# 5. 임베딩 모델 로드
# 6. FAISS 인덱스 로드 또는 생성
# 7. Retriever 생성
# 8. LLM Generator 로드
# 9. RAG 실행
# 10. 평가 및 저장

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import time
import gc
import os
import pickle
import shutil

import json
import hashlib
import pandas as pd
import torch

from src.utils.config_utils import (
    load_yaml_config,
    resolve_project_path,
    print_config_summary,
)
from src.utils.path_utils import find_project_root
from src.utils.file_utils import load_jsonl
from src.utils.eval_dataset_utils import (
    load_json,
    save_json,
    create_and_save_eval_sample,
)
from src.utils.progress_utils import progress_iter, log_step
from src.utils.seed import set_seed
from src.utils.device import get_device

from src.embeddings import load_embedding_model
from src.vectorstores import FAISSVectorStore
from src.retrieval import RAGRetriever
from src.generation import load_llm_generator
from src.evaluation.evaluator import RAGEvaluator
from src.constants.human_eval_questions import HUMAN_EVAL_QUESTIONS


# =========================================================
# Provider helpers
# =========================================================
SUPPORTED_VECTOR_STORES = ("faiss", "chroma", "qdrant", "supabase")
SUPPORTED_EMBEDDING_PROVIDERS = ("huggingface", "hf", "openai")
SUPPORTED_LLM_PROVIDERS = ("huggingface", "hf", "openai")


def _safe_name(value: Any) -> str:
    """파일/collection 이름에 안전하게 쓸 수 있도록 문자열을 정리합니다."""
    text = str(value or "none").strip()

    for old, new in [("/", "_"), ("\\", "_"), (":", "_"), (" ", "_")]:
        text = text.replace(old, new)

    return "".join(
        ch
        for ch in text
        if ch.isalnum() or ch in {"_", "-", "."}
    )[:120]


def _normalize_provider(
    provider: Optional[str],
    default: str = "huggingface",
) -> str:
    provider = (provider or default).lower().strip()

    if provider == "hf":
        return "huggingface"

    return provider


def _flat_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vector DB metadata에 넣을 수 있는 primitive 타입만 남깁니다.

    pdf_page 청킹에서 생성되는 page_start, page_end, page_chunk_index도
    metadata에 포함합니다.
    """
    metadata = dict(chunk.get("metadata") or {})

    metadata.update(
        {
            "chunk_id": str(chunk.get("chunk_id", "")),
            "doc_id": str(chunk.get("doc_id", "")),
            "file_name": str(chunk.get("file_name") or metadata.get("file_name", "")),
            "file_type": str(chunk.get("file_type") or metadata.get("file_type", "")),
            "project_name": str(chunk.get("project_name") or metadata.get("project_name", "")),
            "organization": str(chunk.get("organization") or metadata.get("organization", "")),
            "section_title": str(chunk.get("section_title") or metadata.get("section_title", "")),
            "section_id": str(chunk.get("section_id") or metadata.get("section_id", "")),
            "section_path": str(chunk.get("section_path") or metadata.get("section_path", "")),
            "chunking_strategy": str(
                chunk.get("chunking_strategy") or metadata.get("chunking_strategy", "")
            ),
        }
    )

    if chunk.get("page_start") is not None:
        metadata["page_start"] = chunk.get("page_start")

    if chunk.get("page_end") is not None:
        metadata["page_end"] = chunk.get("page_end")

    if chunk.get("page_chunk_index") is not None:
        metadata["page_chunk_index"] = chunk.get("page_chunk_index")

    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) and value is not None
    }


def _json_safe(value: Any) -> Any:
    """
    json.dump에서 자주 터지는 타입(Path, numpy scalar, pandas NA, set 등)을
    저장 가능한 기본 타입으로 변환합니다.

    결과 파일이 생성되지 않는 문제를 막기 위한 마지막 안전망입니다.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=lambda x: str(x))]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return _json_safe(value.detach().cpu().tolist())
        except Exception:
            pass

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def _remove_dir_force(path: Path) -> None:
    """
    Chroma/FAISS 로컬 DB 디렉토리를 최대한 안전하게 삭제합니다.

    중요:
    - Chroma PersistentClient가 열리기 전에 호출되어야 합니다.
    - ignore_errors=True로 조용히 실패하면 readonly DB 파일이 남을 수 있으므로,
      chmod 후 재시도하고 그래도 실패하면 명시적으로 예외를 올립니다.
    """
    path = Path(path)

    if not path.exists():
        return

    def _onerror(func, failed_path, exc_info):
        try:
            os.chmod(failed_path, 0o700)
            func(failed_path)
        except Exception as exc:
            raise RuntimeError(
                f"vector DB 디렉토리 삭제 실패: {failed_path}. "
                "권한/소유자를 확인하세요. "
                f"원인: {repr(exc)}"
            ) from exc

    shutil.rmtree(path, onerror=_onerror)


def _assert_writable_dir(path: Path) -> None:
    """SQLite/Chroma가 실제로 쓸 수 있는 디렉토리인지 미리 확인합니다."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    test_file = path / ".write_test"

    try:
        with test_file.open("w", encoding="utf-8") as f:
            f.write("ok")

        test_file.unlink(missing_ok=True)

    except Exception as exc:
        raise RuntimeError(
            f"Chroma persist_dir에 쓰기 권한이 없습니다: {path}\n"
            "해결: 해당 폴더를 삭제 후 재생성하거나, 소유자를 user3로 바꾸세요.\n"
            f"예: rm -rf {path} && mkdir -p {path} && chmod -R u+rwX {path}\n"
            f"원인: {repr(exc)}"
        ) from exc


class OpenAIEmbeddingModel:
    """OpenAI embedding adapter. HF embedder와 비슷하게 encode_chunks/encode_query를 제공합니다."""

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package가 필요합니다. 설치: pip install openai python-dotenv"
            ) from exc

        api_key = os.getenv(api_key_env)

        if not api_key:
            raise RuntimeError(
                f"{api_key_env} 환경변수가 없습니다. "
                f".env에 {api_key_env}=... 를 넣거나 export 해주세요."
            )

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        vectors: List[List[float]] = []
        texts = [str(text or "") for text in texts]

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )

            vectors.extend(item.embedding for item in response.data)

        return vectors

    def encode_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
        show_progress: bool = True,
        log_every: int = 10,
    ) -> List[List[float]]:
        """
        청크 dict 목록에서 embedding_text를 우선 사용해 임베딩합니다.

        pdf_page_chunker.py는 기관명/사업명/파일명/페이지 정보를 포함한
        embedding_text를 생성하므로, 검색 성능 개선을 위해 이를 우선 사용합니다.

        embedding_text가 없거나 비어 있으면 기존 text를 fallback으로 사용합니다.
        """
        texts = []

        for chunk in chunks:
            text = chunk.get("embedding_text")

            if text is None or not str(text).strip():
                text = chunk.get("text", "")

            texts.append(str(text or ""))

        vectors: List[List[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start:start + batch_size]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )

            vectors.extend(item.embedding for item in response.data)

            if show_progress and (
                batch_idx == 1
                or batch_idx % log_every == 0
                or batch_idx == total_batches
            ):
                print(f"OpenAI embedding batch {batch_idx}/{total_batches}")

        return vectors

    def encode_query(self, query: str) -> List[float]:
        return self.encode_texts([query], batch_size=1)[0]

    def unload(self) -> None:
        self.client = None


class OpenAILLMGenerator:
    """OpenAI Responses API 기반 generator. 기존 HF generator와 같은 반환 포맷을 맞춥니다."""

    def __init__(
        self,
        model_name: str,
        fallback_model_name: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_output_tokens: int = 512,
        temperature: Optional[float] = 0.0,
        prompt_type: str = "default",
        max_chars_per_chunk: Optional[int] = None,
        include_metadata: bool = True,
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package가 필요합니다. 설치: pip install openai python-dotenv"
            ) from exc

        api_key = os.getenv(api_key_env)

        if not api_key:
            raise RuntimeError(
                f"{api_key_env} 환경변수가 없습니다. "
                f".env에 {api_key_env}=... 를 넣거나 export 해주세요."
            )

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = temperature
        self.prompt_type = prompt_type
        self.max_chars_per_chunk = max_chars_per_chunk
        self.include_metadata = include_metadata
        self.last_model_name = model_name

    def generate_from_retrieved_chunks(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        return_prompt: bool = False,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(question, retrieved_chunks)
        start = time.perf_counter()

        try:
            response_text, usage, used_model = self._call_model(self.model_name, prompt)
        except Exception as exc:
            if not self.fallback_model_name:
                raise

            print(f"OpenAI primary model 실패. fallback 사용: {repr(exc)}")
            response_text, usage, used_model = self._call_model(
                self.fallback_model_name,
                prompt,
            )

        latency = time.perf_counter() - start
        self.last_model_name = used_model

        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))

        result = {
            "response": response_text,
            "generation_latency_sec": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model_name": used_model,
        }

        if return_prompt:
            result["prompt"] = prompt

        return result

    def _call_model(
        self,
        model: str,
        prompt: str,
    ) -> tuple[str, Dict[str, int], str]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        try:
            response = self.client.responses.create(**kwargs)
        except Exception as exc:
            if "temperature" not in str(exc).lower():
                raise

            kwargs.pop("temperature", None)
            response = self.client.responses.create(**kwargs)

        text = (getattr(response, "output_text", "") or "").strip()
        usage_obj = getattr(response, "usage", None)

        usage = {
            "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
        }

        return text, usage, model

    def _build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        context_blocks = []
        max_chars = self.max_chars_per_chunk

        for index, chunk in enumerate(retrieved_chunks, start=1):
            text = str(chunk.get("text", ""))

            if max_chars:
                text = text[:int(max_chars)]

            if self.include_metadata:
                metadata = chunk.get("metadata", {}) or {}
                doc_id = chunk.get("doc_id") or metadata.get("doc_id", "")
                chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id", "")
                file_name = chunk.get("file_name") or metadata.get("file_name", "")
                project_name = chunk.get("project_name") or metadata.get("project_name", "")
                organization = chunk.get("organization") or metadata.get("organization", "")
                page_start = chunk.get("page_start") or metadata.get("page_start", "")
                page_end = chunk.get("page_end") or metadata.get("page_end", "")

                if page_start and page_end:
                    page_info = (
                        str(page_start)
                        if str(page_start) == str(page_end)
                        else f"{page_start}-{page_end}"
                    )
                else:
                    page_info = ""

                header = (
                    f"[Evidence {index}] "
                    f"doc_id={doc_id} "
                    f"chunk_id={chunk_id} "
                    f"file_name={file_name} "
                    f"project_name={project_name} "
                    f"organization={organization} "
                    f"page={page_info}"
                )
            else:
                header = f"[Evidence {index}]"

            context_blocks.append(f"{header}\n{text}")

        contexts = "\n\n".join(context_blocks)

        return (
            "너는 RFP 문서를 분석하는 한국어 RAG 어시스턴트다.\n"
            "아래 근거에 있는 내용만 사용해서 답변하라. 근거가 부족하면 부족하다고 말하라.\n"
            "가능하면 문서명, 공고번호, 기관명 같은 근거 정보를 함께 언급하라.\n\n"
            f"{contexts}\n\n"
            f"질문: {question}\n"
            "답변:"
        )

    def unload(self) -> None:
        self.client = None


class OpenAIFaissStore:
    def __init__(
        self,
        persist_dir: Path,
        index_file: str = "index.faiss",
        chunk_meta_file: str = "chunks.pkl",
    ):
        self.persist_dir = Path(persist_dir)
        self.index_path = self.persist_dir / index_file
        self.chunk_meta_path = self.persist_dir / chunk_meta_file
        self.index = None
        self.chunks: List[Dict[str, Any]] = []

    def exists(self) -> bool:
        return self.index_path.exists() and self.chunk_meta_path.exists()

    def is_loaded(self) -> bool:
        return self.index is not None and bool(self.chunks)

    def clear(self) -> None:
        for path in [self.index_path, self.chunk_meta_path]:
            if path.exists():
                path.unlink()

    def build(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        import faiss
        import numpy as np

        vectors = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        self.chunks = list(chunks)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))

        with self.chunk_meta_path.open("wb") as handle:
            pickle.dump(self.chunks, handle)

    def load(self) -> None:
        import faiss

        self.index = faiss.read_index(str(self.index_path))

        with self.chunk_meta_path.open("rb") as handle:
            self.chunks = pickle.load(handle)

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        import faiss
        import numpy as np

        if self.index is None:
            self.load()

        vector = np.asarray([query_embedding], dtype="float32")
        faiss.normalize_L2(vector)

        scores, indices = self.index.search(vector, int(top_k))
        rows = []

        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue

            rows.append(
                {
                    **self.chunks[int(idx)],
                    "rank": rank,
                    "score": float(score),
                }
            )

        return rows


class OpenAIChromaStore:
    """
    Chroma PersistentClient adapter.

    주의:
    - Chroma는 내부적으로 SQLite를 사용합니다.
    - 같은 프로세스에서 PersistentClient를 연 뒤 디렉토리를 삭제/재생성하면
      readonly database 오류가 날 수 있습니다.
    - 따라서 rebuild 시에는 이 클래스의 client를 열기 전에 파이프라인 쪽에서
      persist_dir를 먼저 정리합니다.
    """

    def __init__(self, persist_dir: Path, collection: str):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection
        self._client_instance = None

    def _client(self):
        import chromadb

        _assert_writable_dir(self.persist_dir)

        if self._client_instance is None:
            self._client_instance = chromadb.PersistentClient(path=str(self.persist_dir))

        return self._client_instance

    def _collection(self):
        return self._client().get_or_create_collection(self.collection_name)

    def exists(self) -> bool:
        """
        기존 collection 존재 여부를 확인합니다.

        force rebuild가 필요한 경우에는 파이프라인에서 이 메서드를 호출하지 않도록 합니다.
        이 메서드는 PersistentClient를 열 수 있기 때문입니다.
        """
        if not self.persist_dir.exists():
            return False

        try:
            client = self._client()
            collections = client.list_collections()

            names = []

            for col in collections:
                names.append(
                    getattr(
                        col,
                        "name",
                        col if isinstance(col, str) else None,
                    )
                )

            if self.collection_name not in set(filter(None, names)):
                return False

            return client.get_collection(self.collection_name).count() > 0

        except Exception:
            return False

    def is_loaded(self) -> bool:
        return self.exists()

    def clear(self) -> None:
        """
        열린 client가 있을 때 collection만 삭제합니다.

        전체 디렉토리 삭제는 파이프라인의 clear_vector_db_files()에서
        client를 열기 전에 수행합니다.
        """
        if not self.persist_dir.exists():
            return

        try:
            self._client().delete_collection(self.collection_name)
        except Exception:
            pass

    def build(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        _assert_writable_dir(self.persist_dir)

        collection = self._collection()
        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [_flat_metadata(chunk) for chunk in chunks]

        batch_size = 1000

        for start in range(0, len(ids), batch_size):
            end = start + batch_size

            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

            print(f"Chroma add batch {min(end, len(ids))}/{len(ids)}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        result = self._collection().query(
            query_embeddings=[query_embedding],
            n_results=int(top_k),
            include=["documents", "metadatas", "distances"],
        )

        rows = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for rank, chunk_id in enumerate(ids, start=1):
            metadata = metadatas[rank - 1] or {}

            rows.append(
                {
                    **metadata,
                    "chunk_id": str(chunk_id),
                    "doc_id": str(metadata.get("doc_id", "")),
                    "text": docs[rank - 1] if rank - 1 < len(docs) else "",
                    "rank": rank,
                    "score": distances[rank - 1] if rank - 1 < len(distances) else None,
                    "metadata": metadata,
                }
            )

        return rows


class OpenAIQdrantStore:
    def __init__(self, config: Dict[str, Any]):
        self.url = config.get("url", "http://localhost:6333")
        self.collection_name = config.get("collection", "rfp_openai_rag")
        self.api_key = os.getenv(config.get("api_key_env", "QDRANT_API_KEY"))

    def _client(self):
        from qdrant_client import QdrantClient

        return QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )

    def exists(self) -> bool:
        try:
            self._client().get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def is_loaded(self) -> bool:
        return self.exists()

    def clear(self) -> None:
        try:
            self._client().delete_collection(self.collection_name)
        except Exception:
            pass

    def build(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = self._client()
        self.clear()

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=len(embeddings[0]),
                distance=Distance.COSINE,
            ),
        )

        points = [
            PointStruct(
                id=index,
                vector=embedding,
                payload={"chunk": chunk},
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        for start in range(0, len(points), 256):
            client.upsert(
                collection_name=self.collection_name,
                points=points[start:start + 256],
            )

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        client = self._client()

        try:
            hits = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=int(top_k),
            )
        except AttributeError:
            hits = client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=int(top_k),
            ).points

        rows = []

        for rank, hit in enumerate(hits, start=1):
            chunk = dict((hit.payload or {}).get("chunk", {}))
            rows.append({**chunk, "rank": rank, "score": float(hit.score)})

        return rows


class OpenAISupabaseStore:
    def __init__(self, config: Dict[str, Any]):
        self.url = os.getenv(config.get("url_env", "SUPABASE_URL"), "")
        self.key = os.getenv(config.get("key_env", "SUPABASE_SERVICE_ROLE_KEY"), "")
        self.table = config.get("table", "rfp_chunks")
        self.match_function = config.get("match_function", "match_rfp_chunks")

    def _client(self):
        if not self.url or not self.key:
            raise RuntimeError("Supabase URL/key 환경변수가 없습니다.")

        from supabase import create_client

        return create_client(self.url, self.key)

    def exists(self) -> bool:
        return bool(self.url and self.key)

    def is_loaded(self) -> bool:
        return self.exists()

    def clear(self) -> None:
        return None

    def build(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        client = self._client()

        rows = [
            {
                "id": chunk["chunk_id"],
                "content": chunk["text"],
                "metadata": _flat_metadata(chunk),
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        for start in range(0, len(rows), 100):
            client.table(self.table).upsert(rows[start:start + 100]).execute()

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        result = self._client().rpc(
            self.match_function,
            {
                "query_embedding": query_embedding,
                "match_count": int(top_k),
            },
        ).execute()

        rows = []

        for rank, row in enumerate(result.data or [], start=1):
            metadata = row.get("metadata") or {}

            rows.append(
                {
                    **metadata,
                    "chunk_id": str(row.get("id", "")),
                    "doc_id": str(metadata.get("doc_id", "")),
                    "text": row.get("content", ""),
                    "rank": rank,
                    "score": row.get("similarity", row.get("score")),
                    "metadata": metadata,
                }
            )

        return rows


class RAGEvalPipeline:
    """
    RAG 자동 평가 파이프라인 클래스입니다.

    YAML config를 기준으로 다음을 수행합니다.

    - section_chunks.jsonl 로드
    - FAISS 인덱스 로드 또는 생성
    - 평가 샘플 로드 또는 생성
    - RAG 답변 생성
    - RAGEvaluator 기반 평가
    - 결과 저장
    """

    def __init__(
        self,
        config_path: str | Path,
        project_root: Optional[str | Path] = None,
        project_name: str = "RFP-RAG-Extractor",
    ):
        self.project_name = project_name

        if project_root is None:
            self.project_root = find_project_root(project_name)
        else:
            self.project_root = Path(project_root)

        self.config_path = resolve_project_path(
            self.project_root,
            config_path,
        )

        self.config: Dict[str, Any] = load_yaml_config(self.config_path)

        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        self.evaluator = None

        self.chunks: List[Dict[str, Any]] = []
        self.standard_chunks: List[Dict[str, Any]] = []
        self.eval_dataset: List[Dict[str, Any]] = []
        self.sample_eval_dataset: List[Dict[str, Any]] = []
        self.rag_outputs: List[Dict[str, Any]] = []
        self.scored_outputs: List[Dict[str, Any]] = []

        self.paths: Dict[str, Path] = {}

        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """
        YAML config에 있는 상대 경로들을 프로젝트 루트 기준 절대 경로로 변환합니다.

        provider/model/vector_db 조합별로 결과와 vector DB 경로가 덮어써지지 않도록
        experiment_key를 자동으로 붙입니다.
        """
        cfg = self.config

        embedding_provider = self._embedding_provider()
        llm_provider = self._llm_provider()
        vector_type = self._vector_db_type()
        sample_size = cfg["evaluation"]["sample_size"]

        self.experiment_key = self._build_experiment_key()

        self.paths["chunk_path"] = resolve_project_path(
            self.project_root,
            cfg["paths"]["chunk_path"],
        )
        self.paths["eval_dataset_path"] = resolve_project_path(
            self.project_root,
            cfg["paths"]["eval_dataset_path"],
        )
        self.paths["eval_sample_path"] = resolve_project_path(
            self.project_root,
            cfg["paths"]["eval_sample_path"],
        )

        store_cfg = cfg.get("vector_db", {}).get("stores", {}).get(vector_type, {})
        base_vector_dir = store_cfg.get("persist_dir") or cfg["paths"].get("vector_db_dir", "data/vector_db")
        
        # vector DB는 YAML에 설정된 persist_dir 바로 아래에 저장합니다.
        # 예:
        # - vector_db.type: faiss  -> data/vector_db/faiss/
        # - vector_db.type: chroma -> data/vector_db/chroma/
        #
        # LLM 모델명이나 experiment_key를 vector DB 경로에 붙이지 않습니다.
        # 이렇게 해야 LLM만 바꿔도 기존 vector DB를 재사용할 수 있습니다.
        self.paths["vector_db_dir"] = resolve_project_path(
            self.project_root,
            base_vector_dir,
        )

        self.paths["report_dir"] = resolve_project_path(
            self.project_root,
            cfg["paths"].get("report_dir", "reports/evaluation"),
        ) / self.experiment_key

        report_dir = self.paths["report_dir"]
        prefix = f"sample{sample_size}"

        self.paths["rag_output_path"] = report_dir / f"{prefix}_rag_outputs.json"
        self.paths["rag_output_scored_path"] = report_dir / f"{prefix}_rag_outputs_scored.json"

        self.paths["vector_db_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["report_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["rag_output_path"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["rag_output_scored_path"].parent.mkdir(parents=True, exist_ok=True)

        self.paths["metrics_path"] = report_dir / f"{prefix}_metrics.json"
        self.paths["metrics_by_question_type_path"] = report_dir / f"{prefix}_by_question_type.json"
        self.paths["metrics_by_source_type_path"] = report_dir / f"{prefix}_by_source_type.json"
        self.paths["metrics_by_answer_format_path"] = report_dir / f"{prefix}_by_answer_format.json"
        self.paths["metrics_by_file_type_path"] = report_dir / f"{prefix}_by_file_type.json"
        self.paths["retrieval_failure_path"] = report_dir / f"{prefix}_retrieval_failures.csv"
        self.paths["keyword_failure_path"] = report_dir / f"{prefix}_keyword_failures.csv"
        self.paths["summary_csv_path"] = report_dir / f"{prefix}_summary.csv"
        self.paths["experiment_summary_path"] = report_dir / f"{prefix}_experiment_summary.json"
        self.paths["chunk_fingerprint_path"] = self.paths["vector_db_dir"] / "chunk_fingerprint.json"
        self.paths["vector_config_path"] = self.paths["vector_db_dir"] / "vector_config.json"

        print("experiment_key:", self.experiment_key)
        print("embedding_provider:", embedding_provider)
        print("llm_provider:", llm_provider)
        print("vector_db_type:", vector_type)
        print("rag_output_path:", self.paths["rag_output_path"])
        print("rag_output_scored_path:", self.paths["rag_output_scored_path"])
        print("report_dir:", self.paths["report_dir"])

    def _embedding_provider(self) -> str:
        return _normalize_provider(
            self.config.get("embedding", {}).get("provider"),
            default="huggingface",
        )

    def _llm_provider(self) -> str:
        return _normalize_provider(
            self.config.get("llm", {}).get("provider"),
            default="huggingface",
        )

    def _vector_db_type(self) -> str:
        return str(self.config.get("vector_db", {}).get("type", "faiss")).lower().strip()

    def _active_embedding_model_name(self) -> str:
        embedding_cfg = self.config.get("embedding", {})
        openai_cfg = self.config.get("openai", {})

        if self._embedding_provider() == "openai":
            return str(
                embedding_cfg.get("openai_model_name")
                or openai_cfg.get("embedding_model")
            )

        return str(embedding_cfg.get("hf_model_name"))

    def _active_llm_model_name(self) -> str:
        llm_cfg = self.config.get("llm", {})
        openai_cfg = self.config.get("openai", {})

        if self._llm_provider() == "openai":
            return str(
                llm_cfg.get("openai_model_name")
                or openai_cfg.get("llm_model")
            )

        return str(llm_cfg.get("hf_model_name"))

    def _build_experiment_key(self) -> str:
        """
        결과 폴더명과 vector DB 폴더명에 사용할 실험 key를 생성합니다.
    
        experiment.name은 제외하고,
        embedding model / vector DB / LLM model만 포함합니다.
    
        형식:
        emb-{embedding_model}_vdb-{vector_db_type}_llm-{llm_model}
    
        예:
        emb-BAAI_bge-m3_vdb-faiss_llm-Qwen_Qwen2.5-1.5B-Instruct
        """
        embedding_model = _safe_name(self._active_embedding_model_name())
        vector_db_type = _safe_name(self._vector_db_type())
        llm_model = _safe_name(self._active_llm_model_name())
    
        parts = [
            f"emb-{embedding_model}",
            f"vdb-{vector_db_type}",
            f"llm-{llm_model}",
        ]
    
        return "_".join(parts)

    def _vector_snapshot(self) -> Dict[str, Any]:
        return {
            "embedding_provider": self._embedding_provider(),
            "embedding_model_name": self._active_embedding_model_name(),
            "vector_db_type": self._vector_db_type(),
            "chunking_strategy": self.config.get("chunking", {}).get("strategy"),
            "chunk_path": str(self.paths.get("chunk_path", "")),
        }

    def _load_vector_snapshot(self) -> Optional[Dict[str, Any]]:
        path = self.paths.get("vector_config_path")

        if not path or not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_vector_snapshot(self) -> None:
        path = self.paths["vector_config_path"]
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._vector_snapshot(),
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _ensure_output_dirs(self) -> None:
        """
        결과/리포트 저장 전에 필요한 디렉토리를 항상 생성합니다.
        기존 save_json 유틸이 parent mkdir을 하지 않는 경우를 방어합니다.
        """
        for key in [
            "rag_output_path",
            "rag_output_scored_path",
            "metrics_path",
            "metrics_by_question_type_path",
            "metrics_by_source_type_path",
            "metrics_by_answer_format_path",
            "metrics_by_file_type_path",
            "retrieval_failure_path",
            "keyword_failure_path",
            "summary_csv_path",
            "experiment_summary_path",
        ]:
            path = self.paths.get(key)

            if path is not None:
                Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _write_json_atomic(
        self,
        data: Any,
        path: str | Path,
        label: str = "json",
    ) -> None:
        """
        JSON 결과를 원자적으로 저장합니다.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix(path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(
                _json_safe(data),
                f,
                ensure_ascii=False,
                indent=2,
            )

        tmp_path.replace(path)
        print(f"{label} 저장:", path)

    def _save_rag_outputs(self) -> None:
        """
        raw RAG 결과 저장 전용 helper입니다.
        save_json 유틸 실패 시 자체 json 저장으로 fallback합니다.
        """
        path = self.paths["rag_output_path"]
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_json(_json_safe(self.rag_outputs), path)
            print("RAG 실행 결과 저장:", path)

        except Exception as exc:
            print(f"[WARN] save_json 실패. fallback json 저장 사용: {repr(exc)}")
            self._write_json_atomic(
                self.rag_outputs,
                path,
                label="RAG 실행 결과",
            )

        if not path.exists():
            raise RuntimeError(f"RAG 결과 파일 생성 실패: {path}")

        print("결과 수:", len(self.rag_outputs))

    def _save_scored_outputs(self) -> None:
        """
        평가 점수가 붙은 RAG 결과 저장 helper입니다.
        evaluator.save_rows_as_json 실패 시 fallback합니다.
        """
        path = self.paths["rag_output_scored_path"]
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.evaluator.save_rows_as_json(
                _json_safe(self.scored_outputs),
                str(path),
            )
            print("Scored RAG 결과 저장:", path)

        except Exception as exc:
            print(f"[WARN] evaluator.save_rows_as_json 실패. fallback json 저장 사용: {repr(exc)}")
            self._write_json_atomic(
                self.scored_outputs,
                path,
                label="Scored RAG 결과",
            )

        if not path.exists():
            raise RuntimeError(f"Scored RAG 결과 파일 생성 실패: {path}")

    def _save_metrics_safe(
        self,
        metrics: Dict[str, Any],
        path: str | Path,
        label: str = "metrics",
    ) -> None:
        """
        metric JSON 저장 helper입니다.
        evaluator.save_metrics 실패 시 자체 저장으로 fallback합니다.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.evaluator.save_metrics(
                _json_safe(metrics),
                str(path),
            )
            print(f"{label} 저장:", path)

        except Exception as exc:
            print(f"[WARN] evaluator.save_metrics 실패. fallback json 저장 사용: {repr(exc)}")
            self._write_json_atomic(
                metrics,
                path,
                label=label,
            )

    def print_summary(self) -> None:
        """
        현재 config와 주요 경로를 출력합니다.
        """
        print_config_summary(self.config)

        print("\n===== Path Summary =====")
        print("project_root:", self.project_root)
        print("config_path:", self.config_path)

        for key, value in self.paths.items():
            print(f"{key}: {value}")

    def setup_runtime(self) -> None:
        """
        seed 고정과 device 확인을 수행합니다.
        """
        seed = self.config["experiment"].get("random_seed", 42)

        set_seed(seed)

        self.device = get_device()

    def load_eval_dataset(self) -> List[Dict[str, Any]]:
        """
        전체 평가 데이터셋을 로드합니다.
        """
        eval_dataset_path = self.paths["eval_dataset_path"]

        if not eval_dataset_path.exists():
            raise FileNotFoundError(f"평가 데이터셋이 없습니다: {eval_dataset_path}")

        self.eval_dataset = load_json(eval_dataset_path)

        print("전체 평가 문항 수:", len(self.eval_dataset))
        print(pd.Series([x.get("question_type") for x in self.eval_dataset]).value_counts())

        return self.eval_dataset

    def load_or_create_eval_sample(self) -> List[Dict[str, Any]]:
        """
        샘플 평가셋을 로드하거나, 없으면 생성합니다.
        """
        eval_sample_path = self.paths["eval_sample_path"]

        if eval_sample_path.exists():
            print("기존 샘플 평가셋 로드:", eval_sample_path)
            self.sample_eval_dataset = load_json(eval_sample_path)

        else:
            print("샘플 평가셋 새로 생성:", eval_sample_path)
            self.sample_eval_dataset = create_and_save_eval_sample(
                input_path=self.paths["eval_dataset_path"],
                output_path=eval_sample_path,
                sample_size=self.config["evaluation"]["sample_size"],
                random_seed=self.config["experiment"]["random_seed"],
            )

        print("샘플 문항 수:", len(self.sample_eval_dataset))
        print(pd.Series([x.get("question_type") for x in self.sample_eval_dataset]).value_counts())

        return self.sample_eval_dataset

    def load_chunks(self) -> List[Dict[str, Any]]:
        """
        section_chunks.jsonl을 로드합니다.
        """
        chunk_path = self.paths["chunk_path"]

        if not chunk_path.exists():
            raise FileNotFoundError(
                f"청크 파일이 없습니다: {chunk_path}\n"
                "먼저 01_extract_clean_chunk.ipynb 또는 청킹 파이프라인을 실행하세요."
            )

        self.chunks = load_jsonl(chunk_path)

        print("로드된 청크 수:", len(self.chunks))

        if self.chunks:
            print("첫 번째 청크 keys:", self.chunks[0].keys())

        return self.chunks

    @staticmethod
    def _get_chunk_text(chunk: Dict[str, Any]) -> str:
        """
        청크에서 본문 text를 안전하게 추출합니다.
        """
        for key in ["text", "page_content", "content", "chunk_text"]:
            value = chunk.get(key)

            if value:
                return str(value)

        return ""

    @staticmethod
    def _get_chunk_doc_id(chunk: Dict[str, Any]) -> str:
        """
        청크에서 doc_id를 안전하게 추출합니다.
        """
        if chunk.get("doc_id") is not None:
            return str(chunk["doc_id"])

        metadata = chunk.get("metadata", {}) or {}

        if metadata.get("doc_id") is not None:
            return str(metadata["doc_id"])

        return ""

    @staticmethod
    def _get_chunk_id(chunk: Dict[str, Any], idx: int) -> str:
        """
        청크에서 chunk_id를 안전하게 추출합니다.
        없으면 index 기반으로 생성합니다.
        """
        if chunk.get("chunk_id") is not None:
            return str(chunk["chunk_id"])

        metadata = chunk.get("metadata", {}) or {}

        if metadata.get("chunk_id") is not None:
            return str(metadata["chunk_id"])

        return f"chunk_{idx:06d}"

    def standardize_chunks(self) -> List[Dict[str, Any]]:
        """
        청크 필드를 표준화합니다.

        필수 필드:
        - chunk_id
        - doc_id
        - text
        """
        standard_chunks = []

        for idx, chunk in enumerate(self.chunks):
            text = self._get_chunk_text(chunk)
            doc_id = self._get_chunk_doc_id(chunk)
            chunk_id = self._get_chunk_id(chunk, idx)

            if not text.strip():
                continue

            if not doc_id.strip():
                continue

            standard_chunks.append(
                {
                    **chunk,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": text,
                }
            )

        self.standard_chunks = standard_chunks

        print("표준화 전 청크 수:", len(self.chunks))
        print("표준화 후 청크 수:", len(self.standard_chunks))

        return self.standard_chunks

    def print_chunk_stats(self) -> pd.DataFrame:
        """
        청크 통계를 출력하고 DataFrame을 반환합니다.
        """
        if not self.standard_chunks:
            raise RuntimeError(
                "standard_chunks가 비어 있습니다. standardize_chunks()를 먼저 호출하세요."
            )

        df = pd.DataFrame(
            [
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "file_type": chunk.get("file_type"),
                    "chunking_strategy": chunk.get(
                        "chunking_strategy",
                        chunk.get("chunking_method"),
                    ),
                    "text_len": len(chunk.get("text", "")),
                    "embedding_text_len": len(chunk.get("embedding_text", "")),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                }
                for chunk in self.standard_chunks
            ]
        )

        print("청크 수:", len(df))

        print("\ntext_len describe:")
        print(df["text_len"].describe())

        if "embedding_text_len" in df.columns:
            print("\nembedding_text_len describe:")
            print(df["embedding_text_len"].describe())

        print("\n문서별 청크 수 describe:")
        print(df.groupby("doc_id")["chunk_id"].count().describe())

        print("\nfile_type 분포:")
        print(df["file_type"].value_counts(dropna=False))

        return df

    def compute_chunk_fingerprint(self) -> Dict[str, Any]:
        """
        현재 standard_chunks의 내용을 기반으로 fingerprint를 계산합니다.

        목적:
        - section_chunks.jsonl의 경로가 같아도 내용이 바뀌면 감지
        - 청크 순서, chunk_id, doc_id, text, embedding_text가 바뀌면 vector DB 재생성
        - pdf_page 청킹의 page_start/page_end 변경도 감지
        """
        if not self.standard_chunks:
            raise RuntimeError(
                "standard_chunks가 비어 있습니다. "
                "load_chunks()와 standardize_chunks()를 먼저 호출하세요."
            )

        hasher = hashlib.sha256()

        for idx, chunk in enumerate(self.standard_chunks):
            record = {
                "order": idx,
                "chunk_id": str(chunk.get("chunk_id", "")),
                "doc_id": str(chunk.get("doc_id", "")),
                "text": str(chunk.get("text", "")),
                "embedding_text": str(chunk.get("embedding_text", "")),
                "file_type": str(chunk.get("file_type", "")),
                "chunking_strategy": str(
                    chunk.get("chunking_strategy", chunk.get("chunking_method", ""))
                ),
                "page_start": str(chunk.get("page_start", "")),
                "page_end": str(chunk.get("page_end", "")),
            }

            encoded = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")

            hasher.update(encoded)
            hasher.update(b"\n")

        fingerprint = {
            "chunk_count": len(self.standard_chunks),
            "chunk_content_sha256": hasher.hexdigest(),
            "chunk_path": str(self.paths["chunk_path"]),
        }

        return fingerprint

    def load_saved_chunk_fingerprint(self) -> Optional[Dict[str, Any]]:
        """
        기존 vector DB와 함께 저장된 chunk fingerprint를 로드합니다.
        """
        fingerprint_path = self.paths["chunk_fingerprint_path"]

        if not fingerprint_path.exists():
            return None

        with open(fingerprint_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_chunk_fingerprint(self, fingerprint: Dict[str, Any]) -> None:
        """
        새로 생성한 vector DB에 대응되는 chunk fingerprint를 저장합니다.
        """
        fingerprint_path = self.paths["chunk_fingerprint_path"]
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(
                fingerprint,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("chunk fingerprint 저장:", fingerprint_path)

    def should_rebuild_by_chunk_fingerprint(self) -> tuple[bool, List[str]]:
        """
        현재 청크 fingerprint와 저장된 fingerprint를 비교해서
        vector DB 재생성 여부를 판단합니다.
        """
        current_fingerprint = self.compute_chunk_fingerprint()
        saved_fingerprint = self.load_saved_chunk_fingerprint()

        reasons = []

        if saved_fingerprint is None:
            reasons.append("저장된 chunk fingerprint가 없음")
            return True, reasons

        if saved_fingerprint.get("chunk_count") != current_fingerprint.get("chunk_count"):
            reasons.append(
                "chunk_count 변경: "
                f"{saved_fingerprint.get('chunk_count')} -> {current_fingerprint.get('chunk_count')}"
            )

        if saved_fingerprint.get("chunk_content_sha256") != current_fingerprint.get("chunk_content_sha256"):
            reasons.append("chunk_content_sha256 변경")

        if saved_fingerprint.get("chunk_path") != current_fingerprint.get("chunk_path"):
            reasons.append(
                "chunk_path 변경: "
                f"{saved_fingerprint.get('chunk_path')} -> {current_fingerprint.get('chunk_path')}"
            )

        return len(reasons) > 0, reasons

    def clear_vector_db_files(self) -> None:
        """
        현재 provider/vector_db 조합에 해당하는 기존 vector DB 파일을 삭제합니다.

        Chroma 주의:
        - Chroma는 SQLite DB를 PersistentClient가 열고 있습니다.
        - client를 먼저 연 뒤 디렉토리를 삭제하면 readonly database 오류가 날 수 있습니다.
        - 따라서 chroma/faiss 로컬 디렉토리는 client.clear()를 부르지 않고
          디렉토리를 먼저 삭제한 뒤 vector_store 객체를 새로 만듭니다.
        """
        vector_type = self._vector_db_type()
        embedding_provider = self._embedding_provider()
        vector_dir = self.paths["vector_db_dir"]

        if vector_type in {"faiss", "chroma"}:
            self.vector_store = None
            _remove_dir_force(vector_dir)
            vector_dir.mkdir(parents=True, exist_ok=True)
            _assert_writable_dir(vector_dir)
            print("기존 vector DB 디렉토리 삭제 후 재생성:", vector_dir)
            return

        if (
            embedding_provider == "openai"
            and self.vector_store is not None
            and hasattr(self.vector_store, "clear")
        ):
            self.vector_store.clear()

        for path_key in ["chunk_fingerprint_path", "vector_config_path"]:
            path = self.paths.get(path_key)

            if path and path.exists():
                path.unlink()
                print("기존 vector DB 메타 파일 삭제:", path)

    def load_embedder(self, device: Optional[str] = None):
        """
        embedding.provider 값에 따라 HF 또는 OpenAI 임베딩 모델을 로드합니다.
        """
        embedding_cfg = self.config["embedding"]
        provider = self._embedding_provider()

        if provider == "openai":
            openai_cfg = self.config.get("openai", {})
            model_name = (
                embedding_cfg.get("openai_model_name")
                or openai_cfg.get("embedding_model")
            )

            self.embedder = OpenAIEmbeddingModel(
                model_name=model_name,
                api_key_env=openai_cfg.get("api_key_env", "OPENAI_API_KEY"),
            )

            print("OpenAI embedder 로드:", model_name)
            return self.embedder

        if provider != "huggingface":
            raise ValueError(f"Unsupported embedding.provider: {provider}")

        self.embedder = load_embedding_model(
            model_name=embedding_cfg["hf_model_name"],
            normalize_embeddings=embedding_cfg.get("normalize_embeddings", True),
            device=device,
            trust_remote_code=embedding_cfg.get("trust_remote_code", True),
        )

        print("HF embedder 로드:", embedding_cfg["hf_model_name"])
        return self.embedder

    def setup_vector_store(self):
        """
        embedding.provider와 vector_db.type에 맞춰 vector store를 생성합니다.

        - HuggingFace embedding: 기존 FAISSVectorStore 사용
        - OpenAI embedding: FAISS/Chroma/Qdrant/Supabase 중 선택
        """
        vector_cfg = self.config["vector_db"]
        vector_type = self._vector_db_type()
        store_cfg = vector_cfg.get("stores", {}).get(vector_type, {})
        embedding_provider = self._embedding_provider()

        if vector_type not in SUPPORTED_VECTOR_STORES:
            raise ValueError(f"Unsupported vector_db.type: {vector_type}")

        if embedding_provider == "huggingface":
            if vector_type != "faiss":
                raise ValueError(
                    "현재 HF embedding 경로는 기존 FAISSVectorStore를 사용하므로 "
                    "vector_db.type='faiss'만 지원합니다. "
                    "Chroma/Qdrant/Supabase 실험은 embedding.provider='openai'로 실행하세요."
                )

            self.vector_store = FAISSVectorStore(
                vector_dir=self.paths["vector_db_dir"],
                index_file=store_cfg.get("index_file", "index.faiss"),
                chunk_meta_file=store_cfg.get("chunk_meta_file", "chunks.pkl"),
                config_file=store_cfg.get("config_file", "config.json"),
            )

            return self.vector_store

        if embedding_provider != "openai":
            raise ValueError(f"Unsupported embedding.provider: {embedding_provider}")

        if vector_type == "faiss":
            self.vector_store = OpenAIFaissStore(
                persist_dir=self.paths["vector_db_dir"],
                index_file=store_cfg.get("index_file", "index.faiss"),
                chunk_meta_file=store_cfg.get("chunk_meta_file", "chunks.pkl"),
            )

        elif vector_type == "chroma":
            self.vector_store = OpenAIChromaStore(
                persist_dir=self.paths["vector_db_dir"],
                collection=f"{store_cfg.get('collection', 'rfp_openai_rag')}_{_safe_name(self.experiment_key)}",
            )

        elif vector_type == "qdrant":
            qdrant_cfg = dict(store_cfg)
            qdrant_cfg["collection"] = (
                f"{store_cfg.get('collection', 'rfp_openai_rag')}_{_safe_name(self.experiment_key)}"
            )
            self.vector_store = OpenAIQdrantStore(qdrant_cfg)

        elif vector_type == "supabase":
            self.vector_store = OpenAISupabaseStore(dict(store_cfg))

        return self.vector_store

    def build_or_load_vector_store(self):
        """
        provider별 vector DB를 로드하거나 새로 생성합니다.
        """
        if self.vector_store is None:
            self.setup_vector_store()

        if not self.standard_chunks:
            raise RuntimeError(
                "standard_chunks가 비어 있습니다. "
                "load_chunks()와 standardize_chunks()를 먼저 호출하세요."
            )

        if self._embedding_provider() == "openai":
            return self._build_or_load_openai_vector_store()

        return self._build_or_load_hf_vector_store()

    def _build_or_load_hf_vector_store(self):
        """기존 HuggingFace + FAISS 경로입니다."""
        embedding_cfg = self.config["embedding"]
        config_force_rebuild = embedding_cfg.get("force_rebuild_index", False)
        reload_query_embedder_on_cpu = embedding_cfg.get("reload_query_embedder_on_cpu", False)

        fingerprint_rebuild, fingerprint_reasons = self.should_rebuild_by_chunk_fingerprint()
        force_rebuild = config_force_rebuild or fingerprint_rebuild

        rebuild, reasons = self.vector_store.should_rebuild(
            current_config=self.config,
            force_rebuild=force_rebuild,
            keys_to_check=[
                "embedding.provider",
                "embedding.hf_model_name",
                "embedding.normalize_embeddings",
                "chunking.strategy",
                "paths.chunk_path",
            ],
        )

        reasons = list(reasons) + fingerprint_reasons

        if fingerprint_rebuild:
            print("청크 내용 변경 감지. 기존 vector DB를 교체합니다.")
            print("fingerprint reasons:", fingerprint_reasons)
            self.clear_vector_db_files()
            self.setup_vector_store()

        if rebuild:
            print("HF FAISS 인덱스 새로 생성")
            print("rebuild reasons:", reasons)

            if self.embedder is None:
                self.load_embedder(device=embedding_cfg.get("device"))

            embeddings = self.embedder.encode_chunks(
                chunks=self.standard_chunks,
                batch_size=embedding_cfg.get("batch_size", 32),
                show_progress=True,
                log_every=10,
            )

            self.vector_store.build(
                embeddings=embeddings,
                chunks=self.standard_chunks,
            )
            self.vector_store.save(config_snapshot=self.config)
            self.save_chunk_fingerprint(self.compute_chunk_fingerprint())
            self._save_vector_snapshot()

            del embeddings
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            if reload_query_embedder_on_cpu:
                print("FAISS 생성 완료. GPU embedding model 해제 후 CPU query embedder 재로드")

                if self.embedder is not None:
                    if hasattr(self.embedder, "unload"):
                        self.embedder.unload()
                    else:
                        del self.embedder

                    self.embedder = None
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                self.load_embedder(device="cpu")

        else:
            print("기존 HF FAISS 인덱스 로드")
            self.vector_store.load()

            if self.embedder is None:
                self.load_embedder(
                    device=(
                        "cpu"
                        if reload_query_embedder_on_cpu
                        else embedding_cfg.get("device")
                    )
                )

        return self.vector_store

    def _build_or_load_openai_vector_store(self):
        """OpenAI embedding + 선택 vector DB 경로입니다."""
        embedding_cfg = self.config["embedding"]
        force_rebuild = embedding_cfg.get("force_rebuild_index", False)

        if self.embedder is None:
            self.load_embedder()

        fingerprint_rebuild, fingerprint_reasons = self.should_rebuild_by_chunk_fingerprint()
        saved_snapshot = self._load_vector_snapshot()
        snapshot_rebuild = saved_snapshot != self._vector_snapshot()

        pre_rebuild = bool(
            force_rebuild
            or fingerprint_rebuild
            or snapshot_rebuild
        )

        if pre_rebuild:
            exists = False
        else:
            exists = self.vector_store.exists()

        rebuild = pre_rebuild or not exists

        reasons = []

        if force_rebuild:
            reasons.append("embedding.force_rebuild_index=True")

        if fingerprint_rebuild:
            reasons.extend(fingerprint_reasons)

        if snapshot_rebuild:
            reasons.append("embedding/vector snapshot 변경")

        if not exists:
            reasons.append("저장된 vector store 없음")

        if rebuild:
            print(f"OpenAI {self._vector_db_type()} vector store 새로 생성")
            print("rebuild reasons:", reasons)

            self.clear_vector_db_files()
            self.setup_vector_store()

            embeddings = self.embedder.encode_chunks(
                chunks=self.standard_chunks,
                batch_size=int(embedding_cfg.get("batch_size", 32)),
                show_progress=True,
                log_every=10,
            )

            self.vector_store.build(self.standard_chunks, embeddings)
            self.save_chunk_fingerprint(self.compute_chunk_fingerprint())
            self._save_vector_snapshot()

            del embeddings
            gc.collect()

        else:
            print(f"기존 OpenAI {self._vector_db_type()} vector store 로드")

            if hasattr(self.vector_store, "load"):
                self.vector_store.load()

        return self.vector_store

    def setup_retriever(self):
        """
        provider별 retriever를 준비합니다.

        OpenAI vector store는 run 단계에서 direct search를 사용하므로 별도 RAGRetriever를 만들지 않습니다.
        """
        if self.embedder is None:
            self.load_embedder()

        if self.vector_store is None:
            self.setup_vector_store()

        if self._embedding_provider() == "openai":
            if (
                self.vector_store is None
                or not hasattr(self.vector_store, "is_loaded")
                or not self.vector_store.is_loaded()
            ):
                self.build_or_load_vector_store()

            self.retriever = None
            return self.vector_store

        if not self.vector_store.is_loaded():
            self.build_or_load_vector_store()

        self.retriever = RAGRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.config["retrieval"]["top_k"],
        )

        return self.retriever

    def _retrieve(
        self,
        question: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if self._embedding_provider() == "openai":
            query_embedding = self.embedder.encode_query(question)
            return self.vector_store.search(query_embedding, top_k)

        if self.retriever is None:
            raise RuntimeError("retriever가 없습니다. setup_retriever()를 먼저 호출하세요.")

        return self.retriever.retrieve(query=question, top_k=top_k)

    def _get_retrieved_ids(
        self,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[str]:
        if self.retriever is not None and hasattr(self.retriever, "get_retrieved_ids"):
            return self.retriever.get_retrieved_ids(retrieved_chunks)

        return [str(chunk.get("doc_id", "")) for chunk in retrieved_chunks]

    def _get_retrieved_chunk_ids(
        self,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[str]:
        if self.retriever is not None and hasattr(self.retriever, "get_retrieved_chunk_ids"):
            return self.retriever.get_retrieved_chunk_ids(retrieved_chunks)

        return [str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks]

    def _get_retrieved_contexts(
        self,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[str]:
        if self.retriever is not None and hasattr(self.retriever, "get_retrieved_contexts"):
            return self.retriever.get_retrieved_contexts(retrieved_chunks)

        return [str(chunk.get("text", "")) for chunk in retrieved_chunks]

    def _compact_retrieved_chunks(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        max_text_chars: int = 1500,
    ) -> List[Dict[str, Any]]:
        if self.retriever is not None and hasattr(self.retriever, "compact_retrieved_chunks"):
            return self.retriever.compact_retrieved_chunks(
                retrieved_chunks,
                max_text_chars=max_text_chars,
            )

        compact_rows = []

        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {}) or {}

            compact_rows.append(
                {
                    "rank": chunk.get("rank"),
                    "score": chunk.get("score"),
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "file_name": chunk.get("file_name") or metadata.get("file_name"),
                    "project_name": chunk.get("project_name") or metadata.get("project_name"),
                    "organization": chunk.get("organization") or metadata.get("organization"),
                    "page_start": chunk.get("page_start") or metadata.get("page_start"),
                    "page_end": chunk.get("page_end") or metadata.get("page_end"),
                    "text": str(chunk.get("text", ""))[:max_text_chars],
                    "metadata": metadata,
                }
            )

        return compact_rows

    def _get_first_eval_value(
        self,
        item: Dict[str, Any],
        keys: Sequence[str],
        default: str = "",
    ) -> str:
        """
        평가 item에서 여러 후보 key 중 첫 번째 유효 값을 문자열로 반환합니다.
        """
        for key in keys:
            value = item.get(key)

            if value is None:
                continue

            text = str(value).strip()

            if text and text.lower() != "nan":
                return text

        return default

    def build_eval_retrieval_query(
        self,
        eval_item: Dict[str, Any],
    ) -> str:
        """
        자동 평가용 retrieval query를 생성합니다.

        자동 평가 질문에는 "이 사업", "이 제안요청서"처럼 지시어가 많습니다.
        전체 문서 검색에서 정답 문서를 더 잘 찾기 위해 검색 단계에만
        기관명, 사업명, doc_id를 함께 넣습니다.

        주의:
        - 이 query는 retrieval에만 사용합니다.
        - LLM 생성에는 원래 question을 그대로 사용합니다.
        """
        question = self._get_first_eval_value(
            eval_item,
            ["question", "질문"],
        )

        organization = self._get_first_eval_value(
            eval_item,
            ["organization", "발주 기관", "기관명"],
        )

        project_name = self._get_first_eval_value(
            eval_item,
            ["project_name", "사업명", "project"],
        )

        doc_id = self._get_first_eval_value(
            eval_item,
            ["doc_id", "공고 번호"],
        )

        parts = []

        if organization:
            parts.append(f"기관명: {organization}")

        if project_name:
            parts.append(f"사업명: {project_name}")

        if doc_id:
            parts.append(f"doc_id: {doc_id}")

        parts.append(f"질문: {question}")

        return "\n".join(parts)

    def load_generator(self):
        """
        llm.provider 값에 따라 HF 또는 OpenAI generator를 로드합니다.

        HuggingFace provider에서는 선택적으로 bitsandbytes 4bit quantization을 사용할 수 있습니다.

        YAML 예:
        llm:
          provider: huggingface
          hf_model_name: Qwen/Qwen3-14B
          load_in_4bit: true
          bnb_4bit_quant_type: nf4
          bnb_4bit_compute_dtype: float16
          bnb_4bit_use_double_quant: true

        load_in_4bit가 없거나 false이면 기존 FP16/FP32 로딩 방식으로 동작합니다.
        """
        llm_cfg = self.config["llm"]
        provider = self._llm_provider()

        if provider == "openai":
            openai_cfg = self.config.get("openai", {})
            model_name = (
                llm_cfg.get("openai_model_name")
                or openai_cfg.get("llm_model")
            )

            self.generator = OpenAILLMGenerator(
                model_name=model_name,
                fallback_model_name=(
                    llm_cfg.get("fallback_model_name")
                    or openai_cfg.get("fallback_llm_model")
                ),
                api_key_env=openai_cfg.get("api_key_env", "OPENAI_API_KEY"),
                max_output_tokens=llm_cfg.get(
                    "max_output_tokens",
                    llm_cfg.get("max_new_tokens", 512),
                ),
                temperature=llm_cfg.get("temperature", 0.0),
                prompt_type=llm_cfg.get("prompt_type", "default"),
                max_chars_per_chunk=llm_cfg.get("max_chars_per_chunk"),
                include_metadata=llm_cfg.get("include_metadata", True),
            )

            print("OpenAI generator 로드:", model_name)
            return self.generator

        if provider != "huggingface":
            raise ValueError(f"Unsupported llm.provider: {provider}")

        self.generator = load_llm_generator(
            model_name=llm_cfg["hf_model_name"],
            max_new_tokens=llm_cfg.get(
                "max_new_tokens",
                llm_cfg.get("max_output_tokens", 512),
            ),
            temperature=llm_cfg.get("temperature", 0.0),
            do_sample=llm_cfg.get("do_sample", False),
            trust_remote_code=llm_cfg.get("trust_remote_code", True),
            prompt_type=llm_cfg.get("prompt_type", "default"),
            max_chars_per_chunk=llm_cfg.get("max_chars_per_chunk"),
            include_metadata=llm_cfg.get("include_metadata", True),

            # =====================================================
            # bitsandbytes 4bit quantization options
            # =====================================================
            # 기존 모델은 load_in_4bit가 없거나 false이면 그대로 동작합니다.
            # Qwen3-14B처럼 22GB GPU에서 FP16 OOM이 나는 모델은 true로 설정합니다.
            load_in_4bit=llm_cfg.get("load_in_4bit", False),
            bnb_4bit_quant_type=llm_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=llm_cfg.get(
                "bnb_4bit_compute_dtype",
                "float16",
            ),
            bnb_4bit_use_double_quant=llm_cfg.get(
                "bnb_4bit_use_double_quant",
                True,
            ),
        )

        print("HF generator 로드:", llm_cfg["hf_model_name"])
        print("HF generator 4bit:", llm_cfg.get("load_in_4bit", False))

        return self.generator

    def run_single_rag(
        self,
        eval_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        평가 문항 1개에 대해 RAG를 실행합니다.

        자동 평가에서는 retrieval query와 generation question을 분리합니다.

        - retrieval_query:
          기관명/사업명/doc_id/질문을 함께 넣어 정답 문서 검색률을 높입니다.

        - question:
          LLM 답변 생성에는 원래 질문을 그대로 사용합니다.
        """
        if self.vector_store is None:
            raise RuntimeError("vector_store가 없습니다. setup_retriever()를 먼저 호출하세요.")

        if self.generator is None:
            raise RuntimeError("generator가 없습니다. load_generator()를 먼저 호출하세요.")

        question = eval_item["question"]
        top_k = int(self.config["retrieval"]["top_k"])

        retrieval_query = self.build_eval_retrieval_query(eval_item)

        start_total = time.perf_counter()

        start_retrieval = time.perf_counter()
        retrieved_chunks = self._retrieve(retrieval_query, top_k=top_k)
        retrieval_latency_sec = time.perf_counter() - start_retrieval

        generation_result = self.generator.generate_from_retrieved_chunks(
            question=question,
            retrieved_chunks=retrieved_chunks,
            return_prompt=False,
        )

        total_latency_sec = time.perf_counter() - start_total

        result = {
            **eval_item,
            "retrieval_query": retrieval_query,
            "embedding_provider": self._embedding_provider(),
            "embedding_model": self._active_embedding_model_name(),
            "llm_provider": self._llm_provider(),
            "llm_model": generation_result.get("model_name", self._active_llm_model_name()),
            "vector_db_type": self._vector_db_type(),
            "experiment_key": self.experiment_key,
            "retrieved_ids": self._get_retrieved_ids(retrieved_chunks),
            "retrieved_chunk_ids": self._get_retrieved_chunk_ids(retrieved_chunks),
            "retrieved_chunks": self._compact_retrieved_chunks(
                retrieved_chunks,
                max_text_chars=1500,
            ),
            "retrieved_contexts": self._get_retrieved_contexts(retrieved_chunks),
            "response": generation_result["response"],
            "retrieval_latency_sec": retrieval_latency_sec,
            "generation_latency_sec": generation_result["generation_latency_sec"],
            "total_latency_sec": total_latency_sec,
            "input_tokens": generation_result.get("input_tokens", 0),
            "output_tokens": generation_result.get("output_tokens", 0),
            "total_tokens": generation_result.get("total_tokens", 0),
            "estimated_cost": self._estimate_generation_cost(generation_result),
        }

        return result

    def _estimate_generation_cost(
        self,
        generation_result: Dict[str, Any],
    ) -> float:
        """
        OpenAI 모델일 때만 대략 비용을 계산합니다.
        config.openai.pricing에 모델별 단가를 넣으면 그 값을 우선 사용합니다.
        단위: USD / 1 token.
        """
        if self._llm_provider() != "openai":
            return 0.0

        openai_cfg = self.config.get("openai", {})
        pricing_cfg = openai_cfg.get("pricing", {})
        model_name = (
            generation_result.get("model_name")
            or self._active_llm_model_name()
        )

        default_pricing = {
            "gpt-5-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-5-nano": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
        }

        pricing = pricing_cfg.get(model_name) or default_pricing.get(model_name)

        if not pricing:
            return 0.0

        return (
            float(generation_result.get("input_tokens", 0)) * float(pricing.get("input", 0.0))
            + float(generation_result.get("output_tokens", 0)) * float(pricing.get("output", 0.0))
        )

    def run_user_query(
        self,
        question: str,
        log_human_eval: bool = True,
        human_eval_csv: str | Path = "outputs/human_eval/real_user_eval_sheet.csv",
    ) -> Dict[str, Any]:
        """
        실제 사용자 질문 1개에 대해 RAG를 실행합니다.

        실제 사용자 질문은 자동 평가셋과 달리 별도의 doc_id/project_name row가 없으므로
        원래 question을 그대로 retrieval에 사용합니다.
        """
        if self.vector_store is None:
            self.setup_retriever()

        if self.generator is None:
            self.load_generator()

        top_k = int(self.config["retrieval"]["top_k"])
        start_total = time.perf_counter()

        start_retrieval = time.perf_counter()
        retrieved_chunks = self._retrieve(question, top_k=top_k)
        retrieval_latency_sec = time.perf_counter() - start_retrieval

        generation_result = self.generator.generate_from_retrieved_chunks(
            question=question,
            retrieved_chunks=retrieved_chunks,
            return_prompt=False,
        )

        total_latency_sec = time.perf_counter() - start_total

        result = {
            "question": question,
            "embedding_provider": self._embedding_provider(),
            "embedding_model": self._active_embedding_model_name(),
            "llm_provider": self._llm_provider(),
            "llm_model": generation_result.get("model_name", self._active_llm_model_name()),
            "vector_db_type": self._vector_db_type(),
            "experiment_key": self.experiment_key,
            "retrieved_ids": self._get_retrieved_ids(retrieved_chunks),
            "retrieved_chunk_ids": self._get_retrieved_chunk_ids(retrieved_chunks),
            "retrieved_chunks": self._compact_retrieved_chunks(
                retrieved_chunks,
                max_text_chars=1500,
            ),
            "retrieved_contexts": self._get_retrieved_contexts(retrieved_chunks),
            "response": generation_result["response"],
            "retrieval_latency_sec": retrieval_latency_sec,
            "generation_latency_sec": generation_result["generation_latency_sec"],
            "total_latency_sec": total_latency_sec,
            "input_tokens": generation_result.get("input_tokens", 0),
            "output_tokens": generation_result.get("output_tokens", 0),
            "total_tokens": generation_result.get("total_tokens", 0),
            "estimated_cost": self._estimate_generation_cost(generation_result),
        }

        if log_human_eval:
            if self.evaluator is None:
                self.setup_evaluator()

            output_csv = resolve_project_path(
                self.project_root,
                human_eval_csv,
            )

            self.evaluator.log_for_human_eval(
                question=question,
                rag_result=result,
                output_csv=str(output_csv),
            )

        return result

    def run_human_eval_queries_if_enabled(self) -> List[Dict[str, Any]]:
        """
        팀원 수동 평가용 실제 사용자 질문 리스트를 실행합니다.

        실행 시점:
        - 자동 평가셋 RAG 실행과 evaluate()가 끝난 직후
        - pipeline.cleanup()이 호출되기 전
        """
        human_eval_cfg = self.config.get("human_eval", {})

        if not human_eval_cfg.get("enabled", False):
            print("Human eval query logging skipped: human_eval.enabled=False")
            return []

        output_csv = human_eval_cfg.get(
            "output_csv",
            "outputs/human_eval/real_user_eval_sheet.csv",
        )

        questions = HUMAN_EVAL_QUESTIONS

        if not questions:
            print("Human eval query logging skipped: HUMAN_EVAL_QUESTIONS is empty")
            return []

        print("\n===== Run Human Eval Queries =====")
        print("reuse existing retriever/generator/vector DB")
        print("num_questions:", len(questions))
        print("output_csv:", output_csv)

        human_eval_outputs = []

        for question in progress_iter(
            questions,
            total=len(questions),
            desc="Running human eval queries",
            log_every=1,
            min_interval_sec=0.0,
        ):
            try:
                result = self.run_user_query(
                    question=question,
                    log_human_eval=True,
                    human_eval_csv=output_csv,
                )
                human_eval_outputs.append(result)

            except Exception as e:
                print(f"[Human Eval][ERROR] question={question} | error={repr(e)}")

                human_eval_outputs.append(
                    {
                        "question": question,
                        "retrieved_ids": [],
                        "retrieved_chunk_ids": [],
                        "retrieved_chunks": [],
                        "retrieved_contexts": [],
                        "response": "",
                        "error": repr(e),
                        "retrieval_latency_sec": 0.0,
                        "generation_latency_sec": 0.0,
                        "total_latency_sec": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost": 0.0,
                    }
                )

        print("Human eval query logging 완료:", output_csv)

        return human_eval_outputs

    def run_rag_on_sample(self) -> List[Dict[str, Any]]:
        """
        sample_eval_dataset 전체에 대해 RAG를 실행하고 결과를 저장합니다.
        """
        if not self.sample_eval_dataset:
            raise RuntimeError(
                "sample_eval_dataset이 비어 있습니다. load_or_create_eval_sample()을 먼저 호출하세요."
            )

        if self.embedder is None:
            self.load_embedder()

        if self.vector_store is None:
            self.setup_vector_store()

        if (
            self.vector_store is None
            or not hasattr(self.vector_store, "is_loaded")
            or not self.vector_store.is_loaded()
        ):
            self.build_or_load_vector_store()

        if self._embedding_provider() != "openai" and self.retriever is None:
            self.setup_retriever()

        if self.generator is None:
            self.load_generator()

        self._ensure_output_dirs()

        rag_outputs = []

        for item in progress_iter(
            self.sample_eval_dataset,
            total=len(self.sample_eval_dataset),
            desc="Running RAG evaluation sample",
            log_every=1,
            min_interval_sec=0.0,
        ):
            try:
                result = self.run_single_rag(item)
                rag_outputs.append(result)

            except Exception as e:
                error_result = {
                    **item,
                    "retrieval_query": "",
                    "retrieved_ids": [],
                    "retrieved_chunk_ids": [],
                    "retrieved_chunks": [],
                    "retrieved_contexts": [],
                    "response": "",
                    "error": repr(e),
                    "retrieval_latency_sec": 0.0,
                    "generation_latency_sec": 0.0,
                    "total_latency_sec": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0,
                }
                rag_outputs.append(error_result)

        self.rag_outputs = rag_outputs
        self._save_rag_outputs()

        return self.rag_outputs

    def setup_evaluator(self) -> RAGEvaluator:
        """
        RAGEvaluator를 생성합니다.
        """
        self.evaluator = RAGEvaluator(
            auto_download_nltk=False,
            use_nltk_tokenizer=False,
        )

        return self.evaluator

    def evaluate(self) -> Dict[str, Any]:
        """
        rag_outputs에 대해 전체 평가를 수행하고 결과를 저장합니다.
        """
        if not self.rag_outputs:
            raise RuntimeError("rag_outputs가 비어 있습니다. run_rag_on_sample()을 먼저 호출하세요.")

        if self.evaluator is None:
            self.setup_evaluator()

        top_k = self.config["retrieval"]["top_k"]

        with log_step("Evaluate all metrics"):
            metrics = self.evaluator.evaluate_all(
                self.rag_outputs,
                k=top_k,
            )

        valid_costs = [
            row.get("estimated_cost", 0.0)
            for row in self.rag_outputs
            if "error" not in row
        ]

        total_cost = float(sum(valid_costs))
        avg_cost_per_query = total_cost / len(valid_costs) if valid_costs else 0.0

        metrics["total_cost"] = total_cost
        metrics["avg_cost_per_query"] = avg_cost_per_query

        self._save_metrics_safe(
            metrics,
            self.paths["metrics_path"],
            label="전체 metrics",
        )

        with log_step("Evaluate by question_type"):
            by_question_type = self.evaluator.evaluate_by_group(
                self.rag_outputs,
                group_key="question_type",
                k=top_k,
            )

        self._save_metrics_safe(
            by_question_type,
            self.paths["metrics_by_question_type_path"],
            label="question_type metrics",
        )

        with log_step("Evaluate by source_type"):
            by_source_type = self.evaluator.evaluate_by_group(
                self.rag_outputs,
                group_key="source_type",
                k=top_k,
            )

        self._save_metrics_safe(
            by_source_type,
            self.paths["metrics_by_source_type_path"],
            label="source_type metrics",
        )

        with log_step("Evaluate by answer_format"):
            by_answer_format = self.evaluator.evaluate_by_group(
                self.rag_outputs,
                group_key="answer_format",
                k=top_k,
            )

        self._save_metrics_safe(
            by_answer_format,
            self.paths["metrics_by_answer_format_path"],
            label="answer_format metrics",
        )

        with log_step("Evaluate by file_type"):
            by_file_type = self.evaluator.evaluate_by_group(
                self.rag_outputs,
                group_key="file_type",
                k=top_k,
            )

        self._save_metrics_safe(
            by_file_type,
            self.paths["metrics_by_file_type_path"],
            label="file_type metrics",
        )

        with log_step("Extract retrieval failure cases"):
            retrieval_failures = self.evaluator.get_retrieval_failure_cases(
                self.rag_outputs,
                k=top_k,
            )

        self.evaluator.save_rows_as_csv(
            retrieval_failures,
            str(self.paths["retrieval_failure_path"]),
        )

        with log_step("Extract keyword failure cases"):
            keyword_failures = self.evaluator.get_keyword_failure_cases(
                self.rag_outputs,
                threshold=self.config["evaluation"].get("keyword_failure_threshold", 1.0),
            )

        self.evaluator.save_rows_as_csv(
            keyword_failures,
            str(self.paths["keyword_failure_path"]),
        )

        with log_step("Attach keyword scores"):
            self.scored_outputs = self.evaluator.attach_keyword_scores(
                self.rag_outputs,
            )

        self._save_scored_outputs()

        summary_df = self._save_summary_csv(self.scored_outputs)

        experiment_summary = {
            "experiment_key": self.experiment_key,
            "embedding_provider": self._embedding_provider(),
            "embedding_model": self._active_embedding_model_name(),
            "llm_provider": self._llm_provider(),
            "llm_model": self._active_llm_model_name(),
            "vector_db_type": self._vector_db_type(),
            "config": self.config,
            "paths": {key: str(value) for key, value in self.paths.items()},
            "metrics": metrics,
            "num_rag_outputs": len(self.rag_outputs),
            "num_retrieval_failures": len(retrieval_failures),
            "num_keyword_failures": len(keyword_failures),
            "total_cost": metrics.get("total_cost", 0.0),
            "avg_cost_per_query": metrics.get("avg_cost_per_query", 0.0),
        }

        self._save_metrics_safe(
            experiment_summary,
            self.paths["experiment_summary_path"],
            label="experiment_summary",
        )

        print("평가 완료")
        print("metrics:", metrics)
        print("retrieval_failures:", len(retrieval_failures))
        print("keyword_failures:", len(keyword_failures))

        return {
            "metrics": metrics,
            "by_question_type": by_question_type,
            "by_source_type": by_source_type,
            "by_answer_format": by_answer_format,
            "by_file_type": by_file_type,
            "retrieval_failures": retrieval_failures,
            "keyword_failures": keyword_failures,
            "summary_df": summary_df,
        }

    def _save_summary_csv(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        사람이 보기 좋은 요약 CSV를 생성합니다.
        """
        top_k = self.config["retrieval"]["top_k"]
        summary_rows = []

        for row in rows:
            retrieved_ids_topk = row.get("retrieved_ids", [])[:top_k]

            summary_rows.append(
                {
                    "qid": row.get("qid"),
                    "doc_id": row.get("doc_id"),
                    "question_type": row.get("question_type"),
                    "source_type": row.get("source_type"),
                    "answer_format": row.get("answer_format"),
                    "file_type": row.get("file_type"),
                    "project_name": row.get("project_name"),
                    "organization": row.get("organization"),
                    "question": row.get("question"),
                    "retrieval_query": row.get("retrieval_query"),
                    "reference": row.get("reference"),
                    "response": row.get("response"),
                    "embedding_provider": row.get("embedding_provider"),
                    "embedding_model": row.get("embedding_model"),
                    "llm_provider": row.get("llm_provider"),
                    "llm_model": row.get("llm_model"),
                    "vector_db_type": row.get("vector_db_type"),
                    "experiment_key": row.get("experiment_key"),
                    "retrieved_ids": str(row.get("retrieved_ids", [])),
                    "retrieval_hit": row.get("doc_id") in set(retrieved_ids_topk),
                    "keyword_group_recall": row.get("keyword_group_recall"),
                    "matched_keyword_group_count": row.get("matched_keyword_group_count"),
                    "total_keyword_group_count": row.get("total_keyword_group_count"),
                    "missed_groups": str(row.get("missed_groups", [])),
                    "retrieval_latency_sec": row.get("retrieval_latency_sec"),
                    "generation_latency_sec": row.get("generation_latency_sec"),
                    "total_latency_sec": row.get("total_latency_sec"),
                    "input_tokens": row.get("input_tokens"),
                    "output_tokens": row.get("output_tokens"),
                    "estimated_cost": row.get("estimated_cost", 0.0),
                    "error": row.get("error", ""),
                }
            )

        summary_df = pd.DataFrame(summary_rows)

        self.paths["summary_csv_path"].parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(
            self.paths["summary_csv_path"],
            index=False,
            encoding="utf-8-sig",
        )

        print("요약 CSV 저장:", self.paths["summary_csv_path"])

        return summary_df

    def run(self) -> Dict[str, Any]:
        """
        전체 RAG 평가 파이프라인을 실행합니다.
        """
        self.print_summary()
        self.setup_runtime()

        self.load_eval_dataset()
        self.load_or_create_eval_sample()

        self.load_chunks()
        self.standardize_chunks()
        self.print_chunk_stats()

        self.load_embedder()
        self.setup_vector_store()
        self.build_or_load_vector_store()

        if self._embedding_provider() != "openai":
            self.setup_retriever()

        self.load_generator()
        self.run_rag_on_sample()

        results = self.evaluate()

        human_eval_outputs = self.run_human_eval_queries_if_enabled()
        results["human_eval_outputs"] = human_eval_outputs

        return results

    def cleanup(self) -> None:
        """
        GPU/CPU 메모리 정리를 수행합니다.
        """
        if self.generator is not None:
            self.generator.unload()
            self.generator = None

        if self.embedder is not None:
            if hasattr(self.embedder, "unload"):
                self.embedder.unload()
            else:
                del self.embedder

            self.embedder = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("Pipeline cleanup 완료")