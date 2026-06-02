from __future__ import annotations

import copy
import gc
import json
import math
import os
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.evaluation.evaluator import RAGEvaluator
from src.utils.config_utils import load_yaml_config, resolve_project_path
from src.utils.eval_dataset_utils import create_and_save_eval_sample, load_json, save_json
from src.utils.file_utils import load_jsonl
try:
    from src.utils.seed import set_seed
except ImportError:
    pass


SUPPORTED_VECTOR_STORES = ("faiss", "chroma", "qdrant", "supabase")
DEFAULT_MATRIX = [
    ("gpt-5-mini", "qdrant"),
    ("gpt-5-mini", "supabase"),
    ("gpt-5-mini", "faiss"),
    ("gpt-5-mini", "chroma"),
    ("gpt-5-nano", "qdrant"),
    ("gpt-5-nano", "supabase"),
    ("gpt-5-nano", "faiss"),
    ("gpt-5-nano", "chroma"),
]


class OpenAIEmbeddingModel:
    def __init__(self, model_name: str, api_key_env: str = "OPENAI_API_KEY"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAI RAG experiments.") from exc

        if not os.getenv(api_key_env):
            raise RuntimeError(f"{api_key_env} is not set.")

        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv(api_key_env))

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def encode_query(self, query: str) -> List[float]:
        return self.encode_texts([query], batch_size=1)[0]


class SimpleBM25Index:
    """
    외부 의존성 없이 동작하는 간단한 BM25 인덱스입니다.

    목적:
    - Dense vector search가 놓치는 정확한 키워드 매칭을 보완합니다.
    - RFP 문서의 '입찰참가자격', '평가기준', '제출기한' 같은 용어 검색에 유리합니다.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        text_key: str = "embedding_text",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunks = chunks
        self.text_key = text_key
        self.k1 = k1
        self.b = b

        self.doc_term_freqs: List[Counter[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_freqs: Counter[str] = Counter()
        self.idf: Dict[str, float] = {}
        self.avgdl = 0.0

        self._build()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = str(text or "").lower()
        return re.findall(r"[가-힣A-Za-z0-9]+", text)

    def _chunk_text(self, chunk: Dict[str, Any]) -> str:
        return str(chunk.get(self.text_key) or chunk.get("text") or "")

    def _build(self) -> None:
        total_len = 0

        for chunk in self.chunks:
            tokens = self.tokenize(self._chunk_text(chunk))
            tf = Counter(tokens)

            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)

            for term in tf.keys():
                self.doc_freqs[term] += 1

        doc_count = len(self.chunks)
        self.avgdl = total_len / doc_count if doc_count else 0.0

        self.idf = {
            term: math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freqs.items()
        }

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_terms = self.tokenize(query)

        if not query_terms or not self.chunks:
            return []

        query_tf = Counter(query_terms)
        scores: List[tuple[int, float]] = []

        for doc_idx, tf in enumerate(self.doc_term_freqs):
            doc_len = self.doc_lengths[doc_idx]
            score = 0.0

            for term, q_count in query_tf.items():
                if term not in tf:
                    continue

                term_freq = tf[term]
                idf = self.idf.get(term, 0.0)
                denom = term_freq + self.k1 * (
                    1.0 - self.b + self.b * doc_len / (self.avgdl or 1.0)
                )
                score += q_count * idf * (term_freq * (self.k1 + 1.0)) / (denom or 1.0)

            if score > 0:
                scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        rows = []
        for rank, (doc_idx, score) in enumerate(scores[:top_k], start=1):
            chunk = dict(self.chunks[doc_idx])
            chunk["rank"] = rank
            chunk["bm25_score"] = float(score)
            rows.append(chunk)

        return rows


class CrossEncoderReranker:
    """
    sentence-transformers CrossEncoder 기반 reranker입니다.

    config에서 retrieval.reranker.enabled=true일 때만 로드됩니다.
    예: upskyy/ko-reranker-8k, BAAI/bge-reranker-v2-m3 등
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Reranker를 사용하려면 sentence-transformers가 필요합니다. "
                "설치: pip install sentence-transformers"
            ) from exc

        kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "max_length": max_length,
        }
        if device:
            kwargs["device"] = device

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(**kwargs)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        pairs = [
            (
                query,
                str(chunk.get("embedding_text") or chunk.get("text") or ""),
            )
            for chunk in chunks
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        final = reranked[:top_k]
        for rank, item in enumerate(final, start=1):
            item["rank"] = rank

        return final


class OpenAILLMGenerator:
    def __init__(
        self,
        model_name: str,
        fallback_model_name: Optional[str],
        api_key_env: str,
        max_output_tokens: int,
        temperature: float,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAI RAG experiments.") from exc

        if not os.getenv(api_key_env):
            raise RuntimeError(f"{api_key_env} is not set.")

        self.client = OpenAI(api_key=os.getenv(api_key_env))
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def generate_from_retrieved_chunks(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(question, retrieved_chunks)
        start = time.perf_counter()
        try:
            response, usage = self._call_model(self.model_name, prompt)
        except Exception:
            if not self.fallback_model_name:
                raise
            response, usage = self._call_model(self.fallback_model_name, prompt)

        latency = time.perf_counter() - start
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        return {
            "response": response,
            "generation_latency_sec": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _call_model(self, model: str, prompt: str) -> tuple[str, Dict[str, int]]:
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
        return text, usage

    @staticmethod
    def _build_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_blocks = []
    
        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {}) or {}
    
            doc_id = chunk.get("doc_id") or metadata.get("doc_id", "")
            chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id", "")
    
            project_name = chunk.get("project_name") or metadata.get("project_name", "")
            organization = chunk.get("organization") or metadata.get("organization", "")
            section_title = chunk.get("section_title") or metadata.get("section_title", "")
            section_path = chunk.get("section_path") or metadata.get("section_path", "")
    
            if isinstance(section_path, list):
                section_path = " > ".join(str(x) for x in section_path if x)
    
            header = (
                f"[Evidence {index}] "
                f"doc_id={doc_id} chunk_id={chunk_id}\n"
                f"사업명: {project_name}\n"
                f"발주기관: {organization}\n"
                f"섹션경로: {section_path}\n"
                f"섹션제목: {section_title}\n"
            )
    
            context_blocks.append(
                f"{header}\n{chunk.get('text', '')}"
            )
    
        contexts = "\n\n".join(context_blocks)
    
        return (
            "You are a Korean RAG assistant for analyzing RFP documents.\n"
            "Answer in Korean. Use only the evidence below. If the evidence is insufficient, say so.\n\n"
            f"{contexts}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )


class OpenAIFaissStore:
    def __init__(self, persist_dir: Path, index_file: str, chunk_meta_file: str):
        self.persist_dir = persist_dir
        self.index_path = persist_dir / index_file
        self.chunk_meta_path = persist_dir / chunk_meta_file
        self.index = None
        self.chunks: List[Dict[str, Any]] = []

    def exists(self) -> bool:
        return self.index_path.exists() and self.chunk_meta_path.exists()

    def build(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        import faiss
        import numpy as np

        vectors = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.chunk_meta_path.open("wb") as handle:
            pickle.dump(chunks, handle)

    def load(self) -> None:
        import faiss
        self.index = faiss.read_index(str(self.index_path))
        with self.chunk_meta_path.open("rb") as handle:
            self.chunks = pickle.load(handle)

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        import faiss
        import numpy as np

        if self.index is None:
            self.load()
        vector = np.asarray([query_embedding], dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, top_k)
        return [
            {**self.chunks[int(idx)], "rank": rank, "score": float(score)}
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1)
            if idx >= 0
        ]


class OpenAIChromaStore:
    def __init__(self, persist_dir: Path, collection: str):
        self.persist_dir = persist_dir
        self.collection_name = collection

    def _collection(self):
        import chromadb
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        return client.get_or_create_collection(self.collection_name)

    def exists(self) -> bool:
        return self.persist_dir.exists()

    def build(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        collection = self._collection()
        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        if ids:
            try:
                collection.delete(ids=ids)
            except Exception:
                pass
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[_flat_metadata(chunk) for chunk in chunks],
        )

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        result = self._collection().query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        for rank, chunk_id in enumerate(result.get("ids", [[]])[0], start=1):
            metadata = result.get("metadatas", [[]])[0][rank - 1] or {}
            rows.append(
                {
                    **metadata,
                    "chunk_id": chunk_id,
                    "doc_id": str(metadata.get("doc_id", "")),
                    "text": result.get("documents", [[]])[0][rank - 1],
                    "rank": rank,
                    "score": result.get("distances", [[]])[0][rank - 1],
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
        return QdrantClient(url=self.url, api_key=self.api_key)

    def exists(self) -> bool:
        try:
            self._client().get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def build(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams
        client = self._client()
        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=index, vector=embedding, payload={"chunk": chunk})
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        hits = self._client().search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )
        rows = []
        for rank, hit in enumerate(hits, start=1):
            chunk = dict(hit.payload.get("chunk", {}))
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
            raise RuntimeError("Supabase URL/key environment variables are not set.")
        from supabase import create_client
        return create_client(self.url, self.key)

    def exists(self) -> bool:
        return bool(self.url and self.key)

    def build(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
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
            client.table(self.table).upsert(rows[start : start + 100]).execute()

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        result = self._client().rpc(
            self.match_function,
            {"query_embedding": query_embedding, "match_count": top_k},
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


class OpenAIRAGEvalPipeline:
    def __init__(
        self,
        config_path: str | Path,
        project_root: str | Path,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self.project_root = Path(project_root)
        self.config_path = resolve_project_path(self.project_root, config_path)
        self.config = load_yaml_config(self.config_path)
        if overrides:
            self._apply_overrides(overrides)

        self.paths: Dict[str, Path] = {}
        self.chunks: List[Dict[str, Any]] = []
        self.eval_dataset: List[Dict[str, Any]] = []
        self.sample_eval_dataset: List[Dict[str, Any]] = []
        self.rag_outputs: List[Dict[str, Any]] = []
        self.scored_outputs: List[Dict[str, Any]] = []
        self.embedder = None
        self.generator = None
        self.vector_store = None
        self.bm25_index = None
        self.reranker = None
        self.evaluator = None

        self._resolve_paths()

    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        llm_model = overrides.get("llm_model")
        vector_db_type = overrides.get("vector_db_type")
        experiment_name = overrides.get("experiment_name")
        embedding_model = overrides.get("embedding_model")
    
        if llm_model:
            self.config.setdefault("llm", {})["openai_model_name"] = llm_model
            self.config.setdefault("openai", {})["llm_model"] = llm_model
    
        if embedding_model:
            self.config.setdefault("embedding", {})["openai_model_name"] = embedding_model
            self.config.setdefault("openai", {})["embedding_model"] = embedding_model
    
        if vector_db_type:
            self.config.setdefault("vector_db", {})["type"] = vector_db_type
    
        if experiment_name:
            self.config.setdefault("experiment", {})["name"] = experiment_name
    
    
    @staticmethod
    def _safe_name(value: str, max_len: int = 40) -> str:
        """
        파일/폴더/collection 이름으로 안전하게 쓸 수 있는 짧은 slug 생성.
        """
        value = str(value or "").strip()

        if not value:
            return "unknown"

        value = value.replace("/", "_")
        value = value.replace("\\", "_")
        value = re.sub(r"\s+", "_", value)
        value = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value)
        value = re.sub(r"_+", "_", value)
        value = value.strip("._-")

        if not value:
            return "unknown"

        return value[:max_len].strip("._-") or "unknown"


    @staticmethod
    def _short_model_name(model_name: str, max_len: int = 24) -> str:
        """
        모델명을 저장용 짧은 이름으로 변환.

        예:
        text-embedding-3-small -> emb3s
        text-embedding-3-large -> emb3l
        gpt-5-mini -> gpt5mini
        gpt-5-nano -> gpt5nano
        LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct -> EXAONE-3.5-7.8B
        """
        raw = str(model_name or "").strip()

        if not raw:
            return "unknown"

        aliases = {
            "text-embedding-3-small": "emb3s",
            "text-embedding-3-large": "emb3l",
            "gpt-5-mini": "gpt5mini",
            "gpt-5-nano": "gpt5nano",
            "gpt-4o-mini": "gpt4omini",
            "gpt-4o": "gpt4o",
        }

        key = raw.lower()
        if key in aliases:
            return aliases[key]

        # HuggingFace 스타일 모델명은 마지막 이름만 사용
        last = raw.replace("\\", "/").split("/")[-1]

        # 너무 긴 instruct 계열 이름 축약
        last = last.replace("-Instruct", "")
        last = last.replace("_Instruct", "")
        last = last.replace("-instruction", "")
        last = last.replace("_instruction", "")

        return OpenAIRAGEvalPipeline._safe_name(last, max_len=max_len)


    def _make_run_name(self) -> str:
        """
        최종 저장 이름 생성 규칙:

        exp-{실험명}__emb-{임베딩}__llm-{LLM}__vec-{벡터DB}

        예:
        exp-baseline__emb-emb3s__llm-gpt5mini__vec-faiss
        exp-gpt5mini_faiss__emb-emb3s__llm-gpt5mini__vec-faiss
        """
        cfg = self.config
        embedding_cfg = cfg.get("embedding", {})
        llm_cfg = cfg.get("llm", {})
        openai_cfg = cfg.get("openai", {})

        experiment_name = (
            cfg.get("experiment", {}).get("short_name")
            or cfg.get("experiment", {}).get("name")
            or "exp"
        )

        vector_type = cfg.get("vector_db", {}).get("type", "vec")

        embedding_model = (
            embedding_cfg.get("openai_model_name")
            or openai_cfg.get("embedding_model")
            or embedding_cfg.get("hf_model_name")
            or "embedding"
        )

        llm_model = (
            llm_cfg.get("openai_model_name")
            or openai_cfg.get("llm_model")
            or llm_cfg.get("hf_model_name")
            or "llm"
        )

        exp_name = self._safe_name(experiment_name, max_len=28)
        emb_name = self._short_model_name(embedding_model, max_len=24)
        llm_name = self._short_model_name(llm_model, max_len=24)
        vec_name = self._safe_name(vector_type, max_len=16)

        return f"exp-{exp_name}__emb-{emb_name}__llm-{llm_name}__vec-{vec_name}"


    def _resolve_paths(self) -> None:
        cfg = self.config
        vector_type = cfg["vector_db"]["type"]

        # 모든 저장 경로에서 공통으로 사용할 단일 run name
        self.run_name = self._make_run_name()

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

        # report 저장 폴더
        self.paths["report_dir"] = (
            resolve_project_path(self.project_root, cfg["paths"]["report_dir"])
            / self.run_name
        )

        # RAG output 저장 폴더
        eval_output_dir = (
            resolve_project_path(self.project_root, "data/processed/eval")
            / self.run_name
        )

        self.paths["rag_output_path"] = eval_output_dir / "rag.json"
        self.paths["rag_output_scored_path"] = eval_output_dir / "rag_scored.json"

        store_cfg = cfg["vector_db"].get("stores", {}).get(vector_type, {})
        persist_dir = store_cfg.get("persist_dir") or cfg["paths"].get(
            "vector_db_dir",
            "data/vector_db",
        )

        # vector DB 저장 폴더
        self.paths["vector_db_dir"] = (
            resolve_project_path(self.project_root, persist_dir)
            / self.run_name
        )

        self.paths["report_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["vector_db_dir"].mkdir(parents=True, exist_ok=True)
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        report_dir = self.paths["report_dir"]

        # 파일명은 짧게 고정
        self.paths["metrics_path"] = report_dir / "metrics.json"
        self.paths["metrics_by_question_type_path"] = report_dir / "by_qtype.json"
        self.paths["metrics_by_source_type_path"] = report_dir / "by_source.json"
        self.paths["metrics_by_answer_format_path"] = report_dir / "by_answer.json"
        self.paths["metrics_by_file_type_path"] = report_dir / "by_file.json"
        self.paths["retrieval_failure_path"] = report_dir / "retrieval_failures.csv"
        self.paths["keyword_failure_path"] = report_dir / "keyword_failures.csv"
        self.paths["summary_csv_path"] = report_dir / "summary.csv"
        self.paths["experiment_summary_path"] = report_dir / "experiment.json"

    def load_eval_dataset(self) -> None:
        self.eval_dataset = load_json(self.paths["eval_dataset_path"])
        if self.paths["eval_sample_path"].exists():
            self.sample_eval_dataset = load_json(self.paths["eval_sample_path"])
        else:
            self.sample_eval_dataset = create_and_save_eval_sample(
                input_path=self.paths["eval_dataset_path"],
                output_path=self.paths["eval_sample_path"],
                sample_size=self.config["evaluation"]["sample_size"],
                random_seed=self.config["experiment"].get("random_seed", 42),
            )

    def load_chunks(self) -> None:
        rows = load_jsonl(self.paths["chunk_path"])
        self.chunks = [self._standardize_chunk(row, index) for index, row in enumerate(rows)]
        self.chunks = [row for row in self.chunks if row["text"].strip() and row["doc_id"].strip()]

    @staticmethod
    def _build_embedding_text(row: Dict[str, Any], text: str, metadata: Dict[str, Any]) -> str:
        project_name = row.get("project_name") or metadata.get("project_name", "")
        organization = row.get("organization") or metadata.get("organization", "")
        section_title = row.get("section_title") or metadata.get("section_title", "")
        section_path = row.get("section_path") or metadata.get("section_path", [])
    
        if isinstance(section_path, list):
            section_path_text = " > ".join(str(x) for x in section_path if x)
        else:
            section_path_text = str(section_path or "")
    
        header_parts = []
    
        if project_name:
            header_parts.append(f"사업명: {project_name}")
    
        if organization:
            header_parts.append(f"발주기관: {organization}")
    
        if section_path_text:
            header_parts.append(f"섹션경로: {section_path_text}")
    
        if section_title:
            header_parts.append(f"섹션제목: {section_title}")
    
        header = "\n".join(header_parts).strip()
        body = str(text or "").strip()
    
        if header:
            return f"{header}\n\n본문:\n{body}"
    
        return body


    @staticmethod
    def _standardize_chunk(row: Dict[str, Any], index: int) -> Dict[str, Any]:
        metadata = dict(row.get("metadata") or {})
    
        text = (
            row.get("text")
            or row.get("page_content")
            or row.get("content")
            or row.get("chunk_text")
            or ""
        )
    
        doc_id = row.get("doc_id") or metadata.get("doc_id") or ""
        chunk_id = row.get("chunk_id") or metadata.get("chunk_id") or f"chunk_{index:06d}"
    
        for key in [
            "file_name",
            "file_type",
            "project_name",
            "organization",
            "section_title",
            "section_path",
            "section_level",
        ]:
            if row.get(key) is not None:
                metadata.setdefault(key, row.get(key))
    
        embedding_text = (
            row.get("embedding_text")
            or metadata.get("embedding_text")
            or OpenAIRAGEvalPipeline._build_embedding_text(row, str(text), metadata)
        )
    
        return {
            **row,
            "chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "text": str(text),
            "embedding_text": str(embedding_text),
            "metadata": metadata,
        }

    def load_embedder(self) -> None:
        openai_cfg = self.config.get("openai", {})
        embedding_cfg = self.config.get("embedding", {})
        model_name = embedding_cfg.get("openai_model_name") or openai_cfg.get("embedding_model")
        self.embedder = OpenAIEmbeddingModel(
            model_name=model_name,
            api_key_env=openai_cfg.get("api_key_env", "OPENAI_API_KEY"),
        )

    def setup_vector_store(self) -> None:
        vector_type = self.config["vector_db"]["type"]
        store_cfg = copy.deepcopy(
            self.config["vector_db"].get("stores", {}).get(vector_type, {})
        )

        collection_suffix = self._safe_name(
            getattr(self, "run_name", self._make_run_name()),
            max_len=80,
        )

        if vector_type == "faiss":
            self.vector_store = OpenAIFaissStore(
                persist_dir=self.paths["vector_db_dir"],
                index_file=store_cfg.get("index_file", "index.faiss"),
                chunk_meta_file=store_cfg.get("chunk_meta_file", "chunks.pkl"),
            )

        elif vector_type == "chroma":
            base_collection = self._safe_name(
                store_cfg.get("collection", "rfp_rag"),
                max_len=32,
            )
            collection_name = f"{base_collection}_{collection_suffix}"

            self.vector_store = OpenAIChromaStore(
                persist_dir=self.paths["vector_db_dir"],
                collection=collection_name,
            )

        elif vector_type == "qdrant":
            base_collection = self._safe_name(
                store_cfg.get("collection", "rfp_rag"),
                max_len=32,
            )
            store_cfg["collection"] = f"{base_collection}_{collection_suffix}"

            self.vector_store = OpenAIQdrantStore(store_cfg)

        elif vector_type == "supabase":
            self.vector_store = OpenAISupabaseStore(store_cfg)

        else:
            raise ValueError(f"Unsupported vector_db.type: {vector_type}")

    def build_or_load_vector_store(self) -> None:
        if self.embedder is None:
            self.load_embedder()
        if self.vector_store is None:
            self.setup_vector_store()
    
        force = self.config.get("embedding", {}).get("force_rebuild_index", False)
    
        if force or not self.vector_store.exists():
            embedding_inputs = [
                chunk.get("embedding_text") or chunk["text"]
                for chunk in self.chunks
            ]
    
            embeddings = self.embedder.encode_texts(
                embedding_inputs,
                batch_size=int(self.config.get("embedding", {}).get("batch_size", 32)),
            )
    
            self.vector_store.build(self.chunks, embeddings)
    
        elif hasattr(self.vector_store, "load"):
            self.vector_store.load()

    def setup_hybrid_search(self) -> None:
        """
        BM25 인덱스를 준비합니다.
        retrieval.hybrid.enabled=true일 때만 사용합니다.
        """
        retrieval_cfg = self.config.get("retrieval", {})
        hybrid_cfg = retrieval_cfg.get("hybrid", {})

        if not bool(hybrid_cfg.get("enabled", False)):
            self.bm25_index = None
            return

        self.bm25_index = SimpleBM25Index(
            chunks=self.chunks,
            text_key=str(hybrid_cfg.get("text_key", "embedding_text")),
            k1=float(hybrid_cfg.get("k1", 1.5)),
            b=float(hybrid_cfg.get("b", 0.75)),
        )

    def setup_reranker(self) -> None:
        """
        CrossEncoder reranker를 준비합니다.
        retrieval.reranker.enabled=true일 때만 로드합니다.
        """
        retrieval_cfg = self.config.get("retrieval", {})
        reranker_cfg = retrieval_cfg.get("reranker", {})

        if not bool(reranker_cfg.get("enabled", False)):
            self.reranker = None
            return

        self.reranker = CrossEncoderReranker(
            model_name=str(reranker_cfg.get("model_name", "upskyy/ko-reranker-8k")),
            batch_size=int(reranker_cfg.get("batch_size", 16)),
            max_length=int(reranker_cfg.get("max_length", 512)),
            device=reranker_cfg.get("device"),
        )

    @staticmethod
    def _chunk_identity(chunk: Dict[str, Any]) -> str:
        chunk_id = chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        doc_id = chunk.get("doc_id") or chunk.get("metadata", {}).get("doc_id", "")
        text = str(chunk.get("text", ""))[:80]
        return f"{doc_id}:{text}"

    @staticmethod
    def _merge_chunk_fields(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        """
        dense 결과와 bm25 결과가 같은 chunk_id를 가리킬 때,
        더 풍부한 필드를 보존하기 위한 병합 함수입니다.
        """
        merged = dict(base)

        for key, value in incoming.items():
            if key in {"rank", "score"}:
                continue

            if key not in merged or merged.get(key) in (None, "", [], {}):
                merged[key] = value

        base_meta = dict(merged.get("metadata") or {})
        incoming_meta = dict(incoming.get("metadata") or {})

        for key, value in incoming_meta.items():
            if key not in base_meta or base_meta.get(key) in (None, "", [], {}):
                base_meta[key] = value

        if base_meta:
            merged["metadata"] = base_meta

        return merged

    def _rrf_fuse(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        candidate_k: int,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Dense 검색 결과와 BM25 검색 결과를 Reciprocal Rank Fusion으로 결합합니다.
        """
        fused: Dict[str, Dict[str, Any]] = {}

        def add_results(results: List[Dict[str, Any]], source: str, weight: float) -> None:
            for default_rank, chunk in enumerate(results, start=1):
                rank = int(chunk.get("rank") or default_rank)
                key = self._chunk_identity(chunk)

                if key not in fused:
                    fused[key] = {
                        "chunk": dict(chunk),
                        "hybrid_score": 0.0,
                    }
                else:
                    fused[key]["chunk"] = self._merge_chunk_fields(
                        fused[key]["chunk"],
                        chunk,
                    )

                fused[key]["hybrid_score"] += weight / (rrf_k + rank)

                if source == "dense":
                    fused[key]["chunk"]["dense_rank"] = rank
                    fused[key]["chunk"]["dense_score"] = chunk.get("score")
                elif source == "bm25":
                    fused[key]["chunk"]["bm25_rank"] = rank
                    fused[key]["chunk"]["bm25_score"] = chunk.get("bm25_score", chunk.get("score"))

        add_results(dense_results, source="dense", weight=dense_weight)
        add_results(bm25_results, source="bm25", weight=bm25_weight)

        rows = []
        for item in fused.values():
            chunk = dict(item["chunk"])
            chunk["hybrid_score"] = float(item["hybrid_score"])
            rows.append(chunk)

        rows.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

        rows = rows[:candidate_k]
        for rank, chunk in enumerate(rows, start=1):
            chunk["rank"] = rank

        return rows

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """
        Dense / BM25 hybrid / reranker를 config에 따라 적용해 최종 검색 결과를 반환합니다.
        """
        retrieval_cfg = self.config.get("retrieval", {})

        top_k = int(retrieval_cfg.get("top_k", 5))
        candidate_k = int(retrieval_cfg.get("candidate_k", max(top_k, 30)))

        hybrid_cfg = retrieval_cfg.get("hybrid", {})
        use_hybrid = bool(hybrid_cfg.get("enabled", False)) and self.bm25_index is not None

        dense_candidate_k = int(hybrid_cfg.get("dense_candidate_k", candidate_k))
        bm25_candidate_k = int(hybrid_cfg.get("bm25_candidate_k", candidate_k))

        rrf_k = int(hybrid_cfg.get("rrf_k", 60))
        dense_weight = float(hybrid_cfg.get("dense_weight", 0.7))
        bm25_weight = float(hybrid_cfg.get("bm25_weight", 0.3))

        query_embedding = self.embedder.encode_query(question)

        if use_hybrid:
            dense_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=dense_candidate_k,
            )
            bm25_results = self.bm25_index.search(
                query=question,
                top_k=bm25_candidate_k,
            )
            retrieved_candidates = self._rrf_fuse(
                dense_results=dense_results,
                bm25_results=bm25_results,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
            )
        else:
            retrieved_candidates = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=candidate_k,
            )

        if self.reranker is not None:
            retrieved = self.reranker.rerank(
                query=question,
                chunks=retrieved_candidates,
                top_k=top_k,
            )
        else:
            retrieved = retrieved_candidates[:top_k]
            for rank, chunk in enumerate(retrieved, start=1):
                chunk["rank"] = rank

        return retrieved

    def load_generator(self) -> None:
        openai_cfg = self.config.get("openai", {})
        llm_cfg = self.config.get("llm", {})
        self.generator = OpenAILLMGenerator(
            model_name=llm_cfg.get("openai_model_name") or openai_cfg.get("llm_model"),
            fallback_model_name=llm_cfg.get("fallback_model_name") or openai_cfg.get("fallback_llm_model"),
            api_key_env=openai_cfg.get("api_key_env", "OPENAI_API_KEY"),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 512)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
        )

    def run_single_rag(self, eval_item: Dict[str, Any]) -> Dict[str, Any]:
        question = eval_item["question"]
        start_total = time.perf_counter()
        start_retrieval = time.perf_counter()
        
        retrieved = self.retrieve(question)
        retrieval_latency = time.perf_counter() - start_retrieval
        
        generation = self.generator.generate_from_retrieved_chunks(question, retrieved)
        total_latency = time.perf_counter() - start_total

        # [추가] 모델별 1M 토큰당 단가 설정(open ai 공식 홈페이지 기준)
        llm_pricing = {
            "gpt-5-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-5-nano": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
        }
        
        # 현재 활성화된 모델 가져오기
        current_model = getattr(self.generator, "model_name", "gpt-5-mini")
        pricing = llm_pricing.get(current_model, {"input": 0.0, "output": 0.0})

        # 실제 비용 계산
        input_cost = generation["input_tokens"] * pricing["input"]
        output_cost = generation["output_tokens"] * pricing["output"]
        calculated_cost = input_cost + output_cost

        return {
            **eval_item,
            "retrieved_ids": [str(chunk.get("doc_id", "")) for chunk in retrieved],
            "retrieved_chunk_ids": [str(chunk.get("chunk_id", "")) for chunk in retrieved],
            "retrieved_chunks": [self._compact_chunk(chunk) for chunk in retrieved],
            "retrieved_contexts": [chunk.get("text", "") for chunk in retrieved],
            "response": generation["response"],
            "retrieval_latency_sec": retrieval_latency,
            "generation_latency_sec": generation["generation_latency_sec"],
            "total_latency_sec": total_latency,
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "total_tokens": generation["total_tokens"],
            "estimated_cost": calculated_cost,  #  비용 반영
        }

    @staticmethod
    def _compact_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        metadata = chunk.get("metadata", {}) or {}
        section_path = chunk.get("section_path") or metadata.get("section_path", "")

        if isinstance(section_path, list):
            section_path = " > ".join(str(x) for x in section_path if x)

        return {
            "rank": chunk.get("rank"),
            "score": chunk.get("score"),
            "dense_score": chunk.get("dense_score"),
            "bm25_score": chunk.get("bm25_score"),
            "hybrid_score": chunk.get("hybrid_score"),
            "rerank_score": chunk.get("rerank_score"),
            "dense_rank": chunk.get("dense_rank"),
            "bm25_rank": chunk.get("bm25_rank"),
            "chunk_id": chunk.get("chunk_id") or metadata.get("chunk_id"),
            "doc_id": chunk.get("doc_id") or metadata.get("doc_id"),
            "project_name": chunk.get("project_name") or metadata.get("project_name"),
            "organization": chunk.get("organization") or metadata.get("organization"),
            "section_title": chunk.get("section_title") or metadata.get("section_title"),
            "section_id": chunk.get("section_id") or metadata.get("section_id"),
            "section_path": section_path,
            "text": str(chunk.get("text", ""))[:1500],
            "metadata": metadata,
        }

    def run_rag_on_sample(self) -> None:
        self.rag_outputs = []
        for item in self.sample_eval_dataset:
            try:
                self.rag_outputs.append(self.run_single_rag(item))
            except Exception as exc:
                self.rag_outputs.append(
                    {
                        **item,
                        "retrieved_ids": [],
                        "retrieved_chunk_ids": [],
                        "retrieved_chunks": [],
                        "retrieved_contexts": [],
                        "response": "",
                        "error": repr(exc),
                        "retrieval_latency_sec": 0.0,
                        "generation_latency_sec": 0.0,
                        "total_latency_sec": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost": 0.0,
                    }
                )
        save_json(self.rag_outputs, self.paths["rag_output_path"])

    def evaluate(self) -> Dict[str, Any]:
        self.evaluator = RAGEvaluator(auto_download_nltk=False, use_nltk_tokenizer=False)
        top_k = int(self.config["retrieval"]["top_k"])
        metrics = self.evaluator.evaluate_all(self.rag_outputs, k=top_k)
        
        # [추가] 전체 총 비용 및 쿼리당 평균 비용 산출
        valid_costs = [row.get("estimated_cost", 0.0) for row in self.rag_outputs if "error" not in row]
        total_cost = sum(valid_costs)
        avg_cost_per_query = total_cost / len(valid_costs) if valid_costs else 0.0

        metrics["total_cost"] = total_cost
        metrics["avg_cost_per_query"] = avg_cost_per_query

        self.evaluator.save_metrics(metrics, str(self.paths["metrics_path"]))
        self.evaluator.save_metrics(
            self.evaluator.evaluate_by_group(self.rag_outputs, "question_type", k=top_k),
            str(self.paths["metrics_by_question_type_path"]),
        )
        self.evaluator.save_metrics(
            self.evaluator.evaluate_by_group(self.rag_outputs, "source_type", k=top_k),
            str(self.paths["metrics_by_source_type_path"]),
        )
        self.evaluator.save_metrics(
            self.evaluator.evaluate_by_group(self.rag_outputs, "answer_format", k=top_k),
            str(self.paths["metrics_by_answer_format_path"]),
        )
        self.evaluator.save_metrics(
            self.evaluator.evaluate_by_group(self.rag_outputs, "file_type", k=top_k),
            str(self.paths["metrics_by_file_type_path"]),
        )
        
        retrieval_failures = self.evaluator.get_retrieval_failure_cases(self.rag_outputs, k=top_k)
        keyword_failures = self.evaluator.get_keyword_failure_cases(
            self.rag_outputs,
            threshold=float(self.config["evaluation"].get("keyword_failure_threshold", 1.0)),
        )
        self.evaluator.save_rows_as_csv(retrieval_failures, str(self.paths["retrieval_failure_path"]))
        self.evaluator.save_rows_as_csv(keyword_failures, str(self.paths["keyword_failure_path"]))
        self.scored_outputs = self.evaluator.attach_keyword_scores(self.rag_outputs)
        self.evaluator.save_rows_as_json(self.scored_outputs, str(self.paths["rag_output_scored_path"]))
        
        summary_df = self._save_summary_csv(self.scored_outputs)
        
        self.evaluator.save_metrics(
            {
                "config": self.config,
                "paths": {key: str(value) for key, value in self.paths.items()},
                "metrics": metrics,
                "num_rag_outputs": len(self.rag_outputs),
                "num_retrieval_failures": len(retrieval_failures),
                "num_keyword_failures": len(keyword_failures),
                "total_cost": total_cost,               # 파일 저장
                "avg_cost_per_query": avg_cost_per_query # 파일 저장
            },
            str(self.paths["experiment_summary_path"]),
        )
        return {"metrics": metrics, "summary_df": summary_df}

    def _save_summary_csv(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        top_k = int(self.config["retrieval"]["top_k"])
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
                    "reference": row.get("reference"),
                    "response": row.get("response"),
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
                    "estimated_cost": row.get("estimated_cost", 0.0),  # CSV에 비용 열 추가
                    "error": row.get("error", ""),
                }
            )
        df = pd.DataFrame(summary_rows)
        df.to_csv(self.paths["summary_csv_path"], index=False, encoding="utf-8-sig")
        return df

    def run(self) -> Dict[str, Any]:
        set_seed(self.config["experiment"].get("random_seed", 42))
        self.load_eval_dataset()
        self.load_chunks()
        self.load_embedder()
        self.setup_vector_store()
        self.build_or_load_vector_store()
        self.setup_hybrid_search()
        self.setup_reranker()
        self.load_generator()
        self.run_rag_on_sample()
        return self.evaluate()

    def cleanup(self) -> None:
        if self.generator is not None:
            self.generator.unload()
        self.embedder = None
        self.vector_store = None
        self.bm25_index = None
        self.reranker = None
        gc.collect()


def _flat_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})

    section_path = chunk.get("section_path") or metadata.get("section_path", "")
    if isinstance(section_path, list):
        section_path = " > ".join(str(x) for x in section_path if x)

    metadata.update(
        {
            "chunk_id": str(chunk.get("chunk_id", "")),
            "doc_id": str(chunk.get("doc_id", "")),
            "file_name": str(chunk.get("file_name") or metadata.get("file_name", "")),
            "file_type": str(chunk.get("file_type") or metadata.get("file_type", "")),
            "project_name": str(chunk.get("project_name") or metadata.get("project_name", "")),
            "organization": str(chunk.get("organization") or metadata.get("organization", "")),
            "section_id": str(chunk.get("section_id") or metadata.get("section_id", "")),
            "section_title": str(chunk.get("section_title") or metadata.get("section_title", "")),
            "section_path": str(section_path or ""),
            "section_level": chunk.get("section_level") or metadata.get("section_level", ""),
            "heading_marker": str(chunk.get("heading_marker") or metadata.get("heading_marker", "")),
            "heading_raw": str(chunk.get("heading_raw") or metadata.get("heading_raw", "")),
        }
    )

    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


def build_openai_experiment_matrix() -> List[Dict[str, str]]:
    return [
        {
            "llm_model": llm_model,
            "vector_db_type": vector_db_type,
            "experiment_name": f"{llm_model}_{vector_db_type}",
        }
        for llm_model, vector_db_type in DEFAULT_MATRIX
    ]