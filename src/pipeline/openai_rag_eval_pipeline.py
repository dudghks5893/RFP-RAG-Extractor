from __future__ import annotations

import copy
import gc
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.evaluation.evaluator import RAGEvaluator
from src.utils.config_utils import load_yaml_config, resolve_project_path
from src.utils.eval_dataset_utils import create_and_save_eval_sample, load_json, save_json
from src.utils.file_utils import load_jsonl
from src.utils.seed import set_seed


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
            doc_id = chunk.get("doc_id") or chunk.get("metadata", {}).get("doc_id", "")
            chunk_id = chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id", "")
            context_blocks.append(
                f"[Evidence {index}] doc_id={doc_id} chunk_id={chunk_id}\n{chunk.get('text', '')}"
            )
        contexts = "\n\n".join(context_blocks)
        return (
            "You are a Korean RAG assistant for analyzing RFP documents.\n"
            "Answer in Korean. Use only the evidence below. If the evidence is insufficient, say so.\n\n"
            f"{contexts}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def unload(self) -> None:
        self.client = None


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
        self.evaluator = None

        self._resolve_paths()

    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        llm_model = overrides.get("llm_model")
        vector_db_type = overrides.get("vector_db_type")
        experiment_name = overrides.get("experiment_name")
        if llm_model:
            self.config.setdefault("llm", {})["openai_model_name"] = llm_model
            self.config.setdefault("openai", {})["llm_model"] = llm_model
        if vector_db_type:
            self.config.setdefault("vector_db", {})["type"] = vector_db_type
        if experiment_name:
            self.config.setdefault("experiment", {})["name"] = experiment_name

    def _resolve_paths(self) -> None:
        cfg = self.config
        experiment_name = cfg["experiment"]["name"]
        sample_size = cfg["evaluation"]["sample_size"]
        vector_type = cfg["vector_db"]["type"]

        self.paths["chunk_path"] = resolve_project_path(self.project_root, cfg["paths"]["chunk_path"])
        self.paths["eval_dataset_path"] = resolve_project_path(self.project_root, cfg["paths"]["eval_dataset_path"])
        self.paths["eval_sample_path"] = resolve_project_path(self.project_root, cfg["paths"]["eval_sample_path"])
        self.paths["report_dir"] = resolve_project_path(self.project_root, cfg["paths"]["report_dir"]) / experiment_name
        self.paths["rag_output_path"] = resolve_project_path(
            self.project_root,
            f"data/processed/eval/{experiment_name}_rag_outputs.json",
        )
        self.paths["rag_output_scored_path"] = resolve_project_path(
            self.project_root,
            f"data/processed/eval/{experiment_name}_rag_outputs_scored.json",
        )
        store_cfg = cfg["vector_db"].get("stores", {}).get(vector_type, {})
        persist_dir = store_cfg.get("persist_dir") or cfg["paths"].get("vector_db_dir", "data/vector_db")
        self.paths["vector_db_dir"] = resolve_project_path(self.project_root, persist_dir) / experiment_name

        self.paths["report_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["vector_db_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["rag_output_path"].parent.mkdir(parents=True, exist_ok=True)

        report_dir = self.paths["report_dir"]
        prefix = f"{experiment_name}_sample{sample_size}"
        self.paths["metrics_path"] = report_dir / f"{prefix}_metrics.json"
        self.paths["metrics_by_question_type_path"] = report_dir / f"{prefix}_by_question_type.json"
        self.paths["metrics_by_source_type_path"] = report_dir / f"{prefix}_by_source_type.json"
        self.paths["metrics_by_answer_format_path"] = report_dir / f"{prefix}_by_answer_format.json"
        self.paths["metrics_by_file_type_path"] = report_dir / f"{prefix}_by_file_type.json"
        self.paths["retrieval_failure_path"] = report_dir / f"{prefix}_retrieval_failures.csv"
        self.paths["keyword_failure_path"] = report_dir / f"{prefix}_keyword_failures.csv"
        self.paths["summary_csv_path"] = report_dir / f"{prefix}_summary.csv"
        self.paths["experiment_summary_path"] = report_dir / f"{prefix}_experiment_summary.json"

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
    def _standardize_chunk(row: Dict[str, Any], index: int) -> Dict[str, Any]:
        metadata = dict(row.get("metadata") or {})
        text = row.get("text") or row.get("page_content") or row.get("content") or row.get("chunk_text") or ""
        doc_id = row.get("doc_id") or metadata.get("doc_id") or ""
        chunk_id = row.get("chunk_id") or metadata.get("chunk_id") or f"chunk_{index:06d}"
        for key in ["file_name", "file_type", "project_name", "organization", "section_title"]:
            if row.get(key) is not None:
                metadata.setdefault(key, row.get(key))
        return {
            **row,
            "chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "text": str(text),
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
        store_cfg = copy.deepcopy(self.config["vector_db"].get("stores", {}).get(vector_type, {}))
        if vector_type == "faiss":
            self.vector_store = OpenAIFaissStore(
                persist_dir=self.paths["vector_db_dir"],
                index_file=store_cfg.get("index_file", "index.faiss"),
                chunk_meta_file=store_cfg.get("chunk_meta_file", "chunks.pkl"),
            )
        elif vector_type == "chroma":
            self.vector_store = OpenAIChromaStore(
                persist_dir=self.paths["vector_db_dir"],
                collection=f"{store_cfg.get('collection', 'rfp_openai_rag')}_{self.config['experiment']['name']}",
            )
        elif vector_type == "qdrant":
            store_cfg["collection"] = f"{store_cfg.get('collection', 'rfp_openai_rag')}_{self.config['experiment']['name']}"
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
            embeddings = self.embedder.encode_texts(
                [chunk["text"] for chunk in self.chunks],
                batch_size=int(self.config.get("embedding", {}).get("batch_size", 32)),
            )
            self.vector_store.build(self.chunks, embeddings)
        elif hasattr(self.vector_store, "load"):
            self.vector_store.load()

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
        query_embedding = self.embedder.encode_query(question)
        retrieved = self.vector_store.search(query_embedding, int(self.config["retrieval"]["top_k"]))
        retrieval_latency = time.perf_counter() - start_retrieval
        generation = self.generator.generate_from_retrieved_chunks(question, retrieved)
        total_latency = time.perf_counter() - start_total

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
            "estimated_cost": 0.0,
        }

    @staticmethod
    def _compact_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rank": chunk.get("rank"),
            "score": chunk.get("score"),
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "text": str(chunk.get("text", ""))[:1500],
            "metadata": chunk.get("metadata", {}),
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
        self.load_generator()
        self.run_rag_on_sample()
        return self.evaluate()

    def cleanup(self) -> None:
        if self.generator is not None:
            self.generator.unload()
        self.embedder = None
        self.vector_store = None
        gc.collect()


def _flat_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    metadata.update(
        {
            "chunk_id": str(chunk.get("chunk_id", "")),
            "doc_id": str(chunk.get("doc_id", "")),
            "file_name": str(chunk.get("file_name") or metadata.get("file_name", "")),
            "file_type": str(chunk.get("file_type") or metadata.get("file_type", "")),
            "project_name": str(chunk.get("project_name") or metadata.get("project_name", "")),
            "organization": str(chunk.get("organization") or metadata.get("organization", "")),
        }
    )
    return {key: value for key, value in metadata.items() if isinstance(value, (str, int, float, bool))}


def build_openai_experiment_matrix() -> List[Dict[str, str]]:
    return [
        {
            "llm_model": llm_model,
            "vector_db_type": vector_db_type,
            "experiment_name": f"{llm_model}_{vector_db_type}",
        }
        for llm_model, vector_db_type in DEFAULT_MATRIX
    ]
