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
from typing import Dict, Any, List, Optional
import time
import gc

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

    Parameters
    ----------
    config_path:
        YAML config 파일 경로입니다.
        예: configs/baseline_rag.yaml

    project_root:
        프로젝트 루트 경로입니다.
        None이면 find_project_root()로 자동 탐색합니다.

    project_name:
        프로젝트 루트 폴더 이름입니다.
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

        # 주요 객체
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        self.evaluator = None

        # 데이터
        self.chunks: List[Dict[str, Any]] = []
        self.standard_chunks: List[Dict[str, Any]] = []
        self.eval_dataset: List[Dict[str, Any]] = []
        self.sample_eval_dataset: List[Dict[str, Any]] = []
        self.rag_outputs: List[Dict[str, Any]] = []
        self.scored_outputs: List[Dict[str, Any]] = []

        # 경로
        self.paths: Dict[str, Path] = {}

        self._resolve_paths()

    # ---------------------------------------------------------
    # Config / Path
    # ---------------------------------------------------------
    def _resolve_paths(self) -> None:
        """
        YAML config에 있는 상대 경로들을 프로젝트 루트 기준 절대 경로로 변환합니다.
        """
        cfg = self.config

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

        self.paths["vector_db_dir"] = resolve_project_path(
            self.project_root,
            cfg["paths"]["vector_db_dir"],
        )

        self.paths["rag_output_path"] = resolve_project_path(
            self.project_root,
            cfg["paths"]["rag_output_path"],
        )

        self.paths["rag_output_scored_path"] = resolve_project_path(
            self.project_root,
            cfg["paths"]["rag_output_scored_path"],
        )

        self.paths["report_dir"] = resolve_project_path(
            self.project_root,
            cfg["paths"]["report_dir"],
        )

        self.paths["vector_db_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["report_dir"].mkdir(parents=True, exist_ok=True)

        experiment_name = cfg["experiment"]["name"]
        sample_size = cfg["evaluation"]["sample_size"]
        report_dir = self.paths["report_dir"]

        self.paths["metrics_path"] = report_dir / f"{experiment_name}_sample{sample_size}_metrics.json"
        self.paths["metrics_by_question_type_path"] = report_dir / f"{experiment_name}_sample{sample_size}_by_question_type.json"
        self.paths["metrics_by_source_type_path"] = report_dir / f"{experiment_name}_sample{sample_size}_by_source_type.json"
        self.paths["metrics_by_answer_format_path"] = report_dir / f"{experiment_name}_sample{sample_size}_by_answer_format.json"
        self.paths["metrics_by_file_type_path"] = report_dir / f"{experiment_name}_sample{sample_size}_by_file_type.json"
        self.paths["retrieval_failure_path"] = report_dir / f"{experiment_name}_sample{sample_size}_retrieval_failures.csv"
        self.paths["keyword_failure_path"] = report_dir / f"{experiment_name}_sample{sample_size}_keyword_failures.csv"
        self.paths["summary_csv_path"] = report_dir / f"{experiment_name}_sample{sample_size}_summary.csv"
        self.paths["experiment_summary_path"] = report_dir / f"{experiment_name}_sample{sample_size}_experiment_summary.json"
        self.paths["chunk_fingerprint_path"] = self.paths["vector_db_dir"] / "chunk_fingerprint.json"

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

    # ---------------------------------------------------------
    # Seed / Device
    # ---------------------------------------------------------
    def setup_runtime(self) -> None:
        """
        seed 고정과 device 확인을 수행합니다.
        """
        seed = self.config["experiment"].get("random_seed", 42)

        set_seed(seed)

        # device.py의 get_device는 현재 사용 가능한 장치를 출력합니다.
        # 실제 모델 로드는 각 모듈의 설정에 따라 이루어집니다.
        self.device = get_device()

    # ---------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------
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

            standard_chunks.append({
                **chunk,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text,
            })

        self.standard_chunks = standard_chunks

        print("표준화 전 청크 수:", len(self.chunks))
        print("표준화 후 청크 수:", len(self.standard_chunks))

        return self.standard_chunks

    def print_chunk_stats(self) -> pd.DataFrame:
        """
        청크 통계를 출력하고 DataFrame을 반환합니다.
        """
        if not self.standard_chunks:
            raise RuntimeError("standard_chunks가 비어 있습니다. standardize_chunks()를 먼저 호출하세요.")

        df = pd.DataFrame([
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "file_type": chunk.get("file_type"),
                "chunking_strategy": chunk.get("chunking_strategy", chunk.get("chunking_method")),
                "text_len": len(chunk.get("text", "")),
            }
            for chunk in self.standard_chunks
        ])

        print("청크 수:", len(df))
        print("\ntext_len describe:")
        print(df["text_len"].describe())

        print("\n문서별 청크 수 describe:")
        print(df.groupby("doc_id")["chunk_id"].count().describe())

        print("\nfile_type 분포:")
        print(df["file_type"].value_counts(dropna=False))

        return df

    # ---------------------------------------------------------
    # Chunk fingerprint
    # ---------------------------------------------------------
    def compute_chunk_fingerprint(self) -> Dict[str, Any]:
        """
        현재 standard_chunks의 내용을 기반으로 fingerprint를 계산합니다.

        목적:
        - section_chunks.jsonl의 경로가 같아도 내용이 바뀌면 감지
        - 청크 순서, chunk_id, doc_id, text가 바뀌면 vector DB 재생성
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
                "file_type": str(chunk.get("file_type", "")),
                "chunking_strategy": str(
                    chunk.get("chunking_strategy", chunk.get("chunking_method", ""))
                ),
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
        기존 FAISS vector DB 파일을 삭제합니다.
        chunk 내용이 바뀌었을 때 오래된 인덱스와 metadata가 섞이는 것을 방지합니다.
        """
        vector_cfg = self.config["vector_db"]
        vector_db_dir = self.paths["vector_db_dir"]

        target_files = [
            vector_db_dir / vector_cfg.get("index_file", "index.faiss"),
            vector_db_dir / vector_cfg.get("chunk_meta_file", "chunks.pkl"),
            vector_db_dir / vector_cfg.get("config_file", "config.json"),
            self.paths["chunk_fingerprint_path"],
        ]

        for path in target_files:
            if path.exists():
                path.unlink()
                print("기존 vector DB 파일 삭제:", path)

    # ---------------------------------------------------------
    # Embedding / Vector Store / Retriever
    # ---------------------------------------------------------
    def load_embedder(self, device: Optional[str] = None):
        """
        임베딩 모델을 로드합니다.

        Parameters
        ----------
        device:
            "cuda", "cpu" 등.
            None이면 SentenceTransformer가 자동 결정합니다.
        """
        embedding_cfg = self.config["embedding"]

        self.embedder = load_embedding_model(
            model_name=embedding_cfg["model_name"],
            normalize_embeddings=embedding_cfg.get("normalize_embeddings", True),
            device=device,
            trust_remote_code=True,
        )

        return self.embedder

    def setup_vector_store(self) -> FAISSVectorStore:
        """
        FAISSVectorStore 객체를 생성합니다.
        아직 build/load는 수행하지 않습니다.
        """
        vector_cfg = self.config["vector_db"]

        self.vector_store = FAISSVectorStore(
            vector_dir=self.paths["vector_db_dir"],
            index_file=vector_cfg.get("index_file", "index.faiss"),
            chunk_meta_file=vector_cfg.get("chunk_meta_file", "chunks.pkl"),
            config_file=vector_cfg.get("config_file", "config.json"),
        )

        return self.vector_store

    def build_or_load_vector_store(self) -> FAISSVectorStore:
        """
        config에 따라 FAISS 인덱스를 로드하거나 새로 생성합니다.
    
        재생성 조건:
        - embedding.force_rebuild_index=True
        - FAISS index/chunk metadata 파일 없음
        - 저장된 config snapshot과 현재 config 불일치
    
        OOM 방지 전략:
        - FAISS 인덱스 생성 시에는 embedder를 GPU에서 사용할 수 있음
        - FAISS 생성/저장 후 embeddings 배열 삭제
        - embedding.reload_query_embedder_on_cpu=True이면
          GPU에 올라간 embedding model을 unload하고,
          query embedding 전용 embedder를 CPU로 다시 로드함
        - 이후 LLM 로드 시 GPU 메모리 여유를 더 확보할 수 있음
    
        Returns
        -------
        FAISSVectorStore
            로드 또는 생성 완료된 FAISS vector store입니다.
        """
        if self.vector_store is None:
            self.setup_vector_store()
    
        if not self.standard_chunks:
            raise RuntimeError(
                "standard_chunks가 비어 있습니다. "
                "load_chunks()와 standardize_chunks()를 먼저 호출하세요."
            )
    
        embedding_cfg = self.config["embedding"]

        
        config_force_rebuild = embedding_cfg.get("force_rebuild_index", False)
        reload_query_embedder_on_cpu = embedding_cfg.get(
            "reload_query_embedder_on_cpu",
            False,
        )

        # 청크 내용 fingerprint 비교
        fingerprint_rebuild, fingerprint_reasons = self.should_rebuild_by_chunk_fingerprint()

        force_rebuild = config_force_rebuild or fingerprint_rebuild

        rebuild, reasons = self.vector_store.should_rebuild(
            current_config=self.config,
            force_rebuild=force_rebuild,
            keys_to_check=[
                "embedding.model_name",
                "chunking.strategy",
                "paths.chunk_path",
            ],
        )

        reasons = list(reasons) + fingerprint_reasons

        if fingerprint_rebuild:
            print("청크 내용 변경 감지. 기존 vector DB를 교체합니다.")
            print("fingerprint reasons:", fingerprint_reasons)
            self.clear_vector_db_files()
    
        if rebuild:
            print("FAISS 인덱스 새로 생성")
            print("rebuild reasons:", reasons)
    
            # 1. 청크 임베딩 생성용 embedder 로드
            #    여기서는 config의 embedding.device가 있으면 사용하고,
            #    없으면 SentenceTransformer 기본 자동 device 선택을 따릅니다.
            if self.embedder is None:
                self.load_embedder(
                    device=embedding_cfg.get("device")
                )
    
            # 2. 청크 전체 임베딩 생성
            embeddings = self.embedder.encode_chunks(
                chunks=self.standard_chunks,
                batch_size=embedding_cfg.get("batch_size", 32),
                show_progress=True,
                log_every=10,
            )
    
            # 3. FAISS 인덱스 build
            self.vector_store.build(
                embeddings=embeddings,
                chunks=self.standard_chunks,
            )
    
            # 4. FAISS 인덱스 및 config snapshot 저장
            self.vector_store.save(
                config_snapshot=self.config,
            )
            
            # 4-1. 현재 청크 fingerprint 저장
            current_fingerprint = self.compute_chunk_fingerprint()
            self.save_chunk_fingerprint(current_fingerprint)
            
            # 5. embeddings 배열은 저장 후 필요 없으므로 제거
            del embeddings
            gc.collect()
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    
            # 6. OOM 방지:
            #    FAISS 생성 후 GPU embedding model을 내리고,
            #    query embedding용 embedder를 CPU로 다시 로드합니다.
            #
            #    이 옵션을 켜면:
            #    - 문서 임베딩 생성은 GPU로 빠르게 처리 가능
            #    - 이후 RAG 검색 query embedding은 CPU에서 처리
            #    - GPU는 LLM에 더 많이 할당 가능
            if reload_query_embedder_on_cpu:
                print(
                    "FAISS 생성 완료. "
                    "GPU embedding model을 해제하고 CPU query embedder를 다시 로드합니다."
                )
    
                if self.embedder is not None:
                    if hasattr(self.embedder, "unload"):
                        self.embedder.unload()
                    else:
                        del self.embedder
                        gc.collect()
    
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
    
                    self.embedder = None
    
                # query embedding 전용 CPU embedder 재로드
                self.load_embedder(device="cpu")
    
        else:
            print("기존 FAISS 인덱스 로드")
            self.vector_store.load()
    
            # 기존 FAISS를 로드하는 경우에도 query embedding용 embedder는 필요합니다.
            #
            # reload_query_embedder_on_cpu=True이면 CPU에 로드하고,
            # 아니면 config의 embedding.device 또는 자동 device를 사용합니다.
            if self.embedder is None:
                if reload_query_embedder_on_cpu:
                    print("기존 FAISS 사용. query embedder를 CPU로 로드합니다.")
                    self.load_embedder(device="cpu")
                else:
                    self.load_embedder(
                        device=embedding_cfg.get("device")
                    )
    
        return self.vector_store

    def setup_retriever(self) -> RAGRetriever:
        """
        RAGRetriever를 생성합니다.
        """
        if self.embedder is None:
            self.load_embedder()

        if self.vector_store is None or not self.vector_store.is_loaded():
            self.build_or_load_vector_store()

        self.retriever = RAGRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.config["retrieval"]["top_k"],
        )

        return self.retriever

    # ---------------------------------------------------------
    # LLM Generator
    # ---------------------------------------------------------
    def load_generator(self):
        """
        LLMGenerator를 로드합니다.
        """
        llm_cfg = self.config["llm"]

        self.generator = load_llm_generator(
            model_name=llm_cfg["model_name"],
            max_new_tokens=llm_cfg.get("max_new_tokens", 512),
            temperature=llm_cfg.get("temperature", 0.0),
            do_sample=llm_cfg.get("do_sample", False),
            trust_remote_code=llm_cfg.get("trust_remote_code", True),
            prompt_type=llm_cfg.get("prompt_type", "default"),
            max_chars_per_chunk=llm_cfg.get("max_chars_per_chunk"),
            include_metadata=llm_cfg.get("include_metadata", True),
        )

        return self.generator

    # ---------------------------------------------------------
    # RAG 실행
    # ---------------------------------------------------------
    def run_single_rag(self, eval_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        평가 문항 1개에 대해 RAG를 실행합니다.

        처리:
        1. retrieval
        2. generation
        3. 평가용 row 생성
        """
        if self.retriever is None:
            raise RuntimeError("retriever가 없습니다. setup_retriever()를 먼저 호출하세요.")

        if self.generator is None:
            raise RuntimeError("generator가 없습니다. load_generator()를 먼저 호출하세요.")

        question = eval_item["question"]

        start_total = time.perf_counter()

        start_retrieval = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=self.config["retrieval"]["top_k"],
        )
        retrieval_latency_sec = time.perf_counter() - start_retrieval

        generation_result = self.generator.generate_from_retrieved_chunks(
            question=question,
            retrieved_chunks=retrieved_chunks,
            return_prompt=False,
        )

        total_latency_sec = time.perf_counter() - start_total

        retrieved_ids = self.retriever.get_retrieved_ids(retrieved_chunks)
        retrieved_chunk_ids = self.retriever.get_retrieved_chunk_ids(retrieved_chunks)
        retrieved_contexts = self.retriever.get_retrieved_contexts(retrieved_chunks)
        compact_chunks = self.retriever.compact_retrieved_chunks(
            retrieved_chunks,
            max_text_chars=1500,
        )

        result = {
            **eval_item,
            "retrieved_ids": retrieved_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_chunks": compact_chunks,
            "retrieved_contexts": retrieved_contexts,
            "response": generation_result["response"],
            "retrieval_latency_sec": retrieval_latency_sec,
            "generation_latency_sec": generation_result["generation_latency_sec"],
            "total_latency_sec": total_latency_sec,
            "input_tokens": generation_result["input_tokens"],
            "output_tokens": generation_result["output_tokens"],
            "total_tokens": generation_result["total_tokens"],

            # 현재 로컬 모델 기준이므로 비용은 0으로 둡니다.
            # API 모델 평가 시 evaluator.attach_costs()로 계산할 수 있습니다.
            "estimated_cost": 0.0,
        }

        return result

    
    def run_user_query(
        self,
        question: str,
        log_human_eval: bool = True,
        human_eval_csv: str | Path = "outputs/human_eval/real_user_eval_sheet.csv",
    ) -> Dict[str, Any]:
        """
        실제 사용자 질문 1개에 대해 RAG를 실행합니다.

        중요:
        - 이 함수는 자동 평가 후 이미 로드되어 있는 retriever/generator/vector DB를 재사용합니다.
        - self.retriever 또는 self.generator가 None인 경우에만 방어적으로 setup합니다.
        - 정상적인 run() 흐름에서는 자동 평가 단계에서 이미 모두 준비되어 있으므로
          모델과 vector DB를 다시 로드하지 않습니다.

        용도:
        - 자동 평가가 끝난 뒤 실제 사용자 질문 리스트를 실행
        - 팀원 수동 평가용 CSV에 질문/context/답변 누적 저장
        """
        if self.retriever is None:
            # 일반적인 run() 흐름에서는 이미 setup_retriever()가 호출되어 있으므로
            # 여기로 들어오지 않습니다.
            self.setup_retriever()

        if self.generator is None:
            # 일반적인 run() 흐름에서는 이미 load_generator()가 호출되어 있으므로
            # 여기로 들어오지 않습니다.
            self.load_generator()

        start_total = time.perf_counter()

        start_retrieval = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=self.config["retrieval"]["top_k"],
        )
        retrieval_latency_sec = time.perf_counter() - start_retrieval

        generation_result = self.generator.generate_from_retrieved_chunks(
            question=question,
            retrieved_chunks=retrieved_chunks,
            return_prompt=False,
        )

        total_latency_sec = time.perf_counter() - start_total

        retrieved_ids = self.retriever.get_retrieved_ids(retrieved_chunks)
        retrieved_chunk_ids = self.retriever.get_retrieved_chunk_ids(retrieved_chunks)
        retrieved_contexts = self.retriever.get_retrieved_contexts(retrieved_chunks)
        compact_chunks = self.retriever.compact_retrieved_chunks(
            retrieved_chunks,
            max_text_chars=1500,
        )

        result = {
            "question": question,
            "retrieved_ids": retrieved_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_chunks": compact_chunks,
            "retrieved_contexts": retrieved_contexts,
            "response": generation_result["response"],
            "retrieval_latency_sec": retrieval_latency_sec,
            "generation_latency_sec": generation_result["generation_latency_sec"],
            "total_latency_sec": total_latency_sec,
            "input_tokens": generation_result["input_tokens"],
            "output_tokens": generation_result["output_tokens"],
            "total_tokens": generation_result["total_tokens"],
            "estimated_cost": 0.0,
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

        중요:
        - 자동 평가에서 이미 로드한 embedder/vector_store/retriever/generator를 그대로 재사용합니다.
        - 여기서 모델이나 vector DB를 새로 로드하지 않습니다.
        - 모든 팀원 평가 질문 실행이 끝난 뒤 run_rag_eval.py의 finally에서
          pipeline.cleanup()과 disk_guard.cleanup()이 실행됩니다.
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

                human_eval_outputs.append({
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
                })

        print("Human eval query logging 완료:", output_csv)

        return human_eval_outputs

        
    def run_rag_on_sample(self) -> List[Dict[str, Any]]:
        """
        sample_eval_dataset 전체에 대해 RAG를 실행하고 결과를 저장합니다.
        """
        if not self.sample_eval_dataset:
            raise RuntimeError("sample_eval_dataset이 비어 있습니다. load_or_create_eval_sample()을 먼저 호출하세요.")

        if self.retriever is None:
            self.setup_retriever()

        if self.generator is None:
            self.load_generator()

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

        save_json(
            self.rag_outputs,
            self.paths["rag_output_path"],
        )

        print("RAG 실행 결과 저장:", self.paths["rag_output_path"])
        print("결과 수:", len(self.rag_outputs))

        return self.rag_outputs

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
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

        self.evaluator.save_metrics(
            metrics,
            str(self.paths["metrics_path"]),
        )

        with log_step("Evaluate by question_type"):
            by_question_type = self.evaluator.evaluate_by_group(
                self.rag_outputs,
                group_key="question_type",
                k=top_k,
            )

        self.evaluator.save_metrics(
            by_question_type,
            str(self.paths["metrics_by_question_type_path"]),
        )

        with log_step("Evaluate by source_type"):
            by_source_type = self.evaluator.evaluate_by_group(
                self.rag_outputs,
                group_key="source_type",
                k=top_k,
            )

        self.evaluator.save_metrics(
            by_source_type,
            str(self.paths["metrics_by_source_type_path"]),
        )

        with log_step("Evaluate by answer_format"):
            by_answer_format = self.evaluator.evaluate_by_group(
                self.rag_outputs,
                group_key="answer_format",
                k=top_k,
            )

        self.evaluator.save_metrics(
            by_answer_format,
            str(self.paths["metrics_by_answer_format_path"]),
        )

        with log_step("Evaluate by file_type"):
            by_file_type = self.evaluator.evaluate_by_group(
                self.rag_outputs,
                group_key="file_type",
                k=top_k,
            )

        self.evaluator.save_metrics(
            by_file_type,
            str(self.paths["metrics_by_file_type_path"]),
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

        self.evaluator.save_rows_as_json(
            self.scored_outputs,
            str(self.paths["rag_output_scored_path"]),
        )

        summary_df = self._save_summary_csv(self.scored_outputs)

        experiment_summary = {
            "config": self.config,
            "paths": {key: str(value) for key, value in self.paths.items()},
            "metrics": metrics,
            "num_rag_outputs": len(self.rag_outputs),
            "num_retrieval_failures": len(retrieval_failures),
            "num_keyword_failures": len(keyword_failures),
        }

        self.evaluator.save_metrics(
            experiment_summary,
            str(self.paths["experiment_summary_path"]),
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

            summary_rows.append({
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
            })

        summary_df = pd.DataFrame(summary_rows)

        summary_df.to_csv(
            self.paths["summary_csv_path"],
            index=False,
            encoding="utf-8-sig",
        )

        print("요약 CSV 저장:", self.paths["summary_csv_path"])

        return summary_df

    # ---------------------------------------------------------
    # Full run
    # ---------------------------------------------------------
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
        self.setup_retriever()

        self.load_generator()
        self.run_rag_on_sample()

        results = self.evaluate()

        # 실제 사용자 질문 리스트를 실행하고 팀원 평가용
        human_eval_outputs = self.run_human_eval_queries_if_enabled()
        results["human_eval_outputs"] = human_eval_outputs

        return results

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
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