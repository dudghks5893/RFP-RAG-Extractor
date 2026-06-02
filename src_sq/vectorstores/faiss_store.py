# src/vectorstores/faiss_store.py
#
# FAISS 기반 Vector Store 모듈입니다.
#
# 주요 역할:
# - chunk embeddings로 FAISS IndexFlatIP 생성
# - FAISS index 저장/로드
# - 청크 메타데이터 저장/로드
# - query embedding으로 top-k 검색
# - config snapshot을 통해 기존 index와 현재 설정 호환성 확인
#
# 전제:
# - cosine similarity처럼 사용하려면 embedding 단계에서 normalize_embeddings=True를 권장합니다.
# - 이 모듈은 embedding 자체를 생성하지 않습니다.
#   embedding 생성은 src/embeddings/embedding_model.py의 EmbeddingModel이 담당합니다.

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pickle
import json

import numpy as np
import faiss

from src.utils.progress_utils import log_step
from src.utils.config_utils import (
    save_config_snapshot,
    load_config_snapshot,
    check_vector_config_compatibility,
)


class FAISSVectorStore:
    """
    FAISS IndexFlatIP 기반 Vector Store 래퍼 클래스입니다.

    Parameters
    ----------
    vector_dir:
        FAISS index, chunk metadata, config snapshot을 저장할 디렉토리입니다.

    index_file:
        FAISS index 파일명입니다.

    chunk_meta_file:
        FAISS index row와 매칭되는 chunk metadata pickle 파일명입니다.

    config_file:
        index 생성 당시 config snapshot 파일명입니다.
    """

    def __init__(
        self,
        vector_dir: str | Path,
        index_file: str = "index.faiss",
        chunk_meta_file: str = "chunks.pkl",
        config_file: str = "config.json",
    ):
        self.vector_dir = Path(vector_dir)
        self.index_path = self.vector_dir / index_file
        self.chunk_meta_path = self.vector_dir / chunk_meta_file
        self.config_path = self.vector_dir / config_file

        self.index: Optional[faiss.Index] = None
        self.chunks: Optional[List[Dict[str, Any]]] = None

    # ---------------------------------------------------------
    # 상태 확인
    # ---------------------------------------------------------
    def exists(self) -> bool:
        """
        저장된 FAISS index와 chunk metadata가 모두 존재하는지 확인합니다.
        """
        return self.index_path.exists() and self.chunk_meta_path.exists()

    def is_loaded(self) -> bool:
        """
        현재 객체에 FAISS index와 chunks가 로드되어 있는지 확인합니다.
        """
        return self.index is not None and self.chunks is not None

    def _ensure_loaded(self) -> None:
        """
        search 전에 index와 chunks가 로드되어 있는지 확인합니다.
        """
        if not self.is_loaded():
            raise RuntimeError(
                "FAISSVectorStore가 아직 로드되지 않았습니다. "
                "먼저 build() 또는 load()를 호출하세요."
            )

    # ---------------------------------------------------------
    # build / save / load
    # ---------------------------------------------------------
    def build(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
    ) -> "FAISSVectorStore":
        """
        embeddings와 chunks를 받아 FAISS IndexFlatIP를 생성합니다.

        Parameters
        ----------
        embeddings:
            shape = (num_chunks, embedding_dim)
            dtype은 float32여야 합니다.
            float32가 아니면 내부에서 변환합니다.

        chunks:
            embeddings의 row 순서와 정확히 대응되는 chunk dict 목록입니다.

        Returns
        -------
        FAISSVectorStore
            자기 자신을 반환합니다.

        Raises
        ------
        ValueError
            embeddings 개수와 chunks 개수가 다르면 발생합니다.
        """
        if embeddings is None:
            raise ValueError("embeddings가 None입니다.")

        if len(chunks) == 0:
            raise ValueError("chunks가 비어 있습니다.")

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"embeddings 개수와 chunks 개수가 다릅니다. "
                f"embeddings={embeddings.shape[0]}, chunks={len(chunks)}"
            )

        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]

        with log_step("FAISS index build"):
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

        self.index = index
        self.chunks = chunks

        print("FAISS index 생성 완료")
        print("embedding shape:", embeddings.shape)
        print("FAISS ntotal:", self.index.ntotal)

        return self

    def save(
        self,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        현재 FAISS index와 chunks를 디스크에 저장합니다.

        저장 파일:
        - index.faiss
        - chunks.pkl
        - config.json, 선택

        Parameters
        ----------
        config_snapshot:
            index 생성 당시의 config입니다.
            제공하면 config_path에 JSON으로 저장합니다.
        """
        self._ensure_loaded()

        self.vector_dir.mkdir(parents=True, exist_ok=True)

        with log_step("FAISS vector store save"):
            faiss.write_index(self.index, str(self.index_path))

            with open(self.chunk_meta_path, "wb") as f:
                pickle.dump(self.chunks, f)

            if config_snapshot is not None:
                save_config_snapshot(config_snapshot, self.config_path)

        print("FAISS 저장 완료")
        print("index_path:", self.index_path)
        print("chunk_meta_path:", self.chunk_meta_path)
        if config_snapshot is not None:
            print("config_path:", self.config_path)

    def load(self) -> "FAISSVectorStore":
        """
        디스크에 저장된 FAISS index와 chunks를 로드합니다.

        Returns
        -------
        FAISSVectorStore
            자기 자신을 반환합니다.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index 파일이 없습니다: {self.index_path}")

        if not self.chunk_meta_path.exists():
            raise FileNotFoundError(f"chunk metadata 파일이 없습니다: {self.chunk_meta_path}")

        with log_step("FAISS vector store load"):
            self.index = faiss.read_index(str(self.index_path))

            with open(self.chunk_meta_path, "rb") as f:
                self.chunks = pickle.load(f)

        print("FAISS 로드 완료")
        print("index_path:", self.index_path)
        print("chunk_meta_path:", self.chunk_meta_path)
        print("FAISS ntotal:", self.index.ntotal)
        print("chunks:", len(self.chunks))

        return self

    # ---------------------------------------------------------
    # config compatibility
    # ---------------------------------------------------------
    def load_saved_config(self) -> Dict[str, Any]:
        """
        저장된 config snapshot을 로드합니다.

        없으면 빈 dict를 반환합니다.
        """
        return load_config_snapshot(self.config_path)

    def check_compatibility(
        self,
        current_config: Dict[str, Any],
        keys_to_check: Optional[List[str]] = None,
    ) -> List[str]:
        """
        현재 config와 저장된 FAISS config snapshot이 호환되는지 확인합니다.

        Returns
        -------
        List[str]
            변경된 key 목록입니다.
            빈 리스트면 호환된다고 볼 수 있습니다.
        """
        saved_config = self.load_saved_config()

        changed_keys = check_vector_config_compatibility(
            current_config=current_config,
            saved_config=saved_config,
            keys_to_check=keys_to_check,
        )

        return changed_keys

    def should_rebuild(
        self,
        current_config: Dict[str, Any],
        force_rebuild: bool = False,
        keys_to_check: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        FAISS 인덱스를 새로 만들어야 하는지 판단합니다.

        재생성 조건:
        1. force_rebuild=True
        2. 저장된 index/chunks가 없음
        3. 저장된 config와 현재 config가 호환되지 않음

        Returns
        -------
        Tuple[bool, List[str]]
            - rebuild 여부
            - 변경된 config key 목록
        """
        if force_rebuild:
            return True, ["force_rebuild=True"]

        if not self.exists():
            return True, ["vector_store_files_missing"]

        changed_keys = self.check_compatibility(
            current_config=current_config,
            keys_to_check=keys_to_check,
        )

        if changed_keys:
            return True, changed_keys

        return False, []

    # ---------------------------------------------------------
    # search
    # ---------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        query embedding으로 FAISS 검색을 수행합니다.

        Parameters
        ----------
        query_embedding:
            shape = (1, embedding_dim) 또는 (embedding_dim,)
            dtype은 float32로 변환됩니다.

        top_k:
            검색할 청크 개수입니다.

        Returns
        -------
        List[Dict[str, Any]]
            [
                {
                    "rank": 1,
                    "score": 0.83,
                    "chunk_id": "...",
                    "doc_id": "...",
                    "text": "...",
                    "metadata": {...}
                },
                ...
            ]
        """
        self._ensure_loaded()

        query_embedding = np.asarray(query_embedding).astype("float32")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Dict[str, Any]] = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue

            chunk = self.chunks[int(idx)]

            result = {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "text": chunk.get("text", ""),
                "metadata": {
                    key: value
                    for key, value in chunk.items()
                    if key not in {"text"}
                },
            }

            results.append(result)

        return results

    def search_by_text(
        self,
        query: str,
        embedder: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        query 문자열을 embedder로 임베딩한 뒤 FAISS 검색을 수행합니다.

        Parameters
        ----------
        query:
            사용자 질문입니다.

        embedder:
            encode_query(query)를 제공하는 임베딩 객체입니다.
            보통 src.embeddings.embedding_model.EmbeddingModel을 사용합니다.

        top_k:
            검색할 청크 개수입니다.

        Returns
        -------
        List[Dict[str, Any]]
            검색 결과 목록입니다.
        """
        query_embedding = embedder.encode_query(query)
        return self.search(query_embedding=query_embedding, top_k=top_k)

    # ---------------------------------------------------------
    # convenience
    # ---------------------------------------------------------
    @property
    def ntotal(self) -> int:
        """
        FAISS index에 저장된 벡터 개수를 반환합니다.
        """
        if self.index is None:
            return 0

        return int(self.index.ntotal)

    def summary(self) -> Dict[str, Any]:
        """
        현재 Vector Store 상태를 dict로 반환합니다.
        """
        return {
            "vector_dir": str(self.vector_dir),
            "index_path": str(self.index_path),
            "chunk_meta_path": str(self.chunk_meta_path),
            "config_path": str(self.config_path),
            "exists": self.exists(),
            "is_loaded": self.is_loaded(),
            "ntotal": self.ntotal,
            "num_chunks": len(self.chunks) if self.chunks is not None else 0,
        }


def build_or_load_faiss_store(
    vector_store: FAISSVectorStore,
    chunks: List[Dict[str, Any]],
    embeddings: Optional[np.ndarray],
    config: Dict[str, Any],
    force_rebuild: bool = False,
    compatibility_keys: Optional[List[str]] = None,
) -> FAISSVectorStore:
    """
    FAISSVectorStore를 build 또는 load하는 편의 함수입니다.

    Parameters
    ----------
    vector_store:
        FAISSVectorStore 객체입니다.

    chunks:
        청크 목록입니다.
        rebuild가 필요한 경우 사용합니다.

    embeddings:
        청크 임베딩입니다.
        rebuild가 필요한 경우 반드시 필요합니다.

    config:
        현재 실행 config입니다.
        저장된 config와 비교하거나 snapshot 저장에 사용합니다.

    force_rebuild:
        True이면 무조건 새로 생성합니다.

    compatibility_keys:
        config 호환성 비교에 사용할 key 목록입니다.
        None이면 config_utils의 기본 key를 사용합니다.

    Returns
    -------
    FAISSVectorStore
        로드 또는 생성이 완료된 vector_store입니다.
    """
    rebuild, reasons = vector_store.should_rebuild(
        current_config=config,
        force_rebuild=force_rebuild,
        keys_to_check=compatibility_keys,
    )

    if rebuild:
        print("FAISS 인덱스 새로 생성")
        print("rebuild reasons:", reasons)

        if embeddings is None:
            raise ValueError(
                "FAISS 인덱스를 새로 생성해야 하지만 embeddings가 None입니다. "
                "먼저 청크 임베딩을 생성하세요."
            )

        vector_store.build(
            embeddings=embeddings,
            chunks=chunks,
        )

        vector_store.save(config_snapshot=config)

    else:
        print("기존 FAISS 인덱스 로드")
        vector_store.load()

    return vector_store