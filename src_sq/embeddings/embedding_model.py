# src/embeddings/embedding_model.py
#
# SentenceTransformer 기반 임베딩 모델 로드 및 임베딩 생성 모듈입니다.
#
# 주요 역할:
# - 임베딩 모델 로드
# - 청크 text 배치 임베딩
# - query 임베딩
# - ProgressLogger를 사용해 JupyterHub/GCP 환경에서도 안정적으로 진행상황 출력
#
# 사용 예:
# from src.embeddings.embedding_model import EmbeddingModel
#
# embedder = EmbeddingModel(
#     model_name="nlpai-lab/KURE-v1",
#     normalize_embeddings=True
# )
#
# embeddings = embedder.encode_texts(texts, batch_size=32)
# query_embedding = embedder.encode_query("사업 범위는 무엇인가요?")

from __future__ import annotations

from typing import List, Sequence, Optional, Any
import numpy as np

from sentence_transformers import SentenceTransformer

from src.utils.progress_utils import ProgressLogger, log_step


class EmbeddingModel:
    """
    SentenceTransformer 기반 임베딩 래퍼 클래스입니다.

    이 클래스는 RAG 파이프라인에서 문서 청크와 사용자 질문을
    같은 임베딩 공간으로 변환하는 역할을 담당합니다.

    Parameters
    ----------
    model_name:
        Hugging Face 또는 SentenceTransformers 모델 이름입니다.
        예: "nlpai-lab/KURE-v1"

    normalize_embeddings:
        True이면 임베딩 벡터를 L2 normalize합니다.
        FAISS IndexFlatIP와 함께 사용하면 cosine similarity와 유사하게 동작합니다.

    device:
        사용할 디바이스입니다.
        None이면 SentenceTransformer가 자동으로 결정합니다.
        예: "cuda", "cpu"

    trust_remote_code:
        일부 Hugging Face 모델에서 필요한 옵션입니다.
        SentenceTransformer에서 지원되는 경우에만 의미가 있습니다.
    """

    def __init__(
        self,
        model_name: str,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.trust_remote_code = trust_remote_code

        self.model: Optional[SentenceTransformer] = None

    def load(self) -> "EmbeddingModel":
        """
        SentenceTransformer 모델을 로드합니다.

        Returns
        -------
        EmbeddingModel
            자기 자신을 반환합니다.
            체이닝 형태로 사용할 수 있습니다.

        예:
        embedder = EmbeddingModel("nlpai-lab/KURE-v1").load()
        """
        with log_step(f"Embedding model load: {self.model_name}"):
            # SentenceTransformer는 버전에 따라 trust_remote_code를 지원합니다.
            # 지원하지 않는 환경을 대비해 try-except로 처리합니다.
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code,
                )
            except TypeError:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )

        return self

    def _ensure_loaded(self) -> None:
        """
        모델이 로드되었는지 확인합니다.
        로드되지 않았으면 RuntimeError를 발생시킵니다.
        """
        if self.model is None:
            raise RuntimeError(
                "Embedding model이 아직 로드되지 않았습니다. "
                "먼저 embedder.load()를 호출하세요."
            )

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = True,
        log_every: int = 10,
    ) -> np.ndarray:
        """
        여러 개의 문서 청크 text를 배치 단위로 임베딩합니다.

        tqdm 대신 ProgressLogger를 사용하므로 JupyterHub 환경에서도
        진행 상황을 안정적으로 확인할 수 있습니다.

        Parameters
        ----------
        texts:
            임베딩할 텍스트 목록입니다.

        batch_size:
            임베딩 배치 크기입니다.
            GPU 메모리가 부족하면 16으로 낮추고, 여유가 있으면 64로 올릴 수 있습니다.

        show_progress:
            True이면 ProgressLogger로 진행 상황을 출력합니다.

        log_every:
            몇 batch마다 로그를 출력할지 설정합니다.

        Returns
        -------
        np.ndarray
            shape = (num_texts, embedding_dim)
            dtype = float32
        """
        self._ensure_loaded()

        texts = [str(text or "") for text in texts]
        total = len(texts)

        if total == 0:
            raise ValueError("임베딩할 texts가 비어 있습니다.")

        num_batches = (total + batch_size - 1) // batch_size
        all_embeddings: List[np.ndarray] = []

        logger = None

        if show_progress:
            logger = ProgressLogger(
                total=num_batches,
                desc="Embedding batches",
                log_every=log_every,
                min_interval_sec=5.0,
            )
            logger.start()

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )

            all_embeddings.append(batch_embeddings)

            if logger is not None:
                logger.update(
                    1,
                    message=f"texts={end}/{total}",
                )

        if logger is not None:
            logger.done(message="embedding complete")

        embeddings = np.vstack(all_embeddings).astype("float32")

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        단일 query를 임베딩합니다.

        Parameters
        ----------
        query:
            사용자 질문 또는 검색 query입니다.

        Returns
        -------
        np.ndarray
            shape = (1, embedding_dim)
            dtype = float32
        """
        self._ensure_loaded()

        query = str(query or "")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return embedding.astype("float32")

    def encode_chunks(
        self,
        chunks: Sequence[dict],
        text_key: str = "embedding_text",
        fallback_text_key: str = "text",
        batch_size: int = 32,
        show_progress: bool = True,
        log_every: int = 10,
    ) -> np.ndarray:
        """
        청크 dict 목록에서 embedding_text를 우선 사용해 임베딩합니다.

        pdf_page_chunker.py에서 생성한 chunk에는 보통 아래 두 텍스트가 있습니다.

        - text:
          실제 문서 본문입니다.

        - embedding_text:
          기관명, 사업명, 파일명, 페이지 정보 등 metadata를 본문 앞에 붙인
          검색 최적화용 텍스트입니다.

        검색 성능 개선을 위해 embedding_text를 우선 사용하고,
        embedding_text가 없거나 비어 있으면 text를 fallback으로 사용합니다.

        Parameters
        ----------
        chunks:
            청크 dict 목록입니다.

        text_key:
            임베딩에 우선 사용할 key입니다.
            기본값은 "embedding_text"입니다.

        fallback_text_key:
            text_key 값이 없거나 비어 있을 때 사용할 fallback key입니다.
            기본값은 "text"입니다.

        batch_size:
            임베딩 배치 크기입니다.

        show_progress:
            진행 상황 출력 여부입니다.

        log_every:
            몇 batch마다 로그를 출력할지 설정합니다.

        Returns
        -------
        np.ndarray
            청크 텍스트 임베딩 배열입니다.
        """
        texts: List[str] = []

        for chunk in chunks:
            text = chunk.get(text_key)

            if text is None or not str(text).strip():
                text = chunk.get(fallback_text_key, "")

            texts.append(str(text or ""))

        return self.encode_texts(
            texts=texts,
            batch_size=batch_size,
            show_progress=show_progress,
            log_every=log_every,
        )

    @property
    def embedding_dimension(self) -> int:
        """
        임베딩 차원을 반환합니다.

        모델이 로드된 상태에서만 사용할 수 있습니다.
        """
        self._ensure_loaded()

        # SentenceTransformer는 get_sentence_embedding_dimension()을 제공합니다.
        dim = self.model.get_sentence_embedding_dimension()

        if dim is None:
            # 혹시 dim이 None이면 더미 문장을 임베딩해서 차원을 확인합니다.
            dummy = self.encode_query("dimension check")
            dim = dummy.shape[1]

        return int(dim)

    def unload(self) -> None:
        """
        SentenceTransformer 모델 객체를 해제하고 CUDA cache를 비웁니다.

        FAISS 인덱스 생성 이후 임베딩 모델이 더 이상 필요 없거나,
        LLM 로드 전에 GPU 메모리를 확보하고 싶을 때 사용합니다.
        """
        import gc
        import torch

        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("EmbeddingModel unloaded and CUDA cache cleared.")


def load_embedding_model(
    model_name: str,
    normalize_embeddings: bool = True,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
) -> EmbeddingModel:
    """
    EmbeddingModel을 생성하고 바로 load까지 수행하는 편의 함수입니다.

    사용 예:
    embedder = load_embedding_model("nlpai-lab/KURE-v1")
    """
    embedder = EmbeddingModel(
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    embedder.load()

    return embedder