# src/retrieval/retriever.py
#
# RAG 검색 모듈입니다.
#
# 주요 역할:
# - 사용자 질문을 임베딩
# - FAISSVectorStore에서 top-k 청크 검색
# - 검색 결과를 RAG 평가에 필요한 형태로 정리
#
# 이 모듈은 임베딩 모델과 벡터스토어를 직접 소유하지 않고,
# 외부에서 주입받아 사용합니다.
#
# 사용 예:
# from src.retrieval import RAGRetriever
#
# retriever = RAGRetriever(
#     embedder=embedder,
#     vector_store=vector_store,
#     top_k=5,
# )
#
# retrieved_chunks = retriever.retrieve("사업 범위는 무엇인가요?")
# retrieved_ids = retriever.get_retrieved_ids(retrieved_chunks)

from __future__ import annotations

from typing import List, Dict, Any, Optional


class RAGRetriever:
    """
    RAG 검색을 담당하는 Retriever 클래스입니다.

    Parameters
    ----------
    embedder:
        query를 embedding으로 변환하는 객체입니다.
        보통 src.embeddings.embedding_model.EmbeddingModel을 사용합니다.
        encode_query(query: str) 메서드를 가지고 있어야 합니다.

    vector_store:
        FAISS 검색을 수행하는 객체입니다.
        보통 src.vectorstores.faiss_store.FAISSVectorStore를 사용합니다.
        search(query_embedding, top_k) 메서드를 가지고 있어야 합니다.

    top_k:
        기본 검색 개수입니다.
    """

    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    # ---------------------------------------------------------
    # 기본 검색
    # ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        사용자 질문을 받아 top-k 청크를 검색합니다.

        처리 순서:
        1. query를 embedding으로 변환
        2. FAISSVectorStore에서 top-k 검색
        3. 검색 결과 반환

        Parameters
        ----------
        query:
            사용자 질문입니다.

        top_k:
            검색할 청크 개수입니다.
            None이면 self.top_k를 사용합니다.

        Returns
        -------
        List[Dict[str, Any]]
            검색 결과 목록입니다.

            각 item 구조:
            {
                "rank": 1,
                "score": 0.83,
                "chunk_id": "...",
                "doc_id": "...",
                "text": "...",
                "metadata": {...}
            }
        """
        if top_k is None:
            top_k = self.top_k

        query_embedding = self.embedder.encode_query(query)

        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return retrieved_chunks

    # ---------------------------------------------------------
    # 검색 결과 후처리
    # ---------------------------------------------------------
    @staticmethod
    def get_retrieved_ids(
        retrieved_chunks: List[Dict[str, Any]],
        unique: bool = False,
    ) -> List[str]:
        """
        검색 결과에서 doc_id 목록을 추출합니다.

        RAGEvaluator의 retrieval 평가는 doc_id 기준입니다.
        따라서 chunk_id가 아니라 doc_id를 추출해야 합니다.

        Parameters
        ----------
        retrieved_chunks:
            retrieve() 결과입니다.

        unique:
            True이면 중복 doc_id를 제거합니다.
            단, 평가에서는 순위 정보가 중요하므로 기본값은 False입니다.

        Returns
        -------
        List[str]
            검색된 doc_id 목록입니다.

        예:
        ["20241001798", "20241001798", "20241002912"]
        """
        retrieved_ids = []

        for item in retrieved_chunks:
            doc_id = item.get("doc_id")

            if doc_id is None:
                continue

            doc_id = str(doc_id).strip()

            if not doc_id:
                continue

            if unique and doc_id in retrieved_ids:
                continue

            retrieved_ids.append(doc_id)

        return retrieved_ids

    @staticmethod
    def get_retrieved_chunk_ids(
        retrieved_chunks: List[Dict[str, Any]],
        unique: bool = False,
    ) -> List[str]:
        """
        검색 결과에서 chunk_id 목록을 추출합니다.

        chunk 단위 실패 분석이나 디버깅에 사용합니다.
        """
        chunk_ids = []

        for item in retrieved_chunks:
            chunk_id = item.get("chunk_id")

            if chunk_id is None:
                continue

            chunk_id = str(chunk_id).strip()

            if not chunk_id:
                continue

            if unique and chunk_id in chunk_ids:
                continue

            chunk_ids.append(chunk_id)

        return chunk_ids

    @staticmethod
    def get_retrieved_contexts(
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[str]:
        """
        검색 결과에서 context text 목록을 추출합니다.

        LLM 프롬프트 구성 또는 human eval 로그 저장에 사용합니다.
        """
        contexts = []

        for item in retrieved_chunks:
            text = item.get("text", "")

            if text:
                contexts.append(str(text))

        return contexts

    @staticmethod
    def compact_retrieved_chunks(
        retrieved_chunks: List[Dict[str, Any]],
        max_text_chars: int = 1500,
    ) -> List[Dict[str, Any]]:
        """
        RAG 출력 JSON에 저장하기 좋은 형태로 검색 결과를 축약합니다.

        전체 text를 모두 저장하면 JSON이 너무 커질 수 있으므로,
        기본적으로 text는 앞부분 max_text_chars까지만 저장합니다.

        Parameters
        ----------
        retrieved_chunks:
            retrieve() 결과입니다.

        max_text_chars:
            저장할 text 최대 길이입니다.

        Returns
        -------
        List[Dict[str, Any]]
            축약된 검색 결과 목록입니다.
        """
        compact = []

        for item in retrieved_chunks:
            text = item.get("text", "") or ""

            compact.append({
                "rank": item.get("rank"),
                "score": item.get("score"),
                "doc_id": item.get("doc_id"),
                "chunk_id": item.get("chunk_id"),
                "section_title": item.get("metadata", {}).get("section_title"),
                "section_id": item.get("metadata", {}).get("section_id"),
                "text": text[:max_text_chars],
            })

        return compact

    # ---------------------------------------------------------
    # 평가 row 생성 보조
    # ---------------------------------------------------------
    def build_retrieval_result(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_text_chars: int = 1500,
    ) -> Dict[str, Any]:
        """
        하나의 query에 대해 검색을 수행하고,
        RAG 평가 row에 붙일 수 있는 검색 결과 dict를 생성합니다.

        이 함수는 generation 없이 retrieval 결과만 만듭니다.

        Returns
        -------
        Dict[str, Any]
            {
                "retrieved_ids": [...],
                "retrieved_chunk_ids": [...],
                "retrieved_chunks": [...],
                "retrieved_contexts": [...]
            }
        """
        retrieved_chunks = self.retrieve(
            query=query,
            top_k=top_k,
        )

        return {
            "retrieved_ids": self.get_retrieved_ids(retrieved_chunks),
            "retrieved_chunk_ids": self.get_retrieved_chunk_ids(retrieved_chunks),
            "retrieved_chunks": self.compact_retrieved_chunks(
                retrieved_chunks,
                max_text_chars=max_text_chars,
            ),
            "retrieved_contexts": self.get_retrieved_contexts(retrieved_chunks),
        }

    def build_eval_retrieval_row(
        self,
        eval_item: Dict[str, Any],
        top_k: Optional[int] = None,
        max_text_chars: int = 1500,
    ) -> Dict[str, Any]:
        """
        평가 데이터셋의 한 문항에 검색 결과를 붙인 row를 생성합니다.

        Parameters
        ----------
        eval_item:
            eval_dataset_v6의 문항 하나입니다.
            question 필드를 포함해야 합니다.

        top_k:
            검색할 top-k 개수입니다.

        max_text_chars:
            retrieved_chunks에 저장할 각 청크 text 최대 길이입니다.

        Returns
        -------
        Dict[str, Any]
            eval_item에 retrieval 결과가 추가된 dict입니다.
        """
        question = eval_item.get("question", "")

        retrieval_result = self.build_retrieval_result(
            query=question,
            top_k=top_k,
            max_text_chars=max_text_chars,
        )

        return {
            **eval_item,
            **retrieval_result,
        }

    # ---------------------------------------------------------
    # 배치 검색
    # ---------------------------------------------------------
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        여러 query에 대해 순차적으로 검색을 수행합니다.

        대량 검색이 필요할 때 사용합니다.
        현재는 단순 순차 처리이며, 필요하면 추후 batch query embedding으로 개선할 수 있습니다.
        """
        results = []

        for query in queries:
            results.append(
                self.retrieve(
                    query=query,
                    top_k=top_k,
                )
            )

        return results

    def batch_build_eval_rows(
        self,
        eval_items: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        max_text_chars: int = 1500,
        progress_iter_fn: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        평가 데이터셋 여러 문항에 대해 retrieval 결과를 붙입니다.

        Parameters
        ----------
        eval_items:
            평가 문항 리스트입니다.

        top_k:
            검색할 top-k 개수입니다.

        max_text_chars:
            retrieved_chunks에 저장할 각 청크 text 최대 길이입니다.

        progress_iter_fn:
            진행률 표시용 iterator wrapper입니다.
            예: src.utils.progress_utils.progress_iter
            None이면 일반 for loop를 사용합니다.

        Returns
        -------
        List[Dict[str, Any]]
            retrieval 결과가 추가된 평가 row 목록입니다.
        """
        rows = []

        iterator = eval_items

        if progress_iter_fn is not None:
            iterator = progress_iter_fn(
                eval_items,
                total=len(eval_items),
                desc="Batch retrieval",
                log_every=1,
                min_interval_sec=0.0,
            )

        for item in iterator:
            rows.append(
                self.build_eval_retrieval_row(
                    eval_item=item,
                    top_k=top_k,
                    max_text_chars=max_text_chars,
                )
            )

        return rows