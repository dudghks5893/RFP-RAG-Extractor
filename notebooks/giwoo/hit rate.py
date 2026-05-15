# =========================================================
# RANX 기반
# Retrieval Hit Rate 평가기
# =========================================================
#
# 기능
# 1. Retriever 독립 평가
#    - Hit Rate@K
#
# 3. Class 기반 평가기
#
# 4. LLM Retriever 결과 평가 가능


# =========================================================
# 1. 라이브러리 설치
# =========================================================

# !pip install ranx pandas


# =========================================================
# 2. 라이브러리 import
# =========================================================

import pandas as pd

from ranx import Qrels
from ranx import Run
from ranx import evaluate


# =========================================================
# 3. Retrieval Evaluator
# =========================================================

class RanxRetrievalEvaluator:

    def __init__(self):

        self.retrieval_results = []


    # =====================================================
    # Retrieval 평가
    # =====================================================

    def evaluate_retrieval(

        self,

        query_id,

        retrieved_doc_ids,

        relevant_doc_id,

        top_k=5
    ):

        """
        Parameters
        ----------
        query_id : str
            Query ID

        retrieved_doc_ids : list
            Retriever가 반환한 문서 ID 리스트

        relevant_doc_id : str
            정답 문서 ID

        top_k : int
            평가할 top-k 범위
        """

        # -------------------------------------------------
        # Qrels 생성
        # -------------------------------------------------

        qrels = Qrels({

            query_id: {

                relevant_doc_id: 1
            }
        })

        # -------------------------------------------------
        # Retriever 결과 생성
        # -------------------------------------------------

        run_dict = {

            query_id: {}
        }

        # rank 기반 score 부여
        for rank, doc_id in enumerate(
            retrieved_doc_ids[:top_k]
        ):

            score = 1 / (rank + 1)

            run_dict[query_id][doc_id] = score

        run = Run(run_dict)

        # -------------------------------------------------
        # Hit Rate 계산
        # -------------------------------------------------

        hit_rate = evaluate(

            qrels=qrels,

            run=run,

            metrics=[f"hit_rate@{top_k}"]
        )

        # -------------------------------------------------
        # 결과 저장
        # -------------------------------------------------

        result = {

            "query_id": query_id,

            "retrieved_doc_ids": retrieved_doc_ids[:top_k],

            "relevant_doc_id": relevant_doc_id,

            "hit_rate": hit_rate
        }

        self.retrieval_results.append(result)

        return result


    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        if len(self.retrieval_results) > 0:

            retrieval_df = pd.DataFrame(
                self.retrieval_results
            )

            retrieval_score = round(

                retrieval_df["hit_rate"].mean(),

                4
            )

        else:

            retrieval_score = 0


        print("\n==========================")
        print("RANX Retrieval Evaluation")
        print("==========================")

        print(
            f"Average Hit Rate : {retrieval_score}"
        )

        return {

            "average_hit_rate": retrieval_score
        }


    # =====================================================
    # Retrieval DataFrame
    # =====================================================

    def retrieval_dataframe(self):

        return pd.DataFrame(
            self.retrieval_results
        )


# =========================================================
# 4. 사용 예시
# =========================================================

evaluator = RanxRetrievalEvaluator()


# =========================================================
# Query 1
# =========================================================

retrieval_result_1 = evaluator.evaluate_retrieval(

    query_id="q1",

    retrieved_doc_ids=[
        "doc_3",
        "doc_1",
        "doc_7"
    ],

    relevant_doc_id="doc_1",

    top_k=3
)

print(retrieval_result_1)


# =========================================================
# Query 2
# =========================================================

retrieval_result_2 = evaluator.evaluate_retrieval(

    query_id="q2",

    retrieved_doc_ids=[
        "doc_10",
        "doc_5",
        "doc_2"
    ],

    relevant_doc_id="doc_2",

    top_k=3
)

print(retrieval_result_2)


# =========================================================
# Query 3
# =========================================================

retrieval_result_3 = evaluator.evaluate_retrieval(

    query_id="q3",

    retrieved_doc_ids=[
        "doc_8",
        "doc_11",
        "doc_15"
    ],

    relevant_doc_id="doc_20",

    top_k=3
)

print(retrieval_result_3)


# =========================================================
# Summary
# =========================================================

summary = evaluator.summary()


# =========================================================
# Retrieval 결과 DataFrame
# =========================================================

retrieval_df = evaluator.retrieval_dataframe()

print("\n[Retrieval Results]")
print(retrieval_df)