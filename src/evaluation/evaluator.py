# src/evaluation/evaluator.py

import os
import json
import collections
import pandas as pd
from typing import List, Dict, Any
from tqdm.auto import tqdm

# ---------------------------------------------------------
# [사전 준비] 라이브러리 로드 (필요시 pip install ranx nltk rouge-score)
# ---------------------------------------------------------
from ranx import Qrels, Run, evaluate as ranx_evaluate
import ssl
import nltk

# SSL 인증서 검증을 무력화하여 다운로드 에러를 우회
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


class RAGEvaluator:
    """
    RAG 시스템 구축 후 평가와 개선을 위한 통합 평가지표 모듈입니다.
    기존에 구상한 함수 구조, 주석, 평가지표 형식을 100% 동일하게 보존합니다.
    """
    def __init__(self):
        # 모듈 호출 시 NLTK 리소스 안전하게 자동 다운로드
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        self.smoothie = SmoothingFunction().method1
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # ---------------------------------------------------------
    # 1. 독립 평가: Retrieval (검색 성능)
    # ---------------------------------------------------------
    def evaluate_retrieval(self, eval_rows: List[Dict[str, Any]], k: int = 3) -> Dict[str, float]:
        """
        [평가 지표 설명]
        - Hit Rate (hits@K): Top-K 안에 정답 문서가 1개라도 있는가? (0 또는 1)
        - Precision@K: 가져온 K개의 문서 중 정답 문서의 비율은? (정확도)
        - Recall@K: 실제 정답 문서 전체 중, K개 안에 찾아온 문서의 비율은? (재현율)
        - MRR (mrr@K): 첫 번째로 찾은 정답 문서가 몇 등(Rank)에 있는가? (상위 노출 점수)
        """
        qrels_dict, run_dict = {}, {}
        for idx, row in enumerate(eval_rows):
            qid = str(row.get("qid", f"q{idx}"))
            # 정답 문서 세팅
            qrels_dict[qid] = {str(doc_id): 1.0 for doc_id in row.get("gold_ids", [])}
            # 검색된 문서 세팅 (순위에 따라 가중치 부여)
            run_dict[qid] = {str(doc_id): 1.0/(rank+1) for rank, doc_id in enumerate(row.get("retrieved_ids", []))}
            
        metrics = [f"hits@{k}", f"precision@{k}", f"recall@{k}", f"mrr@{k}"]
        return ranx_evaluate(Qrels(qrels_dict), Run(run_dict), metrics=metrics)

    # ---------------------------------------------------------
    # 2. 독립 평가: Generator (생성 성능) & 종단간 평가 (F1)
    # ---------------------------------------------------------
    def evaluate_generation(self, eval_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        [평가 지표 설명]
        - BLEU: 생성 답변이 정답(Reference)과 '문장 구조/단어 순서'가 얼마나 일치하는가? (번역/생성 품질)
        - ROUGE-L: 정답에 있는 핵심 문맥(Longest Common Subsequence)을 얼마나 안 빼먹고 요약/생성했는가?
        - Token F1 Score (종단간 평가): RAG가 뱉은 최종 답변과 정답 간의 단어 교집합(조화평균). 쓸데없는 말은 안 하면서 정답 단어는 다 말했는가?
        """
        bleu_scores, rouge_scores, f1_scores = [], [], []
        
        for row in eval_rows:
            pred = row.get("response", "")
            gold = row.get("reference", "")
            
            # 1) BLEU Score 계산
            ref_tokens = [nltk.word_tokenize(gold)]
            pred_tokens = nltk.word_tokenize(pred)
            bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoothie) if gold and pred else 0.0)
            
            # 2) ROUGE-L Score 계산
            rouge_scores.append(self.scorer.score(gold, pred)['rougeL'].fmeasure if gold and pred else 0.0)
            
            # 3) Token F1 Score 계산
            common = sum((collections.Counter(gold.split()) & collections.Counter(pred.split())).values())
            if len(gold.split()) == 0 or len(pred.split()) == 0:
                f1_scores.append(1.0 if gold == pred else 0.0)
            elif common == 0:
                f1_scores.append(0.0)
            else:
                p = common / len(pred.split())
                r = common / len(gold.split())
                f1_scores.append((2 * p * r) / (p + r))

        return {
            "avg_bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            "avg_rougeL": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
            "avg_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        }

    # ---------------------------------------------------------
    # 3. 종단간 평가: LLM-as-a-Judge (뼈대)
    # ---------------------------------------------------------
    def evaluate_llm_as_a_judge(self, question: str, response: str, reference: str) -> str:
        """
        [평가 지표 설명]
        - LLM-as-a-Judge: 문자 그대로 일치하지 않더라도, "의미상으로 정답을 맞혔는지" GPT-4 등이 직접 채점합니다.
        (이 함수는 추후 LangChain이나 OpenAI API를 연결하여 점수를 1~5점 등으로 반환하도록 구현합니다. 현재는 뼈대입니다.)
        """
        # TODO: OpenAI API 호출 로직 추가 부분
        return "OpenAI 자원이 남을 경우 추가 예정."

    # ---------------------------------------------------------
    # 4. 팀원 평가: 실제 사용자의 질문과 답변을 엑셀에 한 줄씩 누적 저장
    # ---------------------------------------------------------
    def log_for_human_eval(self, question: str, rag_result: Dict[str, Any], output_csv: str = "real_user_eval_sheet.csv"):
        """
        미리 만들어둔 답안지(json) 없이, 실제 던진 질문과 RAG의 답변을 엑셀에 한 줄씩 추가합니다.
        """
        # 1. RAG가 참고한 문서 내용 정리 (리스트면 줄바꿈으로 합치기)
        contexts = rag_result.get("retrieved_contexts", ["(참고 문서 없음)"])
        context_text = "\n\n".join(contexts) if isinstance(contexts, list) else str(contexts)
        
        # 2. 엑셀에 추가할 새로운 데이터 한 줄 세팅
        new_row = {
            "1. 실제 사용자 질문 (Question)": question,
            "2. RAG 참고 문서 (Context)": context_text,
            "3. RAG 최종 답변 (Response)": rag_result.get("response", ""),
            "Groundedness 점수 (1~5)": "", # 팀원 평가용 빈칸
            "Faithfulness 점수 (1~5)": "", # 팀원 평가용 빈칸
            "코멘트": ""
        }
        
        # 3. 기존 엑셀 파일이 있으면 불러오고, 없으면 새로 만들기
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])
            
        # 4. 파일 저장 (utf-8-sig로 저장해야 엑셀에서 한글이 안 깨짐)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"📝 새로운 사용자 질문이 '{output_csv}' 파일에 추가되었습니다! (현재 총 {len(df)}건 누적)")