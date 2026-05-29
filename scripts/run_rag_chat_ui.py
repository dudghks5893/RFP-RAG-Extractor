# scripts/run_rag_chat_ui.py
#
# Streamlit 기반 간단 RAG 챗봇 UI입니다.
#
# 실행:
# streamlit run scripts/run_rag_chat_ui.py -- --config configs/baseline_rag.yaml
#
# 주요 동작:
# 1. 프로젝트 루트 찾기
# 2. RAGEvalPipeline 로드
# 3. 기존 vector DB / retriever / generator 준비
# 4. 사용자가 질문하면 run_user_query()로 RAG 답변 생성
# 5. 답변과 참고 문서 chunk를 UI에 표시

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


def find_project_root_from_script(
    project_name: str = "RFP-RAG-Extractor",
) -> Path:
    current = Path(__file__).resolve()

    for path in [current, *current.parents]:
        if path.name == project_name:
            return path

    raise FileNotFoundError(
        f"프로젝트 루트 폴더 '{project_name}'를 찾을 수 없습니다. 현재 위치: {current}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple RAG chatbot UI with Streamlit."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_rag.yaml",
        help="YAML config 파일 경로입니다.",
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="RFP-RAG-Extractor",
        help="프로젝트 루트 폴더 이름입니다.",
    )

    return parser.parse_args()


@st.cache_resource(show_spinner=True)
def load_rag_pipeline(
    config_path: str,
    project_name: str,
):
    """
    Streamlit 세션에서 한 번만 RAG pipeline을 로드합니다.

    주의:
    - LLM 모델 로드가 오래 걸리므로 cache_resource를 사용합니다.
    - force_rebuild_index=false이면 기존 vector DB를 재사용합니다.
    """
    project_root = find_project_root_from_script(project_name)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.pipeline import RAGEvalPipeline

    pipeline = RAGEvalPipeline(
        config_path=config_path,
        project_root=project_root,
        project_name=project_name,
    )

    # UI에서는 자동 평가 데이터셋은 필요 없습니다.
    # 필요한 구성요소만 로드합니다.
    pipeline.setup_runtime()

    pipeline.load_chunks()
    pipeline.standardize_chunks()

    pipeline.load_embedder()
    pipeline.setup_vector_store()
    pipeline.build_or_load_vector_store()

    if pipeline._embedding_provider() != "openai":
        pipeline.setup_retriever()

    pipeline.load_generator()
    pipeline.setup_evaluator()

    return pipeline


def format_retrieved_chunk(chunk: Dict[str, Any]) -> str:
    """
    검색된 chunk를 UI에 표시할 문자열로 변환합니다.
    """
    metadata = chunk.get("metadata", {}) or {}

    rank = chunk.get("rank", "")
    score = chunk.get("score", "")
    doc_id = chunk.get("doc_id") or metadata.get("doc_id", "")
    chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id", "")
    file_name = chunk.get("file_name") or metadata.get("file_name", "")
    project_name = chunk.get("project_name") or metadata.get("project_name", "")
    organization = chunk.get("organization") or metadata.get("organization", "")

    page_start = chunk.get("page_start") or metadata.get("page_start", "")
    page_end = chunk.get("page_end") or metadata.get("page_end", "")

    if page_start and page_end:
        if str(page_start) == str(page_end):
            page_info = str(page_start)
        else:
            page_info = f"{page_start}-{page_end}"
    else:
        page_info = ""

    text = str(chunk.get("text", ""))

    if len(text) > 1200:
        text = text[:1200] + "\n..."

    return f"""
**문서 순위:** {rank}  
**score:** {score}  
**doc_id:** `{doc_id}`  
**chunk_id:** `{chunk_id}`  
**기관명:** {organization}  
**사업명:** {project_name}  
**파일명:** {file_name}  
**페이지:** {page_info}

```text
{text}

""".strip()

def main() -> None:
    args = parse_args()
    
    st.set_page_config(
        page_title="RFP RAG Chatbot",
        page_icon="📄",
        layout="wide",
    )
    
    st.title("📄 RFP RAG Chatbot")
    st.caption("기업 및 정부 제안요청서(RFP) 기반 질의응답 UI")
    
    with st.sidebar:
        st.header("설정")
        st.write("config")
        st.code(args.config)
    
        st.write("project")
        st.code(args.project_name)
    
        st.divider()
    
        if st.button("대화 초기화"):
            st.session_state.messages = []
            st.rerun()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    try:
        with st.spinner("RAG pipeline 로드 중..."):
            pipeline = load_rag_pipeline(
                config_path=args.config,
                project_name=args.project_name,
            )
    
        with st.sidebar:
            st.success("Pipeline loaded")
    
            st.write("Embedding")
            st.code(pipeline._active_embedding_model_name())
    
            st.write("LLM")
            st.code(pipeline._active_llm_model_name())
    
            st.write("Vector DB")
            st.code(pipeline._vector_db_type())
    
            st.write("Experiment")
            st.code(pipeline.experiment_key)
    
    except Exception as e:
        st.error("Pipeline 로드 실패")
        st.exception(e)
        return
    
    # 기존 대화 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
            if message["role"] == "assistant" and message.get("retrieved_chunks"):
                with st.expander("참고 문서 보기"):
                    for chunk in message["retrieved_chunks"]:
                        st.markdown(format_retrieved_chunk(chunk))
                        st.divider()
    
    question = st.chat_input("RFP 문서에 대해 질문하세요.")
    
    if not question:
        return
    
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )
    
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("검색 및 답변 생성 중..."):
                result = pipeline.run_user_query(
                    question=question,
                    log_human_eval=False,
                )
    
            response = result.get("response", "")
            retrieved_chunks = result.get("retrieved_chunks", [])
    
            st.markdown(response)
    
            col1, col2, col3 = st.columns(3)
    
            with col1:
                st.metric(
                    "Retrieval latency",
                    f"{result.get('retrieval_latency_sec', 0):.2f}s",
                )
    
            with col2:
                st.metric(
                    "Generation latency",
                    f"{result.get('generation_latency_sec', 0):.2f}s",
                )
    
            with col3:
                st.metric(
                    "Total latency",
                    f"{result.get('total_latency_sec', 0):.2f}s",
                )
    
            with st.expander("참고 문서 보기"):
                for chunk in retrieved_chunks:
                    st.markdown(format_retrieved_chunk(chunk))
                    st.divider()
    
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "retrieved_chunks": retrieved_chunks,
                }
            )
    
        except Exception as e:
            error_message = f"답변 생성 중 오류가 발생했습니다.\n\n```text\n{repr(e)}\n```"
            st.error(error_message)
            st.code(traceback.format_exc())
    
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "retrieved_chunks": [],
                }
            )

if __name__ == "__main__":
    main()