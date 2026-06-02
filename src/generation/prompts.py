# src/generation/prompts.py
#
# RAG 답변 생성을 위한 prompt 구성 모듈입니다.
#
# 주요 역할:
# - RFP 문서 요약/질의응답에 맞는 system prompt 제공
# - 검색된 청크들을 LLM context 형식으로 변환
# - Qwen Instruct 모델의 tokenizer.apply_chat_template에 넣을 messages 생성
#
# 사용 예:
# from src.generation.prompts import build_rfp_rag_messages
#
# messages = build_rfp_rag_messages(
#     question="이 사업의 기대효과는 무엇인가요?",
#     retrieved_chunks=retrieved_chunks,
# )

from __future__ import annotations

from typing import List, Dict, Any, Optional


'''
default:
- 기본 RFP RAG용
- 문서 근거 기반 답변
- 주요 항목 누락 방지
- 베이스라인 및 일반 평가 실험에 적합
'''
DEFAULT_RFP_SYSTEM_PROMPT = """
당신은 기업 및 정부 제안요청서(RFP)를 분석하는 전문 RAG Assistant입니다.
반드시 제공된 문서 내용에 근거해서만 답변하세요.

답변 규칙:
1. 문서에 근거가 있는 내용만 답변하세요.
2. 문서에서 확인할 수 없는 내용은 추측하지 말고 "제공된 문서에서 확인할 수 없습니다."라고 답변하세요.
3. 질문이 예산, 금액, 기간, 계약방법, 담당자, 연락처처럼 정확한 값을 요구하면 문서에 나온 값을 그대로 사용하세요.
4. 질문이 사업범위, 추진목표, 기대효과, 주요 기능, 요구사항을 묻는 경우 문서에 나온 항목명을 우선적으로 bullet로 정리하세요.
5. 문서에 나열된 고유명사, 시스템명, 기능명, 법령명, 기관명, 금액, 기간, 조건은 가능한 한 누락하지 마세요.
6. 문서에 제시된 고유명사, 기능명, 법령명, 금액, 기간, 평가 배점은 원문 표현을 최대한 유지해서 답변하세요.
7. 일반적인 설명보다 문서에 나온 구체 표현을 우선 사용하세요.
8. 여러 문서가 검색되었더라도 질문 대상 사업과 관련된 문서의 내용만 사용하세요.
9. 검색 문서의 내용이 서로 다르면, 질문과 가장 관련성이 높은 문서의 내용을 우선하세요.
10. 추론 과정, 사고 과정, 분석 과정은 출력하지 마세요.
11. 영어 분석 문장으로 시작하지 말고, 바로 한국어 답변을 작성하세요.
12. 답변은 한국어로 작성하세요.
13. 답변 마지막에는 가능하면 근거가 된 doc_id를 적으세요.

답변 형식:
- 목록형 질문은 bullet로 답변하세요.
- 값 확인 질문은 먼저 값을 제시하고, 필요하면 짧은 근거를 덧붙이세요.
- 비교 질문은 대상별로 구분해서 답변하세요.
""".strip()


'''
strict:
- 문서 근거를 더 엄격하게 요구
- 환각이 많은 경우 사용
- 문서에서 확인되지 않는 내용은 보수적으로 처리
'''
STRICT_RFP_SYSTEM_PROMPT = """
당신은 기업 및 정부 제안요청서(RFP)를 분석하는 전문 RAG Assistant입니다.
반드시 제공된 검색 문서 안에서만 답변해야 합니다.

엄격한 답변 규칙:
1. 검색 문서에 없는 내용은 절대 추측하지 마세요.
2. 검색 문서에서 확인할 수 없으면 "제공된 문서에서 확인할 수 없습니다."라고 답변하세요.
3. 숫자, 금액, 날짜, 기간, 전화번호, 계약방법은 문서에 나온 값을 그대로 사용하세요.
4. 문서에 여러 값이 있으면 가장 질문과 관련성이 높은 값을 선택하고, 불확실하면 불확실하다고 말하세요.
5. 사업범위, 기대효과, 추진목표, 주요 기능, 요구사항은 문서의 표현을 기반으로 요약하세요.
6. 문서에 나열된 고유명사, 시스템명, 기능명, 법령명, 기관명, 금액, 기간, 조건은 가능한 한 누락하지 마세요.
7. 문서에 제시된 고유명사, 기능명, 법령명, 금액, 기간, 평가 배점은 원문 표현을 최대한 유지해서 답변하세요.
8. 일반적인 설명보다 문서에 나온 구체 표현을 우선 사용하세요.
9. 여러 문서가 검색되었더라도 질문 대상 사업과 관련된 문서의 내용만 사용하세요.
10. 추론 과정, 사고 과정, 분석 과정은 출력하지 마세요.
11. 영어 분석 문장으로 시작하지 말고, 바로 한국어 답변을 작성하세요.
12. 답변은 한국어로 작성하세요.
13. 답변 마지막에는 가능하면 근거가 된 doc_id를 적으세요.

답변 형식:
- 목록형 질문은 bullet로 답변하세요.
- 값 확인 질문은 먼저 값을 제시하고, 필요하면 짧은 근거를 덧붙이세요.
- 비교 질문은 대상별로 구분해서 답변하세요.
""".strip()


def get_system_prompt(prompt_type: str = "default") -> str:
    """
    prompt_type에 따라 system prompt를 반환합니다.

    Parameters
    ----------
    prompt_type:
        사용할 prompt 종류입니다.

        지원값:
        - "default": 기본 RFP RAG prompt
        - "strict": 더 엄격한 근거 기반 prompt

    Returns
    -------
    str
        system prompt 문자열입니다.
    """
    if prompt_type == "default":
        return DEFAULT_RFP_SYSTEM_PROMPT

    if prompt_type == "strict":
        return STRICT_RFP_SYSTEM_PROMPT

    raise ValueError(
        f"지원하지 않는 prompt_type입니다: {prompt_type}. "
        "사용 가능 값: 'default', 'strict'"
    )


def _safe_to_string(value: Any) -> str:
    """
    None이나 list/dict 값을 안전하게 문자열로 변환합니다.
    """
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " > ".join(str(v) for v in value if v is not None)

    return str(value)


def _format_page_info(
    page_start: Any,
    page_end: Any,
) -> str:
    """
    pdf_page 청킹 metadata의 page_start/page_end를 사람이 읽기 쉬운 문자열로 변환합니다.
    """
    if page_start is None or page_start == "":
        return ""

    if page_end is None or page_end == "":
        return str(page_start)

    if str(page_start) == str(page_end):
        return str(page_start)

    return f"{page_start}-{page_end}"


def format_single_context_block(
    item: Dict[str, Any],
    max_chars: Optional[int] = None,
    include_metadata: bool = True,
) -> str:
    """
    검색 결과 청크 하나를 prompt context 블록으로 변환합니다.

    Parameters
    ----------
    item:
        retriever.retrieve() 또는 FAISSVectorStore.search()의 결과 item입니다.

        예상 구조:
        {
            "rank": 1,
            "score": 0.83,
            "chunk_id": "...",
            "doc_id": "...",
            "text": "...",
            "metadata": {...}
        }

    max_chars:
        context에 넣을 text 최대 길이입니다.
        None이면 전체 text를 사용합니다.

    include_metadata:
        True이면 doc_id, chunk_id, section_title, page 정보 등을 context에 포함합니다.

    Returns
    -------
    str
        하나의 context block 문자열입니다.
    """
    rank = item.get("rank", "")
    score = item.get("score", "")
    doc_id = item.get("doc_id", "")
    chunk_id = item.get("chunk_id", "")
    text = item.get("text", "") or ""

    if max_chars is not None and max_chars > 0:
        text = text[:max_chars]

    metadata = item.get("metadata", {}) or {}

    section_title = metadata.get("section_title", item.get("section_title", ""))
    section_id = metadata.get("section_id", item.get("section_id", ""))
    section_path = metadata.get("section_path", item.get("section_path", ""))
    file_name = metadata.get("file_name", item.get("file_name", ""))
    project_name = metadata.get("project_name", item.get("project_name", ""))
    organization = metadata.get("organization", item.get("organization", ""))

    # pdf_page 청킹용 metadata
    page_start = metadata.get("page_start", item.get("page_start", ""))
    page_end = metadata.get("page_end", item.get("page_end", ""))
    page_chunk_index = metadata.get(
        "page_chunk_index",
        item.get("page_chunk_index", ""),
    )
    page_info = _format_page_info(page_start, page_end)

    if include_metadata:
        block = f"""
[문서 {rank}]
score: {score}
doc_id: {doc_id}
chunk_id: {chunk_id}
section_id: {_safe_to_string(section_id)}
section_title: {_safe_to_string(section_title)}
section_path: {_safe_to_string(section_path)}
page: {_safe_to_string(page_info)}
page_chunk_index: {_safe_to_string(page_chunk_index)}
project_name: {_safe_to_string(project_name)}
organization: {_safe_to_string(organization)}
file_name: {_safe_to_string(file_name)}

{text}
""".strip()
    else:
        block = f"""
[문서 {rank}]
{text}
""".strip()

    return block


def format_context(
    retrieved_chunks: List[Dict[str, Any]],
    max_chars_per_chunk: Optional[int] = None,
    include_metadata: bool = True,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    검색된 여러 청크를 하나의 context 문자열로 변환합니다.

    Parameters
    ----------
    retrieved_chunks:
        검색 결과 청크 목록입니다.

    max_chars_per_chunk:
        각 청크 text의 최대 길이입니다.
        None이면 자르지 않습니다.

    include_metadata:
        True이면 metadata를 함께 표시합니다.

    separator:
        청크 사이 구분자입니다.

    Returns
    -------
    str
        LLM prompt에 넣을 context 문자열입니다.
    """
    if not retrieved_chunks:
        return "검색된 문서가 없습니다."

    context_blocks = [
        format_single_context_block(
            item=item,
            max_chars=max_chars_per_chunk,
            include_metadata=include_metadata,
        )
        for item in retrieved_chunks
    ]

    return separator.join(context_blocks)


def build_user_prompt(
    question: str,
    context: str,
) -> str:
    """
    user prompt를 생성합니다.

    Parameters
    ----------
    question:
        사용자 질문입니다.

    context:
        검색된 RFP 문서 context입니다.

    Returns
    -------
    str
        user prompt 문자열입니다.
    """
    return f"""
아래는 검색된 RFP 문서 일부입니다.

[검색 문서]
{context}

[질문]
{question}

[답변 작성 지침]
- 위 검색 문서의 내용만 사용하세요.
- 질문에서 요구한 값, 항목, 조건을 빠뜨리지 마세요.
- 문서에 나온 구체 명칭과 표현을 우선 사용하세요.
- 사고 과정이나 분석 과정은 쓰지 말고, 최종 답변만 작성하세요.
- 답변은 한국어로 작성하세요.

[답변]
""".strip()


def build_rfp_rag_messages(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    prompt_type: str = "default",
    max_chars_per_chunk: Optional[int] = None,
    include_metadata: bool = True,
) -> List[Dict[str, str]]:
    """
    RAG 답변 생성을 위한 chat messages를 생성합니다.

    Qwen Instruct 모델에서는 아래 결과를 tokenizer.apply_chat_template에 넣어 사용합니다.

    Parameters
    ----------
    question:
        사용자 질문입니다.

    retrieved_chunks:
        검색된 청크 목록입니다.

    prompt_type:
        system prompt 종류입니다.
        - "default"
        - "strict"

    max_chars_per_chunk:
        각 청크 text를 prompt에 넣을 최대 길이입니다.
        None이면 전체 text를 넣습니다.

    include_metadata:
        True이면 context에 doc_id, chunk_id, section_title, page 정보 등을 포함합니다.

    Returns
    -------
    List[Dict[str, str]]
        chat template 입력용 messages입니다.

        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
    """
    system_prompt = get_system_prompt(prompt_type)

    context = format_context(
        retrieved_chunks=retrieved_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
        include_metadata=include_metadata,
    )

    user_prompt = build_user_prompt(
        question=question,
        context=context,
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def build_prompt_text_for_debug(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    prompt_type: str = "default",
    max_chars_per_chunk: Optional[int] = None,
    include_metadata: bool = True,
) -> str:
    """
    tokenizer.apply_chat_template을 거치기 전,
    사람이 확인하기 쉬운 prompt text를 생성합니다.

    디버깅용 함수입니다.
    실제 생성에는 build_rfp_rag_messages()를 사용하는 것을 권장합니다.
    """
    messages = build_rfp_rag_messages(
        question=question,
        retrieved_chunks=retrieved_chunks,
        prompt_type=prompt_type,
        max_chars_per_chunk=max_chars_per_chunk,
        include_metadata=include_metadata,
    )

    return "\n\n".join(
        f"[{message['role'].upper()}]\n{message['content']}"
        for message in messages
    )