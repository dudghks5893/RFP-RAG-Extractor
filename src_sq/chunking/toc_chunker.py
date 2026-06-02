# src/chunking/toc_chunker.py
#
# 목차 기반 계층적 청킹 모듈입니다.
#
# 역할:
# - RFP 문서 텍스트를 목차/제목 패턴 기준으로 섹션 분리
# - 긴 섹션은 overlap을 적용해 하위 청크로 분할
# - 최종적으로 RAG 인덱싱에 사용할 chunk dict 리스트 생성
#
# 현재 ExtractChunkPipeline에서 실제 사용하는 청킹 로직입니다.

from __future__ import annotations

from typing import List, Dict, Any
import re


def preprocess_text_for_toc_chunking(text: str) -> str:
    """
    목차 기반 청킹에 맞춘 텍스트 전처리 함수입니다.

    주의:
    - 로마 숫자, 원문자, 주요 목차 기호는 유지합니다.
    - RFP 문서에서 의미가 있는 %, -, ·, 괄호 등은 보존합니다.
    - 너무 강한 특수문자 제거는 금액, 단위, 목차 구조를 손상시킬 수 있으므로 피합니다.

    Parameters
    ----------
    text:
        정제 전 텍스트입니다.

    Returns
    -------
    str
        목차 청킹에 적합하게 추가 정제된 텍스트입니다.
    """
    if not text:
        return ""

    text = str(text)

    # 유니코드 제어문자 제거
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)

    # 로마 숫자, 원문자, 주요 목차 기호는 유지
    text = re.sub(
        r"[^가-힣a-zA-Z0-9\s.,?!()\[\]%\-\u2160-\u217F①-⑮·]",
        " ",
        text,
    )

    # 과도한 공백/줄바꿈 정리
    text = re.sub(r"[ ]{3,}", " ", text)
    text = re.sub(r"\n{3,}", "\n", text)
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def split_sections_by_toc(cleaned_text: str) -> List[Dict[str, str]]:
    """
    목차/제목 패턴을 기준으로 문서를 섹션 단위로 분리합니다.

    동작 방식:
    1. 문서에 로마 숫자 또는 '제N장' 패턴이 있으면 해당 패턴을 우선 사용
    2. 없으면 숫자형 목차 패턴 사용
    3. 목차 패턴이 충분히 감지되지 않으면 전체 문서를 하나의 섹션으로 처리

    Parameters
    ----------
    cleaned_text:
        정제된 문서 텍스트입니다.

    Returns
    -------
    List[Dict[str, str]]
        [
            {
                "title": "Ⅰ. 사업 개요",
                "text": "Ⅰ. 사업 개요 ... "
            },
            ...
        ]
    """
    if not cleaned_text:
        return []

    has_roman = bool(
        re.search(
            r"^\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|\b[IVXLCDM]+\b)\.?",
            cleaned_text,
            re.MULTILINE,
        )
    )

    has_chapter = bool(
        re.search(
            r"^\s*제\s*\d+\s*장",
            cleaned_text,
            re.MULTILINE,
        )
    )

    if has_roman or has_chapter:
        regex_str = (
            r"^\s*("
            r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.?"
            r"|\b[IVXLCDM]+\b\.?"
            r"|제\s*\d+\s*장"
            r")\s+[^\n]{1,60}$"
        )
    else:
        regex_str = r"^\s*(\d+\.?)\s+[^\n]{2,50}$"

    toc_pattern = re.compile(regex_str, re.MULTILINE)
    matches = list(toc_pattern.finditer(cleaned_text))

    sections: List[Dict[str, str]] = []

    if len(matches) >= 2:
        for i in range(len(matches) - 1):
            start = matches[i].start()
            end = matches[i + 1].start()

            section_text = cleaned_text[start:end].strip()
            title = matches[i].group().strip()

            # 목차 줄 끝에 붙은 페이지 번호 제거
            # 예: "Ⅰ. 사업개요 ........ 3" -> "Ⅰ. 사업개요"
            title = re.sub(r"[\s\.\-·…]*\d{1,4}\s*$", "", title).strip()

            if len(section_text) > 100:
                sections.append({
                    "title": title,
                    "text": section_text,
                })

        last_title = matches[-1].group().strip()
        last_title = re.sub(r"[\s\.\-·…]*\d{1,4}\s*$", "", last_title).strip()
        last_section = cleaned_text[matches[-1].start():].strip()

        if len(last_section) > 100:
            sections.append({
                "title": last_title,
                "text": last_section,
            })

    # 제목 패턴이 충분하지 않으면 전체 문서를 하나의 섹션으로 처리
    if not sections:
        sections = [{
            "title": "전체문서",
            "text": cleaned_text,
        }]

    return sections


def split_text_with_overlap(
    text: str,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    min_chars: int = 100,
) -> List[str]:
    """
    긴 섹션을 overlap 기반으로 하위 청크로 분할합니다.

    특징:
    - max_chars보다 짧은 섹션은 그대로 하나의 청크로 사용
    - max_chars보다 긴 섹션은 overlap_chars만큼 겹치게 분할
    - 가능하면 줄바꿈이나 문장 경계에서 자름

    Parameters
    ----------
    text:
        분할할 섹션 텍스트입니다.

    max_chars:
        청크 최대 문자 수입니다.

    overlap_chars:
        인접 청크 간 겹치는 문자 수입니다.

    min_chars:
        너무 짧은 청크를 제거하기 위한 최소 문자 수입니다.

    Returns
    -------
    List[str]
        분할된 청크 텍스트 목록입니다.
    """
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()

        # 가능하면 줄바꿈/문장 경계에서 자르기
        if end < len(text):
            cut = max(
                chunk.rfind("\n"),
                chunk.rfind("."),
                chunk.rfind("다."),
                chunk.rfind("함"),
            )

            # 너무 앞에서 자르면 청크가 과도하게 짧아지므로 60% 이후 경계만 사용
            if cut > int(max_chars * 0.6):
                end = start + cut + 1
                chunk = text[start:end].strip()

        if len(chunk) >= min_chars:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap_chars)

    return chunks


def create_toc_based_chunks(
    doc_id: str,
    text: str,
    file_name: str = "",
    file_type: str = "",
    project_name: str = "",
    organization: str = "",
    max_chars: int = 3000,
    overlap_chars: int = 300,
    min_chars: int = 100,
) -> List[Dict[str, Any]]:
    """
    목차 기반 계층적 청크를 생성합니다.

    처리 순서:
    1. split_sections_by_toc()으로 문서를 섹션 단위로 분리
    2. 각 섹션이 길면 split_text_with_overlap()으로 하위 청크 생성
    3. RAG 인덱싱에 필요한 메타데이터를 포함한 chunk dict 생성

    Parameters
    ----------
    doc_id:
        문서 ID입니다. 보통 공고 번호를 사용합니다.

    text:
        청킹할 정제 텍스트입니다.

    file_name:
        원본 파일명입니다.

    file_type:
        원본 파일 형식입니다. 예: hwp, pdf, docx

    project_name:
        사업명입니다.

    organization:
        발주 기관입니다.

    max_chars:
        청크 최대 문자 수입니다.

    overlap_chars:
        청크 간 overlap 문자 수입니다.

    min_chars:
        최소 청크 문자 수입니다.

    Returns
    -------
    List[Dict[str, Any]]
        [
            {
                "chunk_id": "...",
                "doc_id": "...",
                "section_id": "S001",
                "section_title": "...",
                "text": "...",
                ...
            },
            ...
        ]
    """
    sections = split_sections_by_toc(text)
    chunks: List[Dict[str, Any]] = []

    for section_idx, section in enumerate(sections, start=1):
        section_title = section["title"]
        section_text = section["text"]

        sub_chunks = split_text_with_overlap(
            section_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
        )

        for chunk_idx, chunk_text in enumerate(sub_chunks, start=1):
            chunks.append({
                "chunk_id": f"{doc_id}_S{section_idx:03d}_C{chunk_idx:03d}",
                "doc_id": doc_id,
                "file_name": file_name,
                "file_type": file_type,
                "project_name": project_name,
                "organization": organization,
                "section_id": f"S{section_idx:03d}",
                "section_title": section_title,
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "char_len": len(re.sub(r"\s+", "", chunk_text)),
                "chunking_method": "toc_based_hierarchical",
                "chunking_strategy": "section",
            })

    return chunks