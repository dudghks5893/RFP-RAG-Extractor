from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

from src.chunking.outline_detector import detect_headings

try:
    from src.utils.text_cleaner import preprocess_text_for_section_chunking
except ImportError:
    preprocess_text_for_section_chunking = None


# ---------------------------------------------------------
# Text normalization for chunker
# ---------------------------------------------------------
def normalize_text(text: str) -> str:
    """
    - section_chunker 내부에서 사용하는 최소 정규화 함수입니다.
    - 강한 전처리는 src.utils.text_cleaner.preprocess_text_for_section_chunking에서 수행합니다.
    - 여기서는 줄바꿈/공백만 가볍게 정리합니다.
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def prepare_text_for_chunking(
    text: str,
    apply_preprocess: bool = True,
) -> str:
    """
    청킹 직전 텍스트 준비 함수입니다.

    Parameters
    ----------
    text:
        clean_extracted_text를 거친 텍스트 또는 원문 텍스트입니다.

    apply_preprocess:
        True이면 src.utils.text_cleaner.preprocess_text_for_section_chunking을 적용합니다.
        파이프라인에서 이미 해당 전처리를 적용했다면 False로 둘 수 있습니다.

    Returns
    -------
    str
        청킹에 사용할 정제 텍스트입니다.
    """
    text = normalize_text(text)

    if apply_preprocess and preprocess_text_for_section_chunking is not None:
        text = preprocess_text_for_section_chunking(text)

    return normalize_text(text)


# ---------------------------------------------------------
# Long section split
# ---------------------------------------------------------
def split_long_text(
    text: str,
    max_chars: int = 2000,
    overlap_chars: int = 300,
) -> List[str]:
    """
    너무 긴 섹션을 문단/라인 기준으로 재분할합니다.

    문자 기준으로 바로 자르면 표, 요구사항 설명, 문장이 중간에서 잘릴 수 있으므로
    우선 라인 단위로 묶고, 한 줄 자체가 너무 긴 경우에만 문자 기준 fallback을 사용합니다.
    """
    text = normalize_text(text)

    if len(text) <= max_chars:
        return [text]

    units = [line.strip() for line in text.splitlines() if line.strip()]

    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_len

        if not buffer:
            return

        chunk = "\n".join(buffer).strip()

        if chunk:
            chunks.append(chunk)

        if overlap_chars > 0:
            overlap_buffer: List[str] = []
            overlap_len = 0

            for line in reversed(buffer):
                line_len = len(line) + 1

                if overlap_len + line_len > overlap_chars:
                    break

                overlap_buffer.insert(0, line)
                overlap_len += line_len

            buffer = overlap_buffer
            buffer_len = overlap_len
        else:
            buffer = []
            buffer_len = 0

    for unit in units:
        unit_len = len(unit) + 1

        if unit_len > max_chars:
            flush_buffer()

            start = 0

            while start < len(unit):
                end = start + max_chars
                part = unit[start:end].strip()

                if part:
                    chunks.append(part)

                if end >= len(unit):
                    break

                start = max(0, end - overlap_chars)

            buffer = []
            buffer_len = 0
            continue

        if buffer and buffer_len + unit_len > max_chars:
            flush_buffer()

        buffer.append(unit)
        buffer_len += unit_len

    flush_buffer()

    return chunks


# ---------------------------------------------------------
# Heading utilities
# ---------------------------------------------------------
def clean_heading_title(title: str) -> str:
    """
    heading title 후처리.
    목차에서 딸려온 점선/공백을 정리합니다.
    """
    title = str(title or "").strip()
    title = re.sub(r"\.{3,}", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def normalize_heading(heading: Dict[str, Any]) -> Dict[str, Any]:
    """
    detect_headings 결과를 표준 형태로 보정합니다.

    기대 형태:
    {
        "line_idx": int,
        "title": str,
        "level": int,
        "raw": str,
        "marker": str
    }

    level이 없다면 기존 코드 호환을 위해 2로 간주합니다.
    """
    return {
        "line_idx": int(heading["line_idx"]),
        "title": clean_heading_title(heading.get("title", "")),
        "level": int(heading.get("level", 2)),
        "raw": str(heading.get("raw", heading.get("title", ""))).strip(),
        "marker": str(heading.get("marker", "")).strip(),
    }


def is_toc_like_line(line: str) -> bool:
    """
    목차 항목으로 보이는 라인을 판단합니다.

    예:
    I. 사업 개요 3
    1. 사업개요 4
    가. 추진방향 5

    본문 제목과 목차 제목이 중복 chunk로 잡히는 것을 막기 위한 필터입니다.
    """
    line = normalize_text(line)

    if not line:
        return False

    # 목차 라인은 보통 끝에 페이지 번호가 붙음
    if not re.search(r"\s+\d{1,4}$", line):
        return False

    toc_patterns = [
        r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.?\s+.+\s+\d{1,4}$",
        r"^[IVX]+\.?\s+.+\s+\d{1,4}$",
        r"^\d{1,2}[.)]\s*.+\s+\d{1,4}$",
        r"^[가-하]\.\s*.+\s+\d{1,4}$",
        r"^\[?붙임\s*\d+\]?.+\s+\d{1,4}$",
        r"^\[?별지.+\]?.+\s+\d{1,4}$",
        r"^\[?별첨.+\]?.+\s+\d{1,4}$",
        r"^\[?별표.+\]?.+\s+\d{1,4}$",
    ]

    return any(re.fullmatch(pattern, line) for pattern in toc_patterns)


def is_noise_heading(heading: Dict[str, Any], lines: List[str]) -> bool:
    """
    chunk 시작점으로 사용하면 안 되는 heading을 필터링합니다.
    """
    line_idx = heading["line_idx"]

    if line_idx < 0 or line_idx >= len(lines):
        return True

    raw_line = lines[line_idx].strip()
    title = heading["title"].strip()

    if not title:
        return True

    if is_toc_like_line(raw_line):
        return True

    # 단독 목차/표지성 라인
    if title in {"목차", "목 차", "차례", "제안요청서"}:
        return True

    # 제목이 지나치게 길면 본문 문장을 heading으로 오탐한 가능성이 큼
    if len(title) > 100:
        return True

    return False


def deduplicate_headings(
    headings: List[Dict[str, Any]],
    lines: List[str],
) -> List[Dict[str, Any]]:
    """
    같은 line_idx에 중복 탐지된 heading 제거.
    더 낮은 level, 즉 더 상위 제목을 우선합니다.
    """
    normalized = [normalize_heading(h) for h in headings]
    normalized = sorted(normalized, key=lambda x: (x["line_idx"], x["level"]))

    deduped: List[Dict[str, Any]] = []
    seen_line_idx = set()

    for heading in normalized:
        line_idx = heading["line_idx"]

        if line_idx in seen_line_idx:
            continue

        if is_noise_heading(heading, lines):
            continue

        seen_line_idx.add(line_idx)
        deduped.append(heading)

    return deduped


def update_section_stack(
    section_stack: List[Dict[str, Any]],
    heading: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    현재 heading을 기준으로 section path stack을 갱신합니다.

    예:
    I. 사업 개요       level=1
    1. 사업개요        level=2
    가. 추진방향       level=3

    section_path:
    ["사업 개요", "사업개요", "추진방향"]
    """
    level = heading["level"]

    section_stack = [
        item for item in section_stack
        if item["level"] < level
    ]

    section_stack.append({
        "level": level,
        "title": heading["title"],
        "line_idx": heading["line_idx"],
        "marker": heading.get("marker", ""),
    })

    return section_stack


def get_section_path(section_stack: List[Dict[str, Any]]) -> List[str]:
    return [item["title"] for item in section_stack]


def get_section_path_with_level(section_stack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    level 정보를 포함한 section path를 반환합니다.
    필요하면 디버깅이나 상세 metadata에 활용할 수 있습니다.
    """
    return [
        {
            "level": item["level"],
            "title": item["title"],
            "line_idx": item["line_idx"],
            "marker": item.get("marker", ""),
        }
        for item in section_stack
    ]


def should_start_chunk(
    heading: Dict[str, Any],
    target_level: int,
    include_deeper_level: bool = False,
) -> bool:
    """
    어떤 제목을 chunk 시작점으로 볼지 결정합니다.

    target_level=2:
    - level 1 제목은 path로만 사용
    - level 2 제목부터 chunk 생성
    - level 3 이하 제목은 기본적으로 현재 chunk 내부에 포함

    include_deeper_level=True:
    - target_level보다 깊은 제목도 별도 chunk로 분리
    """
    level = heading["level"]

    if include_deeper_level:
        return level >= target_level

    return level == target_level


def make_section_id(
    doc_id: str,
    section_path: List[str],
    heading_idx: int,
) -> str:
    """
    문서 내 section 식별자 생성.
    section_title만 쓰면 중복될 수 있어 heading_idx를 함께 사용합니다.
    """
    safe_path = "_".join(section_path[-3:]) if section_path else "unknown"
    safe_path = re.sub(r"\s+", "_", safe_path)
    safe_path = re.sub(r"[^0-9A-Za-z가-힣_]+", "", safe_path)
    safe_path = safe_path[:80] or "section"

    return f"{doc_id}_sec_{heading_idx:04d}_{safe_path}"


# ---------------------------------------------------------
# Fallback chunking
# ---------------------------------------------------------
def create_fallback_chunks(
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
    제목 탐지 실패 시 전체 문서를 문자/문단 기준으로 분할합니다.
    """
    section_chunks: List[Dict[str, Any]] = []

    split_texts = split_long_text(
        text=text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    for idx, chunk_text in enumerate(split_texts):
        chunk_text = chunk_text.strip()

        if len(chunk_text) < min_chars:
            continue

        chunk_id = f"{doc_id}_section_fallback_{idx:04d}"

        section_chunks.append({
            "chunk_id": chunk_id,
            "section_id": f"{doc_id}_fallback",
            "doc_id": doc_id,
            "file_name": file_name,
            "file_type": file_type,
            "project_name": project_name,
            "organization": organization,
            "chunking_method": "outline_section",
            "chunking_strategy": "section_fallback_char",
            "section_title": "",
            "section_path": [],
            "section_path_with_level": [],
            "section_level": None,
            "start_line": None,
            "end_line": None,
            "split_idx": idx,
            "text": chunk_text,
            "embedding_text": build_embedding_text(
                chunk_text=chunk_text,
                project_name=project_name,
                organization=organization,
                section_path=[],
                section_title="",
            ),
            "char_count": len(chunk_text),
            "char_len": len(chunk_text),
        })

    return section_chunks


# ---------------------------------------------------------
# Main section chunker
# ---------------------------------------------------------
def build_embedding_text(
    chunk_text: str,
    project_name: str = "",
    organization: str = "",
    section_path: list[str] | None = None,
    section_title: str = "",
) -> str:
    section_path = section_path or []

    header_parts = []

    if project_name:
        header_parts.append(f"사업명: {project_name}")

    if organization:
        header_parts.append(f"발주기관: {organization}")

    if section_path:
        header_parts.append(f"섹션경로: {' > '.join(section_path)}")

    if section_title:
        header_parts.append(f"섹션제목: {section_title}")

    header = "\n".join(header_parts)

    if header:
        return f"{header}\n\n본문:\n{chunk_text.strip()}"

    return chunk_text.strip()
    
def create_section_chunks(
    doc_id: str,
    text: str,
    file_name: str = "",
    file_type: str = "",
    project_name: str = "",
    organization: str = "",
    max_chars: int = 1500,
    overlap_chars: int = 200,
    min_chars: int = 30,
    target_level: int = 2,
    include_deeper_level: bool = False,
    keep_heading: bool = True,
    apply_preprocess: bool = True,
    fallback_to_higher_level: bool = True,
    _allow_level_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """
    제목 후보를 기준으로 섹션 단위 청크를 생성합니다.

    Parameters
    ----------
    doc_id:
        문서 식별자입니다.

    text:
        추출/정제된 본문 텍스트입니다.

    target_level:
        어느 수준의 제목을 chunk 시작점으로 볼지 결정합니다.

        일반적인 RFP 기준:
        - 1: I. 사업 개요
        - 2: 1. 사업개요
        - 3: 가. 추진방향
        - 4: □ 사업목적

    include_deeper_level:
        True이면 target_level보다 깊은 제목도 별도 chunk로 분리합니다.
        False이면 하위 제목은 현재 chunk 내부 내용으로 포함합니다.

    keep_heading:
        True이면 chunk text에 제목 줄도 포함합니다.

    apply_preprocess:
        True이면 text_cleaner.preprocess_text_for_section_chunking을 적용합니다.
        파이프라인에서 이미 적용했다면 False로 둘 수 있습니다.

    fallback_to_higher_level:
        target_level이 너무 깊어 chunk가 생성되지 않을 때,
        target_level-1, target_level-2 순으로 재시도합니다.

    Returns
    -------
    List[Dict[str, Any]]
        section 단위 chunk 목록입니다.
    """
    text = prepare_text_for_chunking(
        text=text,
        apply_preprocess=apply_preprocess,
    )

    lines = text.splitlines()

    if not text:
        return []

    raw_headings = detect_headings(text)
    headings = deduplicate_headings(raw_headings, lines)

    if not headings:
        return create_fallback_chunks(
            doc_id=doc_id,
            text=text,
            file_name=file_name,
            file_type=file_type,
            project_name=project_name,
            organization=organization,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
        )

    section_chunks: List[Dict[str, Any]] = []
    section_stack: List[Dict[str, Any]] = []

    current_section: Optional[Dict[str, Any]] = None

    def flush_current_section(end_line: int) -> None:
        """
        현재 section을 end_line 직전까지 잘라 chunk로 저장합니다.
        """
        nonlocal current_section

        if current_section is None:
            return

        start_line = current_section["start_line"]
        end_line = max(start_line, min(end_line, len(lines)))

        section_lines = lines[start_line:end_line]

        if not keep_heading and section_lines:
            section_lines = section_lines[1:]

        section_text = "\n".join(section_lines).strip()

        if len(section_text) < min_chars:
            current_section = None
            return

        split_texts = split_long_text(
            text=section_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        section_id = current_section["section_id"]

        for split_idx, chunk_text in enumerate(split_texts):
            chunk_text = chunk_text.strip()

            if len(chunk_text) < min_chars:
                continue

            chunk_id = (
                f"{doc_id}_section_"
                f"{current_section['heading_idx']:04d}_"
                f"{split_idx:02d}"
            )

            section_chunks.append({
                "chunk_id": chunk_id,
                "section_id": section_id,
                "doc_id": doc_id,
                "file_name": file_name,
                "file_type": file_type,
                "project_name": project_name,
                "organization": organization,
                "chunking_method": "outline_section",
                "chunking_strategy": "section",
                "section_title": current_section["title"],
                "section_path": current_section["section_path"],
                "section_path_with_level": current_section["section_path_with_level"],
                "section_level": current_section["level"],
                "heading_marker": current_section.get("marker", ""),
                "heading_raw": current_section.get("raw", ""),
                "start_line": start_line,
                "end_line": end_line,
                "split_idx": split_idx,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "char_len": len(chunk_text),
            })

        current_section = None

    for h_idx, heading in enumerate(headings):
        line_idx = heading["line_idx"]

        if line_idx < 0 or line_idx >= len(lines):
            continue

        starts_new_chunk = should_start_chunk(
            heading=heading,
            target_level=target_level,
            include_deeper_level=include_deeper_level,
        )

        # 중요:
        # target_level=2일 때 level 1 제목이 새로 등장하면
        # 이전 level 2 section은 그 지점에서 종료되어야 합니다.
        # 그렇지 않으면 다음 대제목까지 이전 chunk에 섞입니다.
        if current_section is not None:
            current_level = current_section["level"]

            closes_current_section = (
                starts_new_chunk
                or heading["level"] <= current_level
            )

            if closes_current_section:
                flush_current_section(end_line=line_idx)

        section_stack = update_section_stack(section_stack, heading)

        if starts_new_chunk:
            section_path = get_section_path(section_stack)
            section_path_with_level = get_section_path_with_level(section_stack)

            section_id = make_section_id(
                doc_id=doc_id,
                section_path=section_path,
                heading_idx=h_idx,
            )

            current_section = {
                "section_id": section_id,
                "heading_idx": h_idx,
                "start_line": line_idx,
                "title": heading["title"],
                "level": heading["level"],
                "marker": heading.get("marker", ""),
                "raw": heading.get("raw", ""),
                "section_path": section_path,
                "section_path_with_level": section_path_with_level,
            }

    flush_current_section(end_line=len(lines))

    if section_chunks:
        return section_chunks

    # target_level이 너무 깊어서 청크가 안 만들어진 경우,
    # level을 한 단계 올려 재시도합니다.
    if (
        fallback_to_higher_level
        and _allow_level_fallback
        and target_level > 1
    ):
        return create_section_chunks(
            doc_id=doc_id,
            text=text,
            file_name=file_name,
            file_type=file_type,
            project_name=project_name,
            organization=organization,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
            target_level=target_level - 1,
            include_deeper_level=include_deeper_level,
            keep_heading=keep_heading,
            apply_preprocess=False,
            fallback_to_higher_level=fallback_to_higher_level,
            _allow_level_fallback=True,
        )

    return create_fallback_chunks(
        doc_id=doc_id,
        text=text,
        file_name=file_name,
        file_type=file_type,
        project_name=project_name,
        organization=organization,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
    )