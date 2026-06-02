# src/chunking/pdf_page_chunker.py
#
# PDF 전용 page 기반 chunker입니다.
#
# 주요 목적:
# - PDF 문서를 페이지 단위로 안정적으로 청킹
# - section heading 과분할 문제 방지
# - page_start / page_end metadata 보존
# - 짧은 페이지는 다음 페이지와 병합
# - 긴 페이지는 max_chars 기준으로 재분할
# - embedding_text에 기관명/사업명/파일명/페이지 정보를 포함

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from src.utils.text_cleaner import clean_extracted_text


def _safe_str(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    return text.strip()


def _normalize_text(text: Any) -> str:
    text = _safe_str(text)

    if not text:
        return ""

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_pdf_pages_with_pymupdf(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    PyMuPDF를 사용해 PDF를 페이지별로 텍스트 추출합니다.

    Returns
    -------
    List[Dict[str, Any]]
        [
            {"page_no": 1, "text": "..."},
            {"page_no": 2, "text": "..."},
            ...
        ]
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "pdf_page chunking을 사용하려면 PyMuPDF가 필요합니다. "
            "아래 명령으로 설치하세요:\n"
            "pip install pymupdf"
        ) from e

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    pages: List[Dict[str, Any]] = []

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            # text 추출 모드:
            # - "text": 일반 텍스트 추출
            # - sort=True: 위에서 아래, 왼쪽에서 오른쪽 순서 정렬
            raw_text = page.get_text("text", sort=True) or ""
            clean_text = clean_extracted_text(raw_text)

            pages.append({
                "page_no": page_idx,
                "text": clean_text,
            })

    return pages


def _split_long_text_by_chars(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> List[str]:
    """
    긴 페이지 텍스트를 max_chars 기준으로 재분할합니다.
    문단 기준으로 먼저 묶고, 문단 하나가 너무 길면 문자 기준으로 강제 분할합니다.
    """
    text = _normalize_text(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph and paragraph.strip()
    ]

    if len(paragraphs) <= 1:
        paragraphs = [
            line.strip()
            for line in text.splitlines()
            if line and line.strip()
        ]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush_current() -> Optional[str]:
        if not current_parts:
            return None

        chunk_text = "\n\n".join(current_parts).strip()
        return chunk_text if chunk_text else None

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > max_chars:
            current_chunk = flush_current()

            if current_chunk:
                chunks.append(current_chunk)

            current_parts = []
            current_len = 0

            start = 0

            while start < len(paragraph):
                end = start + max_chars
                piece = paragraph[start:end].strip()

                if piece:
                    chunks.append(piece)

                if overlap_chars > 0:
                    next_start = end - overlap_chars
                else:
                    next_start = end

                if next_start <= start:
                    next_start = end

                start = next_start

            continue

        additional_len = len(paragraph) + 2

        if current_parts and current_len + additional_len > max_chars:
            current_chunk = flush_current()

            if current_chunk:
                chunks.append(current_chunk)

            overlap_text = ""

            if overlap_chars > 0 and current_chunk:
                overlap_text = current_chunk[-overlap_chars:].strip()

            current_parts = []
            current_len = 0

            if overlap_text:
                current_parts.append(overlap_text)
                current_len += len(overlap_text)

        current_parts.append(paragraph)
        current_len += additional_len

    last_chunk = flush_current()

    if last_chunk:
        chunks.append(last_chunk)

    return chunks


def _merge_short_pages(
    pages: List[Dict[str, Any]],
    min_chars: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    """
    너무 짧은 페이지 텍스트를 다음 페이지와 병합합니다.

    병합 기준:
    - 현재 buffer가 min_chars보다 짧으면 다음 페이지를 붙임
    - 단, 병합 후 max_chars를 크게 넘기면 병합하지 않음
    """
    merged_pages: List[Dict[str, Any]] = []

    buffer_text = ""
    buffer_start_page: Optional[int] = None
    buffer_end_page: Optional[int] = None

    def flush_buffer() -> None:
        nonlocal buffer_text, buffer_start_page, buffer_end_page

        text = _normalize_text(buffer_text)

        if text and buffer_start_page is not None and buffer_end_page is not None:
            merged_pages.append({
                "page_start": buffer_start_page,
                "page_end": buffer_end_page,
                "text": text,
            })

        buffer_text = ""
        buffer_start_page = None
        buffer_end_page = None

    for page in pages:
        page_no = int(page["page_no"])
        page_text = _normalize_text(page.get("text", ""))

        if not page_text:
            continue

        if buffer_start_page is None:
            buffer_text = page_text
            buffer_start_page = page_no
            buffer_end_page = page_no
            continue

        buffer_len = len(buffer_text)
        candidate_len = buffer_len + len(page_text) + 2

        should_merge = (
            buffer_len < min_chars
            and candidate_len <= max_chars
        )

        if should_merge:
            buffer_text = f"{buffer_text}\n\n{page_text}".strip()
            buffer_end_page = page_no
        else:
            flush_buffer()
            buffer_text = page_text
            buffer_start_page = page_no
            buffer_end_page = page_no

    flush_buffer()

    return merged_pages


def _build_embedding_text(
    text: str,
    organization: str = "",
    project_name: str = "",
    file_name: str = "",
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    include_metadata: bool = True,
) -> str:
    text = _normalize_text(text)

    if not include_metadata:
        return text

    meta_lines = []

    if organization:
        meta_lines.append(f"기관명: {organization}")

    if project_name:
        meta_lines.append(f"사업명: {project_name}")

    if file_name:
        meta_lines.append(f"파일명: {file_name}")

    if page_start is not None and page_end is not None:
        if page_start == page_end:
            meta_lines.append(f"페이지: {page_start}")
        else:
            meta_lines.append(f"페이지: {page_start}-{page_end}")

    if not meta_lines:
        return text

    return "\n".join(meta_lines) + "\n\n본문:\n" + text


def chunk_pdf_by_page(
    doc_id: str,
    pdf_path: str | Path,
    file_name: str = "",
    file_type: str = "pdf",
    project_name: str = "",
    organization: str = "",
    max_chars: int = 3000,
    overlap_chars: int = 300,
    min_chars: int = 500,
    merge_short_pages: bool = True,
    include_metadata_in_embedding_text: bool = True,
) -> List[Dict[str, Any]]:
    """
    PDF 파일을 페이지 기반으로 chunking합니다.

    Parameters
    ----------
    doc_id:
        문서 ID입니다.

    pdf_path:
        PDF 파일 경로입니다.

    file_name:
        원본 파일명입니다.

    file_type:
        파일 형식입니다. 보통 pdf입니다.

    project_name:
        사업명입니다.

    organization:
        발주 기관명입니다.

    max_chars:
        chunk 최대 문자 수입니다. 페이지 텍스트가 이보다 길면 재분할합니다.

    overlap_chars:
        긴 페이지를 재분할할 때 overlap 문자 수입니다.

    min_chars:
        페이지 텍스트가 이보다 짧으면 다음 페이지와 병합합니다.

    merge_short_pages:
        짧은 페이지 병합 여부입니다.

    include_metadata_in_embedding_text:
        embedding_text에 metadata 포함 여부입니다.

    Returns
    -------
    List[Dict[str, Any]]
        chunk dict 목록입니다.
    """
    pdf_path = Path(pdf_path)

    pages = _extract_pdf_pages_with_pymupdf(pdf_path)

    if merge_short_pages:
        page_units = _merge_short_pages(
            pages=pages,
            min_chars=min_chars,
            max_chars=max_chars,
        )
    else:
        page_units = [
            {
                "page_start": int(page["page_no"]),
                "page_end": int(page["page_no"]),
                "text": _normalize_text(page.get("text", "")),
            }
            for page in pages
            if _normalize_text(page.get("text", ""))
        ]

    chunks: List[Dict[str, Any]] = []
    global_chunk_index = 0

    for page_unit in page_units:
        page_start = int(page_unit["page_start"])
        page_end = int(page_unit["page_end"])
        page_text = _normalize_text(page_unit.get("text", ""))

        if not page_text:
            continue

        split_texts = _split_long_text_by_chars(
            text=page_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        for page_chunk_index, chunk_text in enumerate(split_texts):
            chunk_text = _normalize_text(chunk_text)

            if not chunk_text:
                continue

            if page_start == page_end:
                page_part = f"p{page_start:04d}"
            else:
                page_part = f"p{page_start:04d}_p{page_end:04d}"

            chunk_id = f"{doc_id}_{page_part}_c{page_chunk_index:03d}"

            embedding_text = _build_embedding_text(
                text=chunk_text,
                organization=organization,
                project_name=project_name,
                file_name=file_name,
                page_start=page_start,
                page_end=page_end,
                include_metadata=include_metadata_in_embedding_text,
            )

            chunks.append({
                "doc_id": str(doc_id),
                "chunk_id": chunk_id,
                "file_name": file_name,
                "file_type": file_type,
                "project_name": project_name,
                "organization": organization,
                "source_path": str(pdf_path),
                "page_start": page_start,
                "page_end": page_end,
                "page_chunk_index": page_chunk_index,
                "chunk_index": global_chunk_index,
                "section_id": f"page_{page_start:04d}_{page_end:04d}",
                "section_title": (
                    f"페이지 {page_start}"
                    if page_start == page_end
                    else f"페이지 {page_start}-{page_end}"
                ),
                "section_path": (
                    f"페이지 {page_start}"
                    if page_start == page_end
                    else f"페이지 {page_start}-{page_end}"
                ),
                "text": chunk_text,
                "embedding_text": embedding_text,
                "chunking_strategy": "pdf_page",
                "chunk_char_len": len(chunk_text),
            })

            global_chunk_index += 1

    return chunks


def analyze_pdf_page_split_lengths(
    pdf_path: str | Path,
    max_chars: int = 3000,
    overlap_chars: int = 300,
    min_chars: int = 500,
    merge_short_pages: bool = True,
) -> Dict[str, Any]:
    """
    PDF page 기반 chunking을 적용했을 때의 split 길이 통계를 계산합니다.

    최종 chunk 생성과 동일한 기준으로 page merge + long page split을 수행합니다.
    """
    pages = _extract_pdf_pages_with_pymupdf(pdf_path)

    if merge_short_pages:
        page_units = _merge_short_pages(
            pages=pages,
            min_chars=min_chars,
            max_chars=max_chars,
        )
    else:
        page_units = [
            {
                "page_start": int(page["page_no"]),
                "page_end": int(page["page_no"]),
                "text": _normalize_text(page.get("text", "")),
            }
            for page in pages
            if _normalize_text(page.get("text", ""))
        ]

    lengths: List[int] = []

    for page_unit in page_units:
        page_text = _normalize_text(page_unit.get("text", ""))

        split_texts = _split_long_text_by_chars(
            text=page_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        for split_text in split_texts:
            split_text = _normalize_text(split_text)

            if split_text:
                lengths.append(len(split_text))

    if not lengths:
        return {
            "num_pages": len(pages),
            "pre_chunk_count": 0,
            "pre_chunk_min_chars": 0,
            "pre_chunk_max_chars": 0,
            "pre_chunk_mean_chars": 0.0,
            "pre_chunk_median_chars": 0.0,
            "pre_chunk_lengths": [],
        }

    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)

    if n % 2 == 1:
        median = float(sorted_lengths[n // 2])
    else:
        median = float((sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2)

    return {
        "num_pages": len(pages),
        "pre_chunk_count": len(lengths),
        "pre_chunk_min_chars": int(min(lengths)),
        "pre_chunk_max_chars": int(max(lengths)),
        "pre_chunk_mean_chars": float(sum(lengths) / len(lengths)),
        "pre_chunk_median_chars": median,
        "pre_chunk_lengths": lengths,
    }