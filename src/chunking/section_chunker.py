from typing import List, Dict, Any

from src.chunking.outline_detector import detect_headings


def split_long_text(
    text: str,
    max_chars: int = 3000,
    overlap_chars: int = 300
) -> List[str]:
    """
    너무 긴 섹션을 문자 기준으로 재분할합니다.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap_chars)

    return chunks


def create_section_chunks(
    doc_id: str,
    text: str,
    file_name: str = "",
    file_type: str = "",
    project_name: str = "",
    organization: str = "",
    max_chars: int = 3000,
    overlap_chars: int = 300,
    min_chars: int = 100
) -> List[Dict[str, Any]]:
    """
    제목 후보를 기준으로 섹션 단위 청크를 생성합니다.

    제목 탐지가 하나도 안 되면 전체 문서를 max_chars 기준으로 분할합니다.
    """
    lines = text.splitlines()
    headings = detect_headings(text)

    section_chunks = []

    # 제목 탐지 실패 시 전체 문서를 길이 기준으로 분할
    if not headings:
        split_texts = split_long_text(
            text=text,
            max_chars=max_chars,
            overlap_chars=overlap_chars
        )

        for idx, chunk_text in enumerate(split_texts):
            if len(chunk_text.strip()) < min_chars:
                continue

            section_chunks.append({
                "chunk_id": f"{doc_id}_section_{idx:04d}",
                "doc_id": doc_id,
                "file_name": file_name,
                "file_type": file_type,
                "project_name": project_name,
                "organization": organization,
                "chunking_strategy": "section_fallback_char",
                "section_title": "",
                "section_path": [],
                "text": chunk_text
            })

        return section_chunks

    # 제목 위치 기준으로 섹션 경계 생성
    for h_idx, heading in enumerate(headings):
        start_line = heading["line_idx"]

        if h_idx + 1 < len(headings):
            end_line = headings[h_idx + 1]["line_idx"]
        else:
            end_line = len(lines)

        title = heading["title"]
        section_text = "\n".join(lines[start_line:end_line]).strip()

        if len(section_text) < min_chars:
            continue

        split_texts = split_long_text(
            text=section_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars
        )

        for split_idx, chunk_text in enumerate(split_texts):
            section_chunks.append({
                "chunk_id": f"{doc_id}_section_{h_idx:04d}_{split_idx:02d}",
                "doc_id": doc_id,
                "file_name": file_name,
                "file_type": file_type,
                "project_name": project_name,
                "organization": organization,
                "chunking_strategy": "section",
                "section_title": title,
                "section_path": [title],
                "text": chunk_text
            })

    return section_chunks