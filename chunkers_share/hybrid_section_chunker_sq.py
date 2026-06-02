from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:
    from src.utils.text_cleaner import preprocess_text_for_section_chunking
except ImportError:
    preprocess_text_for_section_chunking = None


HEADING_SPECS = [
    (1, r"^(?P<marker>[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\.?\s+(?P<title>.+)$"),
    (1, r"^(?P<marker>[IVXLCDM]+)\.?\s+(?P<title>.+)$"),
    (1, r"^(?P<marker>제\s*\d+\s*장)\s+(?P<title>.+)$"),
    (2, r"^(?P<marker>제\s*\d+\s*절)\s+(?P<title>.+)$"),
    (3, r"^(?P<marker>\d+(?:\.\d+)+)\.?\s+(?P<title>.+)$"),
    (2, r"^(?P<marker>\d+)\.\s+(?P<title>.+)$"),
    (3, r"^(?P<marker>[가-힣])\.\s+(?P<title>.+)$"),
    (4, r"^(?P<marker>\(\d+\))\s+(?P<title>.+)$"),
    (4, r"^(?P<marker>[①-⑮])\s*(?P<title>.+)$"),
    (4, r"^(?P<marker>[□■○●\-])\s+(?P<title>.+)$"),
]

TOC_TITLE = {"목차", "목 차", "차례", "Contents", "CONTENTS"}


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare_text_for_chunking(text: str, apply_preprocess: bool = True) -> str:
    text = normalize_text(text)

    if apply_preprocess and preprocess_text_for_section_chunking is not None:
        text = preprocess_text_for_section_chunking(text)

    return normalize_text(text)


def clean_title(title: str) -> str:
    title = str(title or "").strip()
    title = re.sub(r"\.{3,}", " ", title)
    title = re.sub(r"[\s.\-·…]*\d{1,4}\s*$", "", title).strip()
    title = re.sub(r"\s+", " ", title)
    return title


def parse_heading_line(line: str) -> Optional[Dict[str, Any]]:
    raw = str(line or "").strip()

    if not raw or len(raw) > 120:
        return None

    for level, pattern in HEADING_SPECS:
        match = re.match(pattern, raw)

        if not match:
            continue

        title = clean_title(match.group("title"))

        if not title:
            return None

        return {
            "title": title,
            "level": level,
            "marker": match.group("marker").strip(),
            "raw": raw,
        }

    return None


def is_toc_like_line(line: str) -> bool:
    line = normalize_text(line)

    if not line or not re.search(r"\s+\d{1,4}$", line):
        return False

    toc_patterns = [
        r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.?\s+.+\s+\d{1,4}$",
        r"^[IVXLCDM]+\.?\s+.+\s+\d{1,4}$",
        r"^\d+(?:\.\d+)*[.)]?\s+.+\s+\d{1,4}$",
        r"^[가-힣]\.\s+.+\s+\d{1,4}$",
        r"^\[?붙임\s*\d+\]?.+\s+\d{1,4}$",
        r"^\[?별지.+\]?.+\s+\d{1,4}$",
        r"^\[?별첨.+\]?.+\s+\d{1,4}$",
        r"^\[?별표.+\]?.+\s+\d{1,4}$",
    ]
    return any(re.fullmatch(pattern, line) for pattern in toc_patterns)


def detect_headings(text: str) -> List[Dict[str, Any]]:
    headings: List[Dict[str, Any]] = []

    for line_idx, line in enumerate(text.splitlines()):
        parsed = parse_heading_line(line)

        if parsed is None:
            continue

        headings.append({
            "line_idx": line_idx,
            **parsed,
        })

    return headings


def is_noise_heading(heading: Dict[str, Any], lines: List[str]) -> bool:
    line_idx = heading["line_idx"]

    if line_idx < 0 or line_idx >= len(lines):
        return True

    title = heading.get("title", "").strip()
    raw = lines[line_idx].strip()

    if not title or title in TOC_TITLE:
        return True

    if is_toc_like_line(raw):
        return True

    if len(title) > 100:
        return True

    return False


def deduplicate_headings(
    headings: List[Dict[str, Any]],
    lines: List[str],
) -> List[Dict[str, Any]]:
    headings = sorted(headings, key=lambda item: (item["line_idx"], item["level"]))
    deduped: List[Dict[str, Any]] = []
    seen_line_idx = set()

    for heading in headings:
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
    level = heading["level"]
    section_stack = [item for item in section_stack if item["level"] < level]
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
    include_deeper_level: bool,
) -> bool:
    if include_deeper_level:
        return heading["level"] >= target_level

    return heading["level"] == target_level


def make_section_id(doc_id: str, section_path: List[str], heading_idx: int) -> str:
    safe_path = "_".join(section_path[-3:]) if section_path else "section"
    safe_path = re.sub(r"\s+", "_", safe_path)
    safe_path = re.sub(r"[^0-9A-Za-z가-힣_]+", "", safe_path)
    safe_path = safe_path[:80] or "section"
    return f"{doc_id}_hybrid_{heading_idx:04d}_{safe_path}"


def split_long_text(
    text: str,
    max_chars: int = 3000,
    overlap_chars: int = 300,
) -> List[str]:
    text = normalize_text(text)

    if not text:
        return []

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

        if overlap_chars <= 0:
            buffer = []
            buffer_len = 0
            return

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

    for unit in units:
        unit_len = len(unit) + 1

        if unit_len > max_chars:
            flush_buffer()

            start = 0
            while start < len(unit):
                end = min(start + max_chars, len(unit))
                chunk = unit[start:end].strip()

                if end < len(unit):
                    cut = max(
                        chunk.rfind("\n"),
                        chunk.rfind("."),
                        chunk.rfind("다."),
                        chunk.rfind("함"),
                    )

                    if cut > int(max_chars * 0.6):
                        end = start + cut + 1
                        chunk = unit[start:end].strip()

                if chunk:
                    chunks.append(chunk)

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


def split_sections_by_toc(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    has_roman = bool(re.search(r"^\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|\b[IVXLCDM]+\b)\.?", text, re.MULTILINE))
    has_chapter = bool(re.search(r"^\s*제\s*\d+\s*장", text, re.MULTILINE))

    if has_roman or has_chapter:
        regex = (
            r"^\s*("
            r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.?"
            r"|\b[IVXLCDM]+\b\.?"
            r"|제\s*\d+\s*장"
            r")\s+[^\n]{1,80}$"
        )
    else:
        regex = r"^\s*(\d+\.?)\s+[^\n]{2,80}$"

    matches = list(re.finditer(regex, text, re.MULTILINE))

    if len(matches) < 2:
        return []

    sections: List[Dict[str, Any]] = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        title = clean_title(match.group().strip())

        if len(section_text) > 100:
            sections.append({
                "title": title,
                "text": section_text,
            })

    return sections


def base_metadata(
    doc_id: str,
    file_name: str,
    file_type: str,
    project_name: str,
    organization: str,
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "file_type": file_type,
        "project_name": project_name,
        "organization": organization,
    }


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
    chunks: List[Dict[str, Any]] = []

    for idx, chunk_text in enumerate(split_long_text(text, max_chars, overlap_chars)):
        chunk_text = chunk_text.strip()

        if len(chunk_text) < min_chars:
            continue

        chunks.append({
            "chunk_id": f"{doc_id}_hybrid_fallback_{idx:04d}",
            "section_id": f"{doc_id}_hybrid_fallback",
            **base_metadata(doc_id, file_name, file_type, project_name, organization),
            "chunking_method": "hybrid_outline_toc",
            "chunking_strategy": "hybrid_fallback_char",
            "section_title": "",
            "section_path": [],
            "section_path_with_level": [],
            "section_level": None,
            "start_line": None,
            "end_line": None,
            "split_idx": idx,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "char_len": len(chunk_text),
        })

    return chunks


def create_toc_chunks(
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
    chunks: List[Dict[str, Any]] = []

    for section_idx, section in enumerate(split_sections_by_toc(text)):
        section_title = clean_title(section.get("title", ""))
        section_text = str(section.get("text", "")).strip()
        section_id = f"{doc_id}_hybrid_toc_{section_idx:04d}"

        for split_idx, chunk_text in enumerate(split_long_text(section_text, max_chars, overlap_chars)):
            chunk_text = chunk_text.strip()

            if len(chunk_text) < min_chars:
                continue

            chunks.append({
                "chunk_id": f"{doc_id}_hybrid_toc_{section_idx:04d}_{split_idx:02d}",
                "section_id": section_id,
                **base_metadata(doc_id, file_name, file_type, project_name, organization),
                "chunking_method": "hybrid_outline_toc",
                "chunking_strategy": "hybrid_toc_section",
                "section_title": section_title,
                "section_path": [section_title] if section_title else [],
                "section_path_with_level": [{
                    "level": 1,
                    "title": section_title,
                    "line_idx": None,
                    "marker": "",
                }] if section_title else [],
                "section_level": 1 if section_title else None,
                "heading_marker": "",
                "heading_raw": section_title,
                "start_line": None,
                "end_line": None,
                "split_idx": split_idx,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "char_len": len(chunk_text),
            })

    return chunks


def create_outline_chunks(
    doc_id: str,
    text: str,
    headings: List[Dict[str, Any]],
    file_name: str = "",
    file_type: str = "",
    project_name: str = "",
    organization: str = "",
    max_chars: int = 3000,
    overlap_chars: int = 300,
    min_chars: int = 100,
    target_level: int = 2,
    include_deeper_level: bool = False,
    keep_heading: bool = True,
) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    chunks: List[Dict[str, Any]] = []
    section_stack: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, Any]] = None

    def flush_current_section(end_line: int) -> None:
        nonlocal current_section

        if current_section is None:
            return

        start_line = current_section["start_line"]
        bounded_end_line = max(start_line, min(end_line, len(lines)))
        section_lines = lines[start_line:bounded_end_line]

        if not keep_heading and section_lines:
            section_lines = section_lines[1:]

        section_text = "\n".join(section_lines).strip()

        if len(section_text) < min_chars:
            current_section = None
            return

        for split_idx, chunk_text in enumerate(split_long_text(section_text, max_chars, overlap_chars)):
            chunk_text = chunk_text.strip()

            if len(chunk_text) < min_chars:
                continue

            chunks.append({
                "chunk_id": (
                    f"{doc_id}_hybrid_section_"
                    f"{current_section['heading_idx']:04d}_{split_idx:02d}"
                ),
                "section_id": current_section["section_id"],
                **base_metadata(doc_id, file_name, file_type, project_name, organization),
                "chunking_method": "hybrid_outline_toc",
                "chunking_strategy": "hybrid_outline_section",
                "section_title": current_section["title"],
                "section_path": current_section["section_path"],
                "section_path_with_level": current_section["section_path_with_level"],
                "section_level": current_section["level"],
                "heading_marker": current_section.get("marker", ""),
                "heading_raw": current_section.get("raw", ""),
                "start_line": start_line,
                "end_line": bounded_end_line,
                "split_idx": split_idx,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "char_len": len(chunk_text),
            })

        current_section = None

    for heading_idx, heading in enumerate(headings):
        line_idx = heading["line_idx"]

        if line_idx < 0 or line_idx >= len(lines):
            continue

        starts_new_chunk = should_start_chunk(heading, target_level, include_deeper_level)

        if current_section is not None:
            current_level = current_section["level"]
            closes_current_section = starts_new_chunk or heading["level"] <= current_level

            if closes_current_section:
                flush_current_section(line_idx)

        section_stack = update_section_stack(section_stack, heading)

        if starts_new_chunk:
            section_path = get_section_path(section_stack)
            current_section = {
                "section_id": make_section_id(doc_id, section_path, heading_idx),
                "heading_idx": heading_idx,
                "start_line": line_idx,
                "title": heading["title"],
                "level": heading["level"],
                "marker": heading.get("marker", ""),
                "raw": heading.get("raw", ""),
                "section_path": section_path,
                "section_path_with_level": get_section_path_with_level(section_stack),
            }

    flush_current_section(len(lines))
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
    min_chars: int = 100,
    target_level: int = 2,
    include_deeper_level: bool = False,
    keep_heading: bool = True,
    apply_preprocess: bool = True,
    fallback_to_higher_level: bool = True,
    use_toc_fallback: bool = True,
) -> List[Dict[str, Any]]:
    text = prepare_text_for_chunking(text, apply_preprocess=apply_preprocess)

    if not text:
        return []

    lines = text.splitlines()
    headings = deduplicate_headings(detect_headings(text), lines)

    if headings:
        for level in range(target_level, 0, -1):
            chunks = create_outline_chunks(
                doc_id=doc_id,
                text=text,
                headings=headings,
                file_name=file_name,
                file_type=file_type,
                project_name=project_name,
                organization=organization,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
                target_level=level,
                include_deeper_level=include_deeper_level,
                keep_heading=keep_heading,
            )

            if chunks or not fallback_to_higher_level:
                return chunks

    if use_toc_fallback:
        chunks = create_toc_chunks(
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

        if chunks:
            return chunks

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


create_hybrid_section_chunks = create_section_chunks
