import re
from typing import List, Dict, Any


HEADING_PATTERNS = [
    # Ⅰ. 사업 개요
    r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s+.+",

    # I. 사업 개요
    r"^[IVX]+\.\s+.+",

    # 제1장 사업 개요 / 제 1 장 사업 개요
    r"^제\s*\d+\s*[장절]\s+.+",

    # 1. 사업 개요
    r"^\d+\.\s+.+",

    # 1.1 사업 목적
    r"^\d+(\.\d+)+\s+.+",

    # 가. 추진 배경
    r"^[가-힣]\.\s+.+",

    # (1) 주요 내용
    r"^\(\d+\)\s+.+",

    # □ 과업 범위
    r"^[□■○●\-]\s+.+",
]


def is_heading_line(line: str) -> bool:
    """
    한 줄이 제목/목차 후보인지 판단합니다.
    """
    line = line.strip()

    if not line:
        return False

    # 너무 긴 줄은 제목보다는 본문일 가능성이 높음
    if len(line) > 120:
        return False

    for pattern in HEADING_PATTERNS:
        if re.match(pattern, line):
            return True

    return False


def detect_headings(text: str) -> List[Dict[str, Any]]:
    """
    텍스트에서 제목 후보를 탐지합니다.

    Returns
    -------
    [
        {
            "line_idx": 10,
            "title": "1. 사업 개요"
        }
    ]
    """
    lines = text.splitlines()
    headings = []

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if is_heading_line(stripped):
            headings.append({
                "line_idx": idx,
                "title": stripped
            })

    return headings