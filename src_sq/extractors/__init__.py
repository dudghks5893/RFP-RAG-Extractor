from pathlib import Path
from typing import Dict, Any

from src.extractors.pdf_extractor import extract_pdf_text
from src.extractors.docx_extractor import extract_docx_text
from src.extractors.hwp_extractor import extract_hwp_text


def extract_text_by_file_type(file_path: str | Path) -> Dict[str, Any]:
    """
    파일 확장자에 따라 적절한 추출기를 호출합니다.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(file_path)

    if suffix == ".docx":
        return extract_docx_text(file_path)

    if suffix == ".hwp":
        return extract_hwp_text(file_path)

    raise ValueError(f"지원하지 않는 파일 형식입니다: {file_path}")