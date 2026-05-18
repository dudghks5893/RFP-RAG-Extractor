from pathlib import Path
from typing import Dict, Any, List

import fitz  # PyMuPDF


def extract_pdf_text(file_path: str | Path) -> Dict[str, Any]:
    """
    PDF 파일에서 페이지별 텍스트를 추출합니다.

    Returns
    -------
    Dict[str, Any]
        {
            "file_path": "...",
            "file_type": "pdf",
            "num_pages": 10,
            "pages": [
                {"page": 1, "text": "..."},
                {"page": 2, "text": "..."}
            ],
            "text": "전체 텍스트"
        }
    """
    file_path = Path(file_path)

    pages: List[Dict[str, Any]] = []

    with fitz.open(file_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append({
                "page": page_idx,
                "text": text
            })

    full_text = "\n\n".join(page["text"] for page in pages)

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_type": "pdf",
        "num_pages": len(pages),
        "pages": pages,
        "text": full_text
    }