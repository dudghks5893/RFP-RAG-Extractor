from pathlib import Path
from typing import Dict, Any, List

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl


def iter_block_items(doc: Document):
    """
    DOCX 문서의 body를 순회하면서 Paragraph와 Table을 원문 순서대로 반환합니다.
    """
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def table_to_text(table: Table) -> str:
    """
    Table 객체를 텍스트로 변환합니다.
    """
    rows = []

    for row in table.rows:
        cells = []

        for cell in row.cells:
            cell_text = cell.text.strip()

            if cell_text:
                cell_text = " ".join(cell_text.split())
                cells.append(cell_text)

        if cells:
            rows.append(" | ".join(cells))

    return "\n".join(rows)


def extract_docx_text(file_path: str | Path) -> Dict[str, Any]:
    """
    python-docx 기반 DOCX 텍스트 추출기입니다.

    문단과 표의 원문 순서를 최대한 유지합니다.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"DOCX 파일을 찾을 수 없습니다: {file_path}")

    doc = Document(file_path)

    blocks: List[Dict[str, Any]] = []
    paragraphs: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    table_idx = 0

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()

            if not text:
                continue

            style_name = block.style.name if block.style is not None else ""

            item = {
                "type": "paragraph",
                "style": style_name,
                "text": text
            }

            blocks.append(item)
            paragraphs.append(item)

        elif isinstance(block, Table):
            table_text = table_to_text(block)

            if not table_text.strip():
                continue

            item = {
                "type": "table",
                "table_idx": table_idx,
                "text": table_text
            }

            blocks.append(item)
            tables.append(item)
            table_idx += 1

    text_parts = []

    for block in blocks:
        if block["type"] == "paragraph":
            style = block.get("style", "")
            text = block.get("text", "")

            if style and style.lower().startswith("heading"):
                text_parts.append(f"\n{text}\n")
            else:
                text_parts.append(text)

        elif block["type"] == "table":
            text_parts.append("\n[표]\n")
            text_parts.append(block.get("text", ""))

    full_text = "\n".join(part for part in text_parts if part).strip()

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_type": "docx",
        "num_pages": None,
        "blocks": blocks,
        "paragraphs": paragraphs,
        "tables": tables,
        "pages": [
            {
                "page": None,
                "text": full_text
            }
        ],
        "text": full_text
    }