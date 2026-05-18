from pathlib import Path
from typing import Dict, Any, List
import re
import zlib

import olefile


def is_compressed_hwp(ole: olefile.OleFileIO) -> bool:
    """
    HWP 파일의 FileHeader를 읽어 압축 여부를 확인합니다.

    HWP 5.x의 FileHeader 36번째 byte flag에서 압축 여부를 확인할 수 있습니다.
    flag & 0x01 이면 BodyText가 zlib 압축되어 있습니다.
    """
    if not ole.exists("FileHeader"):
        return False

    header = ole.openstream("FileHeader").read()

    if len(header) < 37:
        return False

    flags = header[36]
    return bool(flags & 0x01)


def get_bodytext_section_names(ole: olefile.OleFileIO) -> List[str]:
    """
    HWP의 BodyText/Section* 스트림 목록을 가져옵니다.
    """
    streams = ole.listdir()

    section_names = []

    for stream in streams:
        # stream 예시: ["BodyText", "Section0"]
        if len(stream) == 2 and stream[0] == "BodyText" and stream[1].startswith("Section"):
            section_names.append("/".join(stream))

    # Section0, Section1, Section2 순서 정렬
    section_names.sort(
        key=lambda x: int(re.sub(r"[^0-9]", "", x.split("/")[-1]) or 0)
    )

    return section_names


def read_section_stream(
    ole: olefile.OleFileIO,
    section_name: str,
    compressed: bool
) -> bytes:
    """
    BodyText/Section* 스트림을 읽고, 압축되어 있으면 zlib로 해제합니다.
    """
    raw_data = ole.openstream(section_name).read()

    if compressed:
        try:
            # HWP BodyText는 raw deflate인 경우가 많아서 -15 사용
            return zlib.decompress(raw_data, -15)
        except zlib.error:
            # 환경/파일에 따라 일반 zlib header가 있는 경우 대비
            return zlib.decompress(raw_data)

    return raw_data


def extract_text_from_hwp_section(section_data: bytes) -> str:
    """
    HWP Section 바이너리에서 텍스트 레코드만 추출합니다.

    HWP 5.x 레코드 구조:
    - record header 4 bytes
    - tag_id: 하위 10 bits
    - level: 다음 10 bits
    - size: 다음 12 bits

    텍스트 레코드 tag_id는 일반적으로 67입니다.
    """
    pos = 0
    texts = []

    while pos + 4 <= len(section_data):
        header = int.from_bytes(section_data[pos:pos + 4], byteorder="little")
        pos += 4

        tag_id = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF

        # size가 0xFFF이면 확장 크기 4 bytes가 뒤에 옵니다.
        if size == 0xFFF:
            if pos + 4 > len(section_data):
                break

            size = int.from_bytes(section_data[pos:pos + 4], byteorder="little")
            pos += 4

        record_data = section_data[pos:pos + size]
        pos += size

        # HWP PARA_TEXT tag_id
        if tag_id == 67:
            try:
                text = record_data.decode("utf-16le", errors="ignore")
                if text:
                    texts.append(text)
            except Exception:
                continue

    return "\n".join(texts)


def clean_hwp_extracted_text(text: str) -> str:
    """
    olefile 기반 추출 후 HWP 특유의 제어문자/깨진 문자를 1차 정리합니다.

    본격적인 정제는 src/preprocessing/text_cleaner.py에서 다시 수행합니다.
    """
    if text is None:
        return ""

    text = str(text)

    # HWP 텍스트 안에 섞이는 제어문자 일부 제거
    text = text.replace("\x00", "")
    text = text.replace("\r", "\n")

    # 유니코드 replacement character 제거
    text = text.replace("\ufffd", "")

    # 과도한 공백/줄바꿈 1차 정리
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_hwp_text(file_path: str | Path) -> Dict[str, Any]:
    """
    olefile 기반 HWP 텍스트 추출기입니다.

    Parameters
    ----------
    file_path:
        .hwp 파일 경로

    Returns
    -------
    Dict[str, Any]
        {
            "file_path": "...",
            "file_name": "...",
            "file_type": "hwp",
            "num_pages": None,
            "sections": [
                {"section": "BodyText/Section0", "text": "..."},
                ...
            ],
            "pages": [
                {"page": None, "text": "전체 텍스트"}
            ],
            "text": "전체 텍스트"
        }
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"HWP 파일을 찾을 수 없습니다: {file_path}")

    if not olefile.isOleFile(str(file_path)):
        raise ValueError(f"OLE 기반 HWP 파일이 아닙니다: {file_path}")

    sections = []

    with olefile.OleFileIO(str(file_path)) as ole:
        compressed = is_compressed_hwp(ole)
        section_names = get_bodytext_section_names(ole)

        if not section_names:
            raise ValueError(f"BodyText/Section 스트림을 찾을 수 없습니다: {file_path}")

        for section_name in section_names:
            section_data = read_section_stream(
                ole=ole,
                section_name=section_name,
                compressed=compressed
            )

            section_text = extract_text_from_hwp_section(section_data)
            section_text = clean_hwp_extracted_text(section_text)

            if section_text:
                sections.append({
                    "section": section_name,
                    "text": section_text
                })

    full_text = "\n\n".join(section["text"] for section in sections)
    full_text = clean_hwp_extracted_text(full_text)

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_type": "hwp",
        "num_pages": None,
        "sections": sections,
        "pages": [
            {
                "page": None,
                "text": full_text
            }
        ],
        "text": full_text
    }