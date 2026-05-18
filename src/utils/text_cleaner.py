import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """
    유니코드 정규화 함수입니다.

    NFKC는 전각/반각 문자, 호환 문자 등을 정규화합니다.
    RFP 문서에서 추출된 텍스트의 문자 표현을 통일하는 데 사용합니다.
    """
    return unicodedata.normalize("NFKC", text)


def remove_control_chars(text: str) -> str:
    """
    텍스트 추출 과정에서 섞일 수 있는 제어 문자를 제거합니다.

    단, 줄바꿈(\n), 탭(\t)은 뒤 단계에서 처리하기 위해 여기서는 직접 제거하지 않습니다.
    """
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)


def remove_toc_dots(text: str) -> str:
    """
    목차에서 자주 나타나는 점선 패턴을 정리합니다.

    예:
    제1장 사업개요 .......... 3
    → 제1장 사업개요 3
    """
    return re.sub(r"\.{3,}", " ", text)


def normalize_whitespace(text: str) -> str:
    """
    과도한 공백과 줄바꿈을 정리합니다.

    처리 내용:
    - 탭을 공백으로 변환
    - 2개 이상 연속 공백 축소
    - 줄바꿈 주변 공백 제거
    - 3개 이상 연속 줄바꿈을 2개로 축소
    """
    text = text.replace("\t", " ")

    # 줄 내부의 과도한 공백 정리
    text = re.sub(r"[ ]{2,}", " ", text)

    # 줄바꿈 앞뒤 공백 제거
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

    # 과도한 줄바꿈 정리
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_common_symbols(text: str) -> str:
    """
    문서마다 다르게 추출될 수 있는 일부 기호를 통일합니다.

    주의:
    RFP 문서에서는 특수문자가 의미를 가질 수 있으므로 무조건 제거하지 않습니다.
    예: %, 원, ㎡, ㎞, ※, ○, □, ① 등은 보존하는 것이 좋습니다.
    """
    replacements = {
        "–": "-",
        "—": "-",
        "−": "-",
        "‐": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "ㆍ": "·",
        "･": "·",
        "・": "·",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


def clean_extracted_text(text: str) -> str:
    """
    추출된 원시 텍스트에 대한 기본 정제 파이프라인입니다.

    이 함수는 PDF/HWP/DOCX에서 추출한 텍스트를 청킹 전에 정제할 때 사용합니다.

    처리 순서:
    1. None 방어
    2. 유니코드 정규화
    3. 기호 통일
    4. 제어문자 제거
    5. 목차 점선 제거
    6. 공백/줄바꿈 정리
    """
    if text is None:
        return ""

    text = str(text)
    text = normalize_unicode(text)
    text = normalize_common_symbols(text)
    text = remove_control_chars(text)
    text = remove_toc_dots(text)
    text = normalize_whitespace(text)

    return text