# scripts/convert_documents_to_pdf_v2.py
#
# data/raw 하위의 HWP/HWPX/DOC/DOCX/PDF 문서를 PDF 기준 데이터셋으로 변환합니다.
#
# 주요 동작:
# 1. data/raw/data_list.csv를 읽음
# 2. data_list.csv의 파일명 컬럼 기준으로 원본 파일을 찾음
# 3. hwp/hwpx/doc/docx 파일은 LibreOffice headless로 PDF 변환
# 4. 기존 pdf 파일은 변환하지 않고 data/raw/v2/로 복사
# 5. data/raw/v2/data_list_pdf.csv 생성
#
# 사용 예:
# 프로젝트 루트에서 실행
# python scripts/convert_documents_to_pdf_v2.py
#
# 기존 v2 PDF를 덮어쓰기
# python scripts/convert_documents_to_pdf_v2.py --overwrite
#
# data_list 경로 변경
# python scripts/convert_documents_to_pdf_v2.py --data-list-path data/raw/data_list.csv
#
# 출력 폴더 변경
# python scripts/convert_documents_to_pdf_v2.py --output-dir data/raw/v2

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import unicodedata
import re
from pathlib import Path
from typing import Optional

import pandas as pd


SUPPORTED_CONVERT_EXTENSIONS = {".hwp", ".hwpx", ".doc", ".docx"}
SUPPORTED_COPY_EXTENSIONS = {".pdf"}
SUPPORTED_INPUT_EXTENSIONS = SUPPORTED_CONVERT_EXTENSIONS | SUPPORTED_COPY_EXTENSIONS


def find_project_root_from_script(
    project_name: str = "RFP-RAG-Extractor",
) -> Path:
    """
    현재 스크립트 위치를 기준으로 프로젝트 루트 폴더를 찾습니다.
    """
    current = Path(__file__).resolve()

    for path in [current, *current.parents]:
        if path.name == project_name:
            return path

    raise FileNotFoundError(
        f"프로젝트 루트 폴더 '{project_name}'를 찾을 수 없습니다. 현재 위치: {current}"
    )


def normalize_file_name(name: str) -> str:
    """
    파일명 비교용 정규화 함수입니다.

    처리:
    - Unicode NFC 정규화
    - 소문자화
    - 모든 공백 제거
    - 일부 유사 기호 통일
    """
    if name is None or pd.isna(name):
        return ""

    name = str(name).strip()
    name = unicodedata.normalize("NFC", name)
    name = name.lower()
    name = re.sub(r"\s+", "", name)

    replacements = {
        "－": "-",
        "–": "-",
        "—": "-",
        "＿": "_",
        "（": "(",
        "）": ")",
        "㈜": "(주)",
        "（주）": "(주)",
    }

    for src, dst in replacements.items():
        name = name.replace(src, dst)

    return name


def check_libreoffice() -> str:
    """
    LibreOffice 실행 파일이 있는지 확인합니다.
    """
    candidates = ["libreoffice", "soffice"]

    for command in candidates:
        path = shutil.which(command)
        if path:
            return path

    raise RuntimeError(
        "LibreOffice 실행 파일을 찾을 수 없습니다.\n"
        "Ubuntu에서 아래 명령으로 설치하세요:\n"
        "sudo apt update && sudo apt install -y libreoffice"
    )


def collect_raw_files(raw_dir: Path, output_dir: Path) -> list[Path]:
    """
    data/raw 하위의 입력 문서를 수집합니다.

    단, output_dir(data/raw/v2) 내부 파일은 제외합니다.
    """
    raw_files = []

    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue

        # v2 출력 폴더 내부 파일은 입력 후보에서 제외
        try:
            path.relative_to(output_dir)
            continue
        except ValueError:
            pass

        if path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS:
            raw_files.append(path)

    return sorted(raw_files)


def find_source_file(
    raw_files: list[Path],
    file_name: str,
) -> Optional[Path]:
    """
    data_list.csv의 파일명에 해당하는 원본 파일을 찾습니다.

    매칭 순서:
    1. 파일명 완전 일치
    2. 정규화 후 파일명 전체 일치
    3. 정규화된 stem 일치
    4. stem 포함 관계

    같은 stem의 파일이 여러 개면 우선순위:
    1. 기존 PDF
    2. HWP/HWPX
    3. DOCX/DOC
    """
    if file_name is None or pd.isna(file_name):
        return None

    file_name = str(file_name).strip()

    target_name_norm = normalize_file_name(file_name)
    target_stem_norm = normalize_file_name(Path(file_name).stem)

    def sort_priority(path: Path) -> tuple[int, str]:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            priority = 0
        elif suffix in {".hwp", ".hwpx"}:
            priority = 1
        elif suffix in {".docx", ".doc"}:
            priority = 2
        else:
            priority = 9

        return priority, path.name

    # 1. 파일명 완전 일치
    exact_matches = [path for path in raw_files if path.name == file_name]
    if exact_matches:
        return sorted(exact_matches, key=sort_priority)[0]

    # 2. 정규화 후 파일명 전체 일치
    norm_matches = [
        path
        for path in raw_files
        if normalize_file_name(path.name) == target_name_norm
    ]
    if norm_matches:
        return sorted(norm_matches, key=sort_priority)[0]

    # 3. 정규화된 stem 일치
    stem_matches = [
        path
        for path in raw_files
        if normalize_file_name(path.stem) == target_stem_norm
    ]
    if stem_matches:
        return sorted(stem_matches, key=sort_priority)[0]

    # 4. stem 포함 관계
    contains_matches = []

    for path in raw_files:
        path_stem_norm = normalize_file_name(path.stem)

        if target_stem_norm:
            if target_stem_norm in path_stem_norm or path_stem_norm in target_stem_norm:
                contains_matches.append(path)

    if contains_matches:
        return sorted(contains_matches, key=sort_priority)[0]

    return None


def copy_pdf_to_v2(
    source_path: Path,
    output_dir: Path,
    output_pdf_name: str,
    overwrite: bool = False,
) -> tuple[bool, str, Path]:
    """
    기존 PDF 파일을 data/raw/v2로 복사합니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = output_dir / output_pdf_name

    if output_pdf_path.exists() and not overwrite:
        return True, "skipped_existing_pdf", output_pdf_path

    shutil.copy2(source_path, output_pdf_path)

    return True, "copied_pdf", output_pdf_path


def convert_office_to_pdf(
    libreoffice_cmd: str,
    source_path: Path,
    output_dir: Path,
    output_pdf_name: str,
    overwrite: bool = False,
    timeout_sec: int = 240,
) -> tuple[bool, str, Path]:
    """
    HWP/HWPX/DOC/DOCX 파일을 PDF로 변환합니다.

    LibreOffice는 기본적으로 source_path.stem + '.pdf' 이름으로 출력하므로,
    변환 후 원하는 output_pdf_name으로 rename합니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_pdf_path = output_dir / output_pdf_name
    libreoffice_pdf_path = output_dir / f"{source_path.stem}.pdf"

    if expected_pdf_path.exists() and not overwrite:
        return True, "skipped_existing_pdf", expected_pdf_path

    if expected_pdf_path.exists() and overwrite:
        expected_pdf_path.unlink()

    if libreoffice_pdf_path.exists() and overwrite:
        libreoffice_pdf_path.unlink()

    command = [
        libreoffice_cmd,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(source_path),
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )

        if result.returncode != 0:
            message = (
                "conversion_failed"
                f"\nstdout:\n{result.stdout}"
                f"\nstderr:\n{result.stderr}"
            )
            return False, message, expected_pdf_path

        if not libreoffice_pdf_path.exists():
            message = (
                "conversion_finished_but_pdf_not_found"
                f"\nexpected:\n{libreoffice_pdf_path}"
                f"\nstdout:\n{result.stdout}"
                f"\nstderr:\n{result.stderr}"
            )
            return False, message, expected_pdf_path

        if libreoffice_pdf_path != expected_pdf_path:
            libreoffice_pdf_path.rename(expected_pdf_path)

        return True, "converted_to_pdf", expected_pdf_path

    except subprocess.TimeoutExpired:
        return False, "timeout", expected_pdf_path

    except Exception as e:
        return False, repr(e), expected_pdf_path


def build_data_list_pdf(
    df: pd.DataFrame,
    output_pdf_names: list[str],
    file_name_col: str,
    file_type_col: str,
) -> pd.DataFrame:
    """
    변환 결과 기준으로 data_list_pdf.csv용 DataFrame을 생성합니다.
    """
    pdf_df = df.copy()
    pdf_df[file_name_col] = output_pdf_names

    if file_type_col in pdf_df.columns:
        pdf_df[file_type_col] = "pdf"

    return pdf_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HWP/HWPX/DOC/DOCX files to PDF and copy existing PDFs into data/raw/v2."
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="RFP-RAG-Extractor",
        help="프로젝트 루트 폴더 이름입니다.",
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="원본 문서와 data_list.csv가 있는 폴더입니다.",
    )

    parser.add_argument(
        "--data-list-path",
        type=str,
        default="data/raw/data_list.csv",
        help="원본 data_list.csv 경로입니다.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/v2",
        help="PDF 변환/복사 결과를 저장할 폴더입니다.",
    )

    parser.add_argument(
        "--output-data-list-name",
        type=str,
        default="data_list_pdf.csv",
        help="output-dir 안에 저장할 PDF용 data_list 파일명입니다.",
    )

    parser.add_argument(
        "--file-name-col",
        type=str,
        default="파일명",
        help="data_list.csv에서 파일명을 담고 있는 컬럼명입니다.",
    )

    parser.add_argument(
        "--file-type-col",
        type=str,
        default="파일형식",
        help="data_list.csv에서 파일형식을 담고 있는 컬럼명입니다.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 v2 PDF가 있어도 다시 변환/복사합니다.",
    )

    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=240,
        help="파일 1개 변환 제한 시간입니다.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = find_project_root_from_script(args.project_name)

    raw_dir = project_root / args.raw_dir
    data_list_path = project_root / args.data_list_path
    output_dir = project_root / args.output_dir
    output_data_list_path = output_dir / args.output_data_list_name

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir를 찾을 수 없습니다: {raw_dir}")

    if not data_list_path.exists():
        raise FileNotFoundError(f"data_list.csv를 찾을 수 없습니다: {data_list_path}")

    libreoffice_cmd = check_libreoffice()

    df = pd.read_csv(data_list_path)

    if args.file_name_col not in df.columns:
        raise KeyError(
            f"'{args.file_name_col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}"
        )

    raw_files = collect_raw_files(raw_dir=raw_dir, output_dir=output_dir)

    print("===== Convert Documents to PDF v2 =====")
    print("project_root          :", project_root)
    print("raw_dir               :", raw_dir)
    print("data_list_path        :", data_list_path)
    print("output_dir            :", output_dir)
    print("output_data_list_path :", output_data_list_path)
    print("libreoffice_cmd       :", libreoffice_cmd)
    print("num_rows              :", len(df))
    print("num_raw_files         :", len(raw_files))
    print("overwrite             :", args.overwrite)
    print("=======================================")

    output_dir.mkdir(parents=True, exist_ok=True)

    logs: list[dict] = []
    output_pdf_names: list[str] = []

    converted_count = 0
    copied_count = 0
    skipped_count = 0
    failed_count = 0
    not_found_count = 0

    for idx, row in df.iterrows():
        source_file_name = row.get(args.file_name_col)
        source_path = find_source_file(raw_files, source_file_name)

        # data_list_pdf.csv에는 원래 파일명 stem + .pdf로 기록
        output_pdf_name = f"{Path(str(source_file_name)).stem}.pdf"
        output_pdf_names.append(output_pdf_name)

        if source_path is None:
            not_found_count += 1

            log = {
                "row_index": idx,
                "source_file_name": source_file_name,
                "source_path": "",
                "output_pdf_name": output_pdf_name,
                "output_pdf_path": str(output_dir / output_pdf_name),
                "status": "source_not_found",
                "message": "원본 파일을 찾지 못했습니다.",
            }
            logs.append(log)

            print(
                f"[{idx + 1}/{len(df)}] NOT FOUND | "
                f"{source_file_name} -> {output_pdf_name}"
            )
            continue

        suffix = source_path.suffix.lower()

        if suffix == ".pdf":
            ok, message, output_pdf_path = copy_pdf_to_v2(
                source_path=source_path,
                output_dir=output_dir,
                output_pdf_name=output_pdf_name,
                overwrite=args.overwrite,
            )
        elif suffix in SUPPORTED_CONVERT_EXTENSIONS:
            ok, message, output_pdf_path = convert_office_to_pdf(
                libreoffice_cmd=libreoffice_cmd,
                source_path=source_path,
                output_dir=output_dir,
                output_pdf_name=output_pdf_name,
                overwrite=args.overwrite,
                timeout_sec=args.timeout_sec,
            )
        else:
            ok = False
            message = f"unsupported_extension: {suffix}"
            output_pdf_path = output_dir / output_pdf_name

        if ok and message == "copied_pdf":
            copied_count += 1
        elif ok and message == "converted_to_pdf":
            converted_count += 1
        elif ok and message == "skipped_existing_pdf":
            skipped_count += 1
        else:
            failed_count += 1

        logs.append({
            "row_index": idx,
            "source_file_name": source_file_name,
            "source_path": str(source_path),
            "source_suffix": suffix,
            "output_pdf_name": output_pdf_name,
            "output_pdf_path": str(output_pdf_path),
            "status": "success" if ok else "failed",
            "message": message,
        })

        print(
            f"[{idx + 1}/{len(df)}] {source_path.name} -> "
            f"{output_pdf_name} | {message}"
        )

    # data_list_pdf.csv 생성
    pdf_df = build_data_list_pdf(
        df=df,
        output_pdf_names=output_pdf_names,
        file_name_col=args.file_name_col,
        file_type_col=args.file_type_col,
    )

    pdf_df.to_csv(
        output_data_list_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 변환 로그 저장
    log_path = output_dir / "convert_documents_to_pdf_log.csv"
    log_df = pd.DataFrame(logs)
    log_df.to_csv(
        log_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== Summary =====")
    print("converted :", converted_count)
    print("copied    :", copied_count)
    print("skipped   :", skipped_count)
    print("not_found :", not_found_count)
    print("failed    :", failed_count)
    print("output data_list_pdf:", output_data_list_path)
    print("conversion log      :", log_path)

    if failed_count > 0 or not_found_count > 0:
        print("\n일부 파일 변환/매칭에 실패했습니다. 로그를 확인하세요:")
        print(log_path)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())