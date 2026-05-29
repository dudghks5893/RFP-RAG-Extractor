# scripts/retry_failed_pdf_conversions.py
#
# 기존 convert_documents_to_pdf_log.csv에서 실패한 파일만 다시 PDF 변환합니다.
#
# 목적:
# - 이미 성공한 PDF는 건드리지 않음
# - 실패한 파일만 /tmp의 짧은 ASCII 파일명으로 복사 후 변환
# - LibreOffice가 긴 한글/NFD 출력 경로에 직접 저장하면서 발생하는
#   impl_store 0x507 오류를 피함
#
# 사용 예:
# python scripts/retry_failed_pdf_conversions.py
#
# 기존 실패 PDF가 일부 생성되어 있으면 덮어쓰기:
# python scripts/retry_failed_pdf_conversions.py --overwrite

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def find_project_root_from_script(
    project_name: str = "RFP-RAG-Extractor",
) -> Path:
    current = Path(__file__).resolve()

    for path in [current, *current.parents]:
        if path.name == project_name:
            return path

    raise FileNotFoundError(
        f"프로젝트 루트 폴더 '{project_name}'를 찾을 수 없습니다. 현재 위치: {current}"
    )


def check_libreoffice() -> str:
    for command in ["libreoffice", "soffice"]:
        path = shutil.which(command)
        if path:
            return path

    raise RuntimeError(
        "LibreOffice 실행 파일을 찾을 수 없습니다. "
        "sudo apt install -y libreoffice 로 설치하세요."
    )


def convert_with_temp_ascii_paths(
    libreoffice_cmd: str,
    source_path: Path,
    output_pdf_path: Path,
    overwrite: bool = False,
    timeout_sec: int = 300,
) -> tuple[bool, str]:
    """
    실패 파일 재시도용 변환 함수입니다.

    원본 파일을 /tmp의 짧은 ASCII 파일명으로 복사한 뒤,
    LibreOffice가 /tmp/input.pdf로 저장하게 하고,
    최종 output_pdf_path로 Python이 복사합니다.

    이렇게 하면 LibreOffice가 긴 한글/NFD 파일명으로 직접 저장하지 않으므로
    impl_store 0x507 저장 실패를 줄일 수 있습니다.
    """
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if output_pdf_path.exists() and not overwrite:
        return True, "skipped_existing_pdf"

    if output_pdf_path.exists() and overwrite:
        output_pdf_path.unlink()

    source_suffix = source_path.suffix.lower()

    with tempfile.TemporaryDirectory(prefix="retry_lo_convert_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        temp_input_path = tmp_dir / f"input{source_suffix}"
        temp_output_pdf_path = tmp_dir / "input.pdf"

        shutil.copy2(source_path, temp_input_path)

        command = [
            libreoffice_cmd,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(tmp_dir),
            str(temp_input_path),
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
                return (
                    False,
                    "conversion_failed"
                    f"\nstdout:\n{result.stdout}"
                    f"\nstderr:\n{result.stderr}",
                )

            if not temp_output_pdf_path.exists():
                return (
                    False,
                    "conversion_finished_but_pdf_not_found"
                    f"\nexpected:\n{temp_output_pdf_path}"
                    f"\nstdout:\n{result.stdout}"
                    f"\nstderr:\n{result.stderr}",
                )

            shutil.copy2(temp_output_pdf_path, output_pdf_path)

            return True, "converted_to_pdf_temp_ascii"

        except subprocess.TimeoutExpired:
            return False, "timeout"

        except Exception as e:
            return False, repr(e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry only failed HWP/DOC/DOCX to PDF conversions."
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="RFP-RAG-Extractor",
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="data/raw/v2/convert_documents_to_pdf_log.csv",
        help="기존 변환 로그 CSV 경로입니다.",
    )

    parser.add_argument(
        "--output-log-path",
        type=str,
        default="data/raw/v2/retry_failed_pdf_conversions_log.csv",
        help="재시도 결과 로그 CSV 경로입니다.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 출력 PDF가 있으면 덮어씁니다.",
    )

    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="파일 1개 변환 제한 시간입니다.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = find_project_root_from_script(args.project_name)

    log_path = project_root / args.log_path
    output_log_path = project_root / args.output_log_path

    if not log_path.exists():
        raise FileNotFoundError(f"변환 로그 파일을 찾을 수 없습니다: {log_path}")

    libreoffice_cmd = check_libreoffice()

    df = pd.read_csv(log_path)

    required_cols = ["source_path", "output_pdf_path", "status"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(
            f"로그 CSV에 필요한 컬럼이 없습니다: {missing_cols}. "
            f"현재 컬럼: {list(df.columns)}"
        )

    failed_df = df[df["status"] != "success"].copy()

    print("===== Retry Failed PDF Conversions =====")
    print("project_root    :", project_root)
    print("log_path        :", log_path)
    print("output_log_path :", output_log_path)
    print("libreoffice_cmd :", libreoffice_cmd)
    print("failed_count    :", len(failed_df))
    print("overwrite       :", args.overwrite)
    print("========================================")

    if len(failed_df) == 0:
        print("재시도할 실패 파일이 없습니다.")
        return 0

    retry_logs = []

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for retry_idx, (_, row) in enumerate(failed_df.iterrows(), start=1):
        source_path = Path(str(row["source_path"]))
        output_pdf_path = Path(str(row["output_pdf_path"]))

        if not source_path.exists():
            ok = False
            message = "source_not_found"
        else:
            ok, message = convert_with_temp_ascii_paths(
                libreoffice_cmd=libreoffice_cmd,
                source_path=source_path,
                output_pdf_path=output_pdf_path,
                overwrite=args.overwrite,
                timeout_sec=args.timeout_sec,
            )

        if ok and message == "skipped_existing_pdf":
            skipped_count += 1
        elif ok:
            success_count += 1
        else:
            failed_count += 1

        retry_logs.append({
            "retry_index": retry_idx,
            "original_row_index": row.get("row_index"),
            "source_file_name": row.get("source_file_name"),
            "source_path": str(source_path),
            "output_pdf_name": row.get("output_pdf_name"),
            "output_pdf_path": str(output_pdf_path),
            "previous_status": row.get("status"),
            "previous_message": row.get("message"),
            "retry_status": "success" if ok else "failed",
            "retry_message": message,
        })

        print(
            f"[{retry_idx}/{len(failed_df)}] "
            f"{source_path.name} -> {output_pdf_path.name} | {message}"
        )

    retry_log_df = pd.DataFrame(retry_logs)

    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    retry_log_df.to_csv(
        output_log_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== Retry Summary =====")
    print("success:", success_count)
    print("skipped:", skipped_count)
    print("failed :", failed_count)
    print("retry log:", output_log_path)

    # 기존 log도 retry 성공분 반영해서 업데이트하고 싶다면 여기서 status를 수정합니다.
    # 원본 convert_documents_to_pdf_log.csv를 최신 상태로 유지하기 위함입니다.
    updated_df = df.copy()

    for retry_row in retry_logs:
        original_row_index = retry_row["original_row_index"]

        if pd.isna(original_row_index):
            continue

        mask = updated_df["row_index"] == int(original_row_index)

        if retry_row["retry_status"] == "success":
            updated_df.loc[mask, "status"] = "success"
            updated_df.loc[mask, "message"] = retry_row["retry_message"]

    updated_df.to_csv(
        log_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("updated original log:", log_path)

    if failed_count > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())