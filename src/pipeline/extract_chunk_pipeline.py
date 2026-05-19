# src/pipeline/extract_chunk_pipeline.py
#
# 원본 PDF/HWP/DOCX 파일을 직접 읽어서 텍스트 추출, 정제, 청킹을 수행하는 파이프라인입니다.
#
# 이 파이프라인은 notebooks/01_extract_clean_chunk.ipynb에서 검증한 흐름을
# 재사용 가능한 Python 모듈로 옮긴 것입니다.
#
# 주요 흐름:
# 1. YAML config 로드
# 2. data_list.csv 로드
# 3. data/raw에서 원본 파일 매칭
# 4. PDF/HWP/DOCX 직접 텍스트 추출
# 5. 텍스트 정제
# 6. 목차 기반 계층 청킹
# 7. section_chunks.jsonl 저장
# 8. 처리 로그 CSV 저장

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import unicodedata

import pandas as pd

from src.utils.config_utils import (
    load_yaml_config,
    resolve_project_path,
)
from src.utils.path_utils import find_project_root
from src.utils.file_utils import save_jsonl
from src.utils.progress_utils import progress_iter
from src.utils.seed import set_seed

from src.extractors import extract_text_by_file_type
from src.utils.text_cleaner import clean_extracted_text

# 기우님의 청킹 방식
# from src.chunking.toc_chunker import (
#     preprocess_text_for_toc_chunking,
#     create_toc_based_chunks as create_chunks,
# )

# 기존 베이스라인 청킹 방식
from src.chunking.section_chunker import (
    create_section_chunks as create_chunks
)

class ExtractChunkPipeline:
    """
    원본 RFP 파일에서 텍스트를 추출하고 목차 기반 청킹을 수행하는 파이프라인입니다.

    Parameters
    ----------
    config_path:
        YAML config 파일 경로입니다.

    project_root:
        프로젝트 루트 경로입니다.
        None이면 find_project_root()로 자동 탐색합니다.

    project_name:
        프로젝트 루트 폴더 이름입니다.
    """

    def __init__(
        self,
        config_path: str | Path,
        project_root: Optional[str | Path] = None,
        project_name: str = "RFP-RAG-Extractor",
    ):
        self.project_name = project_name

        if project_root is None:
            self.project_root = find_project_root(project_name)
        else:
            self.project_root = Path(project_root)

        self.config_path = resolve_project_path(
            self.project_root,
            config_path,
        )

        self.config = load_yaml_config(self.config_path)

        if "extract" not in self.config:
            raise KeyError(
                "config에 'extract' 설정이 없습니다. "
                "configs/baseline_rag.yaml에 extract 블록을 추가하세요."
            )

        self.extract_cfg = self.config["extract"]

        self.paths: Dict[str, Path] = {}
        self.data_list: Optional[pd.DataFrame] = None

        self.all_chunks: List[Dict[str, Any]] = []
        self.process_logs: List[Dict[str, Any]] = []

        self._resolve_paths()

    # ---------------------------------------------------------
    # Path / Config
    # ---------------------------------------------------------
    def _resolve_paths(self) -> None:
        """
        extract config의 경로를 프로젝트 루트 기준 절대 경로로 변환합니다.
        """
        cfg = self.extract_cfg

        self.paths["raw_dir"] = resolve_project_path(
            self.project_root,
            cfg["raw_dir"],
        )

        self.paths["data_list_path"] = resolve_project_path(
            self.project_root,
            cfg["data_list_path"],
        )

        self.paths["extracted_dir"] = resolve_project_path(
            self.project_root,
            cfg["extracted_dir"],
        )

        self.paths["cleaned_dir"] = resolve_project_path(
            self.project_root,
            cfg["cleaned_dir"],
        )

        self.paths["output_chunk_path"] = resolve_project_path(
            self.project_root,
            cfg["output_chunk_path"],
        )

        self.paths["process_log_path"] = resolve_project_path(
            self.project_root,
            cfg["process_log_path"],
        )

        self.paths["extracted_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["cleaned_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["output_chunk_path"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["process_log_path"].parent.mkdir(parents=True, exist_ok=True)

    def print_summary(self) -> None:
        """
        파이프라인 설정 요약을 출력합니다.
        """
        print("===== Extract/Chunk Pipeline Summary =====")
        print("project_root:", self.project_root)
        print("config_path:", self.config_path)

        for key, value in self.paths.items():
            print(f"{key}: {value}")

        print("\nchunking config:")
        print(self.extract_cfg.get("chunking", {}))
        print("=========================================")

    # ---------------------------------------------------------
    # File name matching
    # ---------------------------------------------------------
    @staticmethod
    def normalize_file_name(name: str) -> str:
        """
        파일명 비교용 정규화 함수입니다.

        raw 파일명이 NFD, data_list.csv 파일명이 NFC인 경우를 맞추기 위해
        Unicode NFC 정규화를 적용합니다.

        처리:
        - Unicode NFC 정규화
        - 소문자화
        - 공백 제거
        - 일부 기호 통일
        """
        if name is None or pd.isna(name):
            return ""

        name = str(name).strip()
        name = unicodedata.normalize("NFC", name)
        name = name.lower()

        # 모든 공백 제거
        name = re.sub(r"\s+", "", name)

        # 유사 기호 통일
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

    def find_raw_file(self, file_name: str) -> Optional[Path]:
        """
        data/raw 하위에서 data_list.csv의 파일명에 해당하는 원본 파일을 찾습니다.

        매칭 순서:
        1. 파일명 완전 일치
        2. Unicode/공백/기호 정규화 후 파일명 전체 일치
        3. stem 정규화 후 일치
        4. stem 포함 관계 + 확장자 동일
        """
        if file_name is None or pd.isna(file_name):
            return None

        file_name = str(file_name).strip()
        raw_dir = self.paths["raw_dir"]

        raw_files = [path for path in raw_dir.rglob("*") if path.is_file()]

        target_name_norm = self.normalize_file_name(file_name)
        target_stem_norm = self.normalize_file_name(Path(file_name).stem)
        target_suffix = Path(file_name).suffix.lower()

        # 1. 파일명 완전 일치
        for path in raw_files:
            if path.name == file_name:
                return path

        # 2. 정규화 후 파일명 전체 일치
        for path in raw_files:
            if self.normalize_file_name(path.name) == target_name_norm:
                return path

        # 3. 정규화된 stem 일치
        for path in raw_files:
            if self.normalize_file_name(path.stem) == target_stem_norm:
                return path

        # 4. stem 포함 관계 + 확장자 동일
        for path in raw_files:
            path_stem_norm = self.normalize_file_name(path.stem)
            path_suffix = path.suffix.lower()

            suffix_ok = not target_suffix or path_suffix == target_suffix

            if suffix_ok and target_stem_norm:
                if target_stem_norm in path_stem_norm or path_stem_norm in target_stem_norm:
                    return path

        return None

    # ---------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------
    def load_data_list(self) -> pd.DataFrame:
        """
        data_list.csv를 로드합니다.

        Returns
        -------
        pd.DataFrame
            로드된 메타데이터 DataFrame입니다.
        """
        data_list_path = self.paths["data_list_path"]

        if not data_list_path.exists():
            raise FileNotFoundError(f"data_list.csv를 찾을 수 없습니다: {data_list_path}")

        self.data_list = pd.read_csv(data_list_path)

        print("문서 수:", len(self.data_list))
        print("columns:", list(self.data_list.columns))

        return self.data_list

    def make_doc_id(self, row: pd.Series) -> str:
        """
        row에서 doc_id를 생성합니다.

        기본적으로 config.extract.columns.doc_id에 해당하는 컬럼을 사용합니다.
        값이 없으면 DOC_0000 형태로 생성합니다.
        """
        columns = self.extract_cfg.get("columns", {})
        doc_id_col = columns.get("doc_id", "공고 번호")

        value = row.get(doc_id_col)

        if pd.notna(value) and str(value).strip():
            return str(value).strip()

        return f"DOC_{int(row.name):04d}"

    def attach_file_paths(self) -> pd.DataFrame:
        """
        data_list에 doc_id, file_path, file_exists 컬럼을 추가합니다.

        Returns
        -------
        pd.DataFrame
            파일 경로 매칭 결과가 추가된 data_list입니다.
        """
        if self.data_list is None:
            self.load_data_list()

        columns = self.extract_cfg.get("columns", {})
        file_name_col = columns.get("file_name", "파일명")

        self.data_list["doc_id"] = self.data_list.apply(self.make_doc_id, axis=1)
        self.data_list["file_path"] = self.data_list[file_name_col].apply(self.find_raw_file)
        self.data_list["file_exists"] = self.data_list["file_path"].apply(
            lambda x: x is not None and Path(x).exists()
        )

        print("file_exists 분포:")
        print(self.data_list["file_exists"].value_counts(dropna=False))

        missing = self.data_list[~self.data_list["file_exists"]]

        if len(missing) > 0:
            print("못 찾은 파일 수:", len(missing))
            print(missing[[file_name_col]].head(20))

        return self.data_list

    # ---------------------------------------------------------
    # Main extraction/chunking
    # ---------------------------------------------------------
    def process_single_row(self, row: pd.Series) -> List[Dict[str, Any]]:
        """
        data_list의 한 row에 대해 텍스트 추출, 정제, 청킹을 수행합니다.

        처리 순서:
        1. 파일 존재 여부 확인
        2. PDF/HWP/DOCX 직접 텍스트 추출
        3. 공통 텍스트 정제
        4. 목차 기반 청킹용 추가 전처리
        5. 추출/정제 텍스트 저장
        6. 목차 기반 계층 청킹 수행
        7. 처리 로그 기록

        Parameters
        ----------
        row:
            data_list의 한 행입니다.

        Returns
        -------
        List[Dict[str, Any]]
            생성된 chunk dict 리스트입니다.
        """
        columns = self.extract_cfg.get("columns", {})

        file_name_col = columns.get("file_name", "파일명")
        file_type_col = columns.get("file_type", "파일형식")
        project_name_col = columns.get("project_name", "사업명")
        organization_col = columns.get("organization", "발주 기관")

        doc_id = row["doc_id"]
        file_path = row["file_path"]

        if file_path is None or not Path(file_path).exists():
            self.process_logs.append({
                "doc_id": doc_id,
                "file_name": row.get(file_name_col),
                "file_type": row.get(file_type_col),
                "file_path": str(file_path),
                "status": "file_not_found",
                "raw_text_len": 0,
                "clean_text_len": 0,
                "num_sections": 0,
                "num_chunks": 0,
                "error": "file_path not found",
            })
            return []

        try:
            # 1. 원본 파일에서 직접 텍스트 추출
            extracted = extract_text_by_file_type(file_path)
            raw_text = extracted.get("text", "") or ""

            # 2. 텍스트 정제
            # base_clean_text = clean_extracted_text(raw_text)

            # 3. 목차 기반 청킹용 추가 전처리
            # clean_text = preprocess_text_for_toc_chunking(base_clean_text)

            # 기본 베이스 라인 적용 시 위 2개 주석하고 아래 코드 사용.
            clean_text = clean_extracted_text(raw_text)

            # 4. 추출/정제 텍스트 저장
            extracted_path = self.paths["extracted_dir"] / f"{doc_id}.txt"
            cleaned_path = self.paths["cleaned_dir"] / f"{doc_id}.txt"

            extracted_path.write_text(raw_text, encoding="utf-8")
            cleaned_path.write_text(clean_text, encoding="utf-8")

            # 5. 목차 기반 계층 청킹
            chunking_cfg = self.extract_cfg.get("chunking", {})

            # 청킹 모듈 cfg 값으로 적용
            chunks = create_chunks(
                doc_id=doc_id,
                text=clean_text,
                file_name=row.get(file_name_col, ""),
                file_type=row.get(file_type_col, ""),
                project_name=row.get(project_name_col, ""),
                organization=row.get(organization_col, ""),
                max_chars=chunking_cfg.get("max_chars", 3000),
                overlap_chars=chunking_cfg.get("overlap_chars", 300),
                min_chars=chunking_cfg.get("min_chars", 100),
            )

            # section_count = len(set(chunk["section_id"] for chunk in chunks))
            section_count = len(set(
                chunk.get("section_id")
                or chunk.get("section_title")
                or chunk.get("metadata", {}).get("section_id")
                or chunk.get("metadata", {}).get("section_title")
                or "unknown"
                for chunk in chunks
            ))

            self.process_logs.append({
                "doc_id": doc_id,
                "file_name": row.get(file_name_col),
                "file_type": row.get(file_type_col),
                "file_path": str(file_path),
                "status": "success" if chunks else "no_chunks_created",
                "raw_text_len": len(raw_text),
                "clean_text_len": len(clean_text),
                "num_sections": section_count,
                "num_chunks": len(chunks),
                "error": "",
            })

            return chunks

        except Exception as e:
            self.process_logs.append({
                "doc_id": doc_id,
                "file_name": row.get(file_name_col),
                "file_type": row.get(file_type_col),
                "file_path": str(file_path),
                "status": "failed",
                "raw_text_len": 0,
                "clean_text_len": 0,
                "num_sections": 0,
                "num_chunks": 0,
                "error": repr(e),
            })
            return []

    def run(self) -> Dict[str, Any]:
        """
        전체 추출/정제/청킹 파이프라인을 실행합니다.

        Returns
        -------
        Dict[str, Any]
            {
                "chunk_path": Path,
                "process_log_path": Path,
                "num_chunks": int,
                "process_log_df": pd.DataFrame,
                "chunk_stats_df": pd.DataFrame,
                "skipped": bool
            }
        """
        self.print_summary()

        set_seed(self.config["experiment"].get("random_seed", 42))

        output_chunk_path = self.paths["output_chunk_path"]
        force_rebuild = self.extract_cfg.get("force_rebuild_chunks", True)

        if output_chunk_path.exists() and not force_rebuild:
            print("기존 청크 파일이 존재하며 force_rebuild_chunks=false 입니다.")
            print("청킹을 재실행하지 않습니다:", output_chunk_path)

            return {
                "chunk_path": output_chunk_path,
                "skipped": True,
            }

        self.load_data_list()
        self.attach_file_paths()

        self.all_chunks = []
        self.process_logs = []

        for _, row in progress_iter(
            self.data_list.iterrows(),
            total=len(self.data_list),
            desc="Extract/Clean/Chunk",
            log_every=5,
            min_interval_sec=5.0,
        ):
            chunks = self.process_single_row(row)
            self.all_chunks.extend(chunks)

        # 청크 저장
        save_jsonl(
            self.all_chunks,
            output_chunk_path,
        )

        # 처리 로그 저장
        process_log_df = pd.DataFrame(self.process_logs)
        process_log_df.to_csv(
            self.paths["process_log_path"],
            index=False,
            encoding="utf-8-sig",
        )

        print("section_chunks 저장 완료:", output_chunk_path)
        print("총 청크 수:", len(self.all_chunks))
        print("로그 저장:", self.paths["process_log_path"])

        if len(process_log_df) > 0:
            print("\nstatus 분포:")
            print(process_log_df["status"].value_counts(dropna=False))

            print("\nnum_chunks describe:")
            print(process_log_df["num_chunks"].describe())

        chunk_stats_df = self.build_chunk_stats_df()

        return {
            "chunk_path": output_chunk_path,
            "process_log_path": self.paths["process_log_path"],
            "num_chunks": len(self.all_chunks),
            "process_log_df": process_log_df,
            "chunk_stats_df": chunk_stats_df,
            "skipped": False,
        }

    def build_chunk_stats_df(self) -> pd.DataFrame:
        """
        생성된 청크의 통계를 DataFrame으로 반환합니다.

        Returns
        -------
        pd.DataFrame
            청크별 길이, 문서 ID, 파일 형식, 섹션 제목 등을 담은 DataFrame입니다.
        """
        rows = []

        for chunk in self.all_chunks:
            rows.append({
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "file_type": chunk.get("file_type"),
                "project_name": chunk.get("project_name"),
                "section_title": chunk.get("section_title"),
                "chunking_method": chunk.get("chunking_method"),
                "chunking_strategy": chunk.get("chunking_strategy"),
                "text_len": len(chunk.get("text", "")),
                "char_len": chunk.get("char_len"),
            })

        df = pd.DataFrame(rows)

        if len(df) > 0:
            print("\nchunk text_len describe:")
            print(df["text_len"].describe())

            print("\n문서별 청크 수 describe:")
            print(df.groupby("doc_id")["chunk_id"].count().describe())

            print("\nfile_type 분포:")
            print(df["file_type"].value_counts(dropna=False))

        return df