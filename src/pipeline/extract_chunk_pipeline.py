# src/pipeline/extract_chunk_pipeline.py
#
# 원본 문서를 읽어서 텍스트 추출, 정제, 청킹을 수행하는 파이프라인입니다.
#
# 지원 전략:
# 1. chunking.strategy: pdf_page
#    - PDF 파일만 대상으로 페이지 기반 청킹
#    - data/raw/v2 + data_list_pdf.csv 사용 권장
#
# 2. chunking.strategy: section
#    - PDF/HWP/HWPX/DOCX/DOC/TXT 등 원본 문서에서 텍스트 추출 후 section/subheading 기반 청킹
#    - data/raw + data_list.csv 사용 가능
#
# 주요 흐름:
# 1. YAML config 로드
# 2. data_list.csv 로드
# 3. raw_dir에서 파일 매칭
# 4. 파일 타입별 텍스트 추출
# 5. 텍스트 정제본 저장
# 6. chunking.strategy에 따라 청킹 수행
# 7. embedding_text 공통 보강
# 8. section_chunks.jsonl 저장
# 9. 처리 로그 CSV 저장
# 10. 청킹 전 split 길이 통계 CSV 저장

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import inspect
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

# =========================================================
# Chunker imports
# =========================================================
# PDF page 기반 청킹용입니다.
# chunking.strategy: pdf_page 일 때 사용합니다.
from src.chunking.pdf_page_chunker import (
    chunk_pdf_by_page,
    analyze_pdf_page_split_lengths,
)

# 기존 section/subheading 기반 청킹용입니다.
# chunking.strategy: section 일 때 사용합니다.
from src.chunking.subheading_chunker import (
    create_section_chunks as chunk_sections_by_subheading,
)


class ExtractChunkPipeline:
    """
    원본 문서에서 텍스트를 추출하고 chunking.strategy에 따라 청킹을 수행하는 파이프라인입니다.

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

    SUPPORTED_SECTION_EXTENSIONS = {
        ".pdf",
        ".hwp",
        ".hwpx",
        ".docx",
        ".doc",
        ".txt",
    }

    SUPPORTED_PDF_PAGE_EXTENSIONS = {
        ".pdf",
    }

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

        if "chunking" not in self.config:
            raise KeyError(
                "config에 top-level 'chunking' 설정이 없습니다. "
                "configs/baseline_rag.yaml에 chunking 블록을 추가하세요."
            )

        self.extract_cfg = self.config["extract"]
        self.chunking_cfg = self.config["chunking"]

        self.paths: Dict[str, Path] = {}
        self.data_list: Optional[pd.DataFrame] = None

        self.all_chunks: List[Dict[str, Any]] = []
        self.process_logs: List[Dict[str, Any]] = []

        # 정제 후, 최종 청킹 전 split 길이 통계를 저장하기 위한 로그입니다.
        self.pre_chunk_stats_logs: List[Dict[str, Any]] = []
        self.pre_chunk_length_rows: List[Dict[str, Any]] = []

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

        # 청킹 전 split 통계 저장 경로입니다.
        self.paths["pre_chunk_stats_path"] = (
            self.paths["process_log_path"].parent / "pre_chunk_split_stats.csv"
        )

        self.paths["pre_chunk_lengths_path"] = (
            self.paths["process_log_path"].parent / "pre_chunk_split_lengths.csv"
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
        print(self.config.get("chunking", {}))

        print("=========================================")

    def get_chunking_strategy(self) -> str:
        """
        현재 chunking.strategy 값을 반환합니다.

        지원:
        - pdf_page
        - section
        - subheading
        """
        return str(
            self.config.get("chunking", {}).get("strategy", "pdf_page")
        ).strip().lower()

    def get_candidate_extensions(self) -> set[str]:
        """
        chunking.strategy에 따라 raw_dir에서 찾을 파일 확장자 집합을 반환합니다.

        pdf_page:
            PDF page 기반 청킹이므로 .pdf만 대상으로 합니다.

        section/subheading:
            원본 문서 직접 텍스트 추출 후 section 청킹이므로
            PDF/HWP/HWPX/DOCX/DOC/TXT를 대상으로 합니다.
        """
        strategy = self.get_chunking_strategy()

        if strategy == "pdf_page":
            return self.SUPPORTED_PDF_PAGE_EXTENSIONS

        if strategy in {"section", "subheading"}:
            return self.SUPPORTED_SECTION_EXTENSIONS

        raise ValueError(
            f"지원하지 않는 chunking.strategy입니다: {strategy}. "
            "사용 가능 값: pdf_page, section"
        )

    # ---------------------------------------------------------
    # File name matching
    # ---------------------------------------------------------
    @staticmethod
    def normalize_file_name(name: str) -> str:
        """
        파일명 비교용 정규화 함수입니다.

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
        raw_dir 하위에서 data_list.csv의 파일명에 해당하는 원본 파일을 찾습니다.

        매칭 순서:
        1. 파일명 완전 일치
        2. Unicode/공백/기호 정규화 후 파일명 전체 일치
        3. stem 정규화 후 일치
        4. stem 포함 관계

        chunking.strategy에 따라 후보 확장자가 달라집니다.

        - pdf_page:
          .pdf만 찾습니다.

        - section/subheading:
          .pdf, .hwp, .hwpx, .docx, .doc, .txt를 찾습니다.
        """
        if file_name is None or pd.isna(file_name):
            return None

        file_name = str(file_name).strip()
        raw_dir = self.paths["raw_dir"]
        candidate_extensions = self.get_candidate_extensions()

        raw_files = [
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in candidate_extensions
        ]

        target_name_norm = self.normalize_file_name(file_name)
        target_stem_norm = self.normalize_file_name(Path(file_name).stem)

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

        # 4. stem 포함 관계
        for path in raw_files:
            path_stem_norm = self.normalize_file_name(path.stem)

            if target_stem_norm:
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
    # Logging helpers
    # ---------------------------------------------------------
    def _append_process_log(
        self,
        row: pd.Series,
        doc_id: str,
        file_path: Any,
        status: str,
        raw_text_len: int = 0,
        clean_text_len: int = 0,
        num_sections: int = 0,
        num_pages: int = 0,
        num_chunks: int = 0,
        error: str = "",
    ) -> None:
        """
        처리 로그 row를 추가합니다.
        """
        columns = self.extract_cfg.get("columns", {})

        file_name_col = columns.get("file_name", "파일명")
        file_type_col = columns.get("file_type", "파일형식")

        self.process_logs.append({
            "doc_id": doc_id,
            "file_name": row.get(file_name_col),
            "file_type": row.get(file_type_col),
            "file_path": str(file_path),
            "status": status,
            "raw_text_len": raw_text_len,
            "clean_text_len": clean_text_len,
            "num_sections": num_sections,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "error": error,
        })

    def _append_empty_pre_chunk_stats(
        self,
        row: pd.Series,
        doc_id: str,
        clean_text_len: int = 0,
    ) -> None:
        """
        파일 없음/처리 실패 시에도 pre_chunk 통계 CSV 컬럼이 유지되도록
        기본값 row를 추가합니다.
        """
        self._append_pre_chunk_stats_from_lengths(
            row=row,
            doc_id=doc_id,
            clean_text_len=clean_text_len,
            chunking_strategy=self.get_chunking_strategy(),
            lengths=[],
            num_sections=0,
            num_pages=0,
        )

    def _append_pre_chunk_stats_from_lengths(
        self,
        row: pd.Series,
        doc_id: str,
        clean_text_len: int,
        chunking_strategy: str,
        lengths: List[int],
        num_sections: int = 0,
        num_pages: int = 0,
    ) -> None:
        """
        split/chunk 길이 목록을 기반으로 pre_chunk 통계를 추가합니다.
        """
        columns = self.extract_cfg.get("columns", {})

        file_name_col = columns.get("file_name", "파일명")
        file_type_col = columns.get("file_type", "파일형식")
        project_name_col = columns.get("project_name", "사업명")
        organization_col = columns.get("organization", "발주 기관")

        if lengths:
            series = pd.Series(lengths)
            min_len = int(series.min())
            max_len = int(series.max())
            mean_len = float(series.mean())
            median_len = float(series.median())
        else:
            min_len = 0
            max_len = 0
            mean_len = 0.0
            median_len = 0.0

        self.pre_chunk_stats_logs.append({
            "doc_id": doc_id,
            "file_name": row.get(file_name_col),
            "file_type": row.get(file_type_col),
            "project_name": row.get(project_name_col),
            "organization": row.get(organization_col),
            "clean_text_len": clean_text_len,
            "chunking_strategy": chunking_strategy,
            "num_sections_pre": num_sections,
            "num_pages_pre": num_pages,
            "pre_chunk_count": len(lengths),
            "pre_chunk_min_chars": min_len,
            "pre_chunk_max_chars": max_len,
            "pre_chunk_mean_chars": mean_len,
            "pre_chunk_median_chars": median_len,
        })

        for split_idx, split_len in enumerate(lengths):
            self.pre_chunk_length_rows.append({
                "doc_id": doc_id,
                "file_name": row.get(file_name_col),
                "file_type": row.get(file_type_col),
                "project_name": row.get(project_name_col),
                "organization": row.get(organization_col),
                "chunking_strategy": chunking_strategy,
                "split_index": split_idx,
                "split_char_len": split_len,
            })

    @staticmethod
    def _get_section_count(chunks: List[Dict[str, Any]]) -> int:
        """
        chunk 목록에서 section 수를 추정합니다.
        """
        if not chunks:
            return 0

        section_keys = set()

        for chunk in chunks:
            metadata = chunk.get("metadata", {}) or {}

            key = (
                chunk.get("section_id")
                or chunk.get("section_title")
                or metadata.get("section_id")
                or metadata.get("section_title")
                or chunk.get("page_start")
                or "unknown"
            )

            section_keys.add(str(key))

        return len(section_keys)

    @staticmethod
    def _get_page_count_from_chunks(chunks: List[Dict[str, Any]]) -> int:
        """
        chunk 목록에서 page_start/page_end를 기반으로 page 수를 추정합니다.
        section 청킹에서는 보통 0입니다.
        """
        pages = set()

        for chunk in chunks:
            metadata = chunk.get("metadata", {}) or {}

            page_start = chunk.get("page_start", metadata.get("page_start"))
            page_end = chunk.get("page_end", metadata.get("page_end"))

            if page_start is None:
                continue

            try:
                start = int(page_start)
                end = int(page_end) if page_end is not None else start
            except Exception:
                continue

            for page in range(start, end + 1):
                pages.add(page)

        return len(pages)

    # ---------------------------------------------------------
    # Embedding text helpers
    # ---------------------------------------------------------
    def _build_embedding_text_for_chunk(
        self,
        chunk: Dict[str, Any],
        file_name: str = "",
        file_type: str = "",
        project_name: str = "",
        organization: str = "",
    ) -> str:
        """
        검색 성능 향상을 위해 chunk 본문 앞에 metadata를 붙인 embedding 전용 텍스트를 생성합니다.

        역할:
        - text:
          LLM prompt에 들어가는 실제 본문입니다.

        - embedding_text:
          FAISS/Vector DB에 넣을 임베딩 생성용 텍스트입니다.

        section 방식에서도 기관명/사업명/파일명/섹션명이 벡터에 반영되도록 합니다.
        pdf_page 방식에서 embedding_text가 비어 있는 경우에도 동일한 형식으로 보강합니다.
        """
        text = str(chunk.get("text", "") or "").strip()

        metadata = chunk.get("metadata", {}) or {}

        chunk_file_name = (
            chunk.get("file_name")
            or metadata.get("file_name")
            or file_name
            or ""
        )

        chunk_file_type = (
            chunk.get("file_type")
            or metadata.get("file_type")
            or file_type
            or ""
        )

        chunk_project_name = (
            chunk.get("project_name")
            or metadata.get("project_name")
            or project_name
            or ""
        )

        chunk_organization = (
            chunk.get("organization")
            or metadata.get("organization")
            or organization
            or ""
        )

        section_id = (
            chunk.get("section_id")
            or metadata.get("section_id")
            or ""
        )

        section_title = (
            chunk.get("section_title")
            or metadata.get("section_title")
            or ""
        )

        section_path = (
            chunk.get("section_path")
            or metadata.get("section_path")
            or ""
        )

        page_start = (
            chunk.get("page_start")
            or metadata.get("page_start")
            or ""
        )

        page_end = (
            chunk.get("page_end")
            or metadata.get("page_end")
            or ""
        )

        if page_start and page_end:
            page_info = (
                str(page_start)
                if str(page_start) == str(page_end)
                else f"{page_start}-{page_end}"
            )
        elif page_start:
            page_info = str(page_start)
        else:
            page_info = ""

        parts = []

        if chunk_organization:
            parts.append(f"기관명: {chunk_organization}")

        if chunk_project_name:
            parts.append(f"사업명: {chunk_project_name}")

        if chunk_file_name:
            parts.append(f"파일명: {chunk_file_name}")

        if chunk_file_type:
            parts.append(f"파일형식: {chunk_file_type}")

        if section_id:
            parts.append(f"섹션ID: {section_id}")

        if section_title:
            parts.append(f"섹션명: {section_title}")

        if section_path:
            parts.append(f"섹션경로: {section_path}")

        if page_info:
            parts.append(f"페이지: {page_info}")

        if parts:
            return "\n".join(parts) + "\n\n본문:\n" + text

        return text

    def _add_embedding_text_to_chunks(
        self,
        chunks: List[Dict[str, Any]],
        file_name: str = "",
        file_type: str = "",
        project_name: str = "",
        organization: str = "",
    ) -> List[Dict[str, Any]]:
        """
        chunk 목록에 embedding_text가 없거나 비어 있으면 embedding_text를 추가합니다.

        chunking.include_metadata_in_embedding_text=true:
            metadata + 본문을 embedding_text로 사용합니다.

        chunking.include_metadata_in_embedding_text=false:
            본문 text만 embedding_text로 복사합니다.

        이미 chunker가 embedding_text를 생성한 경우:
            기존 embedding_text를 유지합니다.
        """
        include_metadata = self.chunking_cfg.get(
            "include_metadata_in_embedding_text",
            True,
        )

        updated_chunks = []

        for chunk in chunks:
            chunk = dict(chunk)

            current_embedding_text = str(
                chunk.get("embedding_text", "") or ""
            ).strip()

            if current_embedding_text:
                updated_chunks.append(chunk)
                continue

            text = str(chunk.get("text", "") or "").strip()

            if include_metadata:
                chunk["embedding_text"] = self._build_embedding_text_for_chunk(
                    chunk=chunk,
                    file_name=file_name,
                    file_type=file_type,
                    project_name=project_name,
                    organization=organization,
                )
            else:
                chunk["embedding_text"] = text

            updated_chunks.append(chunk)

        return updated_chunks

    # ---------------------------------------------------------
    # Chunker call helpers
    # ---------------------------------------------------------
    @staticmethod
    def _call_with_supported_kwargs(
        func: Callable,
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        함수 signature에 존재하는 인자만 골라 호출합니다.

        chunker 함수별로 parameter 이름이 조금 달라도 pipeline 전체가
        바로 깨지지 않도록 하기 위한 helper입니다.
        """
        signature = inspect.signature(func)
        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }

        return func(**supported_kwargs)

    def _create_pdf_page_chunks(
        self,
        row: pd.Series,
        doc_id: str,
        file_path: Path,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        PDF page 기반 청킹을 수행하고 pre_chunk_stats를 반환합니다.
        """
        chunking_cfg = self.config.get("chunking", {})
        columns = self.extract_cfg.get("columns", {})

        file_name_col = columns.get("file_name", "파일명")
        file_type_col = columns.get("file_type", "파일형식")
        project_name_col = columns.get("project_name", "사업명")
        organization_col = columns.get("organization", "발주 기관")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"pdf_page 전략은 PDF 파일만 지원합니다. file_path={file_path}"
            )

        pre_chunk_stats = analyze_pdf_page_split_lengths(
            pdf_path=file_path,
            max_chars=chunking_cfg.get("max_chars", 3000),
            overlap_chars=chunking_cfg.get("overlap_chars", 300),
            min_chars=chunking_cfg.get("min_chars", 500),
            merge_short_pages=chunking_cfg.get("merge_short_pages", True),
        )

        chunks = chunk_pdf_by_page(
            doc_id=doc_id,
            pdf_path=file_path,
            file_name=row.get(file_name_col, ""),
            file_type=row.get(file_type_col, "pdf"),
            project_name=row.get(project_name_col, ""),
            organization=row.get(organization_col, ""),
            max_chars=chunking_cfg.get("max_chars", 3000),
            overlap_chars=chunking_cfg.get("overlap_chars", 300),
            min_chars=chunking_cfg.get("min_chars", 500),
            merge_short_pages=chunking_cfg.get("merge_short_pages", True),
            include_metadata_in_embedding_text=chunking_cfg.get(
                "include_metadata_in_embedding_text",
                True,
            ),
        )

        chunks = self._add_embedding_text_to_chunks(
            chunks=chunks or [],
            file_name=row.get(file_name_col, ""),
            file_type=row.get(file_type_col, "pdf"),
            project_name=row.get(project_name_col, ""),
            organization=row.get(organization_col, ""),
        )

        return chunks, pre_chunk_stats

    def _create_section_chunks(
        self,
        row: pd.Series,
        doc_id: str,
        clean_text: str,
    ) -> List[Dict[str, Any]]:
        """
        section/subheading 기반 청킹을 수행합니다.

        현재 section 전략은 src.chunking.subheading_chunker.create_section_chunks를 사용합니다.
        """
        chunking_cfg = self.config.get("chunking", {})
        columns = self.extract_cfg.get("columns", {})

        file_name_col = columns.get("file_name", "파일명")
        file_type_col = columns.get("file_type", "파일형식")
        project_name_col = columns.get("project_name", "사업명")
        organization_col = columns.get("organization", "발주 기관")

        kwargs = {
            "doc_id": doc_id,
            "text": clean_text,
            "clean_text": clean_text,
            "file_name": row.get(file_name_col, ""),
            "file_type": row.get(file_type_col, ""),
            "project_name": row.get(project_name_col, ""),
            "organization": row.get(organization_col, ""),
            "max_chars": chunking_cfg.get("max_chars", 3000),
            "overlap_chars": chunking_cfg.get("overlap_chars", 300),
            "min_chars": chunking_cfg.get("min_chars", 100),
            "merge_short_chunks": chunking_cfg.get("merge_short_chunks", True),
            "include_metadata_in_embedding_text": chunking_cfg.get(
                "include_metadata_in_embedding_text",
                True,
            ),
        }

        chunks = self._call_with_supported_kwargs(
            func=chunk_sections_by_subheading,
            kwargs=kwargs,
        )

        chunks = self._add_embedding_text_to_chunks(
            chunks=chunks or [],
            file_name=row.get(file_name_col, ""),
            file_type=row.get(file_type_col, ""),
            project_name=row.get(project_name_col, ""),
            organization=row.get(organization_col, ""),
        )

        return chunks

    # ---------------------------------------------------------
    # Main extraction/chunking
    # ---------------------------------------------------------
    def process_single_row(self, row: pd.Series) -> List[Dict[str, Any]]:
        """
        data_list의 한 row에 대해 텍스트 추출, 정제, 청킹을 수행합니다.

        공통 처리:
        1. 파일 존재 여부 확인
        2. 파일 타입별 텍스트 추출
        3. 텍스트 정제
        4. extracted/cleaned txt 저장

        전략별 처리:
        - pdf_page:
          PDF page 기반 청킹

        - section/subheading:
          정제된 전체 텍스트 기반 section/subheading 청킹
        """
        doc_id = row["doc_id"]
        file_path = row["file_path"]

        if file_path is None or not Path(file_path).exists():
            self._append_process_log(
                row=row,
                doc_id=doc_id,
                file_path=file_path,
                status="file_not_found",
                error="file_path not found",
            )

            self._append_empty_pre_chunk_stats(
                row=row,
                doc_id=doc_id,
                clean_text_len=0,
            )

            return []

        try:
            file_path = Path(file_path)
            chunking_strategy = self.get_chunking_strategy()

            # 1. 원본 파일에서 직접 텍스트 추출
            extracted = extract_text_by_file_type(file_path)
            raw_text = extracted.get("text", "") or ""

            # 2. 텍스트 정제
            clean_text = clean_extracted_text(raw_text)

            # 3. 추출/정제 텍스트 저장
            extracted_path = self.paths["extracted_dir"] / f"{doc_id}.txt"
            cleaned_path = self.paths["cleaned_dir"] / f"{doc_id}.txt"

            extracted_path.write_text(raw_text, encoding="utf-8")
            cleaned_path.write_text(clean_text, encoding="utf-8")

            # 4. 전략별 청킹
            if chunking_strategy == "pdf_page":
                chunks, pre_chunk_stats = self._create_pdf_page_chunks(
                    row=row,
                    doc_id=doc_id,
                    file_path=file_path,
                )

                lengths = pre_chunk_stats.get("pre_chunk_lengths", [])
                page_count = int(pre_chunk_stats.get("num_pages", 0))
                section_count = page_count

                self._append_pre_chunk_stats_from_lengths(
                    row=row,
                    doc_id=doc_id,
                    clean_text_len=len(clean_text),
                    chunking_strategy=chunking_strategy,
                    lengths=lengths,
                    num_sections=section_count,
                    num_pages=page_count,
                )

            elif chunking_strategy in {"section", "subheading"}:
                chunks = self._create_section_chunks(
                    row=row,
                    doc_id=doc_id,
                    clean_text=clean_text,
                )

                lengths = [
                    len(str(chunk.get("text", "") or ""))
                    for chunk in chunks
                ]

                section_count = self._get_section_count(chunks)
                page_count = self._get_page_count_from_chunks(chunks)

                self._append_pre_chunk_stats_from_lengths(
                    row=row,
                    doc_id=doc_id,
                    clean_text_len=len(clean_text),
                    chunking_strategy=chunking_strategy,
                    lengths=lengths,
                    num_sections=section_count,
                    num_pages=page_count,
                )

            else:
                raise ValueError(
                    f"지원하지 않는 chunking.strategy입니다: {chunking_strategy}. "
                    "사용 가능 값: pdf_page, section"
                )

            self._append_process_log(
                row=row,
                doc_id=doc_id,
                file_path=file_path,
                status="success" if chunks else "no_chunks_created",
                raw_text_len=len(raw_text),
                clean_text_len=len(clean_text),
                num_sections=section_count,
                num_pages=page_count,
                num_chunks=len(chunks),
                error="",
            )

            return chunks

        except Exception as e:
            self._append_process_log(
                row=row,
                doc_id=doc_id,
                file_path=file_path,
                status="failed",
                raw_text_len=0,
                clean_text_len=0,
                num_sections=0,
                num_pages=0,
                num_chunks=0,
                error=repr(e),
            )

            self._append_empty_pre_chunk_stats(
                row=row,
                doc_id=doc_id,
                clean_text_len=0,
            )

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
                "pre_chunk_stats_path": Path,
                "pre_chunk_lengths_path": Path,
                "num_chunks": int,
                "process_log_df": pd.DataFrame,
                "pre_chunk_stats_df": pd.DataFrame,
                "pre_chunk_lengths_df": pd.DataFrame,
                "chunk_stats_df": pd.DataFrame,
                "skipped": bool
            }
        """
        self.print_summary()

        set_seed(self.config["experiment"].get("random_seed", 42))

        output_chunk_path = self.paths["output_chunk_path"]

        chunking_cfg = self.config.get("chunking", {})
        force_rebuild = chunking_cfg.get("force_rebuild_chunks", True)

        if output_chunk_path.exists() and not force_rebuild:
            print("기존 청크 파일이 존재하며 chunking.force_rebuild_chunks=false 입니다.")
            print("청킹을 재실행하지 않습니다:", output_chunk_path)

            return {
                "chunk_path": output_chunk_path,
                "skipped": True,
            }

        self.load_data_list()
        self.attach_file_paths()

        self.all_chunks = []
        self.process_logs = []
        self.pre_chunk_stats_logs = []
        self.pre_chunk_length_rows = []

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

        # 문서별 pre-chunk split 통계 저장
        pre_chunk_stats_df = pd.DataFrame(self.pre_chunk_stats_logs)
        pre_chunk_stats_df.to_csv(
            self.paths["pre_chunk_stats_path"],
            index=False,
            encoding="utf-8-sig",
        )

        # split 하나하나의 길이 상세 저장
        pre_chunk_lengths_df = pd.DataFrame(self.pre_chunk_length_rows)
        pre_chunk_lengths_df.to_csv(
            self.paths["pre_chunk_lengths_path"],
            index=False,
            encoding="utf-8-sig",
        )

        print("section_chunks 저장 완료:", output_chunk_path)
        print("총 청크 수:", len(self.all_chunks))
        print("로그 저장:", self.paths["process_log_path"])
        print("청킹 전 split 문서별 통계 저장:", self.paths["pre_chunk_stats_path"])
        print("청킹 전 split 길이 상세 저장:", self.paths["pre_chunk_lengths_path"])

        if len(process_log_df) > 0:
            print("\nstatus 분포:")
            print(process_log_df["status"].value_counts(dropna=False))

            print("\nnum_chunks describe:")
            print(process_log_df["num_chunks"].describe())

            if "num_pages" in process_log_df.columns:
                print("\nnum_pages describe:")
                print(process_log_df["num_pages"].describe())

        if len(pre_chunk_stats_df) > 0:
            print("\n===== 문서별 pre-chunk split 통계 =====")

            if "pre_chunk_count" in pre_chunk_stats_df.columns:
                print("\npre_chunk_count describe:")
                print(pre_chunk_stats_df["pre_chunk_count"].describe())

            if "pre_chunk_mean_chars" in pre_chunk_stats_df.columns:
                print("\npre_chunk_mean_chars describe:")
                print(pre_chunk_stats_df["pre_chunk_mean_chars"].describe())

            if "pre_chunk_max_chars" in pre_chunk_stats_df.columns:
                print("\npre_chunk_max_chars describe:")
                print(pre_chunk_stats_df["pre_chunk_max_chars"].describe())

        if len(pre_chunk_lengths_df) > 0:
            print("\n===== 전체 문서 기준 pre-chunk split 길이 통계 =====")
            print(
                pre_chunk_lengths_df["split_char_len"].describe(
                    percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
                )
            )

            print("\n전체 split min:", int(pre_chunk_lengths_df["split_char_len"].min()))
            print("전체 split max:", int(pre_chunk_lengths_df["split_char_len"].max()))
            print("전체 split mean:", float(pre_chunk_lengths_df["split_char_len"].mean()))
            print("전체 split median:", float(pre_chunk_lengths_df["split_char_len"].median()))

        chunk_stats_df = self.build_chunk_stats_df()

        return {
            "chunk_path": output_chunk_path,
            "process_log_path": self.paths["process_log_path"],
            "pre_chunk_stats_path": self.paths["pre_chunk_stats_path"],
            "pre_chunk_lengths_path": self.paths["pre_chunk_lengths_path"],
            "num_chunks": len(self.all_chunks),
            "process_log_df": process_log_df,
            "pre_chunk_stats_df": pre_chunk_stats_df,
            "pre_chunk_lengths_df": pre_chunk_lengths_df,
            "chunk_stats_df": chunk_stats_df,
            "skipped": False,
        }

    def build_chunk_stats_df(self) -> pd.DataFrame:
        """
        생성된 청크의 통계를 DataFrame으로 반환합니다.

        Returns
        -------
        pd.DataFrame
            청크별 길이, 문서 ID, 파일 형식, 페이지 정보 등을 담은 DataFrame입니다.
        """
        rows = []

        for chunk in self.all_chunks:
            rows.append({
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "file_type": chunk.get("file_type"),
                "file_name": chunk.get("file_name"),
                "project_name": chunk.get("project_name"),
                "organization": chunk.get("organization"),
                "section_title": chunk.get("section_title"),
                "section_id": chunk.get("section_id"),
                "section_path": chunk.get("section_path"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "page_chunk_index": chunk.get("page_chunk_index"),
                "chunking_strategy": chunk.get("chunking_strategy"),
                "text_len": len(chunk.get("text", "")),
                "chunk_char_len": chunk.get("chunk_char_len"),
                "embedding_text_len": len(chunk.get("embedding_text", "")),
            })

        df = pd.DataFrame(rows)

        if len(df) > 0:
            print("\nchunk text_len describe:")
            print(df["text_len"].describe())

            print("\n문서별 청크 수 describe:")
            print(df.groupby("doc_id")["chunk_id"].count().describe())

            print("\nfile_type 분포:")
            print(df["file_type"].value_counts(dropna=False))

            if "chunk_char_len" in df.columns:
                print("\nchunk_char_len describe:")
                print(df["chunk_char_len"].describe())

            if "embedding_text_len" in df.columns:
                print("\nembedding_text_len describe:")
                print(df["embedding_text_len"].describe())

            if "page_start" in df.columns:
                print("\npage_start describe:")
                print(df["page_start"].describe())

        return df