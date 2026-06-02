from __future__ import annotations

import os
import gc
import sys
import time
import shutil
import getpass
from pathlib import Path
from typing import Optional


class RunDiskGuard:
    """
    공용 디스크 환경용 실행 캐시 관리자.

    핵심 원칙:
    - 전역 ~/.cache 를 직접 지우지 않는다.
    - 이번 실행에서 생기는 캐시만 프로젝트 내부 .runtime_cache/run_xxx 로 몰아넣는다.
    - 실행 종료 후 해당 run cache만 삭제한다.
    - results/report/raw/processed 등 결과 파일은 보존한다.
    """

    def __init__(
        self,
        project_root: str | Path = ".",
        cache_root: str | Path = ".runtime_cache",
        run_name: Optional[str] = None,
        verbose: bool = True,
    ):
        self.project_root = Path(project_root).resolve()
        self.cache_root = (self.project_root / cache_root).resolve()
        self.verbose = verbose

        username = getpass.getuser()
        pid = os.getpid()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if run_name is None:
            run_name = f"{username}_pid{pid}_{timestamp}"

        self.run_cache_dir = (self.cache_root / run_name).resolve()
        self.extra_delete_paths: list[Path] = []

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[RunDiskGuard] {message}")

    def setup_env(self) -> None:
        """
        모델 관련 라이브러리 import 전에 반드시 호출해야 함.
        transformers, sentence_transformers, torch 관련 캐시 위치를
        이번 실행 전용 폴더로 고정한다.
        """
        self.run_cache_dir.mkdir(parents=True, exist_ok=True)

        env_map = {
            # HuggingFace / Transformers
            "HF_HOME": self.run_cache_dir / "huggingface",
            "HF_HUB_CACHE": self.run_cache_dir / "huggingface" / "hub",
            "HUGGINGFACE_HUB_CACHE": self.run_cache_dir / "huggingface" / "hub",
            "HF_DATASETS_CACHE": self.run_cache_dir / "huggingface" / "datasets",
            "TRANSFORMERS_CACHE": self.run_cache_dir / "huggingface" / "transformers",

            # SentenceTransformers
            "SENTENCE_TRANSFORMERS_HOME": self.run_cache_dir / "sentence_transformers",

            # PyTorch
            "TORCH_HOME": self.run_cache_dir / "torch",

            # pip cache
            "PIP_CACHE_DIR": self.run_cache_dir / "pip",

            # temp
            "TMPDIR": self.run_cache_dir / "tmp",
            "TEMP": self.run_cache_dir / "tmp",
            "TMP": self.run_cache_dir / "tmp",

            # 기타 자주 생기는 캐시
            "XDG_CACHE_HOME": self.run_cache_dir / "xdg",
            "TRITON_CACHE_DIR": self.run_cache_dir / "triton",
            "NUMBA_CACHE_DIR": self.run_cache_dir / "numba",
            "MPLCONFIGDIR": self.run_cache_dir / "matplotlib",

            # wandb를 쓸 경우
            "WANDB_DIR": self.run_cache_dir / "wandb",
            "WANDB_CACHE_DIR": self.run_cache_dir / "wandb_cache",
        }

        for key, path in env_map.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)

        self.log(f"Run cache dir: {self.run_cache_dir}")

    def add_delete_path(self, path: str | Path) -> None:
        """
        추가로 삭제할 경로를 등록한다.
        예: FAISS vector_db_dir

        단, project_root 내부 경로만 삭제 허용.
        """
        self.extra_delete_paths.append(Path(path).resolve())

    def _get_size(self, path: Path) -> int:
        if not path.exists():
            return 0

        if path.is_file():
            try:
                return path.stat().st_size
            except Exception:
                return 0

        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except Exception:
                pass

        return total

    @staticmethod
    def _human_size(size: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(size)

        for unit in units:
            if value < 1024:
                return f"{value:.2f} {unit}"
            value /= 1024

        return f"{value:.2f} PB"

    def _is_safe_to_delete(self, path: Path) -> bool:
        """
        위험 경로 삭제 방지.
        """
        path = path.resolve()

        forbidden = {
            Path("/").resolve(),
            Path.home().resolve(),
            self.project_root.resolve(),
            Path(sys.prefix).resolve(),  # 현재 가상환경 자체 삭제 방지
        }

        if path in forbidden:
            return False

        # 1순위: 이번 실행 캐시 내부는 삭제 허용
        try:
            path.relative_to(self.run_cache_dir)
            return True
        except ValueError:
            pass

        # 2순위: project_root 내부의 명시 등록 경로만 허용
        try:
            path.relative_to(self.project_root)
            return True
        except ValueError:
            return False

    def _delete_path(self, path: Path) -> None:
        path = path.resolve()

        if not path.exists():
            return

        if not self._is_safe_to_delete(path):
            self.log(f"Skip unsafe delete path: {path}")
            return

        size = self._get_size(path)

        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)

            self.log(f"Deleted: {path}")
            self.log(f"Freed approximately: {self._human_size(size)}")

        except Exception as e:
            self.log(f"Failed to delete {path}: {e}")

    def release_memory(self) -> None:
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                self.log("CUDA memory cache cleared.")
        except Exception:
            pass

    def cleanup(self) -> None:
        """
        이번 실행 캐시와 명시적으로 등록된 임시 산출물만 삭제.
        """
        self.log("Cleanup started.")

        self.release_memory()

        # 이번 실행 캐시 삭제
        self._delete_path(self.run_cache_dir)

        # 추가 삭제 경로 삭제
        for path in self.extra_delete_paths:
            self._delete_path(path)

        self.log("Cleanup finished.")

    def __enter__(self):
        self.setup_env()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False