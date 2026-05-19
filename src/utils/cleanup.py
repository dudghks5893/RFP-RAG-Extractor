import os
import shutil
import subprocess
from pathlib import Path


def get_dir_size(path: Path) -> int:
    """
    디렉토리 크기를 byte 단위로 계산합니다.
    """
    if not path.exists():
        return 0

    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() or p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def format_size(num_bytes: int) -> str:
    """
    byte 크기를 사람이 읽기 쉬운 단위로 변환합니다.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f}PB"


def remove_path(path: str | Path, dry_run: bool = False) -> None:
    """
    파일 또는 디렉토리를 안전하게 삭제합니다.
    """
    path = Path(path).expanduser()

    if not path.exists():
        print(f"[Cleanup] skip not found: {path}")
        return

    size = get_dir_size(path) if path.is_dir() else path.stat().st_size
    print(f"[Cleanup] remove target: {path} | size={format_size(size)}")

    if dry_run:
        print(f"[Cleanup] dry-run skip remove: {path}")
        return

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError as e:
            print(f"[Cleanup] failed to remove file: {path} | error={repr(e)}")


def remove_pycache(root_dir: str | Path = ".", dry_run: bool = False) -> None:
    """
    프로젝트 내부 __pycache__ 폴더를 삭제합니다.
    """
    root_dir = Path(root_dir).expanduser()

    for pycache_dir in root_dir.rglob("__pycache__"):
        remove_path(pycache_dir, dry_run=dry_run)


def run_command(command: list[str], dry_run: bool = False) -> None:
    """
    shell 명령어를 실행합니다.
    """
    print(f"[Cleanup] command: {' '.join(command)}")

    if dry_run:
        print("[Cleanup] dry-run skip command")
        return

    try:
        subprocess.run(command, check=False)
    except Exception as e:
        print(f"[Cleanup] command failed: {' '.join(command)} | error={repr(e)}")


def cleanup_huggingface_cache(dry_run: bool = False) -> None:
    """
    Hugging Face 캐시를 삭제합니다.

    주의:
    다음 실행 때 모델을 다시 다운로드합니다.
    """
    hf_home = os.environ.get("HF_HOME")
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")

    candidates = []

    if hf_home:
        candidates.append(Path(hf_home))

    if hf_hub_cache:
        candidates.append(Path(hf_hub_cache))

    candidates.extend([
        Path.home() / ".cache" / "huggingface",
    ])

    # 중복 제거
    unique_candidates = []
    seen = set()

    for path in candidates:
        resolved = str(path.expanduser())
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(path)

    for path in unique_candidates:
        remove_path(path, dry_run=dry_run)


def cleanup_pip_cache(dry_run: bool = False) -> None:
    """
    pip cache를 삭제합니다.
    """
    run_command(["python", "-m", "pip", "cache", "purge"], dry_run=dry_run)

    # pip cache purge가 실패하거나 pip 명령이 다른 환경일 수 있으므로 직접 삭제도 보조 수행
    remove_path(Path.home() / ".cache" / "pip", dry_run=dry_run)


def cleanup_trash(dry_run: bool = False) -> None:
    """
    현재 사용자 휴지통을 비웁니다.
    """
    remove_path(Path.home() / ".local" / "share" / "Trash" / "files", dry_run=dry_run)
    remove_path(Path.home() / ".local" / "share" / "Trash" / "info", dry_run=dry_run)

def remove_ipynb_checkpoints(root_dir: str | Path = ".", dry_run: bool = False) -> None:
    """
    Jupyter Notebook이 생성하는 .ipynb_checkpoints 폴더를 삭제합니다.
    """
    root_dir = Path(root_dir).expanduser()

    for checkpoint_dir in root_dir.rglob(".ipynb_checkpoints"):
        remove_path(checkpoint_dir, dry_run=dry_run)


def cleanup_project_outputs(
    project_root: str | Path,
    dry_run: bool = False,
    remove_vector_db: bool = False,
    remove_embeddings: bool = False,
) -> None:
    """
    RAG 프로젝트에서 재생성 가능한 산출물을 정리합니다.

    remove_vector_db=False 기본값:
        최종 검색 인덱스를 실수로 삭제하지 않기 위함입니다.

    remove_embeddings=False 기본값:
        임베딩 결과를 재사용할 수 있으므로 기본적으로 보존합니다.
    """
    project_root = Path(project_root).expanduser().resolve()

    # 비교적 안전한 임시/로그성 폴더
    safe_targets = [
        project_root / "tmp",
        project_root / "temp",
        project_root / "logs",
        project_root / "runs",
        project_root / ".cache",
    ]

    for target in safe_targets:
        remove_path(target, dry_run=dry_run)

    if remove_embeddings:
        embedding_targets = [
            project_root / "data" / "embeddings",
            project_root / "embeddings",
        ]
        for target in embedding_targets:
            remove_path(target, dry_run=dry_run)

    if remove_vector_db:
        vector_targets = [
            project_root / "data" / "vector_db",
            project_root / "vector_db",
            project_root / "chroma",
            project_root / "faiss_index",
        ]
        for target in vector_targets:
            remove_path(target, dry_run=dry_run)


def cleanup_after_pipeline(
    project_root: str | Path = ".",
    dry_run: bool = False,
    clean_hf_cache: bool = False,
    clean_pip_cache: bool = False,
    clean_trash: bool = True,
    clean_pycache: bool = True,
    clean_ipynb_checkpoints: bool = True,
    clean_project_outputs: bool = True,
    remove_logs: bool = False,
    remove_outputs: bool = False,
    remove_eval_results: bool = False,
    remove_vector_db: bool = False,
    remove_embeddings: bool = False,
) -> None:
    print("\n===== Cleanup Started =====")

    if clean_hf_cache:
        cleanup_huggingface_cache(dry_run=dry_run)

    if clean_pip_cache:
        cleanup_pip_cache(dry_run=dry_run)

    if clean_trash:
        cleanup_trash(dry_run=dry_run)

    if clean_pycache:
        remove_pycache(project_root, dry_run=dry_run)

    if clean_ipynb_checkpoints:
        remove_ipynb_checkpoints(project_root, dry_run=dry_run)

    if clean_project_outputs:
        cleanup_project_outputs(
            project_root=project_root,
            dry_run=dry_run,
            remove_logs=remove_logs,
            remove_outputs=remove_outputs,
            remove_eval_results=remove_eval_results,
            remove_vector_db=remove_vector_db,
            remove_embeddings=remove_embeddings,
        )

    print("===== Cleanup Finished =====\n")