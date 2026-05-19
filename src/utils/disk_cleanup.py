import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def format_size(num_bytes: int) -> str:
    """
    byte 크기를 사람이 읽기 쉬운 문자열로 변환합니다.
    """
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}PB"


def get_path_size(path: Path) -> int:
    """
    파일 또는 디렉토리 크기를 byte 단위로 계산합니다.
    """
    path = path.expanduser()

    if not path.exists():
        return 0

    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() or p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue

    return total


def print_disk_usage(title: str = "Disk usage") -> None:
    """
    현재 루트 디스크 사용량을 출력합니다.
    """
    usage = shutil.disk_usage("/")
    used_percent = usage.used / usage.total * 100

    print(f"\n[{title}]")
    print(f"  total : {format_size(usage.total)}")
    print(f"  used  : {format_size(usage.used)} ({used_percent:.2f}%)")
    print(f"  free  : {format_size(usage.free)}")


def remove_path(path: str | Path, dry_run: bool = True) -> int:
    """
    파일 또는 디렉토리를 삭제합니다.

    dry_run=True이면 실제 삭제하지 않고 삭제 대상만 출력합니다.
    반환값은 삭제 예정 또는 삭제된 크기(byte)입니다.
    """
    path = Path(path).expanduser()

    if not path.exists():
        return 0

    size = get_path_size(path)
    print(f"[Cleanup] target: {path} | size={format_size(size)}")

    if dry_run:
        print(f"[Cleanup] dry-run skip: {path}")
        return size

    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[Cleanup][WARN] failed to remove: {path} | error={repr(e)}")

    return size


def remove_matching_dirs(
    root_dir: str | Path,
    dir_names: Iterable[str],
    dry_run: bool = True,
) -> int:
    """
    root_dir 아래에서 특정 이름의 디렉토리들을 찾아 삭제합니다.
    예: __pycache__, .ipynb_checkpoints, .pytest_cache
    """
    root_dir = Path(root_dir).expanduser().resolve()

    if not root_dir.exists():
        print(f"[Cleanup] root not found: {root_dir}")
        return 0

    total = 0
    dir_names = set(dir_names)

    for path in root_dir.rglob("*"):
        if path.is_dir() and path.name in dir_names:
            total += remove_path(path, dry_run=dry_run)

    return total


def run_command(command: list[str], dry_run: bool = True) -> None:
    """
    시스템 명령을 실행합니다.
    """
    print(f"[Cleanup] command: {' '.join(command)}")

    if dry_run:
        print("[Cleanup] dry-run skip command")
        return

    try:
        subprocess.run(command, check=False)
    except Exception as e:
        print(f"[Cleanup][WARN] command failed: {' '.join(command)} | error={repr(e)}")


def get_huggingface_cache_candidates() -> list[Path]:
    """
    Hugging Face 캐시 후보 경로를 반환합니다.

    HF_HOME, HF_HUB_CACHE 환경변수를 쓰는 경우도 고려합니다.
    """
    candidates: list[Path] = []

    hf_home = os.environ.get("HF_HOME")
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")

    if hf_home:
        candidates.append(Path(hf_home))

    if hf_hub_cache:
        candidates.append(Path(hf_hub_cache))

    if transformers_cache:
        candidates.append(Path(transformers_cache))

    candidates.append(Path.home() / ".cache" / "huggingface")

    unique = []
    seen = set()

    for path in candidates:
        resolved = str(path.expanduser())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    return unique


def cleanup_user_caches(dry_run: bool = True) -> int:
    """
    현재 사용자 홈 디렉토리 기준 주요 캐시를 삭제합니다.
    """
    home = Path.home()

    targets = [
        home / ".cache" / "pip",
        home / ".cache" / "torch",
        home / ".cache" / "matplotlib",
        home / ".cache" / "sentence_transformers",
        home / ".cache" / "datasets",
        home / ".cache" / "wandb",
        home / ".local" / "share" / "Trash" / "files",
        home / ".local" / "share" / "Trash" / "info",
    ]

    targets.extend(get_huggingface_cache_candidates())

    total = 0

    for target in targets:
        total += remove_path(target, dry_run=dry_run)

    return total


def cleanup_project_temp_files(
    project_root: str | Path,
    dry_run: bool = True,
    remove_outputs: bool = False,
    remove_logs: bool = False,
    remove_eval_results: bool = False,
    remove_embeddings: bool = False,
    remove_vector_db: bool = False,
) -> int:
    """
    프로젝트 내부 임시 파일을 정리합니다.

    기본값은 보수적입니다.
    성능평가 결과, logs, outputs, embeddings, vector_db는 기본 삭제하지 않습니다.
    """
    project_root = Path(project_root).expanduser().resolve()

    total = 0

    # 기본 삭제 대상
    safe_targets = [
        project_root / "tmp",
        project_root / "temp",
        project_root / ".cache",
        project_root / ".pytest_cache",
    ]

    for target in safe_targets:
        total += remove_path(target, dry_run=dry_run)

    # 프로젝트 내부 자동 생성 디렉토리
    total += remove_matching_dirs(
        project_root,
        dir_names=[
            "__pycache__",
            ".ipynb_checkpoints",
        ],
        dry_run=dry_run,
    )

    # 선택 삭제 대상
    if remove_outputs:
        total += remove_path(project_root / "outputs", dry_run=dry_run)

    if remove_logs:
        total += remove_path(project_root / "logs", dry_run=dry_run)

    if remove_eval_results:
        eval_targets = [
            project_root / "eval_results",
            project_root / "evaluation_results",
            project_root / "reports",
            project_root / "metrics",
        ]

        for target in eval_targets:
            total += remove_path(target, dry_run=dry_run)

    if remove_embeddings:
        embedding_targets = [
            project_root / "embeddings",
            project_root / "data" / "embeddings",
        ]

        for target in embedding_targets:
            total += remove_path(target, dry_run=dry_run)

    if remove_vector_db:
        vector_targets = [
            project_root / "vector_db",
            project_root / "data" / "vector_db",
            project_root / "chroma",
            project_root / "faiss_index",
        ]

        for target in vector_targets:
            total += remove_path(target, dry_run=dry_run)

    return total


def cleanup_disk(
    project_root: str | Path = ".",
    dry_run: bool = True,
    clean_user_caches: bool = True,
    clean_project_temp: bool = True,
    clean_apt_cache: bool = False,
    clean_docker: bool = False,
    remove_outputs: bool = False,
    remove_logs: bool = False,
    remove_eval_results: bool = False,
    remove_embeddings: bool = False,
    remove_vector_db: bool = False,
) -> None:
    """
    팀원들이 수동으로 실행할 수 있는 공용 디스크 정리 함수입니다.

    기본적으로 안전한 항목만 삭제합니다.

    dry_run=True:
        실제 삭제하지 않고 삭제 대상과 예상 정리 용량만 출력합니다.

    dry_run=False:
        실제 삭제합니다.

    기본 보존:
        outputs, logs, eval_results, reports, metrics,
        embeddings, vector_db, chroma, faiss_index, data/raw
    """
    project_root = Path(project_root).expanduser().resolve()

    print("\n===== Disk Cleanup Started =====")
    print(f"[Cleanup] project_root: {project_root}")
    print(f"[Cleanup] dry_run: {dry_run}")

    print_disk_usage("Before cleanup")

    total_target_size = 0

    if clean_user_caches:
        print("\n[Cleanup] User caches")
        total_target_size += cleanup_user_caches(dry_run=dry_run)

    if clean_project_temp:
        print("\n[Cleanup] Project temporary files")
        total_target_size += cleanup_project_temp_files(
            project_root=project_root,
            dry_run=dry_run,
            remove_outputs=remove_outputs,
            remove_logs=remove_logs,
            remove_eval_results=remove_eval_results,
            remove_embeddings=remove_embeddings,
            remove_vector_db=remove_vector_db,
        )

    if clean_apt_cache:
        print("\n[Cleanup] apt cache")
        run_command(["sudo", "apt", "clean"], dry_run=dry_run)
        run_command(["sudo", "apt", "autoremove", "-y"], dry_run=dry_run)

    if clean_docker:
        print("\n[Cleanup] docker cache")
        run_command(["docker", "system", "prune", "-a", "-f"], dry_run=dry_run)

    print(f"\n[Cleanup] total target size: {format_size(total_target_size)}")

    print_disk_usage("After cleanup")
    print("===== Disk Cleanup Finished =====\n")