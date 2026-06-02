from pathlib import Path


def find_project_root(project_name: str = "RFP-RAG-Extractor") -> Path:
    """
    현재 작업 디렉토리에서 상위 폴더를 탐색하며 프로젝트 루트를 찾습니다.

    예:
    현재 위치가 RFP-RAG-Extractor/notebooks/yh 라면
    RFP-RAG-Extractor 를 PROJECT_ROOT로 반환합니다.
    """
    current = Path.cwd().resolve()

    for path in [current, *current.parents]:
        if path.name == project_name:
            return path

    raise FileNotFoundError(
        f"프로젝트 루트 폴더 '{project_name}'를 찾을 수 없습니다. 현재 위치: {current}"
    )