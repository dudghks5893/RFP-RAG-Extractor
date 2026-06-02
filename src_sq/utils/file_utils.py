import json
from pathlib import Path
from typing import List, Dict, Any


def save_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    """
    dict 리스트를 JSONL로 저장합니다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    JSONL 파일을 로드합니다.
    """
    path = Path(path)

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows