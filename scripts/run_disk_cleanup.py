import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.disk_cleanup import cleanup_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description="RFP-RAG-Extractor 디스크 정리 스크립트"
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 파일을 삭제합니다. 이 옵션이 없으면 dry-run으로 동작합니다.",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="프로젝트 루트 경로입니다.",
    )

    parser.add_argument(
        "--apt",
        action="store_true",
        help="sudo apt clean, sudo apt autoremove도 실행합니다.",
    )

    parser.add_argument(
        "--docker",
        action="store_true",
        help="docker system prune -a -f도 실행합니다.",
    )

    parser.add_argument(
        "--remove-outputs",
        action="store_true",
        help="outputs 폴더를 삭제합니다.",
    )

    parser.add_argument(
        "--remove-logs",
        action="store_true",
        help="logs 폴더를 삭제합니다.",
    )

    parser.add_argument(
        "--remove-eval-results",
        action="store_true",
        help="eval_results, reports, metrics 등 평가 산출물을 삭제합니다.",
    )

    parser.add_argument(
        "--remove-embeddings",
        action="store_true",
        help="embeddings, data/embeddings 폴더를 삭제합니다.",
    )

    parser.add_argument(
        "--remove-vector-db",
        action="store_true",
        help="vector_db, chroma, faiss_index 등을 삭제합니다.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cleanup_disk(
        project_root=args.project_root,
        dry_run=not args.apply,
        clean_user_caches=True,
        clean_project_temp=True,
        clean_apt_cache=args.apt,
        clean_docker=args.docker,
        remove_outputs=args.remove_outputs,
        remove_logs=args.remove_logs,
        remove_eval_results=args.remove_eval_results,
        remove_embeddings=args.remove_embeddings,
        remove_vector_db=args.remove_vector_db,
    )


if __name__ == "__main__":
    main()