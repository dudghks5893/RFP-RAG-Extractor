import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def format_seconds(seconds: float) -> str:
    """
    초 단위 시간을 사람이 보기 쉬운 문자열로 변환합니다.
    """
    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds}s"

    minutes, sec = divmod(seconds, 60)

    if minutes < 60:
        return f"{minutes}m {sec}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


class ProgressLogger:
    """
    JupyterHub/GCP VM 환경에서 tqdm 렌더링 문제가 있을 때 사용할 수 있는
    print 기반 진행상황 로거입니다.

    특징:
    - tqdm처럼 화면을 계속 리렌더링하지 않음
    - 일정 step마다 print 출력
    - elapsed time, ETA 출력
    - embedding, LLM loading, RAG 실행, 평가 루프 등에 공통 사용 가능
    """

    def __init__(
        self,
        total: Optional[int] = None,
        desc: str = "Progress",
        log_every: int = 10,
        min_interval_sec: float = 5.0,
    ):
        """
        Parameters
        ----------
        total:
            전체 작업 개수입니다. 모르면 None으로 둡니다.

        desc:
            출력에 표시할 작업 이름입니다.

        log_every:
            몇 step마다 출력할지 설정합니다.

        min_interval_sec:
            너무 자주 출력되지 않도록 최소 출력 간격을 설정합니다.
        """
        self.total = total
        self.desc = desc
        self.log_every = max(1, log_every)
        self.min_interval_sec = min_interval_sec

        self.start_time = None
        self.last_log_time = None
        self.current = 0

    def start(self):
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.current = 0

        if self.total is not None:
            print(f"[{self.desc}] start | total={self.total}")
        else:
            print(f"[{self.desc}] start")

    def update(self, n: int = 1, message: str = ""):
        """
        진행 step을 증가시키고 조건에 맞으면 로그를 출력합니다.
        """
        if self.start_time is None:
            self.start()

        self.current += n

        now = time.perf_counter()
        should_log = (
            self.current == 1
            or self.current % self.log_every == 0
            or (now - self.last_log_time) >= self.min_interval_sec
            or (self.total is not None and self.current >= self.total)
        )

        if not should_log:
            return

        elapsed = now - self.start_time

        if self.total:
            progress_ratio = min(self.current / self.total, 1.0)
            percent = progress_ratio * 100

            if self.current > 0:
                estimated_total = elapsed / progress_ratio if progress_ratio > 0 else 0
                eta = max(0, estimated_total - elapsed)
            else:
                eta = 0

            log = (
                f"[{self.desc}] "
                f"{self.current}/{self.total} "
                f"({percent:.1f}%) "
                f"| elapsed={format_seconds(elapsed)} "
                f"| eta={format_seconds(eta)}"
            )
        else:
            log = (
                f"[{self.desc}] "
                f"step={self.current} "
                f"| elapsed={format_seconds(elapsed)}"
            )

        if message:
            log += f" | {message}"

        print(log)
        self.last_log_time = now

    def done(self, message: str = ""):
        """
        작업 완료 로그를 출력합니다.
        """
        if self.start_time is None:
            self.start()

        elapsed = time.perf_counter() - self.start_time

        log = f"[{self.desc}] done | elapsed={format_seconds(elapsed)}"

        if self.total is not None:
            log += f" | completed={self.current}/{self.total}"

        if message:
            log += f" | {message}"

        print(log)


def progress_iter(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "Progress",
    log_every: int = 10,
    min_interval_sec: float = 5.0,
) -> Iterator[T]:
    """
    Iterable을 감싸서 print 기반 진행률을 출력합니다.

    사용 예:
    for item in progress_iter(items, total=len(items), desc="RAG 실행"):
        ...
    """
    logger = ProgressLogger(
        total=total,
        desc=desc,
        log_every=log_every,
        min_interval_sec=min_interval_sec,
    )

    logger.start()

    for item in iterable:
        yield item
        logger.update(1)

    logger.done()


@contextmanager
def log_step(desc: str):
    """
    단일 작업의 시작/종료 시간을 출력하는 context manager입니다.

    사용 예:
    with log_step("LLM 모델 로드"):
        model = AutoModelForCausalLM.from_pretrained(...)
    """
    start_time = time.perf_counter()
    print(f"[{desc}] start")

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"[{desc}] done | elapsed={format_seconds(elapsed)}")