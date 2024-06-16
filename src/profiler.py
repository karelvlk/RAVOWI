from datetime import datetime, timedelta
from logging import Logger
from typing import Literal


class Profiler:
    """Simple profiler to measure `with` block time"""

    def __init__(
        self,
        operation: str,
        logger: Logger,
        log_level: Literal["debug", "info", "warning"] = "info",
    ) -> None:
        self.operation: str = operation
        self.logger: Logger = logger
        self.log_level: Literal["debug", "info", "warning"] = log_level
        self.date_start: datetime = datetime.now()
        self.elapsed: timedelta = timedelta()

    def __enter__(self) -> "Profiler":
        self.date_start = datetime.now()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = datetime.now() - self.date_start
        log_func = getattr(self.logger, self.log_level)
        log_func(f"{self.operation} finished in {self.elapsed.total_seconds():.3f}s")

    async def __aenter__(self) -> "Profiler":
        return self.__enter__()

    async def __aexit__(self, *args) -> None:
        return self.__exit__()
