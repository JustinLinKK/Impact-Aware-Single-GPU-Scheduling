"""Logging helpers."""

from __future__ import annotations

from pathlib import Path
import logging


class _EphemeralFileHandler(logging.Handler):
    """File handler that does not keep the log file open between records."""

    terminator = "\n"

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = Path(log_path)
        self._scheduler_file_handler = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            message = self.format(record)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + self.terminator)
        except Exception:
            self.handleError(record)


def setup_scheduler_logger(log_path: Path | None = None, *, level: int = logging.INFO) -> logging.Logger:
    """Configure a dedicated scheduler logger."""
    logger = logging.getLogger("localml_scheduler")
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not any(getattr(handler, "_scheduler_stream_handler", False) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler._scheduler_stream_handler = True  # type: ignore[attr-defined]
        logger.addHandler(stream_handler)

    if log_path is not None:
        resolved_log_path = Path(log_path).resolve()
        for handler in list(logger.handlers):
            if getattr(handler, "_scheduler_file_handler", False):
                logger.removeHandler(handler)
                handler.close()
        file_handler = _EphemeralFileHandler(resolved_log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
