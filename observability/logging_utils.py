"""Logging helpers."""

from __future__ import annotations

from pathlib import Path
import logging


def setup_scheduler_logger(log_path: Path | None = None, *, level: int = logging.INFO) -> logging.Logger:
    """Configure a dedicated scheduler logger."""
    logger = logging.getLogger("localml_scheduler")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
