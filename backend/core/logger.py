"""Centralized Loguru logger configuration for OMNIMIND.

Importing :data:`logger` from this module is sufficient to get a fully
configured logger that emits coloured records to stdout at ``INFO`` level
and persists ``DEBUG`` records to a rotating file at ``logs/omnimind.log``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_STDOUT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "omnimind.log"


def _configure() -> None:
    """Install sinks for the global Loguru logger.

    The default handler is removed first to guarantee idempotency when the
    module is reloaded (for example by Uvicorn's auto-reloader).
    """

    logger.remove()

    logger.add(
        sys.stdout,
        format=_STDOUT_FORMAT,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        _LOG_FILE,
        format=_STDOUT_FORMAT,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )


_configure()

__all__ = ["logger"]
