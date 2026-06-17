"""
logger.py
---------
Centralised, structured logging configuration for the ADVOCATE backend.

All modules import `get_logger(__name__)` so every log line is tagged with its
origin module, severity level, and an ISO-8601 timestamp.  The format is
human-readable in development and can be piped to log-aggregation tools (e.g.
Datadog, Loki) in production without modification.
"""

import logging
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# ANSI colour codes – used only when output is attached to a real terminal.
# ---------------------------------------------------------------------------
_COLOURS: dict[str, str] = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}


class _ColourFormatter(logging.Formatter):
    """
    Custom log formatter that adds ANSI colour codes to the level name when
    writing to a TTY.  Falls back to plain text when stdout is not a terminal
    (e.g. when redirected to a file or captured by a process supervisor).
    """

    _FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    _DATEFMT = "%Y-%m-%dT%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        # Only colourise when the handler's stream is a real TTY.
        use_colour = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        if use_colour:
            colour = _COLOURS.get(record.levelname, _COLOURS["RESET"])
            reset = _COLOURS["RESET"]
            record.levelname = f"{colour}{record.levelname}{reset}"

        formatter = logging.Formatter(self._FMT, datefmt=self._DATEFMT)
        return formatter.format(record)


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger once at application start-up.

    Parameters
    ----------
    level:
        The minimum severity level to emit (e.g. ``"DEBUG"``, ``"INFO"``).
        Defaults to ``"INFO"``.

    Notes
    -----
    This function is idempotent – calling it multiple times does not add
    duplicate handlers.
    """
    root_logger = logging.getLogger()

    # Guard: don't re-add handlers if already configured.
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColourFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence noisy third-party loggers to keep output clean.
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a named logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module so log lines are
        automatically attributed to their source.

    Returns
    -------
    logging.Logger
        A configured :class:`logging.Logger` instance.

    Example
    -------
    >>> from app.utils.logger import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Service started")
    """
    configure_logging()
    return logging.getLogger(name or "advocate")
