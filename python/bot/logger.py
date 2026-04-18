import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    level_upper = level.upper()
    numeric = getattr(logging, level_upper, None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level {level!r}. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=50_000_000, backupCount=5))

    logging.basicConfig(format="%(message)s", level=numeric, handlers=handlers)

    # Force JSON when writing to a file so Promtail can parse it.
    use_json = log_file is not None or not sys.stdout.isatty()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()
