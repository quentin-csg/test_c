import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    level_upper = level.upper()
    numeric = getattr(logging, level_upper, None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level {level!r}. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL")
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.dev.ConsoleRenderer() if sys.stdout.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()
