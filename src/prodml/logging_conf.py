import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from functools import wraps
from time import perf_counter
from typing import Any, Callable, TypeVar

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


STANDARD_LOG_FIELDS = set(logging.makeLogRecord({}).__dict__.keys())

STANDARD_LOG_FIELDS.update({"message", "asctime"})


class JsonFormatter(logging.Formatter):
    """Convert LogRecord objects to JSON."""

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": (correlation_id_var.get()),
        }

        for key, value in record.__dict__.items():
            if key not in STANDARD_LOG_FIELDS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            payload,
            default=str,
        )


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging for the application."""

    root_logger = logging.getLogger()

    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(JsonFormatter())

    root_logger.addHandler(handler)

    root_logger.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_correlation_id(correlation_id: str) -> Token[str]:
    return correlation_id_var.set(correlation_id)


def reset_correlation_id(token: Token[str]) -> None:
    correlation_id_var.reset(token)


def get_correlation_id() -> str:
    return correlation_id_var.get()


F = TypeVar("F", bound=Callable[..., Any])


# @timed decorator


def timed(func: F) -> F:
    """Log execution time for a function."""

    logger = get_logger("prodml.timing")

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:

        start = perf_counter()

        try:
            return func(*args, **kwargs)

        finally:

            latency_ms = (perf_counter() - start) * 1000

            logger.info(
                "function.timed",
                extra={
                    "function": func.__qualname__,
                    "latency_ms": round(
                        latency_ms,
                        3,
                    ),
                },
            )

    return wrapper  # type: ignore[return-value]
