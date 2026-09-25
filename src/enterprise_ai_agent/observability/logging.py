"""Structured logging helpers built on the standard library."""

import json
import logging
from datetime import UTC, datetime

from .context import get_request_id

OBSERVABILITY_LOGGER_NAME = "enterprise_ai_agent.observability"
_HANDLER_NAME = "enterprise_ai_agent_observability"
_STRUCTURED_FIELDS = (
    "event",
    "request_id",
    "component",
    "duration_ms",
    "status",
    "error_type",
    "tool_name",
    "call_id",
    "method",
    "path",
)


class StructuredFormatter(logging.Formatter):
    """Render selected log record fields as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if "event" not in payload:
            payload["event"] = record.getMessage()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def log_event(
    event: str,
    *,
    component: str,
    duration_ms: float | None = None,
    status: str | int | None = None,
    error_type: str | None = None,
    tool_name: str | None = None,
    call_id: str | None = None,
    method: str | None = None,
    path: str | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit one structured observability event."""

    logger = logging.getLogger(OBSERVABILITY_LOGGER_NAME)
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "request_id": get_request_id(),
            "component": component,
            "duration_ms": duration_ms,
            "status": status,
            "error_type": error_type,
            "tool_name": tool_name,
            "call_id": call_id,
            "method": method,
            "path": path,
        },
    )


def configure_observability_logging(
    level: str | int = "INFO",
    *,
    stream: object | None = None,
) -> logging.Logger:
    """Configure the observability logger with a JSON formatter."""

    logger = logging.getLogger(OBSERVABILITY_LOGGER_NAME)
    logger.setLevel(level)
    if any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        return logger

    handler = logging.StreamHandler(stream)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    return logger
