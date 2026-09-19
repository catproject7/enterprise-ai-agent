"""Logging initialization for the application."""

import logging

_APPLICATION_HANDLER_NAME = "enterprise_ai_agent"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | int = "INFO") -> None:
    """Configure the root logger without adding duplicate application handlers."""

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if any(
        handler.get_name() == _APPLICATION_HANDLER_NAME
        for handler in root_logger.handlers
    ):
        return

    handler = logging.StreamHandler()
    handler.set_name(_APPLICATION_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(handler)
