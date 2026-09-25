"""Request context propagation."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from uuid import uuid4

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def generate_request_id() -> str:
    """Generate a safe request identifier."""

    return uuid4().hex


def resolve_request_id(candidate: str | None = None) -> str:
    """Reuse a safe client request ID or generate a new one."""

    if candidate is not None and _REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return generate_request_id()


def get_request_id() -> str | None:
    """Return the request ID bound to the current context."""

    return _request_id.get()


def set_request_id(request_id: str | None = None) -> Token[str | None]:
    """Bind a request ID and return the reset token."""

    return _request_id.set(resolve_request_id(request_id))


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous request ID context."""

    _request_id.reset(token)


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind one request ID for the duration of a context."""

    resolved = resolve_request_id(request_id)
    token = _request_id.set(resolved)
    try:
        yield resolved
    finally:
        _request_id.reset(token)
