"""FastAPI request observability middleware."""

import logging
from time import perf_counter

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .context import reset_request_id, resolve_request_id, set_request_id
from .logging import log_event


class RequestObservabilityMiddleware:
    """Attach request IDs and emit request lifecycle events."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = resolve_request_id(headers.get("X-Request-ID"))
        token = set_request_id(request_id)
        started = perf_counter()
        method = scope.get("method", "")
        path = scope.get("path", "")
        status_code: int | None = None
        log_event(
            "request.started",
            component="api",
            method=method,
            path=path,
        )

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Request-ID"] = request_id
            await send(message)

        try:
            await self._app(scope, receive, send_with_request_id)
        except Exception as error:
            log_event(
                "request.failed",
                component="api",
                duration_ms=(perf_counter() - started) * 1000,
                status=500,
                error_type=type(error).__name__,
                method=method,
                path=path,
                level=logging.ERROR,
            )
            raise
        else:
            resolved_status = status_code if status_code is not None else 500
            event = "request.failed" if resolved_status >= 500 else "request.completed"
            log_event(
                event,
                component="api",
                duration_ms=(perf_counter() - started) * 1000,
                status=resolved_status,
                method=method,
                path=path,
                level=logging.ERROR if event == "request.failed" else logging.INFO,
            )
        finally:
            reset_request_id(token)
