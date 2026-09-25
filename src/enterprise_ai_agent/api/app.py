"""FastAPI application factory and entrypoint."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from enterprise_ai_agent.agent import Agent, MaxToolStepsExceededError
from enterprise_ai_agent.llm import UnsupportedToolCallsError
from enterprise_ai_agent.tools import ToolArgumentError, ToolNotFoundError

from .models import ErrorResponse
from .routes import router


def create_app(agent: Agent[str] | None = None) -> FastAPI:
    """Create the API application with an optional injected Agent."""

    app = FastAPI(
        title="Enterprise AI Agent API",
        version="0.1.0",
    )
    app.state.agent = agent
    app.include_router(router)
    _register_exception_handlers(app)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(status_code=422, detail="Invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        return _error_response(status_code=error.status_code, detail=str(error.detail))

    @app.exception_handler(MaxToolStepsExceededError)
    async def handle_max_tool_steps_error(
        request: Request,
        error: MaxToolStepsExceededError,
    ) -> JSONResponse:
        return _error_response(status_code=502, detail="Agent tool call limit exceeded")

    @app.exception_handler(ToolNotFoundError)
    async def handle_tool_not_found_error(
        request: Request,
        error: ToolNotFoundError,
    ) -> JSONResponse:
        return _error_response(status_code=502, detail="Agent requested an unavailable tool")

    @app.exception_handler(ToolArgumentError)
    async def handle_tool_argument_error(
        request: Request,
        error: ToolArgumentError,
    ) -> JSONResponse:
        return _error_response(status_code=502, detail="Agent produced invalid tool arguments")

    @app.exception_handler(UnsupportedToolCallsError)
    async def handle_unsupported_tool_calls_error(
        request: Request,
        error: UnsupportedToolCallsError,
    ) -> JSONResponse:
        return _error_response(status_code=502, detail="Agent produced unsupported tool calls")

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        return _error_response(status_code=500, detail="Internal server error")


def _error_response(*, status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(detail=detail).model_dump(),
    )


app = create_app()
