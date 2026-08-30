"""FastAPI application factory for the local coding-agent Web frontend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from coding_agent.web.service import (
    RunAlreadyActiveError,
    WebRunService,
    WebRunStartError,
    WebRunStatePayload,
    WebServiceClosedError,
)

DEFAULT_SHUTDOWN_DRAIN_SECONDS = 5.0


class RunRequest(BaseModel):
    """One bounded task submitted by the local browser."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=20_000)

    @field_validator("task")
    @classmethod
    def normalize_task(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("task cannot be blank")
        return normalized


class TimelineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    category: str
    headline: str
    detail: str | None
    level: str
    offset_seconds: float
    duration_ms: float | None
    preview: list[str]


class ChangedFileResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    added_lines: int
    removed_lines: int
    revision: int
    change_kind: str


class VerificationEvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    kind: str
    passed: bool
    step: int
    epoch: int


class SnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_label: str
    phase: str
    current_step: int
    tools_started: int
    tools_finished: int
    tools_failed: int
    active_tools: list[str]
    verification_status: str
    verification_labels: list[str]
    verification_evidence: list[VerificationEvidenceResponse]
    verification_epoch: int
    invalidation_count: int
    changed_files: list[ChangedFileResponse]
    outcome: str
    plan_lines: list[str]
    timeline: list[TimelineResponse]
    latest_change: TimelineResponse | None


class WebRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    run_id: str | None
    task: str | None
    final_text: str | None
    error: str | None
    snapshot: SnapshotResponse | None


class WebMetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: str
    runtime: str
    task_locked: bool
    default_task: str | None


def create_app(
    service: WebRunService,
    *,
    static_dir: str | Path | None = None,
    workspace_label: str = "本地仓库",
    runtime_label: str = "本地 Agent",
    default_task: str | None = None,
    task_locked: bool = False,
) -> FastAPI:
    """Create an API app bound to one injected run service and asset directory."""
    if task_locked and default_task is None:
        raise ValueError("a locked Web task requires default_task")
    assets = (
        Path(static_dir) if static_dir is not None else Path(__file__).with_name("static")
    ).resolve(strict=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # Give normal completion a short window to flush trace/checkpoint writes.
            # The daemon worker is an exit fallback when a model/tool call outlives it.
            await asyncio.to_thread(
                service.shutdown,
                timeout=DEFAULT_SHUTDOWN_DRAIN_SECONDS,
            )

    app = FastAPI(
        title="Coding Agent 本地界面",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost"],
    )
    app.state.run_service = service
    app.state.static_dir = assets
    metadata = WebMetadataResponse(
        workspace=workspace_label,
        runtime=runtime_label,
        task_locked=task_locked,
        default_task=default_task,
    )

    @app.middleware("http")
    async def secure_local_responses(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # These headers keep this loopback control surface isolated from embedding,
        # MIME sniffing, and external resource loads.
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
            "form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/state", response_model=WebRunResponse)
    def get_state() -> WebRunStatePayload:
        return service.state()

    @app.get("/api/meta", response_model=WebMetadataResponse)
    def get_metadata() -> WebMetadataResponse:
        return metadata

    @app.post(
        "/api/runs",
        response_model=WebRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def start_run(request: RunRequest) -> WebRunStatePayload:
        if metadata.task_locked and request.task != metadata.default_task:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="此本地界面已锁定为预设演示任务",
            )
        try:
            return service.start(request.task)
        except RunAlreadyActiveError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from None
        except (WebRunStartError, WebServiceClosedError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from None

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        index_path = assets / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Web frontend assets are unavailable",
            )
        return FileResponse(index_path)

    app.mount(
        "/static",
        StaticFiles(directory=str(assets), html=False, check_dir=False),
        name="static",
    )
    return app
