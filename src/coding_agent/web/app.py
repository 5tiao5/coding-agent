"""FastAPI application factory for the local coding-agent Web frontend."""

from __future__ import annotations

import asyncio
import hmac
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Protocol
from unicodedata import category

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from coding_agent.dashboard import MAX_WORKSPACE_CHANGES
from coding_agent.web.native_picker import (
    NativeFolderPicker,
    NativeFolderPickerBusyError,
    NativeFolderPickerUnavailableError,
)
from coding_agent.web.service import (
    RunAlreadyActiveError,
    WebResumeUnsupportedError,
    WebRunStartError,
    WebRunStatePayload,
    WebServiceClosedError,
)
from coding_agent.web.workbench import (
    HistoryPayload,
    NoActiveProjectError,
    ProjectListPayload,
    ProjectPayload,
    ProjectRemovalPayload,
    ProjectRunsPayload,
    WebWorkbench,
    WorkbenchBusyError,
    WorkbenchError,
    WorkbenchFollowUpError,
    WorkbenchInputError,
    WorkbenchNotFoundError,
    WorkbenchResumeError,
)

DEFAULT_SHUTDOWN_DRAIN_SECONDS = 5.0


class _RunService(Protocol):
    """Narrow service surface shared by fixed and project-aware hosts."""

    def start(self, task: str) -> WebRunStatePayload: ...

    def state(self) -> WebRunStatePayload: ...

    def shutdown(self, timeout: float | None = None) -> bool: ...


class RunRequest(BaseModel):
    """One bounded task submitted by the local browser."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=20_000)
    use_project_memory: bool = True
    parent_run_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$",
    )

    @field_validator("task")
    @classmethod
    def normalize_task(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("task cannot be blank")
        if any(
            category(character).startswith("C") and character not in {"\t", "\n", "\r"}
            for character in normalized
        ):
            raise ValueError("task cannot contain control or formatting characters")
        return normalized


class ActivityFactResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=32)
    value: str = Field(min_length=1, max_length=16_000)
    format: Literal["text", "code", "pre", "status"]


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
    activity_id: str | None = Field(pattern=r"^act_[0-9a-f]{16}$")
    activity_state: Literal["started", "finished"] | None
    facts: list[ActivityFactResponse] = Field(max_length=12)
    facts_complete: bool


class LatestChangeResponse(TimelineResponse):
    expanded_preview: list[str]
    expanded_preview_complete: bool


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


class RunLimitsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_turns: int = Field(ge=1, le=1_000_000)
    max_calls_per_turn: int = Field(ge=1, le=1_000_000)
    max_total_tool_calls: int = Field(ge=1, le=1_000_000)


class SnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_label: str
    phase: str
    current_step: int
    limits: RunLimitsResponse | None
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
    workspace_changes: list[LatestChangeResponse] = Field(max_length=MAX_WORKSPACE_CHANGES)
    workspace_changes_complete: bool
    omitted_change_count: int = Field(ge=0)
    latest_change: LatestChangeResponse | None


class ProjectMemorySourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task: str
    completed_at: str | None


class ProjectMemoryContextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: bool
    applied: bool
    source_run_ids: list[str] = Field(max_length=6)
    sources: list[ProjectMemorySourceResponse] = Field(max_length=6)
    error: str | None


class WebRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    run_id: str | None
    task: str | None
    final_text: str | None
    error: str | None
    snapshot: SnapshotResponse | None
    memory_context: ProjectMemoryContextResponse | None


class WebMetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: str
    runtime: str
    task_locked: bool
    default_task: str | None
    control_token: str | None = None
    native_folder_picker_available: bool


class ProjectRegistrationRequest(BaseModel):
    """One explicit local project registration or empty-directory creation."""

    model_config = ConfigDict(extra="forbid")

    root: str = Field(min_length=1, max_length=4096)
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    create: bool = False

    @field_validator("root", "display_name")
    @classmethod
    def normalize_bounded_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be blank")
        return normalized


class SelectProjectRequest(BaseModel):
    """Strict empty JSON body used to keep selection non-simple and auditable."""

    model_config = ConfigDict(extra="forbid")


class ResumeRunRequest(BaseModel):
    """Strict empty body for continuing one server-selected checkpoint."""

    model_config = ConfigDict(extra="forbid")


class PickFolderRequest(BaseModel):
    """Strict empty body for the native local-control operation."""

    model_config = ConfigDict(extra="forbid")


class PickFolderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["selected", "cancelled"]
    path: str | None


class ProjectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    display_name: str
    root: str
    created_at: str
    last_opened_at: str


class ProjectListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_project_id: str | None
    projects: list[ProjectResponse]


class ProjectRemovalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    removed_from_sidebar: Literal[True]
    workspace_deleted: Literal[False]
    history_preserved: Literal[True]


class RunContinuationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["resume", "follow_up", "none"]
    available: bool
    reason: str | None = None


class RunCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    parent_run_id: str | None = None
    project_id: str
    task: str
    status: str
    created_at: str
    completed_at: str | None
    final_text: str | None = None
    error: str | None = None
    resume_available: bool
    resume_reason: str | None = None
    continuation: RunContinuationResponse
    memory_context: ProjectMemoryContextResponse


class ProjectRunsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    runs: list[RunCatalogResponse]


class HistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunCatalogResponse
    snapshot: SnapshotResponse


def create_app(
    service: _RunService,
    *,
    static_dir: str | Path | None = None,
    workspace_label: str = "本地仓库",
    runtime_label: str = "本地 Agent",
    default_task: str | None = None,
    task_locked: bool = False,
    workbench: WebWorkbench | None = None,
    native_folder_picker: NativeFolderPicker | None = None,
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
    control_token = secrets.token_urlsafe(32) if workbench is not None else None

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

    @app.get(
        "/api/meta",
        response_model=WebMetadataResponse,
        response_model_exclude_none=True,
    )
    def get_metadata() -> WebMetadataResponse:
        return WebMetadataResponse(
            workspace=(workbench.workspace_label if workbench is not None else workspace_label),
            runtime=runtime_label,
            task_locked=task_locked,
            default_task=default_task,
            control_token=control_token,
            native_folder_picker_available=_picker_available(native_folder_picker),
        )

    @app.post(
        "/api/runs",
        response_model=WebRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def start_run(request: RunRequest, raw_request: Request) -> WebRunStatePayload:
        _authorize_mutation(raw_request, control_token)
        if task_locked and request.task != default_task:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="此本地界面已锁定为预设演示任务",
            )
        if workbench is None and request.parent_run_id is not None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="当前固定工作区界面不支持继续历史任务",
            )
        try:
            if workbench is not None:
                return workbench.start(
                    request.task,
                    use_project_memory=request.use_project_memory,
                    parent_run_id=request.parent_run_id,
                )
            return service.start(request.task)
        except RunAlreadyActiveError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from None
        except WorkbenchNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from None
        except (
            NoActiveProjectError,
            WorkbenchBusyError,
            WorkbenchFollowUpError,
            WorkbenchInputError,
        ) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from None
        except (WebRunStartError, WebServiceClosedError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from None
        except WorkbenchError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from None

    if workbench is not None:

        @app.post("/api/folders/pick", response_model=PickFolderResponse)
        def pick_folder(
            request: PickFolderRequest,
            raw_request: Request,
        ) -> PickFolderResponse:
            del request
            _authorize_mutation(raw_request, control_token)
            picker = native_folder_picker
            if picker is None or not _picker_available(picker):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="本机文件夹选择器暂时不可用",
                )
            try:
                with workbench.reserve_navigation():
                    selected = picker.pick_directory()
            except WorkbenchBusyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(exc),
                ) from None
            except NativeFolderPickerBusyError:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="已有文件夹选择窗口正在等待操作",
                ) from None
            except NativeFolderPickerUnavailableError:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="本机文件夹选择器暂时不可用",
                ) from None
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="本机文件夹选择器暂时不可用",
                ) from None
            if selected is None:
                return PickFolderResponse(status="cancelled", path=None)
            return PickFolderResponse(status="selected", path=str(selected))

        @app.get("/api/projects", response_model=ProjectListResponse)
        def list_projects() -> ProjectListPayload:
            return workbench.projects()

        @app.post(
            "/api/projects",
            response_model=ProjectResponse,
            status_code=status.HTTP_201_CREATED,
        )
        def register_project(
            request: ProjectRegistrationRequest,
            raw_request: Request,
        ) -> ProjectPayload:
            _authorize_mutation(raw_request, control_token)
            try:
                return workbench.register_project(
                    root=request.root,
                    display_name=request.display_name,
                    create=request.create,
                )
            except Exception as exc:  # Normalize persistence and local path failures.
                raise _workbench_http_error(exc) from None

        @app.post(
            "/api/projects/{project_id}/select",
            response_model=ProjectResponse,
        )
        def select_project(
            project_id: str,
            request: SelectProjectRequest,
            raw_request: Request,
        ) -> ProjectPayload:
            del request
            _authorize_mutation(raw_request, control_token)
            try:
                return workbench.select_project(project_id)
            except Exception as exc:
                raise _workbench_http_error(exc) from None

        @app.delete(
            "/api/projects/{project_id}",
            response_model=ProjectRemovalResponse,
        )
        def remove_project(
            project_id: str,
            raw_request: Request,
        ) -> ProjectRemovalPayload:
            _authorize_mutation(raw_request, control_token)
            try:
                return workbench.remove_project(project_id)
            except Exception as exc:
                raise _workbench_http_error(exc) from None

        @app.get(
            "/api/projects/{project_id}/runs",
            response_model=ProjectRunsResponse,
        )
        def list_project_runs(project_id: str) -> ProjectRunsPayload:
            try:
                return workbench.project_runs(project_id)
            except Exception as exc:
                raise _workbench_http_error(exc) from None

        @app.get("/api/history/{run_id}", response_model=HistoryResponse)
        def get_history(run_id: str) -> HistoryPayload:
            try:
                return workbench.history(run_id)
            except Exception as exc:
                raise _workbench_http_error(exc) from None

        @app.post(
            "/api/runs/{run_id}/resume",
            response_model=WebRunResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        def resume_run(
            run_id: str,
            request: ResumeRunRequest,
            raw_request: Request,
        ) -> WebRunStatePayload:
            del request
            _authorize_mutation(raw_request, control_token)
            try:
                return workbench.resume(run_id)
            except RunAlreadyActiveError as exc:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(exc),
                ) from None
            except (WebRunStartError, WebServiceClosedError, WebResumeUnsupportedError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from None
            except Exception as exc:
                raise _workbench_http_error(exc) from None

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


def _authorize_mutation(request: Request, control_token: str | None) -> None:
    """Require same-origin possession of a per-process token for workbench writes."""

    if control_token is None:
        return
    supplied = request.headers.get("x-coding-agent-token", "")
    if not supplied or not hmac.compare_digest(supplied, control_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="缺少有效的本地控制令牌",
        )
    origin = request.headers.get("origin")
    if origin is None:
        return
    expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
    if not hmac.compare_digest(origin.rstrip("/"), expected_origin.rstrip("/")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="拒绝跨来源的本地控制请求",
        )


def _picker_available(picker: NativeFolderPicker | None) -> bool:
    if picker is None:
        return False
    try:
        return picker.available
    except Exception:
        return False


def _workbench_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, WorkbenchBusyError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, NoActiveProjectError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, WorkbenchNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, WorkbenchResumeError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, (WorkbenchInputError, ValueError)):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="本地项目状态暂时不可用",
    )
