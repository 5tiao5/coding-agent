"""Independent regression-oracle harness for small repository coding tasks.

The Agent never decides whether a case passed. Each case is materialized in a fresh
temporary repository, proven red with a public pytest suite, executed through an injected
repository runner, and then judged in a sibling directory by separate regression tests that
receive only allowlisted source files.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from coding_agent._workspace.native import is_link_like
from coding_agent.application import (
    ModelFactory,
    RepositoryRunSpec,
    execute_repository_run,
)
from coding_agent.command import (
    CommandEnvironmentProfile,
    CommandPermissionMode,
    CommandRequest,
    CommandStatus,
    LocalCommandRunner,
    decode_command_output,
)
from coding_agent.errors import CodedError
from coding_agent.evaluation_scenarios import (
    EvaluationScenario,
    FixtureFile,
    built_in_scenarios,
)
from coding_agent.events import EventKind, EventSink, MemoryEventSink
from coding_agent.models import AgentResult, AgentState, StopReason
from coding_agent.openai_model import ReasoningEffort
from coding_agent.state import StatePaths

_MAX_ORACLE_OUTPUT_CHARS = 12_000
_PYTEST_CONTROL_NAMES = frozenset(
    {"conftest.py", "pyproject.toml", "pytest.ini", "pytest.toml", "setup.cfg", "tox.ini"}
)
_INFRASTRUCTURE_STOP_REASONS = frozenset(
    {
        StopReason.MODEL_ERROR,
        StopReason.USER_INTERRUPTED,
        StopReason.COMMAND_CONTROL_FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Host-selected runtime configuration shared by all evaluated cases."""

    model_name: str
    base_url: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    permission_mode: CommandPermissionMode = CommandPermissionMode.SAFE
    max_steps: int = 20
    model_timeout: float = 120.0
    oracle_timeout: float = 60.0
    max_model_retries: int = 1

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("evaluation model_name cannot be blank")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("evaluation max_steps must be an integer of at least 1")
        if any(
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not isfinite(timeout)
            or timeout <= 0
            for timeout in (self.model_timeout, self.oracle_timeout)
        ):
            raise ValueError("evaluation timeouts must be finite positive numbers")
        if (
            isinstance(self.max_model_retries, bool)
            or not isinstance(self.max_model_retries, int)
            or not 0 <= self.max_model_retries <= 2
        ):
            raise ValueError("evaluation max_model_retries must be an integer between 0 and 2")


class EvaluationOutcome(StrEnum):
    """Stable distinction between task failure and evaluation infrastructure failure."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class EvaluationRunCallable(Protocol):
    """Repository execution seam used by production and deterministic offline tests."""

    def __call__(
        self,
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink | None,
        model_factory: ModelFactory | None,
    ) -> AgentResult: ...


@dataclass(frozen=True, slots=True)
class PytestMetrics:
    """One host-owned pytest invocation and its bounded diagnostic output."""

    command: tuple[str, ...]
    return_code: int | None
    duration_seconds: float
    timed_out: bool
    output: str
    command_status: str
    completed_cleanly: bool
    collected_tests: int
    passed_tests: int

    @property
    def passed(self) -> bool:
        return (
            self.completed_cleanly
            and not self.timed_out
            and self.command_status == CommandStatus.EXITED.value
            and self.return_code == 0
            and self.collected_tests > 0
            and self.passed_tests == self.collected_tests
        )


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    """Structured outcome for one isolated evaluation case."""

    case_id: str
    title: str
    run_id: str | None
    outcome: EvaluationOutcome
    agent_state: str | None
    stop_reason: str | None
    steps: int
    duration_seconds: float
    tool_errors: int
    baseline: PytestMetrics
    oracle: PytestMetrics | None
    protected_tests_unchanged: bool
    required_files_present: bool
    required_changes_present: bool
    failure_reasons: tuple[str, ...]

    @property
    def success(self) -> bool:
        return self.outcome is EvaluationOutcome.PASSED


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Suite-level metrics derived only from immutable per-case outcomes."""

    total_cases: int
    successful_cases: int
    failed_cases: int
    error_cases: int
    success_rate: float
    total_steps: int
    average_steps: float
    total_duration_seconds: float
    average_duration_seconds: float
    total_tool_errors: int


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Evaluation cases and their aggregate, suitable for JSON/dataclass export."""

    cases: tuple[CaseMetrics, ...]
    aggregate: AggregateMetrics


def evaluate_case(
    scenario: EvaluationScenario,
    *,
    config: EvaluationConfig,
    run_callable: EvaluationRunCallable | None = None,
    model_factory: ModelFactory | None = None,
    temporary_parent: Path | None = None,
) -> CaseMetrics:
    """Evaluate one scenario in a fresh repository removed before this call returns.

    When ``run_callable`` is omitted, the production ``execute_repository_run`` path is
    used.  Supplying either the whole callable or only ``model_factory`` keeps unit and
    offline evaluation deterministic without adding a second Agent loop.
    """

    parent = _temporary_parent(temporary_parent)
    with TemporaryDirectory(
        prefix=f"coding-agent-eval-{scenario.case_id}-",
        dir=str(parent) if parent is not None else None,
    ) as temporary_directory:
        evaluation_root = Path(temporary_directory).resolve()
        repository = evaluation_root / "repository"
        repository.mkdir()
        _materialize_fixture(repository, scenario)

        protected_manifest = _protected_manifest(repository, scenario)
        changed_hashes = {
            path: _required_file_hash(repository, path) for path in scenario.required_changed_paths
        }
        baseline = _run_host_pytest(repository, timeout=config.oracle_timeout)
        if baseline.timed_out or not baseline.completed_cleanly or baseline.return_code != 1:
            return CaseMetrics(
                case_id=scenario.case_id,
                title=scenario.title,
                run_id=None,
                outcome=EvaluationOutcome.ERROR,
                agent_state=None,
                stop_reason=None,
                steps=0,
                duration_seconds=0.0,
                tool_errors=0,
                baseline=baseline,
                oracle=None,
                protected_tests_unchanged=True,
                required_files_present=False,
                required_changes_present=False,
                failure_reasons=(
                    "baseline pytest must fail with return code 1 before the Agent runs",
                ),
            )

        run_id = uuid4().hex
        paths = StatePaths((evaluation_root / "state").resolve())
        spec = RepositoryRunSpec(
            run_id=run_id,
            task=scenario.task,
            root=repository,
            model_name=config.model_name.strip(),
            base_url=config.base_url,
            reasoning_effort=config.reasoning_effort,
            permission_mode=config.permission_mode,
            paths=paths,
            max_steps=config.max_steps,
            model_timeout=config.model_timeout,
            max_model_retries=config.max_model_retries,
        )
        events = MemoryEventSink()
        selected_runner: EvaluationRunCallable = run_callable or execute_repository_run
        result: AgentResult | None = None
        runner_error: str | None = None
        started = perf_counter()
        try:
            result = selected_runner(
                spec,
                event_sink=events,
                model_factory=model_factory,
            )
        except Exception as exc:  # noqa: BLE001 - one broken case must not abort the suite.
            runner_error = _bounded_error(exc)
        duration_seconds = perf_counter() - started

        oracle, unavailable_oracle_sources = _run_holdout_oracle(
            evaluation_root,
            repository,
            scenario,
            timeout=config.oracle_timeout,
        )
        current_protected_manifest = _protected_manifest(repository, scenario)
        modified_protected = tuple(
            sorted(
                path
                for path in protected_manifest.keys() | current_protected_manifest.keys()
                if protected_manifest.get(path) != current_protected_manifest.get(path)
            )
        )
        missing_required = tuple(
            path
            for path in (*scenario.required_changed_paths, *scenario.required_new_paths)
            if _file_hash(repository, path) is None
        )
        unchanged_required = tuple(
            path
            for path, initial_hash in changed_hashes.items()
            if _file_hash(repository, path) == initial_hash
        )
        protected_unchanged = not modified_protected
        required_files_present = not missing_required
        required_changes_present = not unchanged_required
        tool_errors = sum(
            event.kind is EventKind.TOOL_FINISHED and event.data.get("ok") is False
            for event in events.events
        )

        failure_reasons: list[str] = []
        if runner_error is not None:
            failure_reasons.append(f"runner raised {runner_error}")
        if result is None:
            failure_reasons.append("runner did not return an AgentResult")
        else:
            if result.run_id != run_id:
                failure_reasons.append("runner returned a result for a different run ID")
            if result.state is not AgentState.COMPLETED:
                failure_reasons.append(
                    "AgentResult did not report verified completion: " + result.state.value
                )
        if unavailable_oracle_sources:
            failure_reasons.append(
                "host oracle source files are missing or unsafe: "
                + ", ".join(unavailable_oracle_sources)
            )
        if oracle is None or not oracle.passed:
            failure_reasons.append("independent host pytest oracle did not pass")
        if modified_protected:
            failure_reasons.append("protected test files changed: " + ", ".join(modified_protected))
        if missing_required:
            failure_reasons.append("required files are missing: " + ", ".join(missing_required))
        if unchanged_required:
            failure_reasons.append(
                "required source files were not changed: " + ", ".join(unchanged_required)
            )

        runner_protocol_error = result is not None and result.run_id != run_id
        execution_error = (
            runner_error is not None
            or result is None
            or runner_protocol_error
            or (
                result is not None
                and result.state is AgentState.FAILED
                and result.stop_reason in _INFRASTRUCTURE_STOP_REASONS
            )
            or (oracle is not None and oracle.command_status == CommandStatus.CONTROL_FAILED.value)
        )
        outcome = (
            EvaluationOutcome.ERROR
            if execution_error
            else (EvaluationOutcome.FAILED if failure_reasons else EvaluationOutcome.PASSED)
        )
        return CaseMetrics(
            case_id=scenario.case_id,
            title=scenario.title,
            run_id=run_id,
            outcome=outcome,
            agent_state=result.state.value if result is not None else None,
            stop_reason=result.stop_reason.value if result is not None else None,
            steps=result.steps if result is not None else 0,
            duration_seconds=duration_seconds,
            tool_errors=tool_errors,
            baseline=baseline,
            oracle=oracle,
            protected_tests_unchanged=protected_unchanged,
            required_files_present=required_files_present,
            required_changes_present=required_changes_present,
            failure_reasons=tuple(failure_reasons),
        )


def evaluate_suite(
    *,
    config: EvaluationConfig,
    scenarios: Iterable[EvaluationScenario] | None = None,
    run_callable: EvaluationRunCallable | None = None,
    model_factory: ModelFactory | None = None,
    temporary_parent: Path | None = None,
) -> EvaluationReport:
    """Run cases sequentially and compute success, step, duration, and tool-error metrics."""

    selected = tuple(built_in_scenarios() if scenarios is None else scenarios)
    case_ids = tuple(scenario.case_id for scenario in selected)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation scenario IDs must be unique")
    evaluated: list[CaseMetrics] = []
    for scenario in selected:
        metrics = evaluate_case(
            scenario,
            config=config,
            run_callable=run_callable,
            model_factory=model_factory,
            temporary_parent=temporary_parent,
        )
        evaluated.append(metrics)
        if metrics.outcome is EvaluationOutcome.ERROR:
            break
    cases = tuple(evaluated)
    return EvaluationReport(cases=cases, aggregate=_aggregate(cases))


def _aggregate(cases: tuple[CaseMetrics, ...]) -> AggregateMetrics:
    total = len(cases)
    successful = sum(case.success for case in cases)
    errors = sum(case.outcome is EvaluationOutcome.ERROR for case in cases)
    failed = total - successful - errors
    total_steps = sum(case.steps for case in cases)
    total_duration = sum(case.duration_seconds for case in cases)
    return AggregateMetrics(
        total_cases=total,
        successful_cases=successful,
        failed_cases=failed,
        error_cases=errors,
        success_rate=successful / total if total else 0.0,
        total_steps=total_steps,
        average_steps=total_steps / total if total else 0.0,
        total_duration_seconds=total_duration,
        average_duration_seconds=total_duration / total if total else 0.0,
        total_tool_errors=sum(case.tool_errors for case in cases),
    )


def _materialize_fixture(repository: Path, scenario: EvaluationScenario) -> None:
    _materialize_files(repository, scenario.files)


def _materialize_files(repository: Path, fixtures: Iterable[FixtureFile]) -> None:
    for fixture in fixtures:
        destination = _fixture_path(repository, fixture.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(fixture.content.encode("utf-8"))


def _run_holdout_oracle(
    evaluation_root: Path,
    repository: Path,
    scenario: EvaluationScenario,
    *,
    timeout: float,
) -> tuple[PytestMetrics | None, tuple[str, ...]]:
    oracle_repository = evaluation_root / "host-oracle"
    oracle_repository.mkdir()
    unavailable: list[str] = []
    for relative in scenario.oracle_source_paths:
        source = _safe_repository_file(repository, relative)
        if source is None:
            unavailable.append(relative)
            continue
        try:
            content = source.read_bytes()
        except OSError:
            unavailable.append(relative)
            continue
        destination = _fixture_path(oracle_repository, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    _materialize_files(oracle_repository, scenario.oracle_files)
    if unavailable:
        return None, tuple(unavailable)
    return _run_host_pytest(oracle_repository, timeout=timeout), ()


def _protected_manifest(
    repository: Path,
    scenario: EvaluationScenario,
) -> dict[str, str]:
    """Hash public tests and pytest control files, including newly added ones."""

    explicit = set(scenario.protected_paths)
    manifest: dict[str, str] = {}
    for path in repository.rglob("*"):
        relative = path.relative_to(repository).as_posix()
        parts = PurePosixPath(relative).parts
        if "__pycache__" in parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        is_test_file = bool(parts) and parts[0] == "tests"
        is_control_file = path.name.lower() in _PYTEST_CONTROL_NAMES
        if relative not in explicit and not is_test_file and not is_control_file:
            continue
        if path.is_symlink():
            manifest[relative] = "symlink"
        elif path.is_file():
            try:
                content = path.read_bytes()
            except OSError:
                manifest[relative] = "unreadable"
            else:
                manifest[relative] = "sha256:" + hashlib.sha256(content).hexdigest()
        elif relative in explicit:
            manifest[relative] = "non-file"
    for relative in explicit:
        if relative not in manifest:
            manifest[relative] = "missing"
    return manifest


def _run_host_pytest(repository: Path, *, timeout: float) -> PytestMetrics:
    marker = f"__CODING_AGENT_PYTEST_{uuid4().hex}__"
    command = (
        sys.executable,
        "-I",
        "-B",
        "-c",
        _pytest_bootstrap(marker),
    )
    started = perf_counter()
    result = LocalCommandRunner(max_output_bytes=_MAX_ORACLE_OUTPUT_CHARS * 4).run(
        CommandRequest(
            argv=command,
            cwd=repository,
            timeout_seconds=timeout,
            environment_profile=CommandEnvironmentProfile.VERIFIER,
        )
    )
    output, _ = decode_command_output(result.output)
    public_output, completed_cleanly, collected_tests, passed_tests = _parse_pytest_marker(
        output,
        marker=marker,
        return_code=result.exit_code,
    )
    return PytestMetrics(
        command=command,
        return_code=result.exit_code,
        duration_seconds=perf_counter() - started,
        timed_out=result.status is CommandStatus.TIMED_OUT,
        output=_bounded_output(public_output),
        command_status=result.status.value,
        completed_cleanly=completed_cleanly,
        collected_tests=collected_tests,
        passed_tests=passed_tests,
    )


def _pytest_bootstrap(marker: str) -> str:
    return f"""
import os
import sys
from pathlib import Path

import pytest


class EvaluationPlugin:
    def __init__(self):
        self.collected = 0
        self.passed = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed += 1


os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
sys.path.insert(0, str(Path.cwd()))
plugin = EvaluationPlugin()
exit_code = pytest.main(["-q", "-p", "no:cacheprovider"], plugins=[plugin])
print("{marker}:" + f"{{int(exit_code)}}:{{plugin.collected}}:{{plugin.passed}}")
raise SystemExit(exit_code)
""".strip()


def _parse_pytest_marker(
    output: str,
    *,
    marker: str,
    return_code: int | None,
) -> tuple[str, bool, int, int]:
    prefix = marker + ":"
    public_lines: list[str] = []
    parsed: tuple[int, int, int] | None = None
    for line in output.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(prefix):
            fields = stripped.removeprefix(prefix).split(":")
            try:
                values = tuple(int(field) for field in fields)
            except ValueError:
                public_lines.append(line)
                continue
            if len(values) == 3:
                parsed = (values[0], values[1], values[2])
                continue
        public_lines.append(line)
    if parsed is None:
        return "".join(public_lines), False, 0, 0
    pytest_code, collected, passed = parsed
    completed = return_code is not None and pytest_code == return_code
    return "".join(public_lines), completed, max(0, collected), max(0, passed)


def _temporary_parent(value: Path | None) -> Path | None:
    if value is None:
        return None
    resolved = value.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("temporary_parent must be a directory")
    return resolved


def _fixture_path(repository: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    return repository.joinpath(*parts)


def _required_file_hash(repository: Path, relative: str) -> str:
    digest = _file_hash(repository, relative)
    if digest is None:  # Scenario validation guarantees this unless materialization is broken.
        raise RuntimeError(f"fixture file was not materialized: {relative}")
    return digest


def _file_hash(repository: Path, relative: str) -> str | None:
    path = _safe_repository_file(repository, relative)
    if path is None:
        return None
    try:
        content = path.read_bytes()
    except (OSError, RuntimeError):
        return None
    return hashlib.sha256(content).hexdigest()


def _safe_repository_file(repository: Path, relative: str) -> Path | None:
    """Reject a source path with a link/reparse ancestor or escaped resolution."""

    try:
        root = repository.resolve(strict=True)
        current = root
        for part in PurePosixPath(relative).parts:
            current /= part
            if is_link_like(current):
                return None
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(root) or not resolved.is_file():
        return None
    return resolved


def _bounded_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    if len(text) <= _MAX_ORACLE_OUTPUT_CHARS:
        return text
    return text[: _MAX_ORACLE_OUTPUT_CHARS - 3] + "..."


def _bounded_error(exc: Exception) -> str:
    if isinstance(exc, CodedError):
        return f"{type(exc).__name__}[{exc.code}]"
    return type(exc).__name__


__all__ = [
    "AggregateMetrics",
    "CaseMetrics",
    "EvaluationConfig",
    "EvaluationOutcome",
    "EvaluationReport",
    "EvaluationRunCallable",
    "PytestMetrics",
    "evaluate_case",
    "evaluate_suite",
]
