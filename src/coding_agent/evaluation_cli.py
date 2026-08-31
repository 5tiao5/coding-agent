"""Thin presentation helpers for the opt-in live evaluation command."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from coding_agent.evaluation import EvaluationOutcome, EvaluationReport
from coding_agent.evaluation_scenarios import EvaluationScenario, built_in_scenarios
from coding_agent.presentation import safe_terminal_text

EVALUATION_MAX_MODEL_RETRIES = 1


class EvaluationCase(StrEnum):
    ALL = "all"
    SINGLE_FILE = "single-file"
    CROSS_FILE = "cross-file"
    NEW_FEATURE = "new-feature"
    INDIRECT_DEBUG = "indirect-debug"
    RUNTIME_INTEGRATION = "runtime-integration"


class EvaluationFormat(StrEnum):
    TABLE = "table"
    JSON = "json"


def evaluation_endpoint_label(base_url: str | None) -> str:
    if base_url is None:
        return "default OpenAI endpoint"
    parsed = urlsplit(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def selected_evaluation_scenarios(
    selected_case: EvaluationCase,
) -> tuple[EvaluationScenario, ...]:
    scenarios = built_in_scenarios()
    if selected_case is EvaluationCase.ALL:
        return scenarios
    case_ids = {
        EvaluationCase.SINGLE_FILE: "single_file_fix",
        EvaluationCase.CROSS_FILE: "cross_file_change",
        EvaluationCase.NEW_FEATURE: "new_feature",
        EvaluationCase.INDIRECT_DEBUG: "indirect_debugging",
        EvaluationCase.RUNTIME_INTEGRATION: "runtime_integration",
    }
    case_id = case_ids[selected_case]
    selected = tuple(scenario for scenario in scenarios if scenario.case_id == case_id)
    if len(selected) != 1:
        raise ValueError(f"built-in evaluation case is unavailable: {selected_case.value}")
    return selected


def evaluation_payload(
    report: EvaluationReport,
    *,
    model_name: str,
    planned_cases: int,
    max_steps: int,
    max_model_retries: int,
) -> dict[str, object]:
    """Return a versioned report containing only explicitly public metrics."""

    cases: list[dict[str, object]] = []
    for case in report.cases:
        oracle = case.oracle
        cases.append(
            {
                "id": case.case_id,
                "title": case.title,
                "result": case.outcome.value,
                "agent_state": case.agent_state,
                "stop_reason": case.stop_reason,
                "steps": case.steps,
                "duration_seconds": round(case.duration_seconds, 6),
                "tool_errors": case.tool_errors,
                "verified_but_oracle_failed": case.verified_but_oracle_failed,
                "baseline": {
                    "command_status": case.baseline.command_status,
                    "return_code": case.baseline.return_code,
                    "timed_out": case.baseline.timed_out,
                    "completed_cleanly": case.baseline.completed_cleanly,
                    "collected_tests": case.baseline.collected_tests,
                    "passed_tests": case.baseline.passed_tests,
                    "duration_seconds": round(case.baseline.duration_seconds, 6),
                },
                "oracle": (
                    None
                    if oracle is None
                    else {
                        "passed": oracle.passed,
                        "command_status": oracle.command_status,
                        "return_code": oracle.return_code,
                        "timed_out": oracle.timed_out,
                        "completed_cleanly": oracle.completed_cleanly,
                        "collected_tests": oracle.collected_tests,
                        "passed_tests": oracle.passed_tests,
                        "duration_seconds": round(oracle.duration_seconds, 6),
                    }
                ),
                "integrity": {
                    "protected_tests_unchanged": case.protected_tests_unchanged,
                    "required_files_present": case.required_files_present,
                    "required_changes_present": case.required_changes_present,
                    "only_allowed_paths_changed": case.only_allowed_paths_changed,
                    "allowed_changed_paths": list(case.allowed_changed_paths),
                    "observed_changed_paths": list(case.observed_changed_paths),
                    "unexpected_changed_paths": list(case.unexpected_changed_paths),
                },
                "failure_reasons": list(case.failure_reasons),
            }
        )
    aggregate = report.aggregate
    return {
        "schema_version": "coding-agent.eval.v2",
        "mode": "live",
        "model": model_name,
        "limits": {
            "planned_cases": planned_cases,
            "max_steps_per_case": max_steps,
            "model_attempts_per_step": max_model_retries + 1,
            "request_attempt_ceiling": planned_cases * max_steps * (max_model_retries + 1),
        },
        "cases": cases,
        "summary": {
            "total": aggregate.total_cases,
            "passed": aggregate.successful_cases,
            "failed": aggregate.failed_cases,
            "errors": aggregate.error_cases,
            "verified_but_oracle_failed": aggregate.verified_but_oracle_failed_cases,
            "success_rate": round(aggregate.success_rate, 6),
            "steps_total": aggregate.total_steps,
            "steps_average": round(aggregate.average_steps, 6),
            "tool_errors_total": aggregate.total_tool_errors,
            "duration_seconds": round(aggregate.total_duration_seconds, 6),
            "duration_average_seconds": round(aggregate.average_duration_seconds, 6),
        },
    }


def validate_new_report_path(path: Path) -> Path:
    target = path.expanduser()
    parent = target.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("evaluation report parent must be a directory")
    if target.exists() or target.is_symlink():
        raise ValueError("evaluation report already exists; choose a new path")
    return target


def write_new_report(path: Path, rendered: str) -> None:
    """Write one report atomically with respect to accidental overwrites."""

    target = validate_new_report_path(path)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
        stream.write("\n")


def print_evaluation_report(report: EvaluationReport, *, console: Console) -> None:
    table = Table(title="LIVE EVALUATION", box=box.ASCII, show_lines=False)
    table.add_column("CASE", style="cyan")
    table.add_column("RESULT", no_wrap=True)
    table.add_column("AGENT", no_wrap=True)
    table.add_column("BASELINE", no_wrap=True)
    table.add_column("ORACLE", no_wrap=True)
    table.add_column("INTEGRITY", no_wrap=True)
    table.add_column("STEPS", justify="right")
    table.add_column("ERRORS", justify="right")
    table.add_column("TIME", justify="right")
    for case in report.cases:
        oracle = case.oracle
        integrity_ok = (
            case.protected_tests_unchanged
            and case.required_files_present
            and case.required_changes_present
            and case.only_allowed_paths_changed
        )
        table.add_row(
            case.case_id,
            {
                EvaluationOutcome.PASSED: "[green]PASS[/green]",
                EvaluationOutcome.FAILED: "[red]FAIL[/red]",
                EvaluationOutcome.ERROR: "[yellow]ERROR[/yellow]",
            }[case.outcome],
            (case.agent_state or "not run").upper(),
            "RED" if case.baseline.return_code == 1 and not case.baseline.timed_out else "INVALID",
            "NOT RUN" if oracle is None else ("PASS" if oracle.passed else "FAIL"),
            "PASS" if integrity_ok else "FAIL",
            str(case.steps),
            str(case.tool_errors),
            f"{case.duration_seconds:.2f}s",
        )
    console.print(table)

    aggregate = report.aggregate
    summary_style = "green" if aggregate.successful_cases == aggregate.total_cases else "yellow"
    console.print(
        Text(
            (
                f"Passed {aggregate.successful_cases}/{aggregate.total_cases} "
                f"({aggregate.success_rate:.0%}) · {aggregate.total_steps} steps · "
                f"{aggregate.total_tool_errors} tool errors · "
                f"{aggregate.verified_but_oracle_failed_cases} false-green claims · "
                f"{aggregate.total_duration_seconds:.2f}s"
            ),
            style=summary_style,
        )
    )
    for case in report.cases:
        for reason in case.failure_reasons:
            console.print(
                Text(
                    f"{case.case_id}: {safe_terminal_text(reason, console=console)}",
                    style="red",
                )
            )


__all__ = [
    "EVALUATION_MAX_MODEL_RETRIES",
    "EvaluationCase",
    "EvaluationFormat",
    "evaluation_endpoint_label",
    "evaluation_payload",
    "print_evaluation_report",
    "selected_evaluation_scenarios",
    "validate_new_report_path",
    "write_new_report",
]
