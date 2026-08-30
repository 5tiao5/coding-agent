"""CLI contract tests for the opt-in live evaluation boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import coding_agent.cli as cli_module
from coding_agent.cli import app
from coding_agent.command import CommandPermissionMode
from coding_agent.evaluation import (
    AggregateMetrics,
    CaseMetrics,
    EvaluationConfig,
    EvaluationOutcome,
    EvaluationReport,
    PytestMetrics,
)
from coding_agent.evaluation_scenarios import EvaluationScenario
from coding_agent.local_config import LOCAL_CONFIG_KEYS, load_local_environment
from coding_agent.openai_model import ReasoningEffort

runner = CliRunner(
    charset=getattr(cli_module.console.file, "encoding", None) or "utf-8",
)
_PROVIDER_SECRET = "TEST_PROVIDER_SECRET_SENTINEL"
_RAW_ORACLE_OUTPUT = "TEST_RAW_ORACLE_OUTPUT_SENTINEL"


@pytest.fixture(autouse=True)
def _isolate_process_local_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "load_local_environment", lambda: False)


def test_fixed_local_config_is_available_before_typer_resolves_env_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_secret = "TEST_LOCAL_CONFIG_SECRET_SENTINEL"
    tmp_path.joinpath(".env.local").write_text(
        "\n".join(
            (
                f"OPENAI_API_KEY={local_secret}",
                "OPENAI_BASE_URL=https://config.example/v1",
                "CODING_AGENT_MODEL=config-model",
                "CODING_AGENT_REASONING_EFFORT=none",
            )
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_evaluate_suite(
        *,
        config: EvaluationConfig,
        scenarios: tuple[EvaluationScenario, ...],
    ) -> EvaluationReport:
        captured["config"] = config
        captured["scenarios"] = scenarios
        return _report()

    monkeypatch.setattr(cli_module, "evaluate_suite", fake_evaluate_suite)
    monkeypatch.setattr(
        cli_module,
        "load_local_environment",
        lambda: load_local_environment(directory=tmp_path),
    )

    result = runner.invoke(
        app,
        ["evaluate", "--live", "--allow-paid-api"],
        env={key: None for key in LOCAL_CONFIG_KEYS},
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert isinstance(config, EvaluationConfig)
    assert config.model_name == "config-model"
    assert config.base_url == "https://config.example/v1"
    assert config.reasoning_effort is ReasoningEffort.NONE
    assert local_secret not in result.output


def _pytest_metrics(*, return_code: int, output: str) -> PytestMetrics:
    return PytestMetrics(
        command=("TEST_PRIVATE_COMMAND_SENTINEL",),
        return_code=return_code,
        duration_seconds=0.25,
        timed_out=False,
        output=output,
        command_status="exited",
        completed_cleanly=True,
        collected_tests=1,
        passed_tests=1 if return_code == 0 else 0,
    )


def _report(*, success: bool = True, error: bool = False) -> EvaluationReport:
    outcome = (
        EvaluationOutcome.ERROR
        if error
        else (EvaluationOutcome.PASSED if success else EvaluationOutcome.FAILED)
    )
    reasons = (
        ("runner raised OpenAIModelError",)
        if error
        else (() if success else ("independent host pytest oracle did not pass",))
    )
    case = CaseMetrics(
        case_id="cross_file_change",
        title="Cross-file shipping correction",
        run_id="TEST_PRIVATE_RUN_ID_SENTINEL",
        outcome=outcome,
        agent_state="failed" if error else "completed",
        stop_reason="model_error" if error else "final_response",
        steps=4,
        duration_seconds=1.5,
        tool_errors=1,
        baseline=_pytest_metrics(return_code=1, output=_RAW_ORACLE_OUTPUT),
        oracle=_pytest_metrics(return_code=0 if success else 1, output=_RAW_ORACLE_OUTPUT),
        protected_tests_unchanged=True,
        required_files_present=True,
        required_changes_present=success,
        allowed_changed_paths=("shipping/rates.py", "shipping/quote.py"),
        observed_changed_paths=("shipping/quote.py", "shipping/rates.py"),
        unexpected_changed_paths=(),
        failure_reasons=reasons,
    )
    return EvaluationReport(
        cases=(case,),
        aggregate=AggregateMetrics(
            total_cases=1,
            successful_cases=int(outcome is EvaluationOutcome.PASSED),
            failed_cases=int(outcome is EvaluationOutcome.FAILED),
            error_cases=int(outcome is EvaluationOutcome.ERROR),
            success_rate=float(outcome is EvaluationOutcome.PASSED),
            total_steps=4,
            average_steps=4.0,
            total_duration_seconds=1.5,
            average_duration_seconds=1.5,
            total_tool_errors=1,
        ),
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--allow-paid-api", "--model", "test-model"],
        ["--live", "--model", "test-model"],
    ],
)
def test_evaluate_requires_two_explicit_live_consents_before_running(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "evaluate_suite",
        lambda **_: pytest.fail("evaluation must not run before both consent gates"),
    )

    result = runner.invoke(
        app,
        ["evaluate", *arguments],
        env={"OPENAI_API_KEY": _PROVIDER_SECRET},
    )

    assert result.exit_code == 2
    assert "Error" in result.output
    assert _PROVIDER_SECRET not in result.output


def test_evaluate_rejects_an_unsafe_base_url_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "evaluate_suite",
        lambda **_: pytest.fail("invalid endpoint configuration must not become a case result"),
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--live",
            "--allow-paid-api",
            "--model",
            "test-model",
            "--base-url",
            "http://example.com",
        ],
        env={"OPENAI_API_KEY": _PROVIDER_SECRET},
    )

    assert result.exit_code == 2
    assert "must use HTTPS" in result.output


def test_evaluate_selects_one_safe_case_and_prints_a_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate_suite(
        *,
        config: EvaluationConfig,
        scenarios: tuple[EvaluationScenario, ...],
    ) -> EvaluationReport:
        captured["config"] = config
        captured["scenarios"] = scenarios
        return _report()

    monkeypatch.setattr(cli_module, "evaluate_suite", fake_evaluate_suite)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--live",
            "--allow-paid-api",
            "--model",
            "test-model",
            "--case",
            "cross-file",
        ],
        env={"OPENAI_API_KEY": _PROVIDER_SECRET},
    )

    assert result.exit_code == 0
    config = captured["config"]
    assert isinstance(config, EvaluationConfig)
    assert config.permission_mode is CommandPermissionMode.SAFE
    assert config.max_model_retries == 1
    scenarios = captured["scenarios"]
    assert isinstance(scenarios, tuple)
    assert [scenario.case_id for scenario in scenarios] == ["cross_file_change"]
    assert "LIVE EVALUATION" in result.output
    assert "Passed 1/1 (100%)" in result.output
    assert "request-attempt ceiling: 24" in result.stderr
    assert _PROVIDER_SECRET not in result.output


def test_evaluate_json_uses_a_strict_public_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "evaluate_suite", lambda **_: _report())

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--live",
            "--allow-paid-api",
            "--model",
            "test-model",
            "--format",
            "json",
        ],
        env={"OPENAI_API_KEY": _PROVIDER_SECRET},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "coding-agent.eval.v2"
    assert payload["mode"] == "live"
    assert payload["model"] == "test-model"
    assert payload["cases"][0]["result"] == "passed"
    assert payload["cases"][0]["integrity"] == {
        "protected_tests_unchanged": True,
        "required_files_present": True,
        "required_changes_present": True,
        "only_allowed_paths_changed": True,
        "allowed_changed_paths": ["shipping/rates.py", "shipping/quote.py"],
        "observed_changed_paths": ["shipping/quote.py", "shipping/rates.py"],
        "unexpected_changed_paths": [],
    }
    assert payload["summary"]["passed"] == 1
    assert _PROVIDER_SECRET not in result.output
    assert _RAW_ORACLE_OUTPUT not in result.output
    assert "TEST_PRIVATE_COMMAND_SENTINEL" not in result.output
    assert "TEST_PRIVATE_RUN_ID_SENTINEL" not in result.output


def test_evaluate_returns_three_for_a_completed_report_with_failed_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "evaluate_suite", lambda **_: _report(success=False))

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--live",
            "--allow-paid-api",
            "--model",
            "test-model",
            "--format",
            "json",
        ],
        env={"OPENAI_API_KEY": _PROVIDER_SECRET},
    )

    assert result.exit_code == 3
    assert json.loads(result.stdout)["summary"]["failed"] == 1


def test_evaluate_returns_one_for_a_runner_or_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "evaluate_suite",
        lambda **_: _report(success=False, error=True),
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--live",
            "--allow-paid-api",
            "--model",
            "test-model",
            "--format",
            "json",
        ],
        env={"OPENAI_API_KEY": _PROVIDER_SECRET},
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["cases"][0]["result"] == "error"
    assert payload["cases"][0]["stop_reason"] == "model_error"
    assert payload["summary"]["errors"] == 1


def test_json_mode_keeps_internal_failures_off_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_evaluation(**_: object) -> EvaluationReport:
        raise RuntimeError("TEST_PRIVATE_FAILURE_SENTINEL")

    monkeypatch.setattr(cli_module, "evaluate_suite", fail_evaluation)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--live",
            "--allow-paid-api",
            "--model",
            "test-model",
            "--format",
            "json",
        ],
        env={"OPENAI_API_KEY": _PROVIDER_SECRET},
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "No JSON report was produced" in result.stderr
    assert "TEST_PRIVATE_FAILURE_SENTINEL" not in result.output


def test_existing_report_is_rejected_before_a_paid_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation.json"
    output.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "evaluate_suite",
        lambda **_: pytest.fail("an invalid output path must be rejected before the paid run"),
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--live",
            "--allow-paid-api",
            "--model",
            "test-model",
            "--format",
            "json",
            "--output",
            str(output),
        ],
        env={"OPENAI_API_KEY": _PROVIDER_SECRET},
    )

    assert result.exit_code == 2
    assert "already exists" in result.output
    assert output.read_text(encoding="utf-8") == "keep me"


def test_json_report_is_written_once_without_private_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(cli_module, "evaluate_suite", lambda **_: _report())

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--live",
            "--allow-paid-api",
            "--model",
            "test-model",
            "--format",
            "json",
            "--output",
            str(output),
        ],
        env={"OPENAI_API_KEY": _PROVIDER_SECRET},
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["passed"] == 1
    assert _RAW_ORACLE_OUTPUT not in output.read_text(encoding="utf-8")
