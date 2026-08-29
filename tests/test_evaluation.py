from __future__ import annotations

import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from shutil import copytree, rmtree

import pytest

from coding_agent.application import ModelFactory, RepositoryRunSpec
from coding_agent.evaluation import (
    EvaluationConfig,
    EvaluationOutcome,
    evaluate_case,
    evaluate_suite,
)
from coding_agent.evaluation_scenarios import (
    EvaluationScenario,
    FixtureFile,
    built_in_scenarios,
)
from coding_agent.events import EventKind, EventSink, RunEvent
from coding_agent.model import ModelAdapter, ScriptedModel
from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    ModelResponse,
    StopReason,
    ToolCall,
)
from coding_agent.openai_model import ReasoningEffort


class RepairingRunner:
    def __init__(
        self,
        *,
        tamper_with_tests: bool = False,
        agent_state: AgentState = AgentState.COMPLETED,
    ) -> None:
        self.roots: list[Path] = []
        self.model_factories: list[ModelFactory | None] = []
        self._tamper_with_tests = tamper_with_tests
        self._agent_state = agent_state

    def __call__(
        self,
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink | None,
        model_factory: ModelFactory | None,
    ) -> AgentResult:
        self.roots.append(spec.root)
        self.model_factories.append(model_factory)
        steps = self._repair(spec.root, spec.run_id, event_sink)
        if self._tamper_with_tests:
            protected = next(spec.root.glob("tests/test_*.py"))
            protected.write_text(
                protected.read_text(encoding="utf-8") + "\n# modified by runner\n",
                encoding="utf-8",
            )
        return AgentResult(
            run_id=spec.run_id,
            state=self._agent_state,
            stop_reason=StopReason.FINAL_RESPONSE,
            steps=steps,
            final_text="Fixture repaired and independently verifiable.",
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="Evaluation system prompt"),
                ChatMessage(role=MessageRole.USER, content=spec.task),
                ChatMessage(role=MessageRole.ASSISTANT, content="Fixture repaired."),
            ),
        )

    @staticmethod
    def _repair(root: Path, run_id: str, event_sink: EventSink | None) -> int:
        if root.joinpath("calculator.py").is_file():
            root.joinpath("calculator.py").write_text(
                "def add(left: int, right: int) -> int:\n    return left + right\n",
                encoding="utf-8",
            )
            return 2
        if root.joinpath("shipping").is_dir():
            root.joinpath("shipping/rates.py").write_text(
                "FREE_SHIPPING_THRESHOLD = 50\n\n\n"
                "def qualifies_for_free_shipping(subtotal: int) -> bool:\n"
                "    return subtotal >= FREE_SHIPPING_THRESHOLD\n",
                encoding="utf-8",
            )
            root.joinpath("shipping/quote.py").write_text(
                "from .rates import qualifies_for_free_shipping\n\n\n"
                "def shipping_total(subtotal: int, fee: int = 8) -> int:\n"
                "    if qualifies_for_free_shipping(subtotal):\n"
                "        return subtotal\n"
                "    return subtotal + fee\n",
                encoding="utf-8",
            )
            if event_sink is not None:
                event_sink.emit(
                    RunEvent(
                        run_id=run_id,
                        kind=EventKind.TOOL_FINISHED,
                        message="An exploratory tool call failed",
                        step=1,
                        data={"ok": False, "tool_name": "search_text"},
                    )
                )
            return 4
        if root.joinpath("text_tools").is_dir():
            root.joinpath("text_tools/slug.py").write_text(
                "import re\n\n\n"
                "def slugify(value: str) -> str:\n"
                "    lowered = value.strip().lower()\n"
                "    return re.sub(r'[^a-z0-9]+', '-', lowered).strip('-')\n",
                encoding="utf-8",
            )
            return 3
        if root.joinpath("reports").is_dir():
            root.joinpath("reports/config.py").write_text(
                "DECIMAL_PLACES = 2\n",
                encoding="utf-8",
            )
            return 5
        raise AssertionError("unknown evaluation fixture")


def _offline_model_factory(
    *,
    model: str,
    base_url: str | None,
    reasoning_effort: ReasoningEffort | None,
    timeout_seconds: float,
    max_retries: int,
) -> ModelAdapter:
    del model, base_url, reasoning_effort, timeout_seconds, max_retries
    return ScriptedModel([])


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        model_name="offline-evaluation-model",
        max_steps=12,
        model_timeout=5,
        oracle_timeout=20,
    )


def test_built_in_suite_runs_four_isolated_red_to_green_cases(tmp_path: Path) -> None:
    runner = RepairingRunner()

    report = evaluate_suite(
        config=_config(),
        run_callable=runner,
        model_factory=_offline_model_factory,
        temporary_parent=tmp_path,
    )

    assert [case.case_id for case in report.cases] == [
        "single_file_fix",
        "cross_file_change",
        "new_feature",
        "indirect_debugging",
    ]
    assert all(case.baseline.return_code == 1 for case in report.cases)
    assert all(case.oracle is not None and case.oracle.passed for case in report.cases)
    assert all(case.success for case in report.cases)
    assert all(case.outcome is EvaluationOutcome.PASSED for case in report.cases)
    assert all(case.protected_tests_unchanged for case in report.cases)
    assert all(case.required_files_present for case in report.cases)
    assert all(case.required_changes_present for case in report.cases)
    assert report.aggregate.total_cases == 4
    assert report.aggregate.successful_cases == 4
    assert report.aggregate.failed_cases == 0
    assert report.aggregate.error_cases == 0
    assert report.aggregate.success_rate == 1.0
    assert report.aggregate.total_steps == 14
    assert report.aggregate.average_steps == 3.5
    assert report.aggregate.total_duration_seconds >= 0
    assert report.aggregate.average_duration_seconds >= 0
    assert report.aggregate.total_tool_errors == 1
    assert len(set(runner.roots)) == 4
    assert all(not root.exists() for root in runner.roots)
    assert runner.model_factories == [_offline_model_factory] * 4


def test_host_oracle_rejects_a_runner_that_modifies_protected_tests(tmp_path: Path) -> None:
    metrics = evaluate_case(
        built_in_scenarios()[0],
        config=_config(),
        run_callable=RepairingRunner(tamper_with_tests=True),
        temporary_parent=tmp_path,
    )

    assert metrics.oracle is not None and metrics.oracle.passed
    assert metrics.success is False
    assert metrics.outcome is EvaluationOutcome.FAILED
    assert metrics.protected_tests_unchanged is False
    assert any("protected test files changed" in reason for reason in metrics.failure_reasons)


def test_host_oracle_does_not_upgrade_an_unverified_agent_result(tmp_path: Path) -> None:
    metrics = evaluate_case(
        built_in_scenarios()[0],
        config=_config(),
        run_callable=RepairingRunner(agent_state=AgentState.COMPLETED_UNVERIFIED),
        temporary_parent=tmp_path,
    )

    assert metrics.oracle is not None and metrics.oracle.passed
    assert metrics.success is False
    assert metrics.outcome is EvaluationOutcome.FAILED
    assert metrics.agent_state == AgentState.COMPLETED_UNVERIFIED.value
    assert metrics.failure_reasons == (
        "AgentResult did not report verified completion: completed_unverified",
    )


def test_rewriting_public_tests_cannot_satisfy_the_holdout_oracle(tmp_path: Path) -> None:
    repairer = RepairingRunner()

    def cheating_runner(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink | None,
        model_factory: ModelFactory | None,
    ) -> AgentResult:
        result = repairer(
            spec,
            event_sink=event_sink,
            model_factory=model_factory,
        )
        spec.root.joinpath("calculator.py").write_text(
            "def add(left: int, right: int) -> int:\n    return left - right\n",
            encoding="utf-8",
        )
        spec.root.joinpath("tests/test_calculator.py").write_text(
            "def test_softened() -> None:\n    assert True\n",
            encoding="utf-8",
        )
        return result

    metrics = evaluate_case(
        built_in_scenarios()[0],
        config=_config(),
        run_callable=cheating_runner,
        temporary_parent=tmp_path,
    )

    assert metrics.success is False
    assert metrics.outcome is EvaluationOutcome.FAILED
    assert metrics.oracle is not None and metrics.oracle.passed is False
    assert metrics.protected_tests_unchanged is False
    assert "independent host pytest oracle did not pass" in metrics.failure_reasons


def test_zero_exit_without_a_completed_holdout_suite_is_not_a_pass(tmp_path: Path) -> None:
    repairer = RepairingRunner()

    def exiting_runner(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink | None,
        model_factory: ModelFactory | None,
    ) -> AgentResult:
        result = repairer(
            spec,
            event_sink=event_sink,
            model_factory=model_factory,
        )
        spec.root.joinpath("calculator.py").write_text(
            "import os\nos._exit(0)\n",
            encoding="utf-8",
        )
        return result

    metrics = evaluate_case(
        built_in_scenarios()[0],
        config=_config(),
        run_callable=exiting_runner,
        temporary_parent=tmp_path,
    )

    assert metrics.oracle is not None
    assert metrics.oracle.return_code == 0
    assert metrics.oracle.completed_cleanly is False
    assert metrics.oracle.passed is False
    assert metrics.success is False
    assert metrics.outcome is EvaluationOutcome.FAILED


def test_holdout_rejects_a_source_file_beneath_a_link_like_ancestor(tmp_path: Path) -> None:
    probe_target = tmp_path / "probe-target"
    probe_link = tmp_path / "probe-link"
    probe_target.mkdir()
    try:
        probe_link.symlink_to(probe_target, target_is_directory=True)
    except OSError:
        pytest.skip("directory links are unavailable on this host")
    probe_link.unlink()

    repairer = RepairingRunner()

    def linked_runner(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink | None,
        model_factory: ModelFactory | None,
    ) -> AgentResult:
        result = repairer(
            spec,
            event_sink=event_sink,
            model_factory=model_factory,
        )
        source = spec.root / "shipping"
        outside = spec.root.parent / "outside-shipping"
        copytree(source, outside)
        rmtree(source)
        source.symlink_to(outside, target_is_directory=True)
        return result

    metrics = evaluate_case(
        built_in_scenarios()[1],
        config=_config(),
        run_callable=linked_runner,
        temporary_parent=tmp_path,
    )

    assert metrics.outcome is EvaluationOutcome.FAILED
    assert metrics.oracle is None
    assert metrics.required_files_present is False
    assert any("missing or unsafe" in reason for reason in metrics.failure_reasons)


def test_passing_baseline_is_rejected_before_runner_is_called(tmp_path: Path) -> None:
    scenario = EvaluationScenario(
        case_id="passing_baseline",
        title="Invalid already-green fixture",
        task="This runner must never be called.",
        files=(
            FixtureFile("value.py", "VALUE = 1\n"),
            FixtureFile(
                "tests/test_value.py",
                "from value import VALUE\n\n\ndef test_value() -> None:\n    assert VALUE == 1\n",
            ),
        ),
        protected_paths=("tests/test_value.py",),
        oracle_files=(
            FixtureFile(
                "tests/test_value_oracle.py",
                "from value import VALUE\n\n\ndef test_value() -> None:\n    assert VALUE == 1\n",
            ),
        ),
        oracle_source_paths=("value.py",),
    )

    def forbidden_runner(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink | None,
        model_factory: ModelFactory | None,
    ) -> AgentResult:
        del spec, event_sink, model_factory
        raise AssertionError("green baseline must stop before the Agent")

    metrics = evaluate_case(
        scenario,
        config=_config(),
        run_callable=forbidden_runner,
        temporary_parent=tmp_path,
    )

    assert metrics.baseline.passed
    assert metrics.run_id is None
    assert metrics.oracle is None
    assert metrics.success is False
    assert metrics.outcome is EvaluationOutcome.ERROR
    assert metrics.failure_reasons == (
        "baseline pytest must fail with return code 1 before the Agent runs",
    )


def test_suite_stops_after_an_execution_error(tmp_path: Path) -> None:
    calls: list[str] = []

    def broken_runner(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink | None,
        model_factory: ModelFactory | None,
    ) -> AgentResult:
        del event_sink, model_factory
        calls.append(spec.task)
        raise RuntimeError("TEST_PRIVATE_RUNNER_FAILURE_SENTINEL")

    report = evaluate_suite(
        config=_config(),
        run_callable=broken_runner,
        temporary_parent=tmp_path,
    )

    assert len(calls) == 1
    assert len(report.cases) == 1
    assert report.cases[0].outcome is EvaluationOutcome.ERROR
    assert report.cases[0].failure_reasons[0] == "runner raised RuntimeError"
    assert "TEST_PRIVATE_RUNNER_FAILURE_SENTINEL" not in str(report)
    assert report.aggregate.error_cases == 1


@pytest.mark.parametrize(
    ("stop_reason", "expected_outcome"),
    [
        (StopReason.MAX_STEPS, EvaluationOutcome.FAILED),
        (StopReason.MODEL_ERROR, EvaluationOutcome.ERROR),
    ],
)
def test_failed_agent_result_is_classified_by_stop_reason(
    tmp_path: Path,
    stop_reason: StopReason,
    expected_outcome: EvaluationOutcome,
) -> None:
    repairer = RepairingRunner()

    def terminal_runner(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink | None,
        model_factory: ModelFactory | None,
    ) -> AgentResult:
        repaired = repairer(
            spec,
            event_sink=event_sink,
            model_factory=model_factory,
        )
        return AgentResult(
            run_id=spec.run_id,
            state=AgentState.FAILED,
            stop_reason=stop_reason,
            steps=repaired.steps,
            error="bounded terminal result",
            messages=repaired.messages[:-1],
        )

    metrics = evaluate_case(
        built_in_scenarios()[0],
        config=_config(),
        run_callable=terminal_runner,
        temporary_parent=tmp_path,
    )

    assert metrics.oracle is not None and metrics.oracle.passed
    assert metrics.outcome is expected_outcome


def test_model_factory_drives_the_production_application_path(tmp_path: Path) -> None:
    scenario = built_in_scenarios()[0]
    calculator = next(file for file in scenario.files if file.path == "calculator.py")
    captured: dict[str, object] = {}

    def model_factory(
        *,
        model: str,
        base_url: str | None,
        reasoning_effort: ReasoningEffort | None,
        timeout_seconds: float,
        max_retries: int,
    ) -> ModelAdapter:
        captured.update(
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        return ScriptedModel(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="eval-read",
                            name="read_file",
                            arguments={"path": "calculator.py", "line_count": 20},
                        ),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="eval-replace",
                            name="replace_text",
                            arguments={
                                "path": "calculator.py",
                                "old_text": "return left - right",
                                "new_text": "return left + right",
                                "expected_sha256": sha256(
                                    calculator.content.encode("utf-8")
                                ).hexdigest(),
                            },
                        ),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="eval-pytest",
                            name="run_command",
                            arguments={
                                "argv": [sys.executable, "-I", "-m", "pytest", "-q"],
                                "cwd": ".",
                                "timeout_seconds": 20,
                            },
                        ),
                    )
                ),
                ModelResponse(content="The calculator defect is fixed and pytest passed."),
            ]
        )

    metrics = evaluate_case(
        scenario,
        config=_config(),
        model_factory=model_factory,
        temporary_parent=tmp_path,
    )

    assert metrics.success is True
    assert metrics.agent_state == AgentState.COMPLETED.value
    assert metrics.steps == 4
    assert metrics.tool_errors == 0
    assert captured == {
        "model": "offline-evaluation-model",
        "base_url": None,
        "reasoning_effort": None,
        "timeout_seconds": 5,
        "max_retries": 0,
    }


def test_scenario_rejects_unsafe_or_overlapping_integrity_paths() -> None:
    for unsafe_path in ("../outside.py", "CON.py", "folder/name. "):
        with pytest.raises(ValueError, match="unsafe|normalized|relative"):
            FixtureFile(unsafe_path, "")
    with pytest.raises(ValueError, match="protected paths cannot"):
        EvaluationScenario(
            case_id="overlap_case",
            title="Overlap",
            task="Invalid overlap",
            files=(
                FixtureFile("tests/test_case.py", "def test_case() -> None:\n    assert False\n"),
            ),
            protected_paths=("tests/test_case.py",),
            oracle_files=(
                FixtureFile(
                    "tests/test_case_oracle.py",
                    "def test_case_oracle() -> None:\n    assert True\n",
                ),
            ),
            oracle_source_paths=("tests/test_case.py",),
            required_changed_paths=("tests/test_case.py",),
        )


def test_scenario_schema_rejects_ambiguous_or_unchecked_fixtures() -> None:
    scenario = built_in_scenarios()[0]

    invalid_variants = (
        ("case_id", {"case_id": "Not-Normalized"}),
        ("title", {"title": " "}),
        ("task", {"task": " "}),
        ("at least one fixture", {"files": ()}),
        ("host oracle", {"oracle_files": ()}),
        ("fixture paths must be unique", {"files": scenario.files + scenario.files[:1]}),
        (
            "fixture paths must be unique",
            {"files": scenario.files + (FixtureFile("Calculator.py", "collision"),)},
        ),
        (
            "file-directory prefix conflicts",
            {"files": scenario.files + (FixtureFile("calculator.py/nested.py", "bad"),)},
        ),
        (
            "oracle fixture paths must be unique",
            {"oracle_files": scenario.oracle_files + scenario.oracle_files[:1]},
        ),
        ("protect at least one", {"protected_paths": ()}),
        ("protected paths must exist", {"protected_paths": ("tests/missing.py",)}),
        ("required changed paths must exist", {"required_changed_paths": ("missing.py",)}),
        (
            "required new paths must be absent",
            {"required_new_paths": ("calculator.py",)},
        ),
        (
            "oracle source paths must be initial",
            {"oracle_source_paths": ("missing.py",)},
        ),
        (
            "all required source changes",
            {"oracle_source_paths": ("tests/__init__.py",)},
        ),
        (
            "cannot overlap copied source",
            {"oracle_files": (FixtureFile("calculator.py", "hidden"),)},
        ),
        (
            "protected paths must be unique",
            {"protected_paths": scenario.protected_paths * 2},
        ),
        (
            "oracle source paths must be unique",
            {"oracle_source_paths": scenario.oracle_source_paths * 2},
        ),
        (
            "required changed paths must be unique",
            {"required_changed_paths": scenario.required_changed_paths * 2},
        ),
        (
            "required new paths must be unique",
            {
                "required_new_paths": ("new.py", "new.py"),
                "oracle_source_paths": (*scenario.oracle_source_paths, "new.py"),
            },
        ),
    )
    for expected, changes in invalid_variants:
        with pytest.raises(ValueError, match=expected):
            replace(scenario, **changes)


def test_evaluation_config_and_suite_reject_invalid_host_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_name"):
        replace(_config(), model_name=" ")
    with pytest.raises(ValueError, match="max_steps"):
        replace(_config(), max_steps=0)
    with pytest.raises(ValueError, match="timeouts"):
        replace(_config(), oracle_timeout=0)
    with pytest.raises(ValueError, match="max_model_retries"):
        replace(_config(), max_model_retries=3)

    not_a_directory = tmp_path / "file.txt"
    not_a_directory.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        evaluate_suite(
            config=_config(),
            scenarios=(built_in_scenarios()[0],),
            temporary_parent=not_a_directory,
        )

    duplicate = built_in_scenarios()[0]
    with pytest.raises(ValueError, match="unique"):
        evaluate_suite(
            config=_config(),
            scenarios=(duplicate, duplicate),
            temporary_parent=tmp_path,
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"max_steps": True}, "max_steps"),
        ({"max_steps": 1.5}, "max_steps"),
        ({"max_model_retries": False}, "max_model_retries"),
        ({"max_model_retries": 1.5}, "max_model_retries"),
        ({"model_timeout": True}, "timeouts"),
        ({"model_timeout": float("nan")}, "timeouts"),
        ({"model_timeout": float("inf")}, "timeouts"),
        ({"oracle_timeout": False}, "timeouts"),
        ({"oracle_timeout": float("nan")}, "timeouts"),
        ({"oracle_timeout": float("-inf")}, "timeouts"),
    ],
)
def test_evaluation_config_rejects_noncanonical_numeric_values(
    changes: dict[str, object],
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        replace(_config(), **changes)  # type: ignore[arg-type]  # Exercise runtime guards.
