"""Deterministic repository-repair scenario used by demos and smoke tests.

The scenario is presentation-neutral.  Callers inject any public ``EventSink``
(for example the Rich dashboard, a JSONL trace, or an in-memory recorder) while
the same repository setup and scripted decisions remain reproducible.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from rich.console import Console

from coding_agent.agent import AgentRunner
from coding_agent.command import VerificationCommandSpec
from coding_agent.dashboard import DashboardEventSink
from coding_agent.events import EventSink
from coding_agent.model import ModelAdapter, ScriptedModel
from coding_agent.models import AgentResult, AgentState, ModelResponse, ToolCall, VerificationKind
from coding_agent.presentation import print_agent_response
from coding_agent.runtime import build_runtime

DEMO_TASK = (
    "复现失败的 pricing 测试，定位并修复 calculate_total 中重复扣减 discount 的缺陷；"
    "随后重新运行 pytest，并且只报告对当前代码仍然有效的验证证据。"
)
DEMO_TASK_LABEL = "修复 calculate_total 重复扣减 discount"
_DEMO_PLAN_REPRODUCE = "运行失败的 pricing 测试以复现问题"
_DEMO_PLAN_LOCATE = "定位重复扣减 discount 的缺陷"
_DEMO_PLAN_REPAIR = "执行一次带版本校验的精确修改"
_DEMO_PLAN_VERIFY = "重新运行 pricing 测试套件"
DEMO_SOURCE_PATH = Path("src/pricing.py")
DEMO_SOURCE_BEFORE = (
    b"def calculate_total(subtotal: int, discount: int) -> int:\n"
    b"    discounted = subtotal - discount\n"
    b"    return discounted - discount\n"
)
DEMO_SOURCE_AFTER = (
    b"def calculate_total(subtotal: int, discount: int) -> int:\n"
    b"    discounted = subtotal - discount\n"
    b"    return discounted\n"
)
DEMO_VERIFICATION_ARGV = (sys.executable, "-I", "-m", "pytest", "-q")


@dataclass(frozen=True, slots=True)
class DemoRunResult:
    """Agent outcome plus an independent assertion about the repaired fixture."""

    result: AgentResult
    source_matches_expected: bool

    @property
    def verified(self) -> bool:
        return self.result.state is AgentState.COMPLETED and self.source_matches_expected


def run_demo(*, console: Console) -> AgentResult:
    """Run the packaged scenario with the Rich dashboard used by the CLI."""
    dashboard = DashboardEventSink(
        console,
        task_label=DEMO_TASK_LABEL,
        auto_final_card=False,
    )
    with TemporaryDirectory(prefix="coding-agent-demo-") as temporary_directory:
        try:
            outcome = run_repository_demo(
                Path(temporary_directory),
                event_sink=dashboard,
            )
            if outcome.result.state is AgentState.COMPLETED and not outcome.source_matches_expected:
                raise RuntimeError("verified demo did not produce the expected repository state")
            print_agent_response(outcome.result, console=console)
            dashboard.print_final_card()
            return outcome.result
        finally:
            dashboard.close()


def demo_verification_command() -> VerificationCommandSpec:
    """Return the exact host-injected verifier capability used by the scenario."""
    return VerificationCommandSpec(
        argv=DEMO_VERIFICATION_ARGV,
        cwd=".",
        kind=VerificationKind.TEST,
        label="demo pytest",
    )


def run_repository_demo(
    root: Path,
    *,
    event_sink: EventSink | None = None,
    model: ModelAdapter | None = None,
    run_id: str | None = None,
) -> DemoRunResult:
    """Create and run the deterministic repair scenario under ``root``."""
    write_demo_project(root)
    runtime = build_runtime(
        root,
        verification_commands=(demo_verification_command(),),
    )
    runner = AgentRunner(
        model or repository_demo_model(),
        runtime.tools,
        event_sink=event_sink,
        max_steps=13,
        run_memory=runtime.run_memory,
        verification_profile=runtime.verification_profile,
        completion_contract=runtime.completion_contract,
    )
    result = runner.run(DEMO_TASK, run_id=run_id)
    source_matches_expected = root.joinpath(DEMO_SOURCE_PATH).read_bytes() == DEMO_SOURCE_AFTER
    return DemoRunResult(
        result=result,
        source_matches_expected=source_matches_expected,
    )


def repository_demo_model() -> ScriptedModel:
    """Build a fresh deterministic decision sequence for one demo run."""
    return ScriptedModel(
        [
            ModelResponse(
                content="I will expose a short observable plan before acting.",
                tool_calls=(
                    ToolCall(
                        id="demo-plan-1",
                        name="update_plan",
                        arguments={
                            "items": [
                                {
                                    "id": "reproduce",
                                    "step": _DEMO_PLAN_REPRODUCE,
                                    "status": "in_progress",
                                },
                                {
                                    "id": "locate",
                                    "step": _DEMO_PLAN_LOCATE,
                                    "status": "pending",
                                },
                                {
                                    "id": "repair",
                                    "step": _DEMO_PLAN_REPAIR,
                                    "status": "pending",
                                },
                                {
                                    "id": "verify",
                                    "step": _DEMO_PLAN_VERIFY,
                                    "status": "pending",
                                },
                            ]
                        },
                    ),
                ),
            ),
            ModelResponse(
                content="I will reproduce the defect before reading implementation details.",
                tool_calls=(
                    ToolCall(
                        id="demo-test-fail",
                        name="run_command",
                        arguments={
                            "argv": list(DEMO_VERIFICATION_ARGV),
                            "cwd": ".",
                            "timeout_seconds": 30,
                        },
                    ),
                ),
            ),
            ModelResponse(
                content="The test fails as expected; I will now locate the implementation.",
                tool_calls=(
                    ToolCall(
                        id="demo-plan-2",
                        name="update_plan",
                        arguments={
                            "items": [
                                {
                                    "id": "reproduce",
                                    "step": _DEMO_PLAN_REPRODUCE,
                                    "status": "completed",
                                },
                                {
                                    "id": "locate",
                                    "step": _DEMO_PLAN_LOCATE,
                                    "status": "in_progress",
                                },
                                {
                                    "id": "repair",
                                    "step": _DEMO_PLAN_REPAIR,
                                    "status": "pending",
                                },
                                {
                                    "id": "verify",
                                    "step": _DEMO_PLAN_VERIFY,
                                    "status": "pending",
                                },
                            ]
                        },
                    ),
                ),
            ),
            ModelResponse(
                content="I will map the repository before choosing a file.",
                tool_calls=(
                    ToolCall(
                        id="demo-list-1",
                        name="list_files",
                        arguments={"path": ".", "max_depth": 3, "limit": 50},
                    ),
                ),
            ),
            ModelResponse(
                content="The repository is small; I will locate the target symbol.",
                tool_calls=(
                    ToolCall(
                        id="demo-search-1",
                        name="search_text",
                        arguments={"query": "def calculate_total", "path": "src"},
                    ),
                ),
            ),
            ModelResponse(
                content="I found the definition and will read its local context.",
                tool_calls=(
                    ToolCall(
                        id="demo-read-1",
                        name="read_file",
                        arguments={"path": "src/pricing.py", "start_line": 1, "line_count": 40},
                    ),
                ),
            ),
            ModelResponse(
                content="The second subtraction is the defect; I will apply one exact edit.",
                tool_calls=(
                    ToolCall(
                        id="demo-plan-3",
                        name="update_plan",
                        arguments={
                            "items": [
                                {
                                    "id": "reproduce",
                                    "step": _DEMO_PLAN_REPRODUCE,
                                    "status": "completed",
                                },
                                {
                                    "id": "locate",
                                    "step": _DEMO_PLAN_LOCATE,
                                    "status": "completed",
                                },
                                {
                                    "id": "repair",
                                    "step": _DEMO_PLAN_REPAIR,
                                    "status": "in_progress",
                                },
                                {
                                    "id": "verify",
                                    "step": _DEMO_PLAN_VERIFY,
                                    "status": "pending",
                                },
                            ]
                        },
                    ),
                ),
            ),
            ModelResponse(
                content="The plan now names the exact edit; I will apply it.",
                tool_calls=(
                    ToolCall(
                        id="demo-replace-1",
                        name="replace_text",
                        arguments={
                            "path": "src/pricing.py",
                            "old_text": "return discounted - discount",
                            "new_text": "return discounted",
                            "expected_sha256": sha256(DEMO_SOURCE_BEFORE).hexdigest(),
                        },
                    ),
                ),
            ),
            ModelResponse(
                content="The diff was applied; I will advance the explicit plan to verification.",
                tool_calls=(
                    ToolCall(
                        id="demo-plan-4",
                        name="update_plan",
                        arguments={
                            "items": [
                                {
                                    "id": "reproduce",
                                    "step": _DEMO_PLAN_REPRODUCE,
                                    "status": "completed",
                                },
                                {
                                    "id": "locate",
                                    "step": _DEMO_PLAN_LOCATE,
                                    "status": "completed",
                                },
                                {
                                    "id": "repair",
                                    "step": _DEMO_PLAN_REPAIR,
                                    "status": "completed",
                                },
                                {
                                    "id": "verify",
                                    "step": _DEMO_PLAN_VERIFY,
                                    "status": "in_progress",
                                },
                            ]
                        },
                    ),
                ),
            ),
            ModelResponse(
                content="Only a fresh external test result can satisfy the verification gate.",
                tool_calls=(
                    ToolCall(
                        id="demo-test-pass",
                        name="run_command",
                        arguments={
                            "argv": list(DEMO_VERIFICATION_ARGV),
                            "cwd": ".",
                            "timeout_seconds": 30,
                        },
                    ),
                ),
            ),
            ModelResponse(
                content="The current test run passed; I will close the explicit plan.",
                tool_calls=(
                    ToolCall(
                        id="demo-plan-5",
                        name="update_plan",
                        arguments={
                            "items": [
                                {
                                    "id": "reproduce",
                                    "step": _DEMO_PLAN_REPRODUCE,
                                    "status": "completed",
                                },
                                {
                                    "id": "locate",
                                    "step": _DEMO_PLAN_LOCATE,
                                    "status": "completed",
                                },
                                {
                                    "id": "repair",
                                    "step": _DEMO_PLAN_REPAIR,
                                    "status": "completed",
                                },
                                {
                                    "id": "verify",
                                    "step": _DEMO_PLAN_VERIFY,
                                    "status": "completed",
                                },
                            ]
                        },
                    ),
                ),
            ),
            ModelResponse(
                content=(
                    "已修复 src/pricing.py：calculate_total 现在只会应用一次 discount。"
                    "修改后重新运行的 pytest 已通过，且验证证据对应最后一次代码修改。"
                )
            ),
        ]
    )


def write_demo_project(root: Path) -> None:
    """Materialize the intentionally failing repository fixture under ``root``."""
    source = root / DEMO_SOURCE_PATH
    test = root / "tests" / "test_pricing.py"
    source.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    source.parent.joinpath("__init__.py").write_text("", encoding="utf-8")
    test.parent.joinpath("__init__.py").write_text("", encoding="utf-8")
    source.write_bytes(DEMO_SOURCE_BEFORE)
    root.joinpath(".gitignore").write_text(
        ".pytest_cache/\n__pycache__/\n*.py[cod]\n",
        encoding="utf-8",
    )
    test.write_text(
        "from src.pricing import calculate_total\n\n"
        "def test_discount_is_applied_once() -> None:\n"
        "    assert calculate_total(100, 15) == 85\n",
        encoding="utf-8",
    )
