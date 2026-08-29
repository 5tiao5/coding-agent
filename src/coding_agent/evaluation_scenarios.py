"""Small, deterministic repositories used by the evaluation harness.

The fixtures deliberately contain only source and tests.  Evaluation state, model
configuration, and host-side oracle output live outside the repository presented to
the Agent. The definitions themselves ship with the project and are deliberately not
presented as a secret benchmark boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from textwrap import dedent

_CASE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_WINDOWS_RESERVED_STEMS = frozenset(
    {"aux", "con", "nul", "prn"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


@dataclass(frozen=True, slots=True)
class FixtureFile:
    """One UTF-8 file materialized relative to an isolated repository root."""

    path: str
    content: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)


@dataclass(frozen=True, slots=True)
class EvaluationScenario:
    """Immutable fixture plus host-owned integrity requirements for one case."""

    case_id: str
    title: str
    task: str
    files: tuple[FixtureFile, ...]
    protected_paths: tuple[str, ...]
    oracle_files: tuple[FixtureFile, ...]
    oracle_source_paths: tuple[str, ...]
    required_changed_paths: tuple[str, ...] = ()
    required_new_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if _CASE_ID_PATTERN.fullmatch(self.case_id) is None:
            raise ValueError("case_id must use lowercase letters, digits, and underscores")
        if not self.title.strip() or not self.task.strip():
            raise ValueError("scenario title and task cannot be blank")
        if not self.files:
            raise ValueError("scenario must contain at least one fixture file")
        if not self.oracle_files or not self.oracle_source_paths:
            raise ValueError("scenario must contain a host oracle and its source allowlist")

        fixture_paths = tuple(file.path for file in self.files)
        oracle_paths = tuple(file.path for file in self.oracle_files)
        _require_portable_unique(fixture_paths, label="fixture paths")
        _require_portable_unique(oracle_paths, label="oracle fixture paths")
        _require_no_prefix_conflicts(fixture_paths, label="fixture paths")
        _require_no_prefix_conflicts(oracle_paths, label="oracle fixture paths")
        for path in (
            *self.protected_paths,
            *self.oracle_source_paths,
            *self.required_changed_paths,
            *self.required_new_paths,
        ):
            _validate_relative_path(path)

        fixture_set = set(fixture_paths)
        oracle_set = set(oracle_paths)
        protected = set(self.protected_paths)
        oracle_sources = set(self.oracle_source_paths)
        changed = set(self.required_changed_paths)
        new = set(self.required_new_paths)
        if not protected:
            raise ValueError("scenario must protect at least one oracle test file")
        if not protected <= fixture_set:
            raise ValueError("protected paths must exist in the initial fixture")
        if not changed <= fixture_set:
            raise ValueError("required changed paths must exist in the initial fixture")
        if _portable_keys(changed) & _portable_keys(protected):
            raise ValueError("protected paths cannot also be required source changes")
        if _portable_keys(new) & _portable_keys(fixture_set):
            raise ValueError("required new paths must be absent from the initial fixture")
        if not oracle_sources <= fixture_set | new:
            raise ValueError("oracle source paths must be initial or required-new source files")
        if not (changed | new) <= oracle_sources:
            raise ValueError("all required source changes must be checked by the host oracle")
        if _portable_keys(oracle_set) & _portable_keys(oracle_sources):
            raise ValueError("host oracle files cannot overlap copied source files")
        _require_portable_unique(self.protected_paths, label="protected paths")
        _require_portable_unique(self.oracle_source_paths, label="oracle source paths")
        _require_portable_unique(self.required_changed_paths, label="required changed paths")
        _require_portable_unique(self.required_new_paths, label="required new paths")
        _require_no_prefix_conflicts(
            (*fixture_paths, *self.required_new_paths),
            label="fixture and required-new paths",
        )
        _require_no_prefix_conflicts(
            (*oracle_paths, *self.oracle_source_paths),
            label="oracle materialization paths",
        )


def built_in_scenarios() -> tuple[EvaluationScenario, ...]:
    """Return the four project-owned evaluation cases in stable display order."""

    return _BUILT_IN_SCENARIOS


def _source(value: str) -> str:
    return dedent(value).strip() + "\n"


def _validate_relative_path(value: str) -> None:
    if not value or "\\" in value:
        raise ValueError("fixture paths must be non-empty POSIX-style relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError(f"fixture path is not normalized: {value!r}")
    for part in path.parts:
        stem = part.rstrip(" .").split(".", 1)[0].casefold()
        if (
            part in {"", ".", ".."}
            or ":" in part
            or part.endswith((" ", "."))
            or stem in _WINDOWS_RESERVED_STEMS
        ):
            raise ValueError(f"fixture path is unsafe: {value!r}")


def _portable_path_key(value: str) -> tuple[str, ...]:
    return tuple(part.casefold() for part in PurePosixPath(value).parts)


def _portable_keys(values: set[str]) -> set[tuple[str, ...]]:
    return {_portable_path_key(value) for value in values}


def _require_portable_unique(paths: tuple[str, ...], *, label: str) -> None:
    keys = tuple(_portable_path_key(path) for path in paths)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} must be unique")


def _require_no_prefix_conflicts(paths: tuple[str, ...], *, label: str) -> None:
    keys = tuple(_portable_path_key(path) for path in paths)
    for index, key in enumerate(keys):
        for other in keys[index + 1 :]:
            shorter, longer = sorted((key, other), key=len)
            if len(shorter) < len(longer) and longer[: len(shorter)] == shorter:
                raise ValueError(f"{label} cannot contain file-directory prefix conflicts")


_BUILT_IN_SCENARIOS = (
    EvaluationScenario(
        case_id="single_file_fix",
        title="Single-file arithmetic repair",
        task=(
            "Fix calculator.add so the protected pytest suite passes. Keep the public "
            "function signature unchanged and verify the final repository."
        ),
        files=(
            FixtureFile(
                "calculator.py",
                _source(
                    """
                    def add(left: int, right: int) -> int:
                        return left - right
                    """
                ),
            ),
            FixtureFile("tests/__init__.py", ""),
            FixtureFile(
                "tests/test_calculator.py",
                _source(
                    """
                    from calculator import add


                    def test_adds_positive_numbers() -> None:
                        assert add(7, 5) == 12


                    def test_adds_a_negative_number() -> None:
                        assert add(7, -2) == 5
                    """
                ),
            ),
        ),
        protected_paths=("tests/test_calculator.py",),
        oracle_files=(
            FixtureFile(
                "tests/test_calculator_oracle.py",
                _source(
                    """
                    from calculator import add


                    def test_add_handles_zero_and_two_negative_operands() -> None:
                        assert add(0, 0) == 0
                        assert add(-8, -5) == -13
                    """
                ),
            ),
        ),
        oracle_source_paths=("calculator.py",),
        required_changed_paths=("calculator.py",),
    ),
    EvaluationScenario(
        case_id="cross_file_change",
        title="Cross-file shipping correction",
        task=(
            "Repair the shipping package. Orders at the free-shipping threshold must qualify, "
            "and paid shipping must add the fee exactly once. The fix must address both source "
            "modules and pass the protected tests."
        ),
        files=(
            FixtureFile("shipping/__init__.py", ""),
            FixtureFile(
                "shipping/rates.py",
                _source(
                    """
                    FREE_SHIPPING_THRESHOLD = 50


                    def qualifies_for_free_shipping(subtotal: int) -> bool:
                        return subtotal > FREE_SHIPPING_THRESHOLD
                    """
                ),
            ),
            FixtureFile(
                "shipping/quote.py",
                _source(
                    """
                    from .rates import qualifies_for_free_shipping


                    def shipping_total(subtotal: int, fee: int = 8) -> int:
                        if qualifies_for_free_shipping(subtotal):
                            return subtotal
                        return subtotal + fee + fee
                    """
                ),
            ),
            FixtureFile("tests/__init__.py", ""),
            FixtureFile(
                "tests/test_shipping.py",
                _source(
                    """
                    from shipping.quote import shipping_total


                    def test_threshold_receives_free_shipping() -> None:
                        assert shipping_total(50) == 50


                    def test_paid_shipping_adds_one_fee() -> None:
                        assert shipping_total(49) == 57
                    """
                ),
            ),
        ),
        protected_paths=("tests/test_shipping.py",),
        oracle_files=(
            FixtureFile(
                "tests/test_shipping_oracle.py",
                _source(
                    """
                    from shipping.quote import shipping_total


                    def test_shipping_boundaries_and_custom_fee() -> None:
                        assert shipping_total(100) == 100
                        assert shipping_total(50, fee=99) == 50
                        assert shipping_total(10, fee=3) == 13
                    """
                ),
            ),
        ),
        oracle_source_paths=(
            "shipping/__init__.py",
            "shipping/rates.py",
            "shipping/quote.py",
        ),
        required_changed_paths=("shipping/rates.py", "shipping/quote.py"),
    ),
    EvaluationScenario(
        case_id="new_feature",
        title="Add a slugification feature",
        task=(
            "Implement text_tools.slug.slugify as a new module. It must normalize surrounding "
            "whitespace, lowercase words, collapse non-alphanumeric separators, and satisfy the "
            "protected pytest suite."
        ),
        files=(
            FixtureFile("text_tools/__init__.py", ""),
            FixtureFile("tests/__init__.py", ""),
            FixtureFile(
                "tests/test_slug.py",
                _source(
                    """
                    import importlib


                    def _slugify(value: str) -> str:
                        module = importlib.import_module("text_tools.slug")
                        return module.slugify(value)


                    def test_slugifies_words_and_punctuation() -> None:
                        assert _slugify("  Hello, Relay World!  ") == "hello-relay-world"


                    def test_collapses_repeated_separators() -> None:
                        assert _slugify("one___two---three") == "one-two-three"


                    def test_preserves_ascii_digits() -> None:
                        assert _slugify("Release 2 Ready") == "release-2-ready"
                    """
                ),
            ),
        ),
        protected_paths=("tests/test_slug.py",),
        oracle_files=(
            FixtureFile(
                "tests/test_slug_oracle.py",
                _source(
                    """
                    from text_tools.slug import slugify


                    def test_slugify_handles_empty_and_boundary_separators() -> None:
                        assert slugify("") == ""
                        assert slugify("---") == ""
                        assert slugify("  Already Slugged  ") == "already-slugged"
                        assert slugify("Version 2.0") == "version-2-0"
                    """
                ),
            ),
        ),
        oracle_source_paths=("text_tools/__init__.py", "text_tools/slug.py"),
        required_new_paths=("text_tools/slug.py",),
    ),
    EvaluationScenario(
        case_id="indirect_debugging",
        title="Trace an indirect report-formatting fault",
        task=(
            "Debug the high-level reports.service.build_summary failure. Locate the indirect "
            "configuration defect rather than hard-coding the expected sentence, then run the "
            "protected tests."
        ),
        files=(
            FixtureFile("reports/__init__.py", ""),
            FixtureFile(
                "reports/config.py",
                _source(
                    """
                    DECIMAL_PLACES = 0
                    """
                ),
            ),
            FixtureFile(
                "reports/formatting.py",
                _source(
                    """
                    from .config import DECIMAL_PLACES


                    def format_average(value: float) -> str:
                        return f"{value:.{DECIMAL_PLACES}f}"
                    """
                ),
            ),
            FixtureFile(
                "reports/service.py",
                _source(
                    """
                    from .formatting import format_average


                    def build_summary(values: list[int]) -> str:
                        average = sum(values) / len(values)
                        return f"Average: {format_average(average)}"
                    """
                ),
            ),
            FixtureFile("tests/__init__.py", ""),
            FixtureFile(
                "tests/test_reports.py",
                _source(
                    """
                    from reports.service import build_summary


                    def test_summary_uses_configured_precision() -> None:
                        assert build_summary([1, 2]) == "Average: 1.50"
                    """
                ),
            ),
        ),
        protected_paths=("tests/test_reports.py",),
        oracle_files=(
            FixtureFile(
                "tests/test_reports_oracle.py",
                _source(
                    """
                    from reports.service import build_summary


                    def test_summary_precision_generalizes() -> None:
                        assert build_summary([2, 3]) == "Average: 2.50"
                        assert build_summary([10]) == "Average: 10.00"
                    """
                ),
            ),
        ),
        oracle_source_paths=(
            "reports/__init__.py",
            "reports/config.py",
            "reports/formatting.py",
            "reports/service.py",
        ),
        required_changed_paths=("reports/config.py",),
    ),
)


__all__ = [
    "EvaluationScenario",
    "FixtureFile",
    "built_in_scenarios",
]
