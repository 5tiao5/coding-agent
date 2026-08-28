"""Private per-user state locations kept outside repositories by default."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StatePaths:
    root: Path

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def traces(self) -> Path:
        return self.root / "traces"


def default_state_paths(environment: dict[str, str] | None = None) -> StatePaths:
    """Resolve an explicit override or the native per-user state directory."""
    source = os.environ if environment is None else environment
    override = source.get("CODING_AGENT_STATE_DIR")
    if override:
        root = Path(override).expanduser()
        if not root.is_absolute():
            raise ValueError("CODING_AGENT_STATE_DIR must be an absolute path")
        return StatePaths(root.resolve(strict=False))

    if os.name == "nt":
        base = source.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    else:
        base = source.get("XDG_STATE_HOME")
        root = Path(base).expanduser() if base else Path.home() / ".local" / "state"
    return StatePaths((root / "coding-agent").resolve(strict=False))
