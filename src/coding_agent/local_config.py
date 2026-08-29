"""Load a fixed, untracked local configuration without searching parent paths."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

from dotenv import dotenv_values

from coding_agent.errors import CodedError

LOCAL_CONFIG_FILENAME = ".env.local"
LOCAL_CONFIG_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CODING_AGENT_MODEL",
    "CODING_AGENT_REASONING_EFFORT",
)


def load_local_environment(
    *,
    directory: Path | None = None,
    environment: MutableMapping[str, str] | None = None,
) -> bool:
    """Load supported keys from exactly ``directory/.env.local``.

    Existing process values win, matching ``python-dotenv``'s safe default. Passing the
    directory explicitly keeps lookup deterministic and prevents implicit parent traversal.
    """

    target = os.environ if environment is None else environment
    root = Path.cwd() if directory is None else directory
    config_path = root / LOCAL_CONFIG_FILENAME
    if not config_path.is_file():
        return False

    try:
        values = dotenv_values(
            dotenv_path=config_path,
            encoding="utf-8",
            interpolate=False,
        )
    except (OSError, UnicodeError) as exc:
        raise CodedError(
            "local_config_unreadable",
            f"Could not read {LOCAL_CONFIG_FILENAME} as a UTF-8 configuration file.",
        ) from exc

    for key in LOCAL_CONFIG_KEYS:
        value = values.get(key)
        if value is not None and key not in target:
            target[key] = value
    return True
