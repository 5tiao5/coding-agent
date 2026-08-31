"""Model-facing creation of one explicitly named workspace directory."""

from __future__ import annotations

from pydantic import BaseModel, Field

from coding_agent.models import ToolControlFacts, ToolOutput
from coding_agent.tools._rendering import summarize_path
from coding_agent.tools.base import BaseTool
from coding_agent.workspace import Workspace


class CreateDirectoryArguments(BaseModel):
    path: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "Workspace-relative path for exactly one directory. Its parent must already "
            "exist; call create_directory once per missing level."
        ),
    )


class CreateDirectoryTool(BaseTool[CreateDirectoryArguments]):
    name = "create_directory"
    description = (
        "Create exactly one workspace directory without overwriting anything or creating "
        "missing parents. Existing ordinary directories are a successful no-op; links, "
        "ignored paths, sensitive paths, and paths outside the workspace are rejected."
    )
    args_model = CreateDirectoryArguments

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    def run(self, arguments: CreateDirectoryArguments) -> ToolOutput:
        receipt = self._workspace.create_directory(arguments.path)
        display_path = summarize_path(receipt.relative)
        if receipt.created:
            content = (
                f"Created directory {receipt.relative}. "
                "No parent directories were created implicitly."
            )
            summary = f"Created directory {display_path}"
        else:
            content = f"Directory {receipt.relative} already exists; no change was needed."
            summary = f"Skipped existing directory {display_path}"
        return ToolOutput(
            content=content,
            summary=summary,
            metadata={
                "path": receipt.relative,
                "created": receipt.created,
                "changed": receipt.created,
                "change_kind": "create_directory" if receipt.created else "noop",
            },
            control=ToolControlFacts(
                invalidates_verification=receipt.created,
                made_progress=receipt.created,
            ),
        )
