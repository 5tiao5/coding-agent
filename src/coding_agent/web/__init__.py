"""Local Web frontend adapters for the shared coding-agent runtime."""

from coding_agent.web.app import create_app
from coding_agent.web.service import (
    RunAlreadyActiveError,
    WebRunService,
    WebRunStartError,
    WebServiceClosedError,
)

__all__ = [
    "RunAlreadyActiveError",
    "WebRunService",
    "WebRunStartError",
    "WebServiceClosedError",
    "create_app",
]
