"""Durable execution for Pydantic AI agents with [Hatchet](https://hatchet.run).

Attach [`HatchetDurability`][pydantic_ai.durable_exec.hatchet.HatchetDurability] to an agent and run the
agent in a Hatchet durable task: its model requests, MCP communication and tool calls run as child tasks
whose results the durable task's event log records.
"""

from __future__ import annotations

try:
    import hatchet_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError(
        'Please install the `hatchet-sdk` package to use the Hatchet integration, '
        'you can use the `hatchet` optional group — `pip install "pydantic-ai-slim[hatchet]"`'
    ) from _import_error

from ._durability import HatchetDurability
from ._event_stream import HatchetAgentRunEvents, HatchetEventStream
from ._operation_backend import HatchetTaskConfig
from ._run_context import HatchetRunContext

__all__ = ['HatchetAgentRunEvents', 'HatchetDurability', 'HatchetEventStream', 'HatchetRunContext', 'HatchetTaskConfig']
