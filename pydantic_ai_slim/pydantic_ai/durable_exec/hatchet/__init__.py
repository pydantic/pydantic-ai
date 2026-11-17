"""Durable execution for Pydantic AI agents with [Hatchet](https://hatchet.run).

Attach [`HatchetDurability`][pydantic_ai.durable_exec.hatchet.HatchetDurability] to an agent and run the
agent in a Hatchet durable task: its model requests, MCP communication and tool calls run as child tasks
whose results the durable task's event log records.
"""

from ._durability import HatchetDurability
from ._event_stream import HatchetAgentRunEvents, HatchetEventStream
from ._operation_backend import HatchetTaskConfig
from ._run_context import HatchetRunContext

__all__ = ['HatchetAgentRunEvents', 'HatchetDurability', 'HatchetEventStream', 'HatchetRunContext', 'HatchetTaskConfig']
