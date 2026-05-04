from __future__ import annotations

from typing import Any

from typing_extensions import TypeVar

from pydantic_ai.builtin_tools import ShellTool
from pydantic_ai.tools import RunContext

from ._run_context import TemporalRunContext

AgentDepsT = TypeVar('AgentDepsT', default=None, covariant=True)


class ShellToolTemporalRunContext(TemporalRunContext[AgentDepsT]):
    """A :class:`TemporalRunContext` subclass that serializes the shell container ID across Temporal activity boundaries.

    Use this when your agent uses :class:`~pydantic_ai.builtin_tools.ShellTool` and you need
    the ``container_id`` to be available inside Temporal activities::

        from pydantic_ai.durable_exec.temporal import TemporalAgent, ShellToolTemporalRunContext

        temporal_agent = TemporalAgent(agent, run_context_type=ShellToolTemporalRunContext)

    Inside a Temporal-ized tool, access the container ID via ``ctx.container_id``.
    """

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context, including the shell container ID extracted from message history."""
        data = super().serialize_run_context(ctx)
        data['container_id'] = ShellTool.get_container_id(ctx.messages)
        return data
