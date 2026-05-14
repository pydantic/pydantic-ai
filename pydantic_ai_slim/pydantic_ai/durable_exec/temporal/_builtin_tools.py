from __future__ import annotations

from typing import Any

from typing_extensions import TypeVar

from pydantic_ai.native_tools import ShellTool
from pydantic_ai.tools import RunContext

from ._run_context import TemporalRunContext

AgentDepsT = TypeVar('AgentDepsT', default=None, covariant=True)


class ShellToolTemporalRunContext(TemporalRunContext[AgentDepsT]):
    """A `TemporalRunContext` subclass that serializes the shell container ID across Temporal activity boundaries.

    Use this when your agent uses `ShellTool` and you need the `container_id` to be available inside Temporal activities:

    ```python
    from pydantic_ai.durable_exec.temporal import (
        ShellToolTemporalRunContext,
        TemporalAgent,
    )


    def build_temporal_agent(agent):
        return TemporalAgent(agent, run_context_type=ShellToolTemporalRunContext)
    ```

    Inside a Temporal-ized tool, access the container ID via `ctx.container_id`.
    """

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context, including the shell container ID extracted from message history."""
        data = super().serialize_run_context(ctx)
        data['container_id'] = ShellTool.get_container_id(ctx.messages)
        return data
