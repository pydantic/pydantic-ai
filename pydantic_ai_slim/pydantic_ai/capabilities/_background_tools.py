"""Auto-injected capability that manages background tool execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import SystemPromptPart
from pydantic_ai.tools import RunContext, ToolDefinition

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.result import FinalResult
    from pydantic_ai.run import AgentRunResult
    from pydantic_graph import End


@dataclass
class BackgroundToolCapability(AbstractCapability[Any]):
    """Manages background tool execution.

    When a tool has `background=True`, this capability:
    1. Spawns an asyncio task for the tool execution
    2. Returns an immediate acknowledgment to the agent
    3. When the task completes, enqueues the result as a follow-up message
    4. Prevents the agent from ending while tasks are pending
    5. Cancels remaining tasks on run cleanup

    This capability is auto-injected when any tool has `background=True`
    and is prepended after `_PendingMessageDrainCapability`.
    """

    _tasks: dict[str, asyncio.Task[None]] = field(default_factory=lambda: {})
    _completion_event: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    def get_serialization_name(cls) -> None:
        return None

    async def for_run(self, ctx: RunContext[Any]) -> BackgroundToolCapability:
        """Return a fresh instance for per-run state isolation."""
        return BackgroundToolCapability()

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: Any,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Any,
    ) -> Any:
        """Intercept background tool calls: spawn a task and return an ack."""
        if not tool_def.background:
            return await handler(args)

        task_id = call.tool_call_id
        tool_name = call.tool_name

        async def _run() -> None:
            try:
                result = await handler(args)
                ctx.enqueue_message(
                    SystemPromptPart(f"Background tool '{tool_name}' (task {task_id}) completed.\nResult: {result}"),
                    priority='follow_up',
                )
            except Exception as e:
                ctx.enqueue_message(
                    SystemPromptPart(f"Background tool '{tool_name}' (task {task_id}) failed: {e}"),
                    priority='follow_up',
                )
            finally:
                self._tasks.pop(task_id, None)
                self._completion_event.set()

        self._tasks[task_id] = asyncio.create_task(_run())
        return (
            f"Tool '{tool_name}' is running in background (task {task_id}). "
            f'You will receive the result automatically when it completes. '
            f'Continue with other work in the meantime.'
        )

    async def after_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: _agent_graph.AgentNode[Any, Any],
        result: _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]]:
        """If the agent would end but background tasks are pending, wait for at least one."""
        from pydantic_graph import End

        if not isinstance(result, End) or not self._tasks:
            return result

        # Wait for at least one task to complete
        self._completion_event.clear()
        await self._completion_event.wait()
        # Task completion enqueued follow_up messages.
        # _PendingMessageDrainCapability (runs after us in reverse order)
        # will redirect End -> ModelRequestNode.
        return result

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: Any,
    ) -> AgentRunResult[Any]:
        """Ensure all background tasks are cancelled on run cleanup."""
        try:
            return await handler()
        finally:
            for task in self._tasks.values():
                task.cancel()
            # Wait for cancelled tasks to finish
            if self._tasks:
                await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            self._tasks.clear()
