"""Auto-injected capability that manages background tool execution."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import SystemPromptPart
from pydantic_ai.tools import RunContext, ToolDefinition

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.result import FinalResult
    from pydantic_ai.run import AgentRunResult
    from pydantic_graph import End


@dataclass
class _RunState:
    tasks: dict[str, asyncio.Task[None]]
    completion_event: asyncio.Event


# Per-run state is scoped via ContextVar so the capability instance itself is stateless
# and safe to share across concurrent agent runs. `wrap_run` installs a fresh `_RunState`
# on entry and tears it down on exit; `wrap_tool_execute` and `after_node_run` read
# from it without needing per-run capability instances.
_RUN_STATE: ContextVar[_RunState] = ContextVar('pydantic_ai.background_tools.run_state')


@dataclass
class BackgroundToolCapability(AbstractCapability[Any]):
    """Manages background tool execution.

    When a tool has `background=True`, this capability:
    1. Spawns an asyncio task for the tool execution
    2. Returns an immediate acknowledgment to the agent
    3. When the task completes, enqueues the result as a follow-up message
    4. Prevents the agent from ending while tasks are pending
    5. Cancels remaining tasks on run cleanup

    This capability is auto-injected for all agents and placed outermost via
    [`CapabilityOrdering`][pydantic_ai.capabilities.abstract.CapabilityOrdering]
    so it wraps around other capabilities. When no tool has `background=True`,
    it's a no-op.
    """

    def get_ordering(self) -> CapabilityOrdering:
        # Outermost so `after_node_run` waits for background tasks before
        # `PendingMessageDrainCapability` redirects End to ModelRequestNode.
        return CapabilityOrdering(position='outermost')

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: Any,
    ) -> AgentRunResult[Any]:
        """Install per-run task state; cancel any leftover tasks on cleanup."""
        state = _RunState(tasks={}, completion_event=asyncio.Event())
        token = _RUN_STATE.set(state)
        try:
            return await handler()
        finally:
            # Cancel any tasks still live (e.g. when a run aborts before
            # `after_node_run` had a chance to wait for them).
            for task in state.tasks.values():
                task.cancel()
            if state.tasks:
                await asyncio.gather(*state.tasks.values(), return_exceptions=True)
            _RUN_STATE.reset(token)

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

        state = _RUN_STATE.get()
        task_id = call.tool_call_id
        tool_name = call.tool_name

        async def _run() -> None:
            try:
                result = await handler(args)
                ctx.enqueue_message(
                    SystemPromptPart(f"Background tool '{tool_name}' (task {task_id}) completed.\nResult: {result}"),
                    priority='follow_up',
                )
            except asyncio.CancelledError:
                # Task cancelled during run cleanup — don't enqueue a follow-up;
                # the event is still set via finally so a waiter can proceed.
                raise
            except Exception as e:
                ctx.enqueue_message(
                    SystemPromptPart(f"Background tool '{tool_name}' (task {task_id}) failed: {e}"),
                    priority='follow_up',
                )
            finally:
                state.tasks.pop(task_id, None)
                state.completion_event.set()

        state.tasks[task_id] = asyncio.create_task(_run())
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

        state = _RUN_STATE.get(None)
        if state is None or not isinstance(result, End) or not state.tasks:
            return result

        # Wait for at least one task to complete
        state.completion_event.clear()
        await state.completion_event.wait()
        # Task completion enqueued follow_up messages.
        # `PendingMessageDrainCapability` (runs after us in reverse order)
        # will redirect End -> ModelRequestNode.
        return result
