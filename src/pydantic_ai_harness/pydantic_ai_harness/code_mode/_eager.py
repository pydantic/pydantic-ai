"""Eager `CodeMode` toolset and its streamed-call state."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Generic

from pydantic_ai import AbstractToolset, RunContext
from pydantic_ai.durable_exec._base import BaseDurabilityCapability  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import AgentStreamEvent, PartDeltaEvent, PartStartEvent, ToolCallPart, ToolCallPartDelta
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import ToolsetTool

from ._streaming import MAX_SCAN_CHARS, closed_statements, decode_partial_args
from ._toolset import CodeModeToolset, RunCodeExecution

MAX_SCAN_WORK_CHARS = 1 << 20
"""Cumulative characters a streamed call may hand to host parsers."""

PUMP_CANCEL_TIMEOUT_SECONDS = 5.0
"""How long to wait for a cancelled pump to release a non-cooperative nested tool before
abandoning it. Abandoning is safe: the pump's feed aborts with the cancellation, and the
executor cancels the in-flight nested calls, so an abandoned pump starts no further
tool calls."""


def in_durable_execution(ctx: RunContext[object]) -> bool:
    """Whether a durable executor is active, where eager streaming must stay disabled."""
    return any(
        isinstance(capability, BaseDurabilityCapability) and capability.in_durable_context
        for capability in ctx.capabilities.values()
    )


@dataclass(kw_only=True)
class StreamedCodeCall:
    """State for one `run_code` call while its arguments are streaming."""

    tool_call_id: str
    execution: RunCodeExecution
    args_text: str = ''
    args_dict: dict[str, Any] | None = None
    halted: bool = False
    scan_work_chars: int = 0
    fed_line_count: int = 0
    fed_prefix: str = ''
    queue: deque[str] = field(default_factory=deque[str])
    pump: asyncio.Task[None] | None = None
    error: BaseException | None = None


@dataclass
class EagerCoordinator(Generic[AgentDepsT]):
    """Follow streamed `run_code` calls for one agent run and inject closed statements into the REPL.

    The stream side only speculates: it feeds statements that can no longer change and stops
    when unsure. Dispatch in `EagerCodeModeToolset.call_tool` decides whether the fed prefix
    still matches the final call, handles `restart`, and returns the single combined result.
    """

    calls: dict[str, StreamedCodeCall] = field(default_factory=dict[str, StreamedCodeCall], init=False)
    call_ids_by_part_index: dict[int, str] = field(default_factory=dict[int, str], init=False)
    pumps: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]], init=False)
    feed_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    run_step: int | None = field(default=None, init=False)
    first_tool_part_index: int | None = field(default=None, init=False)

    async def observe(
        self,
        event: AgentStreamEvent,
        ctx: RunContext[AgentDepsT],
        executor: EagerCodeModeToolset[AgentDepsT],
    ) -> None:
        """Consume one stream event and schedule newly completed statements."""
        if self.run_step != ctx.run_step:
            # Argument validation can reject a call before `call_tool` consumes its stream state.
            # Retire that orphan before it can disable eager execution on the retry step.
            stale_calls = list(self.calls.values())
            self.calls.clear()
            self.call_ids_by_part_index.clear()
            self.first_tool_part_index = None
            for call in stale_calls:
                await self.discard(call)
            self.run_step = ctx.run_step

        match event:
            case PartStartEvent(part=ToolCallPart() as part):
                if self.first_tool_part_index is None:
                    self.first_tool_part_index = event.index
                if part.tool_name != 'run_code':
                    return
                call = self.calls.get(part.tool_call_id)
                if call is None:
                    call = StreamedCodeCall(
                        tool_call_id=part.tool_call_id,
                        execution=RunCodeExecution(parent_tool_call_id=part.tool_call_id),
                        # Eager work may start only for the response's first tool call. Anything
                        # earlier must reach normal dispatch before this sequential tool can run.
                        halted=event.index != self.first_tool_part_index,
                    )
                    self.calls[part.tool_call_id] = call
                    self.call_ids_by_part_index[event.index] = part.tool_call_id
                if await self.receive_args(call, part.args, replace_dict=True):
                    await self.scan(call, ctx, executor)

            case PartDeltaEvent(delta=ToolCallPartDelta() as delta):
                call = self.find_call(event.index, delta.tool_call_id)
                if call is not None and await self.receive_args(call, delta.args_delta, replace_dict=False):
                    await self.scan(call, ctx, executor)

            case _:
                return

    async def receive_args(
        self,
        call: StreamedCodeCall,
        args: str | dict[str, Any] | None,
        *,
        replace_dict: bool,
    ) -> bool:
        """Retain arguments while eager scanning is active; report whether a scan is worthwhile.

        Once scanning halts, dict updates can still invalidate work that has already started.
        """
        if call.halted:
            if isinstance(args, dict):
                code = args.get('code')
                invalid_prefix = 'code' in args and (
                    not isinstance(code, str) or code != call.fed_prefix and not code.startswith(f'{call.fed_prefix}\n')
                )
                if (
                    args.get('restart')
                    or call.fed_prefix
                    and (invalid_prefix or replace_dict and not isinstance(code, str))
                ):
                    await self.discard(call)
            return False
        if isinstance(args, str):
            if len(args) > MAX_SCAN_CHARS - len(call.args_text):
                call.halted = True
                self.drop_queued(call)
                return False
            call.args_text += args
            # A statement can only close on a newline, so deltas without one need no decoding.
            return '\\n' in args
        if isinstance(args, dict):
            code = args.get('code')
            if args.get('restart'):
                call.halted = True
                await self.discard(call)
                return False
            if isinstance(code, str) and len(code) > MAX_SCAN_CHARS:
                # A dict replacement is the complete code, so unlike the string and cumulative
                # limits (which fire mid-stream, before the final code is known) we can check
                # whether the already-run prefix still holds. When it does, the queued-but-
                # unrun statements are still a valid continuation, so they are kept; a diverged
                # prefix discards them.
                call.halted = True
                if call.fed_prefix and code != call.fed_prefix and not code.startswith(f'{call.fed_prefix}\n'):
                    await self.discard(call)
                return False
            if replace_dict and call.fed_prefix and not isinstance(code, str):
                call.halted = True
                await self.discard(call)
                return False
            relevant: dict[str, object] = {}
            if isinstance(code, str):
                relevant['code'] = code
            call.args_dict = relevant if replace_dict else {**(call.args_dict or {}), **relevant}
            return True
        return False

    @staticmethod
    def drop_queued(call: StreamedCodeCall) -> None:
        """Return queued statements to the tail that normal dispatch will execute."""
        if not call.queue:
            return
        queued_lines = sum(source.count('\n') + 1 for source in call.queue)
        call.fed_line_count -= queued_lines
        call.fed_prefix = '\n'.join(call.fed_prefix.split('\n')[: call.fed_line_count])
        call.queue.clear()

    def find_call(self, part_index: int, tool_call_id: str | None) -> StreamedCodeCall | None:
        """Route a delta by part index, following a call ID the provider rewrites mid-stream."""
        indexed_call_id = self.call_ids_by_part_index.get(part_index)
        if indexed_call_id is None:
            return None
        call = self.calls[indexed_call_id]
        if tool_call_id is not None and tool_call_id != indexed_call_id:
            del self.calls[indexed_call_id]
            self.calls[tool_call_id] = call
            self.call_ids_by_part_index[part_index] = tool_call_id
            call.tool_call_id = tool_call_id
        return call

    async def scan(
        self,
        call: StreamedCodeCall,
        ctx: RunContext[AgentDepsT],
        executor: EagerCodeModeToolset[AgentDepsT],
    ) -> None:
        """Queue top-level statements that have become stable in the stream."""
        args: Mapping[str, object] | None = call.args_dict
        if args is None:
            if len(call.args_text) > MAX_SCAN_WORK_CHARS - call.scan_work_chars:
                call.halted = True
                self.drop_queued(call)
                return
            call.scan_work_chars += len(call.args_text)
            args = decode_partial_args(call.args_text)
            if args is None:
                return
        if args.get('restart'):
            call.halted = True
            await self.discard(call)
            return
        code = args.get('code')
        if not isinstance(code, str):
            return
        if call.args_dict is not None:
            if len(code) > MAX_SCAN_WORK_CHARS - call.scan_work_chars:
                call.halted = True
                self.drop_queued(call)
                return
            call.scan_work_chars += len(code)

        lines = code.split('\n')
        if call.fed_line_count and '\n'.join(lines[: call.fed_line_count]) != call.fed_prefix:
            call.halted = True
            await self.discard(call)
            return

        # Fed statements are closed top-level statements, so the unfed suffix parses on its own.
        # Parsing only that suffix keeps each scan proportional to what is still open.
        unfed = '\n'.join(lines[call.fed_line_count :])

        for statement in closed_statements(unfed):
            end = call.fed_line_count + (statement.end_lineno or statement.lineno)
            call.queue.append('\n'.join(lines[call.fed_line_count : end]))
            call.fed_line_count = end
        call.fed_prefix = '\n'.join(lines[: call.fed_line_count])

        if call.queue and (call.pump is None or call.pump.done()):
            pump = asyncio.create_task(self.run_pump(call, ctx, executor))
            call.pump = pump
            self.pumps.add(pump)
            pump.add_done_callback(self.pumps.discard)

    async def run_pump(
        self,
        call: StreamedCodeCall,
        ctx: RunContext[AgentDepsT],
        executor: EagerCodeModeToolset[AgentDepsT],
    ) -> None:
        """Feed a streamed call's queued statements in program order."""
        while call.queue and call.error is None:
            source = call.queue.popleft()
            feed_ctx = replace(ctx, tool_call_id=call.tool_call_id, tool_name='run_code')
            try:
                async with self.feed_lock:
                    await executor.feed_fragment(call, source, feed_ctx)
            except Exception as error:
                call.error = error
                call.queue.clear()

    @staticmethod
    async def drain(call: StreamedCodeCall) -> None:
        if call.pump is not None:
            await call.pump

    async def discard(self, call: StreamedCodeCall) -> None:
        call.queue.clear()
        if call.pump is not None:
            await self.cancel_pump(call.pump)

    @staticmethod
    async def cancel_pump(pump: asyncio.Task[None]) -> None:
        pump.cancel()
        # A nested tool that swallows the cancellation can hold the pump past the budget;
        # waiting without a deadline would let one such tool hang run teardown.
        await asyncio.wait({pump}, timeout=PUMP_CANCEL_TIMEOUT_SECONDS)

    async def close(self) -> None:
        """Cancel all background work owned by this run."""
        self.calls.clear()
        self.call_ids_by_part_index.clear()
        self.run_step = None
        self.first_tool_part_index = None
        pumps = list(self.pumps)
        self.pumps.clear()
        for pump in pumps:
            await self.cancel_pump(pump)


@dataclass
class EagerCodeModeToolset(CodeModeToolset[AgentDepsT]):
    """Run-scoped `CodeModeToolset` specialization for streamed execution."""

    execution: EagerCoordinator[AgentDepsT] = field(default_factory=EagerCoordinator, init=False, repr=False)

    @classmethod
    def from_run_context(cls, ctx: RunContext[AgentDepsT]) -> EagerCodeModeToolset[AgentDepsT] | None:
        """Return the active step's eager toolset, or `None` when this run does not use one."""
        tool_manager = ctx.tool_manager
        if tool_manager is None or tool_manager.tools is None:
            return None  # pragma: no cover - the agent installs its tool manager before streaming
        tool = tool_manager.tools.get('run_code')
        if tool is None:
            return None  # pragma: no cover - eager `CodeMode` always contributes `run_code`
        return tool.toolset if isinstance(tool.toolset, cls) else None

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        new_self = await super().for_run_step(ctx)
        if new_self is not self:
            assert isinstance(new_self, EagerCodeModeToolset), '`replace` preserves the class'
            # Step-specific toolset copies participate in the same run-owned stream coordination.
            new_self.execution = self.execution
        return new_self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            await self.execution.close()
        finally:
            result = await super().__aexit__(*args)
        return result

    async def observe_stream_event(self, event: AgentStreamEvent, ctx: RunContext[AgentDepsT]) -> None:
        await self.execution.observe(event, ctx, self)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        run_code_tool = self._as_run_code_tool(tool)
        code = tool_args.get('code')
        call = self.execution.calls.pop(ctx.tool_call_id or 'pyd_ai_code_mode', None)
        if run_code_tool is None or call is None or not isinstance(code, str):
            return await super().call_tool(name, tool_args, ctx, tool)

        if tool_args.get('restart', False):
            await self.execution.discard(call)
            async with self.execution.feed_lock:
                return await super().call_tool(name, tool_args, ctx, tool)

        lines = code.split('\n')
        if '\n'.join(lines[: call.fed_line_count]) != call.fed_prefix:
            await self.execution.discard(call)
            run_state = self._run_state
            assert run_state is not None, '`CodeModeToolset` must be entered before calling `run_code`'
            run_state.reset()
            raise ModelRetry(
                'The submitted code no longer matches the prefix eager execution already ran, '
                'so the session was restarted. Send the snippet again.'
            )

        await self.execution.drain(call)
        if call.error is not None:
            raise call.error

        tail = '\n'.join(lines[call.fed_line_count :])
        async with self.execution.feed_lock:
            result = await self._execute_code(tail, ctx, run_code_tool, call.execution)
        return call.execution.build_tool_return(result)

    async def feed_fragment(
        self,
        call: StreamedCodeCall,
        source: str,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        tools = await self.get_tools(ctx)
        run_code_tool = self._as_run_code_tool(tools['run_code'])
        assert run_code_tool is not None, '`get_tools` always builds the `run_code` tool'
        await self._execute_code(source, ctx, run_code_tool, call.execution)
