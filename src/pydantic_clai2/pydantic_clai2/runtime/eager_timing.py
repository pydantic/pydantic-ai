"""Measure sandbox tool latency that eager `run_code` execution hid behind argument streaming."""

import re
from dataclasses import dataclass, field
from time import perf_counter

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.tools import AgentDepsT, ToolDefinition

NESTED_CALL = re.compile(r'(?P<parent>.+)__(?P<speculative>spec_)?\d+')
"""Harness CodeMode ids: `parent__N` for a sandbox dispatch, `parent__spec_N` for a speculative launch."""


@dataclass(kw_only=True)
class EagerExecutionCompletedEvent(CapabilityEvent, namespace='pydantic_clai2'):
    """A completed snippet whose sandbox calls overlapped `run_code` argument streaming."""

    saved_ms: float
    """Summed tool-call overlap with generation, not wall-clock speedup."""


@dataclass(kw_only=True)
class _StreamWindow:
    ended: float | None = None
    saved_ms: float = 0.0


@dataclass
class EagerTiming(AbstractCapability[AgentDepsT]):
    """Observe execution hooks without changing tool behavior.

    A window opens when a tool call part starts streaming and closes when it ends. Sandbox
    dispatches (`parent__N`) that run inside their parent's open window are charged the
    overlapping time. Speculative launches (`parent__spec_N`) are excluded because the harness
    already reports them, so the two totals do not double-count.
    """

    _windows: dict[tuple[int, str], _StreamWindow] = field(
        default_factory=dict[tuple[int, str], _StreamWindow], init=False, repr=False
    )
    _by_index: dict[tuple[int, int], _StreamWindow] = field(
        default_factory=dict[tuple[int, int], _StreamWindow], init=False, repr=False
    )
    """The same windows by response part index, since some providers re-key a call mid-stream."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> 'EagerTiming[AgentDepsT]':
        """Keep windows per run."""
        return EagerTiming()

    async def on_event(self, ctx: RunContext[AgentDepsT], *, event: AgentStreamEvent) -> None:
        """Open a window at a tool call's first streamed byte and close it at its last."""
        if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
            window = _StreamWindow()
            self._windows[(ctx.run_step, event.part.tool_call_id)] = window
            self._by_index[(ctx.run_step, event.index)] = window
        elif (
            isinstance(event, PartDeltaEvent)
            and isinstance(event.delta, ToolCallPartDelta)
            and event.delta.tool_call_id
        ):
            # Harness re-keys a call when a delta carries a new id; nested dispatches then use that id.
            window = self._by_index.setdefault((ctx.run_step, event.index), _StreamWindow())
            self._windows[(ctx.run_step, event.delta.tool_call_id)] = window
        elif isinstance(event, PartEndEvent) and isinstance(event.part, ToolCallPart):
            window = self._by_index.pop((ctx.run_step, event.index), None) or _StreamWindow()
            window.ended = perf_counter()
            self._windows[(ctx.run_step, event.part.tool_call_id)] = window

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        """Charge nested dispatches to the open window; report the total when `run_code` succeeds."""
        nested = NESTED_CALL.fullmatch(call.tool_call_id)
        if nested is None or nested['speculative']:
            return await self._top_level(ctx, call=call, tool_def=tool_def, args=args, handler=handler)
        window = self._windows.get((ctx.run_step, nested['parent']))
        if window is None or window.ended is not None:
            return await handler(args)
        started = perf_counter()
        try:
            return await handler(args)
        finally:
            finished = perf_counter()
            cutoff = finished if window.ended is None else min(finished, window.ended)
            window.saved_ms += max(0.0, cutoff - started) * 1000

    async def _top_level(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        key = (ctx.run_step, call.tool_call_id)
        window = self._windows.get(key)
        try:
            result = await handler(args)
            if tool_def.name == 'run_code' and window is not None and not args.get('restart') and window.saved_ms > 0:
                await ctx.emit(EagerExecutionCompletedEvent(tool_call_id=call.tool_call_id, saved_ms=window.saved_ms))
            return result
        finally:
            self._windows.pop(key, None)
