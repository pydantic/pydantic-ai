from __future__ import annotations as _annotations

import asyncio
import dataclasses
import inspect
from asyncio import Task
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

from typing_extensions import TypeVar

from pydantic_ai._utils import cancel_and_drain
from pydantic_ai.tool_manager import ToolManager, ValidatedToolCall
from pydantic_graph import GraphRunContext
from pydantic_graph.basenode import NodeRunEndT

from . import _output, exceptions, messages as _messages, result
from .exceptions import ToolRetryError
from .tools import DeferredToolResult, ToolApproved, ToolDenied, ToolKind

if TYPE_CHECKING:
    from ._agent_graph import GraphAgentDeps, GraphAgentState

DepsT = TypeVar('DepsT')


def _make_output_status_part(
    call: _messages.ToolCallPart,
    content: str,
    output_parts: list[_messages.ModelRequestPart],
) -> _messages.ToolReturnPart:
    """Synthesize and append a status `ToolReturnPart` for an output tool call (success or skip).

    Sites that retry use the part returned by validation/execution directly, not a synthesized one.
    """
    part = _messages.ToolReturnPart(
        tool_name=call.tool_name,
        content=content,
        tool_call_id=call.tool_call_id,
    )
    output_parts.append(part)
    return part


def _emit_output_tool_events(
    call: _messages.ToolCallPart,
    part: _messages.ToolReturnPart | _messages.RetryPromptPart,
    *,
    args_valid: bool | None = None,
) -> Iterator[_messages.HandleResponseEvent]:
    """Yield `OutputToolCallEvent` and `OutputToolResultEvent` for an output tool call."""
    yield _messages.OutputToolCallEvent(call, args_valid=args_valid)
    yield _messages.OutputToolResultEvent(part)


@dataclasses.dataclass
class _OutputCallResult(Generic[NodeRunEndT]):
    """Result of validating and executing one output tool call.

    Exactly one of `final_result` (success), `retry_part` (validation/execution retry),
    or `raise_exc` (max retries exceeded — re-raised by the caller only if no other output
    produced a valid result) is set. `args_valid` carries the validation outcome for event
    emission and to distinguish validation failures from execution failures.
    """

    call: _messages.ToolCallPart
    args_valid: bool | None = None
    final_result: result.FinalResult[NodeRunEndT] | None = None
    retry_part: _messages.RetryPromptPart | None = None
    raise_exc: BaseException | None = None


async def _run_output_tool_call(
    tool_manager: ToolManager[DepsT],
    call: _messages.ToolCallPart,
    schema: _output.OutputSchema[NodeRunEndT],
    output_retries_increment: list[int],
    max_output_retries: int,
) -> _OutputCallResult[NodeRunEndT]:
    """Validate and execute an output tool call, returning a structured result.

    The caller interprets the result against the winner (first valid output by emission
    order) and emits events. `output_retries_increment[0]` accumulates retry-budget
    increments so the caller can apply them after a parallel batch settles, avoiding
    interleaved race writes. `UnexpectedModelBehavior` (max retries exceeded) is captured
    into `raise_exc` rather than raised inline so the caller can decide whether to re-raise
    (no other output produced a valid result) or absorb it as a skip.
    """
    try:
        validated = await tool_manager.validate_output_tool_call(call, schema=schema)
    except exceptions.UnexpectedModelBehavior as e:
        tool = tool_manager.tools.get(call.tool_name) if tool_manager.tools else None
        # Defensive: an output tool is always present in the toolset, so the `None` fallback to
        # the agent-level budget isn't expected in normal operation.
        max_retries = tool.max_retries if tool is not None else max_output_retries
        wrapped = exceptions.UnexpectedModelBehavior(f'Exceeded maximum output retries ({max_retries})')
        wrapped.__cause__ = e.__cause__ or e
        return _OutputCallResult(call=call, args_valid=False, raise_exc=wrapped)

    if not validated.args_valid:
        assert validated.validation_error is not None
        output_retries_increment[0] += 1
        return _OutputCallResult(call=call, args_valid=False, retry_part=validated.validation_error.tool_retry)

    try:
        result_data: Any = await tool_manager.execute_output_tool_call(validated, schema=schema)
    except exceptions.UnexpectedModelBehavior as e:
        max_retries = validated.tool.max_retries if validated.tool else max_output_retries
        wrapped = exceptions.UnexpectedModelBehavior(f'Exceeded maximum output retries ({max_retries})')
        wrapped.__cause__ = e.__cause__ or e
        return _OutputCallResult(call=call, args_valid=True, raise_exc=wrapped)
    except ToolRetryError as e:
        output_retries_increment[0] += 1
        return _OutputCallResult(call=call, args_valid=True, retry_part=e.tool_retry)

    final_result = result.FinalResult(result_data, call.tool_name, call.tool_call_id)
    return _OutputCallResult(call=call, args_valid=True, final_result=final_result)


@dataclasses.dataclass
class _ExhaustiveState(Generic[NodeRunEndT]):
    """Mutable out-params for `_process_exhaustive` (async generators can't return values)."""

    final_result: result.FinalResult[NodeRunEndT] | None = None
    retry_wins_triggered: bool = False


def _segment_by_barriers(indices: list[int], is_barrier: Callable[[int], bool]) -> list[list[int]]:
    """Split `indices` into execution segments around barrier tools.

    Each barrier index becomes a single-element segment; consecutive non-barrier indices form a
    parallel segment. Segments run in order, so a barrier completes before later tools start and
    starts only after earlier tools finish.
    """
    segments: list[list[int]] = []
    current: list[int] = []
    for i in indices:
        if is_barrier(i):
            if current:
                segments.append(current)
                current = []
            segments.append([i])
        else:
            current.append(i)
    if current:
        segments.append(current)
    return segments


async def _validate_function_calls(
    tool_manager: ToolManager[DepsT],
    calls: list[_messages.ToolCallPart],
    calls_to_run_results: dict[str, DeferredToolResult],
    tool_call_metadata: dict[str, dict[str, Any]] | None,
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    validated_calls: dict[str, ValidatedToolCall[DepsT]],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    """Validate a batch of function/unknown calls, emitting their `FunctionToolCallEvent`s.

    Populates `validated_calls`. On resume, a supplied result that isn't a `ToolApproved`
    (e.g. `ToolDenied`, `ModelRetry`) short-circuits inside `_call_tool`, so no validation is
    needed — the event is emitted without args-validity.
    """
    for call in calls:
        deferred_result = calls_to_run_results.get(call.tool_call_id)
        if deferred_result is not None and not isinstance(deferred_result, ToolApproved):
            yield _messages.FunctionToolCallEvent(call)
            continue
        try:
            if isinstance(deferred_result, ToolApproved):
                validate_call = call
                if deferred_result.override_args is not None:
                    validate_call = dataclasses.replace(call, args=deferred_result.override_args)
                metadata = tool_call_metadata.get(call.tool_call_id) if tool_call_metadata else None
                validated = await tool_manager.validate_tool_call(validate_call, approved=True, metadata=metadata)
            else:
                validated = await tool_manager.validate_tool_call(call)
        except exceptions.UnexpectedModelBehavior:
            ctx.state.check_incomplete_tool_call()
            yield _messages.FunctionToolCallEvent(call, args_valid=False)
            raise
        validated_calls[call.tool_call_id] = validated
        yield _messages.FunctionToolCallEvent(call, args_valid=validated.args_valid)


async def _process_exhaustive(  # noqa: C901
    *,
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    call_kinds: list[ToolKind | Literal['unknown']],
    output_indices: list[int],
    function_indices: list[int],
    calls_to_run_results: dict[str, DeferredToolResult],
    tool_call_metadata: dict[str, dict[str, Any]] | None,
    schema: _output.OutputSchema[NodeRunEndT],
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    output_retries_increment: list[int],
    output_parts: list[_messages.ModelRequestPart],
    deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]],
    deferred_metadata: dict[str, dict[str, Any]],
    emit_settled_output: Callable[..., Iterator[_messages.HandleResponseEvent]],
    state: _ExhaustiveState[NodeRunEndT],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    """Run all tool calls in parallel (the `'exhaustive'` strategy).

    Output and function tools launch together, segmented only by `sequential=True` barriers
    (which may be output tools via `ToolOutput(sequential=True)`). The first valid output by
    emission order becomes the final result; the rest still execute. Function-tool returns
    and the message-history parts are assembled in emission order; `FunctionToolResultEvent`s
    stream as each task completes.
    """
    final_result_was_set_externally = state.final_result is not None
    externally_won_id = state.final_result.tool_call_id if state.final_result is not None else None

    # Upfront-validate function calls in emission order, emitting their call events.
    validated_calls: dict[str, ValidatedToolCall[DepsT]] = {}
    async for event in _validate_function_calls(
        tool_manager,
        [tool_calls[i] for i in function_indices],
        calls_to_run_results,
        tool_call_metadata,
        ctx,
        validated_calls,
    ):
        yield event

    executable_indices = sorted([*output_indices, *function_indices])

    # An output tool matching a streamed-in `final_result` is already committed; don't re-execute.
    output_results: dict[int, _OutputCallResult[NodeRunEndT]] = {}
    for i in output_indices:
        if externally_won_id is not None and tool_calls[i].tool_call_id == externally_won_id:
            output_results[i] = _OutputCallResult(call=tool_calls[i], args_valid=True, final_result=state.final_result)

    # Segment by barriers: a `sequential=True` tool (or run-scoped 'sequential' mode) runs alone.
    # Pre-committed streamed outputs have no task to launch, so they're excluded from segmentation.
    mode = tool_manager.get_parallel_execution_mode()
    global_sequential = mode == 'sequential'
    ordered_events = mode == 'parallel_ordered_events'
    task_indices = [i for i in executable_indices if i not in output_results]
    segments = _segment_by_barriers(
        task_indices, lambda i: global_sequential or tool_manager.is_sequential(tool_calls[i])
    )

    function_parts: dict[int, _messages.ModelRequestPart] = {}
    function_user_parts: dict[int, _messages.UserPromptPart] = {}
    # Under `parallel_ordered_events`, function-tool result events are buffered and yielded in
    # emission order at the end (alongside output events) instead of streaming as tasks complete.
    function_events: dict[int, _messages.FunctionToolResultEvent] = {}
    deferred_by_index: dict[int, Literal['external', 'unapproved']] = {}
    deferred_meta_by_index: dict[int, dict[str, Any] | None] = {}

    async def run_one(
        index: int,
    ) -> tuple[
        int,
        _OutputCallResult[NodeRunEndT]
        | tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None]
        | exceptions.CallDeferred
        | exceptions.ApprovalRequired,
    ]:
        call = tool_calls[index]
        if call_kinds[index] == 'output':
            return index, await _run_output_tool_call(
                tool_manager, call, schema, output_retries_increment, ctx.deps.max_output_retries
            )
        try:
            return index, await _call_tool(
                tool_manager,
                validated_calls.get(call.tool_call_id, call),
                calls_to_run_results.get(call.tool_call_id),
            )
        except (exceptions.CallDeferred, exceptions.ApprovalRequired) as e:
            return index, e

    appended = False
    try:
        for segment in segments:
            tasks = [asyncio.create_task(run_one(i), name=tool_calls[i].tool_name) for i in segment]
            try:
                pending: set[asyncio.Task[Any]] = set(tasks)
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        index, payload = task.result()
                        if call_kinds[index] == 'output':
                            assert isinstance(payload, _OutputCallResult)
                            output_results[index] = payload
                        elif isinstance(payload, exceptions.CallDeferred):
                            deferred_by_index[index] = 'external'
                            deferred_meta_by_index[index] = payload.metadata
                        elif isinstance(payload, exceptions.ApprovalRequired):
                            deferred_by_index[index] = 'unapproved'
                            deferred_meta_by_index[index] = payload.metadata
                        else:
                            tool_part, tool_user_content = payload
                            function_parts[index] = tool_part
                            if tool_user_content:
                                function_user_parts[index] = _messages.UserPromptPart(content=tool_user_content)
                            # Only retries from actual function tools trigger retry-wins; retries from
                            # unknown/hallucinated tools don't suppress an otherwise-valid output.
                            if isinstance(tool_part, _messages.RetryPromptPart) and call_kinds[index] == 'function':
                                state.retry_wins_triggered = True
                            result_event = _messages.FunctionToolResultEvent(tool_part, content=tool_user_content)
                            if ordered_events:
                                function_events[index] = result_event
                            else:
                                yield result_event
            except asyncio.CancelledError as e:
                await cancel_and_drain(*tasks, msg=e.args[0] if len(e.args) != 0 else None)
                raise
            except BaseException:
                await cancel_and_drain(*tasks)
                raise

        # Pick the winner: first valid output by emission order (or the streamed-in result).
        if state.final_result is None:
            for i in output_indices:
                r = output_results.get(i)
                if r is not None and r.final_result is not None:
                    state.final_result = r.final_result
                    break

        # If no output produced a valid result but one hit max retries, surface that error.
        if state.final_result is None:
            for i in output_indices:
                r = output_results.get(i)
                if r is not None and r.raise_exc is not None:
                    ctx.state.output_retries_used += output_retries_increment[0]
                    ctx.state.check_incomplete_tool_call()  # pragma: lax no cover
                    raise r.raise_exc

        # Append parts and emit output events in emission order.
        for i in executable_indices:
            if call_kinds[i] == 'output':
                r = output_results.get(i)
                if r is None:
                    continue  # pragma: no cover  # every output index is populated above
                is_winner = state.final_result is not None and r.call.tool_call_id == state.final_result.tool_call_id
                if is_winner and final_result_was_set_externally:
                    # Streamed-in winner: record "processed" without claiming it was selected here.
                    part = _make_output_status_part(r.call, 'Final result processed.', output_parts)
                    for event in _emit_output_tool_events(r.call, part, args_valid=True):
                        yield event
                else:
                    for event in emit_settled_output(r, is_winner=is_winner):
                        yield event
            elif i in function_parts:
                output_parts.append(function_parts[i])
                # Under `parallel_ordered_events`, emit the buffered result event here so events
                # stream in emission order; otherwise it was already yielded as the task completed.
                if ordered_events and i in function_events:
                    yield function_events[i]
        for i in executable_indices:
            if i in function_user_parts:
                output_parts.append(function_user_parts[i])
        appended = True
    finally:
        if not appended:
            # Partial capture on exception: surface completed function-tool returns so
            # `CallToolsNode._handle_tool_calls` can record them in the interrupted request.
            for i in executable_indices:
                if i in function_parts:
                    output_parts.append(function_parts[i])
            # `executable_indices` is non-empty whenever this runs, so the empty-loop branch can't happen.
            for i in executable_indices:  # pragma: no branch
                if i in function_user_parts:
                    output_parts.append(function_user_parts[i])

    _populate_deferred_calls(tool_calls, deferred_by_index, deferred_meta_by_index, deferred_calls, deferred_metadata)


async def process_tool_calls(
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None,
    tool_call_metadata: dict[str, dict[str, Any]] | None,
    final_result: result.FinalResult[NodeRunEndT] | None,
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    output_parts: list[_messages.ModelRequestPart],
    output_final_result: deque[result.FinalResult[NodeRunEndT]] = deque(maxlen=1),
) -> AsyncIterator[_messages.HandleResponseEvent]:
    """Process a model response's tool calls, honoring the `end_strategy`.

    Output and function tools are classified by kind and executed per strategy:

    - `'early'`: output tools run sequentially in emission order and stop at the first
      success; function tools run **only if every output tool failed** (so the model can
      correct on the next round). Once an output succeeds, all function tools are stubbed
      as not executed.
    - `'graceful'` (default): tools run in the order the model emitted them — function
      tools that precede an output tool complete before it runs. Output tools run
      sequentially and stop at the first success; subsequent output tools are skipped
      (their side effects don't run). Function tools run in parallel within each segment.
    - `'exhaustive'`: every tool runs in parallel; the first valid output by emission order
      becomes the final result while the rest still execute. Only `sequential=True` tools
      (function or, via `ToolOutput(sequential=True)`, output) act as barriers.

    A `sequential=True` tool is a barrier: tools emitted before it complete first, it runs
    alone, and tools emitted after it start only once it finishes. The run-scoped
    `parallel_execution_mode('sequential')` turns every tool into its own barrier.

    Under `'graceful'`/`'exhaustive'`, the **retry-wins** invariant applies: if any
    function/unknown tool produces a `RetryPromptPart`, `final_result` is suppressed so the
    model addresses the retries on the next round. Output-tool retries don't trigger this
    ("first valid output wins"). Retry-wins doesn't apply when `final_result` was passed in
    by `Agent.run_stream` (the streamed output is already committed) or under `'early'`
    (function tools never run alongside a successful output).

    Deferred tools (`external`, `unapproved`) without supplied results are collected during
    the walk and resolved as a single batch at the end of the step.

    Because async iterators can't have return values, we use `output_parts` and
    `output_final_result` as output arguments.
    """
    processor = _ToolCallProcessor(
        tool_manager=tool_manager,
        tool_calls=tool_calls,
        tool_call_results=tool_call_results,
        tool_call_metadata=tool_call_metadata,
        ctx=ctx,
        output_parts=output_parts,
        final_result=final_result,
    )
    async for event in processor.run():
        yield event
    if processor.final_result:
        output_final_result.append(processor.final_result)


@dataclasses.dataclass
class _ToolCallProcessor(Generic[DepsT, NodeRunEndT]):
    """Executes one model response's tool calls for a single step, honoring the `end_strategy`.

    Holds the step's inputs plus the mutable result state (`final_result`, `retry_wins_triggered`)
    that the per-strategy methods build up. `output_parts` is appended to in place so partially
    completed work survives an exception (partial capture in `CallToolsNode._handle_tool_calls`).
    """

    tool_manager: ToolManager[DepsT]
    tool_calls: list[_messages.ToolCallPart]
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None
    tool_call_metadata: dict[str, dict[str, Any]] | None
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    output_parts: list[_messages.ModelRequestPart]
    final_result: result.FinalResult[NodeRunEndT] | None

    # Derived from the inputs in `__post_init__`.
    call_kinds: list[ToolKind | Literal['unknown']] = dataclasses.field(init=False)
    tool_calls_by_kind: dict[ToolKind | Literal['unknown'], list[_messages.ToolCallPart]] = dataclasses.field(
        init=False
    )
    calls_to_run_results: dict[str, DeferredToolResult] = dataclasses.field(init=False)
    executable_function_kinds: tuple[ToolKind | Literal['unknown'], ...] = dataclasses.field(init=False)
    function_indices: list[int] = dataclasses.field(init=False)
    output_indices: list[int] = dataclasses.field(init=False)
    schema: _output.OutputSchema[NodeRunEndT] = dataclasses.field(init=False)

    # Mutable state built up during execution.
    #
    # `final_result_was_set_externally`: when `final_result` is passed in pre-set (e.g. from
    # `Agent.run_stream`), the streamed output is already committed and retry-wins can't revoke it.
    # `retry_wins_triggered`: set when a function/unknown tool produces a `RetryPromptPart`.
    # `output_retries_increment`: accumulates output-retry-budget increments to apply once execution
    # settles, so parallel output tasks don't race the counter.
    final_result_was_set_externally: bool = dataclasses.field(init=False)
    retry_wins_triggered: bool = dataclasses.field(default=False, init=False)
    output_retries_increment: list[int] = dataclasses.field(default_factory=lambda: [0], init=False)
    deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]] = dataclasses.field(
        init=False
    )
    deferred_metadata: dict[str, dict[str, Any]] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.final_result_was_set_externally = self.final_result is not None
        self.deferred_calls = defaultdict(list)
        self.deferred_metadata = {}

        # Classify each call once, preserving emission order for the per-index views below.
        tool_calls_by_kind: dict[ToolKind | Literal['unknown'], list[_messages.ToolCallPart]] = defaultdict(list)
        call_kinds: list[ToolKind | Literal['unknown']] = []
        for call in self.tool_calls:
            tool_def = self.tool_manager.get_tool_def(call.tool_name)
            kind = tool_def.kind if tool_def else 'unknown'
            call_kinds.append(kind)
            tool_calls_by_kind[kind].append(call)
        self.call_kinds = call_kinds
        self.tool_calls_by_kind = tool_calls_by_kind

        # When resuming with `tool_call_results`, deferred kinds execute via the regular pipeline
        # (their results are supplied) rather than being batched at the end of the step.
        if self.tool_call_results is not None:
            self.executable_function_kinds = ('function', 'unknown', 'external', 'unapproved')
            result_tool_call_ids = set(self.tool_call_results.keys())
            eligible_call_ids = {
                call.tool_call_id
                for call, kind in zip(self.tool_calls, call_kinds)
                if kind in self.executable_function_kinds
            }
            if eligible_call_ids != result_tool_call_ids:
                raise exceptions.UserError(
                    'Tool call results need to be provided for all deferred tool calls. '
                    f'Expected: {eligible_call_ids}, got: {result_tool_call_ids}'
                )
            self.calls_to_run_results = {
                call_id: value for call_id, value in self.tool_call_results.items() if value != 'skip'
            }
        else:
            self.executable_function_kinds = ('function', 'unknown')
            self.calls_to_run_results = {}

        self.function_indices = [i for i in range(len(self.tool_calls)) if self.is_executable_function(i)]
        self.output_indices = [i for i in range(len(self.tool_calls)) if call_kinds[i] == 'output']
        self.schema = self.ctx.deps.output_schema

    def is_executable_function(self, index: int) -> bool:
        if self.call_kinds[index] not in self.executable_function_kinds:
            return False
        # On resume, calls without a supplied result were executed in a previous step; skip.
        if self.tool_call_results is not None and self.tool_calls[index].tool_call_id not in self.calls_to_run_results:
            return False
        return True

    async def run(self) -> AsyncIterator[_messages.HandleResponseEvent]:
        """Run the configured strategy, then apply retry-wins and resolve deferred calls."""
        # Check tool-call usage limits up front for the full count of function-kind calls.
        if self.ctx.deps.usage_limits.tool_calls_limit is not None and self.function_indices:
            projected_usage = deepcopy(self.ctx.state.usage)
            projected_usage.tool_calls += len(self.function_indices)
            self.ctx.deps.usage_limits.check_before_tool_call(projected_usage)

        end_strategy = self.ctx.deps.end_strategy
        if end_strategy == 'exhaustive':
            run_strategy = self._run_exhaustive
        elif end_strategy == 'early':
            run_strategy = self._run_early
        else:
            run_strategy = self._run_graceful
        async for event in run_strategy():
            yield event

        self._apply_retry_wins()
        async for event in self._finalize_deferred():
            yield event

    async def _run_output(self, call: _messages.ToolCallPart) -> AsyncIterator[_messages.HandleResponseEvent]:
        """Run a single output tool call (or stub it if a final result was already chosen)."""
        if self.final_result is not None and self.final_result.tool_call_id == call.tool_call_id:
            part = _make_output_status_part(call, 'Final result processed.', self.output_parts)
            for event in _emit_output_tool_events(call, part, args_valid=True):
                yield event
        elif self.final_result is not None:
            part = _make_output_status_part(
                call, 'Output tool not used - a final result was already processed.', self.output_parts
            )
            for event in _emit_output_tool_events(call, part, args_valid=None):
                yield event
        else:
            r = await _run_output_tool_call(
                self.tool_manager, call, self.schema, self.output_retries_increment, self.ctx.deps.max_output_retries
            )
            if r.raise_exc is not None:
                self.ctx.state.output_retries_used += self.output_retries_increment[0]
                self.ctx.state.check_incomplete_tool_call()  # pragma: lax no cover
                raise r.raise_exc
            if r.final_result is not None:
                self.final_result = r.final_result
            for event in self._emit_settled_output(r, is_winner=r.final_result is not None):
                yield event

    async def _run_function_calls(
        self, calls: list[_messages.ToolCallPart]
    ) -> AsyncIterator[_messages.HandleResponseEvent]:
        """Validate a batch of function/unknown calls upfront, then execute via `_call_tools`."""
        if not calls:
            return
        validated_calls: dict[str, ValidatedToolCall[DepsT]] = {}
        async for event in _validate_function_calls(
            self.tool_manager, calls, self.calls_to_run_results, self.tool_call_metadata, self.ctx, validated_calls
        ):
            yield event

        before = len(self.output_parts)
        async for event in _call_tools(
            tool_manager=self.tool_manager,
            tool_calls=calls,
            tool_call_results=self.calls_to_run_results,
            validated_calls=validated_calls,
            output_parts=self.output_parts,
            output_deferred_calls=self.deferred_calls,
            output_deferred_metadata=self.deferred_metadata,
        ):
            yield event
        # A `RetryPromptPart` from an actual function tool (a `ModelRetry` or arg-validation
        # failure) triggers retry-wins. Retries from unknown/hallucinated tools don't — they
        # aren't work that needs to complete before the output is valid.
        for part in self.output_parts[before:]:
            if isinstance(part, _messages.RetryPromptPart) and part.tool_name is not None:
                tool_def = self.tool_manager.get_tool_def(part.tool_name)
                if tool_def is not None and tool_def.kind == 'function':
                    self.retry_wins_triggered = True

    def _emit_settled_output(
        self, r: _OutputCallResult[NodeRunEndT], *, is_winner: bool
    ) -> Iterator[_messages.HandleResponseEvent]:
        """Append the message-history part and emit events for a settled output result."""
        if r.final_result is not None:
            if is_winner:
                part = _make_output_status_part(r.call, 'Final result processed.', self.output_parts)
                yield from _emit_output_tool_events(r.call, part, args_valid=True)
            else:
                # A successful-but-not-winning output only happens under `'exhaustive'`; `'early'`
                # and `'graceful'` stop running output tools at the first success.
                part = _make_output_status_part(
                    r.call,
                    'Output tool processed, but its value will not be the final result of the agent run.',
                    self.output_parts,
                )
                yield from _emit_output_tool_events(r.call, part, args_valid=True)
        elif r.retry_part is not None:
            self.output_parts.append(r.retry_part)
            yield from _emit_output_tool_events(r.call, r.retry_part, args_valid=r.args_valid)
        else:
            # Absorbed failure: another output won, so this one's max-retries error is recorded
            # as a skip rather than raised. (When no output won, the caller raises `raise_exc`.)
            assert r.raise_exc is not None
            message = (
                'Output tool not used - output function execution failed.'
                if r.args_valid
                else 'Output tool not used - output failed validation.'
            )
            part = _make_output_status_part(r.call, message, self.output_parts)
            yield from _emit_output_tool_events(r.call, part, args_valid=r.args_valid)

    async def _run_early(self) -> AsyncIterator[_messages.HandleResponseEvent]:
        """`'early'`: run all output tools first; run function tools only if every output failed."""
        for call in self.tool_calls_by_kind['output']:
            # `_run_output` always yields ≥1 event, so the empty-iterator branch can't happen.
            async for event in self._run_output(call):  # pragma: no branch
                yield event
        self.ctx.state.output_retries_used += self.output_retries_increment[0]

        function_calls = [self.tool_calls[i] for i in self.function_indices]
        if self.final_result is not None:
            # An output succeeded: function tools are not executed.
            for call in function_calls:
                self.output_parts.append(
                    _messages.ToolReturnPart(
                        tool_name=call.tool_name,
                        content='Tool not executed - a final result was already processed.',
                        tool_call_id=call.tool_call_id,
                    )
                )
        else:
            # Every output failed; run function tools so the model can correct next round.
            async for event in self._run_function_calls(function_calls):  # pragma: no branch
                yield event

    async def _run_graceful(self) -> AsyncIterator[_messages.HandleResponseEvent]:
        """`'graceful'`: walk in emission order, running pending function-tool batches before each output tool."""
        pending_functions: list[_messages.ToolCallPart] = []

        async def flush_pending() -> AsyncIterator[_messages.HandleResponseEvent]:
            nonlocal pending_functions
            if pending_functions:
                batch = pending_functions
                pending_functions = []
                async for event in self._run_function_calls(batch):
                    yield event

        for i, call in enumerate(self.tool_calls):
            if self.call_kinds[i] == 'output':
                async for event in flush_pending():
                    yield event
                # `_run_output` always yields ≥1 event, so the empty-iterator branch can't happen.
                async for event in self._run_output(call):  # pragma: no branch
                    yield event
            elif self.is_executable_function(i):
                pending_functions.append(call)
        async for event in flush_pending():
            yield event
        self.ctx.state.output_retries_used += self.output_retries_increment[0]

    async def _run_exhaustive(self) -> AsyncIterator[_messages.HandleResponseEvent]:
        """`'exhaustive'`: run every tool in parallel, segmented only by `sequential=True` barriers."""
        state = _ExhaustiveState[NodeRunEndT](final_result=self.final_result)
        async for event in _process_exhaustive(
            tool_manager=self.tool_manager,
            tool_calls=self.tool_calls,
            call_kinds=self.call_kinds,
            output_indices=self.output_indices,
            function_indices=self.function_indices,
            calls_to_run_results=self.calls_to_run_results,
            tool_call_metadata=self.tool_call_metadata,
            schema=self.schema,
            ctx=self.ctx,
            output_retries_increment=self.output_retries_increment,
            output_parts=self.output_parts,
            deferred_calls=self.deferred_calls,
            deferred_metadata=self.deferred_metadata,
            emit_settled_output=self._emit_settled_output,
            state=state,
        ):
            yield event
        self.final_result = state.final_result
        self.retry_wins_triggered = self.retry_wins_triggered or state.retry_wins_triggered
        self.ctx.state.output_retries_used += self.output_retries_increment[0]

    def _apply_retry_wins(self) -> None:
        """Suppress the output result if a function tool retried (graceful + exhaustive).

        The suppressed output's `ToolReturnPart` is rewritten so the model addresses the retry
        next round. Doesn't apply when the final result was committed externally (`run_stream`).
        """
        if not (
            self.retry_wins_triggered and self.final_result is not None and not self.final_result_was_set_externally
        ):
            return
        # The winning output always has a 'Final result processed.' part here, so the loop
        # always breaks; the no-match fall-through can't happen.
        for idx, part in enumerate(self.output_parts):  # pragma: no branch
            if (
                isinstance(part, _messages.ToolReturnPart)
                and part.tool_call_id == self.final_result.tool_call_id
                and part.content == 'Final result processed.'
            ):
                self.output_parts[idx] = dataclasses.replace(
                    part, content='Output not used as the final result - addressing tool retries from this round first.'
                )
                break
        self.final_result = None

    async def _finalize_deferred(self) -> AsyncIterator[_messages.HandleResponseEvent]:
        """Stub, collect, or inline-resolve deferred (`external`/`unapproved`) tool calls."""
        # Collect deferred calls (unless they were already included in the run because results were provided).
        if self.tool_call_results is None:
            async for event in self._collect_deferred_calls():
                yield event

        if not self.final_result and self.deferred_calls:
            async for event in self._resolve_deferred_calls():
                yield event

    async def _collect_deferred_calls(self) -> AsyncIterator[_messages.HandleResponseEvent]:
        """Stub deferred calls (a final result was reached) or validate-and-collect them for resolution."""
        calls = [*self.tool_calls_by_kind['external'], *self.tool_calls_by_kind['unapproved']]
        if self.final_result:
            # If the run was already determined to end on deferred tool calls,
            # we shouldn't insert return parts as the deferred tools will still get a real result.
            if not isinstance(self.final_result.output, _output.DeferredToolRequests):
                for call in calls:
                    self.output_parts.append(
                        _messages.ToolReturnPart(
                            tool_name=call.tool_name,
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=call.tool_call_id,
                        )
                    )
        elif calls:
            for call in calls:
                try:
                    validated = await self.tool_manager.validate_tool_call(call)
                except exceptions.UnexpectedModelBehavior:
                    yield _messages.FunctionToolCallEvent(call, args_valid=False)
                    raise

                yield _messages.FunctionToolCallEvent(call, args_valid=validated.args_valid)

                if validated.args_valid:
                    if call in self.tool_calls_by_kind['external']:
                        self.deferred_calls['external'].append(call)
                    else:
                        self.deferred_calls['unapproved'].append(call)
                else:
                    # Call execute_tool_call to raise the validation error inside a trace span;
                    # retries are already tracked by validate_tool_call() via failed_tools.
                    try:
                        await self.tool_manager.execute_tool_call(validated)
                    except ToolRetryError as e:
                        self.output_parts.append(e.tool_retry)
                        yield _messages.FunctionToolResultEvent(e.tool_retry)

    async def _resolve_deferred_calls(self) -> AsyncIterator[_messages.HandleResponseEvent]:
        """Resolve collected deferred calls via capability handlers, else set the `DeferredToolRequests` result."""
        deferred_tool_requests: _output.DeferredToolRequests | None = _output.DeferredToolRequests(
            calls=self.deferred_calls['external'],
            approvals=self.deferred_calls['unapproved'],
            metadata=self.deferred_metadata,
        )

        # Let capability handlers resolve deferred calls inline (one shot).
        # Results are fed back through the existing tool-execution pipeline so that
        # approvals, denials, retries, and ToolReturn unwrapping all behave identically
        # to the UserPromptNode resume path.
        handler_results = await self.tool_manager.resolve_deferred_tool_calls(deferred_tool_requests)
        if handler_results is not None:
            handler_tool_call_results = handler_results.to_tool_call_results()
            resolved_calls = [
                call
                for call in [*self.deferred_calls['unapproved'], *self.deferred_calls['external']]
                if call.tool_call_id in handler_tool_call_results
            ]

            handler_validated_calls: dict[str, ValidatedToolCall[DepsT]] = {}
            for call in resolved_calls:
                handler_result = handler_tool_call_results[call.tool_call_id]
                if not isinstance(handler_result, ToolApproved):
                    continue
                validate_call = call
                if handler_result.override_args is not None:
                    validate_call = dataclasses.replace(call, args=handler_result.override_args)
                call_metadata = handler_results.metadata.get(call.tool_call_id)
                try:
                    handler_validated_calls[call.tool_call_id] = await self.tool_manager.validate_tool_call(
                        validate_call, approved=True, metadata=call_metadata
                    )
                except exceptions.UnexpectedModelBehavior:  # pragma: no cover
                    # Defensive: only reached if the handler's override_args fail validation after
                    # retries were already exhausted in this run step. Mirrors the non-deferred
                    # validation path above; naturally triggered there, not here.
                    yield _messages.FunctionToolCallEvent(call, args_valid=False)
                    raise

            new_deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]] = defaultdict(
                list
            )
            new_deferred_metadata: dict[str, dict[str, Any]] = {}
            async for event in _call_tools(
                tool_manager=self.tool_manager,
                tool_calls=resolved_calls,
                tool_call_results=handler_tool_call_results,
                validated_calls=handler_validated_calls,
                output_parts=self.output_parts,
                output_deferred_calls=new_deferred_calls,
                output_deferred_metadata=new_deferred_metadata,
            ):
                yield event

            deferred_tool_requests = deferred_tool_requests.remaining(handler_results)
            if new_deferred_calls['external'] or new_deferred_calls['unapproved']:
                if deferred_tool_requests is None:
                    deferred_tool_requests = _output.DeferredToolRequests()
                deferred_tool_requests.calls.extend(new_deferred_calls['external'])
                deferred_tool_requests.approvals.extend(new_deferred_calls['unapproved'])
                deferred_tool_requests.metadata.update(new_deferred_metadata)

        if deferred_tool_requests is not None:
            if not self.ctx.deps.output_schema.allows_deferred_tools:
                raise exceptions.UserError(
                    'A deferred tool call was present, but `DeferredToolRequests` is not among output types. '
                    'To resolve this, add `DeferredToolRequests` to the list of output types for this agent, '
                    'or use a `HandleDeferredToolCalls` capability to handle deferred tool calls inline.'
                )
            self.final_result = result.FinalResult(cast(NodeRunEndT, deferred_tool_requests), None, None)


async def _call_tools(  # noqa: C901
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    tool_call_results: dict[str, DeferredToolResult],
    validated_calls: dict[str, ValidatedToolCall[DepsT]],
    output_parts: list[_messages.ModelRequestPart],
    output_deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]],
    output_deferred_metadata: dict[str, dict[str, Any]],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    tool_parts_by_index: dict[int, _messages.ModelRequestPart] = {}
    user_parts_by_index: dict[int, _messages.UserPromptPart] = {}
    deferred_calls_by_index: dict[int, Literal['external', 'unapproved']] = {}
    deferred_metadata_by_index: dict[int, dict[str, Any] | None] = {}

    async def handle_call_or_result(
        coro_or_task: Awaitable[
            tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None]
        ]
        | Task[
            tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None]
        ],
        index: int,
    ) -> _messages.HandleResponseEvent | None:
        try:
            tool_part, tool_user_content = (
                (await coro_or_task) if inspect.isawaitable(coro_or_task) else coro_or_task.result()
            )
        except exceptions.CallDeferred as e:
            deferred_calls_by_index[index] = 'external'
            deferred_metadata_by_index[index] = e.metadata
        except exceptions.ApprovalRequired as e:
            deferred_calls_by_index[index] = 'unapproved'
            deferred_metadata_by_index[index] = e.metadata
        else:
            tool_parts_by_index[index] = tool_part
            if tool_user_content:
                user_parts_by_index[index] = _messages.UserPromptPart(content=tool_user_content)

            return _messages.FunctionToolResultEvent(tool_part, content=tool_user_content)

    def call_tool(
        index: int,
    ) -> Coroutine[
        Any,
        Any,
        tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None],
    ]:
        call = tool_calls[index]
        return _call_tool(
            tool_manager,
            validated_calls.get(call.tool_call_id, call),
            tool_call_results.get(call.tool_call_id),
        )

    mode = tool_manager.get_parallel_execution_mode()
    ordered_events = mode == 'parallel_ordered_events'
    global_sequential = mode == 'sequential'

    # Segment by barriers: a `sequential=True` tool (or the run-scoped 'sequential' mode) runs
    # alone, with tools emitted before it completing first and tools after it starting only once
    # it finishes. Non-barrier tools parallelize within their segment.
    segments = _segment_by_barriers(
        list(range(len(tool_calls))),
        lambda i: global_sequential or tool_manager.is_sequential(tool_calls[i]),
    )

    try:
        for segment in segments:
            if len(segment) == 1:
                # A barrier (or sole call): run inline, event in completion order.
                index = segment[0]
                if event := await handle_call_or_result(call_tool(index), index):
                    yield event
            else:
                tasks_by_index = {
                    index: asyncio.create_task(call_tool(index), name=tool_calls[index].tool_name) for index in segment
                }
                index_by_task = {task: index for index, task in tasks_by_index.items()}
                try:
                    if ordered_events:
                        # Wait for the whole segment, then yield events in emission order.
                        await asyncio.wait(tasks_by_index.values(), return_when=asyncio.ALL_COMPLETED)
                        for index in segment:
                            if event := await handle_call_or_result(tasks_by_index[index], index):
                                yield event
                    else:
                        pending: set[
                            asyncio.Task[
                                tuple[
                                    _messages.ToolReturnPart | _messages.RetryPromptPart,
                                    str | Sequence[_messages.UserContent] | None,
                                ]
                            ]
                        ] = set(tasks_by_index.values())
                        while pending:
                            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                            for task in done:
                                if event := await handle_call_or_result(task, index_by_task[task]):
                                    yield event
                except asyncio.CancelledError as e:
                    await cancel_and_drain(*tasks_by_index.values(), msg=e.args[0] if len(e.args) != 0 else None)
                    raise
                except BaseException:
                    # Cancel any still-running sibling tasks so they don't become
                    # orphaned asyncio tasks when a non-CancelledError exception
                    # (e.g. RuntimeError, ConnectionError) propagates out of
                    # handle_call_or_result().
                    await cancel_and_drain(*tasks_by_index.values())
                    raise
    finally:
        # Populate output_parts even on exception so partial tool returns surface
        # to the outer capture in `CallToolsNode._handle_tool_calls`. We append the
        # results at the end, rather than as they are received, to retain a
        # consistent ordering.
        output_parts.extend([tool_parts_by_index[k] for k in sorted(tool_parts_by_index)])
        output_parts.extend([user_parts_by_index[k] for k in sorted(user_parts_by_index)])

    _populate_deferred_calls(
        tool_calls, deferred_calls_by_index, deferred_metadata_by_index, output_deferred_calls, output_deferred_metadata
    )


def _populate_deferred_calls(
    tool_calls: list[_messages.ToolCallPart],
    deferred_calls_by_index: dict[int, Literal['external', 'unapproved']],
    deferred_metadata_by_index: dict[int, dict[str, Any] | None],
    output_deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]],
    output_deferred_metadata: dict[str, dict[str, Any]],
) -> None:
    """Populate deferred calls and metadata from indexed mappings."""
    for k in sorted(deferred_calls_by_index):
        call = tool_calls[k]
        output_deferred_calls[deferred_calls_by_index[k]].append(call)
        metadata = deferred_metadata_by_index[k]
        if metadata is not None:
            output_deferred_metadata[call.tool_call_id] = metadata


async def _call_tool(
    tool_manager: ToolManager[DepsT],
    tool_call: ValidatedToolCall[DepsT] | _messages.ToolCallPart,
    tool_call_result: DeferredToolResult | None,
) -> tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None]:
    if isinstance(tool_call, ValidatedToolCall):
        validated = tool_call
        call = tool_call.call
    else:
        validated = None
        call = tool_call

    tool_result: Any
    try:
        if tool_call_result is None or isinstance(tool_call_result, ToolApproved):
            if validated is not None:
                tool_result = await tool_manager.execute_tool_call(validated)
            else:
                raise RuntimeError('Expected validated tool call')  # pragma: no cover
        elif isinstance(tool_call_result, ToolDenied):
            return _messages.ToolReturnPart(
                tool_name=call.tool_name,
                content=tool_call_result.message,
                tool_call_id=call.tool_call_id,
                outcome='denied',
            ), None
        elif isinstance(tool_call_result, exceptions.ModelRetry):
            m = _messages.RetryPromptPart(
                content=tool_call_result.message,
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
            )
            raise ToolRetryError(m)
        elif isinstance(tool_call_result, _messages.RetryPromptPart):
            tool_call_result.tool_name = call.tool_name
            tool_call_result.tool_call_id = call.tool_call_id
            raise ToolRetryError(tool_call_result)
        else:
            tool_result = tool_call_result
    except ToolRetryError as e:
        return e.tool_retry, None

    if isinstance(tool_result, _messages.ToolReturn):
        tool_return = cast(_messages.ToolReturn[Any], tool_result)
    elif isinstance(tool_result, list) and any(
        isinstance(i, _messages.ToolReturn) for i in cast(list[Any], tool_result)
    ):
        raise exceptions.UserError(
            f'The return value of tool {call.tool_name!r} contains invalid nested `ToolReturn` objects. '
            f'`ToolReturn` should be used directly.'
        )
    else:
        tool_return = _messages.ToolReturn[Any](return_value=cast(Any, tool_result))

    # If the called tool's `ToolDefinition.tool_kind` declares a registered typed subclass
    # (e.g. `'tool-search'`), promote the return part to that subclass. This keeps the
    # typed identity intact across multi-turn history: the next turn's discovery parser /
    # cross-provider replay sees a typed `ToolSearchReturnPart` instead of a base part.
    tool_def = tool_manager.get_tool_def(call.tool_name)
    return_part = _messages.ToolReturnPart(
        tool_name=call.tool_name,
        tool_call_id=call.tool_call_id,
        content=tool_return.return_value,
        metadata=tool_return.metadata,
        tool_kind=tool_def.tool_kind if tool_def else None,
    )
    return_part = _messages.ToolReturnPart.narrow_type(return_part)

    return return_part, tool_return.content or None
