"""Background tools capability that runs selected tools concurrently."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import anyio
import anyio.abc
import anyio.lowlevel
import pydantic_core
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import ValidationError

from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NodeResult,
    RawToolArgs,
    SetToolMetadata,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import TextContent, ToolCallPart, ToolReturn, ToolReturnPart, UserContent
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolRequests,
    RunContext,
    ToolDefinition,
    ToolSelector,
    matches_tool_selector,
)
from pydantic_ai.toolsets import AbstractToolset, PreparedToolset
from pydantic_ai_harness._background_completion import (
    escape_tool_result_body,
    make_background_completion,
    tool_result_tags,
)

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.capabilities import WrapRunHandler, WrapToolExecuteHandler
    from pydantic_ai.run import AgentRunResult


_INSTRUCTIONS = """\
Some tools reply that a call is running in the background. This means that exact call is pending. \
Do not repeat or poll it. Its result will arrive automatically in a later message. Continue only \
with independent work. If none remains, end your response; the run will resume when the result arrives.\
"""

_RUN_IN_BACKGROUND = 'run_in_background'

_RUN_IN_BACKGROUND_SCHEMA = {
    'type': 'boolean',
    'description': 'Set to true to keep working and get the result later as a follow-up message.',
}

_backgrounded_call: ContextVar[ToolCallPart | None] = ContextVar('_backgrounded_call', default=None)
"""The tool call a background task is running, so another `BackgroundTools` instance does not schedule it again."""


def _instructions(ctx: RunContext[Any]) -> str | None:
    # Realtime sessions run tools concurrently already; see `_background_mode`.
    return None if ctx.realtime else _INSTRUCTIONS


def _dispatched_by_tool(ctx: RunContext[Any]) -> bool:
    """Whether another tool, such as code mode's `run_code`, dispatched this call and awaits its value."""
    # A nested `ToolManager` carries the context of the call that dispatched it.
    dispatcher = ctx.tool_manager.ctx if ctx.tool_manager is not None else None
    return dispatcher is not None and dispatcher.tool_call_id is not None


def _with_run_in_background(tool_def: ToolDefinition) -> ToolDefinition:
    """Add the optional `run_in_background` argument to a tool's schema."""
    properties = tool_def.parameters_json_schema.get('properties', {})
    if _RUN_IN_BACKGROUND in properties:
        if properties[_RUN_IN_BACKGROUND] == _RUN_IN_BACKGROUND_SCHEMA:
            return tool_def  # Another `BackgroundTools` instance already added it.
        raise UserError(
            f"Tool '{tool_def.name}' already has a '{_RUN_IN_BACKGROUND}' parameter, "
            'so it cannot be an optional background tool.'
        )
    schema = {
        **tool_def.parameters_json_schema,
        'properties': {**properties, _RUN_IN_BACKGROUND: _RUN_IN_BACKGROUND_SCHEMA},
    }
    return replace(tool_def, parameters_json_schema=schema)


@dataclass
class _Completion:
    """A finished background call: the follow-up to deliver, and how it settles the tool's retry count."""

    tool_name: str
    max_retries: int
    content: tuple[UserContent, ...]
    retry: Exception | None = None
    """The failure that counts toward the tool's `max_retries`, when the call is to be retried."""
    succeeded: bool = False
    """Whether the call returned a result, which resets the tool's retry count as for a sequential call."""


_Outcome = _Completion | BaseException
"""What a finished background task hands to the run: the completed call, or the error that ends the run."""


_SignalledError = (
    ApprovalRequired | CallDeferred | ToolRetryError | ToolFailedError | ModelRetry | ToolFailed | ValidationError
)
"""Errors a background task reports to the model as a failed result instead of ending the run."""


def _format_background_failure(tool_name: str, task_id: str, error: _SignalledError) -> TextContent:
    """Describe a tool-signalled failure to the model, framed as the call's failed result."""
    if isinstance(error, (ApprovalRequired, CallDeferred)):
        message = f'{type(error).__name__} was raised; background tools cannot defer a running task.'
    else:
        if isinstance(error, (ModelRetry, ToolFailed)):
            content: object = error.message
        elif isinstance(error, ValidationError):
            content = pydantic_core.to_json(
                error.errors(include_url=False, include_context=False), fallback=str
            ).decode()
        elif isinstance(error, ToolRetryError):
            content = error.tool_retry.content
        else:
            content = error.tool_failed.content
        message = escape_tool_result_body(content) if isinstance(content, str) and content else type(error).__name__
    open_tag, close_tag = tool_result_tags(tool_name, task_id, failed=True)
    return make_background_completion(f'{open_tag}\n{message}\n{close_tag}', task_id)


def _format_background_result(
    tool_name: str, task_id: str, result: Any
) -> tuple[TextContent, *tuple[UserContent, ...]]:
    """Format a tool result as model-visible user content with completion metadata."""
    if isinstance(result, ToolReturn):
        return_value: object = result.return_value
        extra_content = result.content
    else:
        return_value = result
        extra_content = None

    return_part = ToolReturnPart(tool_name=tool_name, tool_call_id=task_id, content=return_value)
    # The frame's attributes name the call the "running in background" reply announced, so the result needs no
    # preamble. `ToolReturn.content` follows the frame unwrapped, as core sends it: content the tool author chose
    # to add for the model.
    open_tag, close_tag = tool_result_tags(tool_name, task_id)
    completion_text = f'{open_tag}\n{escape_tool_result_body(return_part.model_response_str())}'
    files = return_part.files
    if not files:
        completion_text = f'{completion_text}\n{close_tag}'
    content: list[UserContent] = [completion_text, *files, close_tag] if files else [completion_text]
    if isinstance(extra_content, str):
        content.append(extra_content)
    elif extra_content is not None:
        content.extend(extra_content)

    if all(isinstance(item, str) for item in content):
        return (make_background_completion('\n'.join(item for item in content if isinstance(item, str)), task_id),)
    return (make_background_completion(completion_text, task_id), *content[1:])


@dataclass
class BackgroundTools(AbstractCapability[AgentDepsT]):
    """Run selected tools in the background while the agent continues.

    The model receives a "started" message right away and the result when the tool finishes.

    ```python
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai_harness import BackgroundTools

    # Default: any tool with `metadata={'background': True}` runs in the background.
    agent = Agent('openai:gpt-5.6-sol', capabilities=[BackgroundTools()])

    @agent.tool_plain(metadata={'background': True})
    async def slow_research(query: str) -> str:
        await asyncio.sleep(60)  # stand-in for a long-running job
        return f'Research findings for {query!r}'
    ```

    Pass `tools=...` to select tools by name, metadata, or a function. Use
    `metadata={'background': 'optional'}` to let the model decide for each call.

    Warning:
        Cancelling a run also cancels its background tools. Async tools must allow cancellation.
        Python cannot stop a synchronous tool before it returns.

    See the Background Tools guide for streaming, realtime, and durable execution.
    """

    tools: ToolSelector[AgentDepsT] = field(default_factory=lambda: {'background': True})
    """Which tools always run in the background.

    Use `'all'`, a list of names, matching metadata, or a function that chooses each tool. The
    default is `{'background': True}`.

    A tool marked with `metadata={'background': 'optional'}` lets the model decide for each call
    unless this selector chooses it. Sequential tools, realtime sessions, and calls that code mode's
    `run_code` makes from its sandbox run normally.
    """

    id: str | None = 'background_tools'

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Combine selectors so each matching tool is scheduled exactly once."""
        merged = super().combine(capabilities)
        # Core only groups instances of the same capability class under one id.
        assert isinstance(merged, cls)

        async def matches_any(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool:
            for capability in capabilities:
                assert isinstance(capability, cls)
                if await matches_tool_selector(capability.tools, ctx, tool_def):
                    return True
            return False

        return replace(merged, tools=matches_any)

    _task_group: anyio.abc.TaskGroup = field(init=False, repr=False, compare=False)
    """Owns the run's background tasks. `wrap_run` opens it around the run, so no task outlives the run."""
    _live: int = field(default=0, init=False, repr=False)
    """Background tasks that have not handed over their outcome yet."""
    _send: MemoryObjectSendStream[_Outcome] = field(init=False, repr=False, compare=False)
    _outcomes: MemoryObjectReceiveStream[_Outcome] = field(init=False, repr=False, compare=False)
    """Outcomes in completion order; `after_node_run` takes them as they arrive."""
    _receipts: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """Tools whose started message core counted as a success since the last node boundary."""
    _unsettled: list[_Completion] = field(default_factory=list[_Completion], init=False, repr=False, compare=False)
    """Delivered calls whose outcome has not been charged to the tool's retry count yet."""
    _prepared_modes: dict[str, Literal['always', 'optional'] | None] = field(
        default_factory=dict[str, Literal['always', 'optional'] | None], init=False, repr=False, compare=False
    )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return _instructions

    def _realtime_uncalled_hooks(self) -> tuple[str, ...]:
        # `before_node_run` and `after_node_run` only settle and deliver background outcomes, and no
        # tool runs in the background in a realtime session (`_background_mode` returns `None` when
        # `ctx.realtime`), so the session needs nothing from them. A private core contract until #7190
        # maps hooks onto sessions.
        # Measured against this class, so a subclass that adds its own hooks still warns.
        return self._realtime_uncalled_hooks_overriding(BackgroundTools)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> BackgroundTools[AgentDepsT]:
        return replace(self)

    async def _background_mode(
        self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition
    ) -> Literal['always', 'optional'] | None:
        """Whether `tool_def` always runs in the background, may on request, or cannot in this run."""
        run_sequential = ctx.tool_manager is not None and ctx.tool_manager.get_parallel_execution_mode() == 'sequential'
        if ctx.realtime or run_sequential or tool_def.sequential:
            return None
        if await matches_tool_selector(self.tools, ctx, tool_def):
            return 'always'
        if (tool_def.metadata or {}).get('background') == 'optional':
            return 'optional'
        return None

    def get_ordering(self) -> CapabilityOrdering:
        # Outside `SetToolMetadata`, so the tools reach `_prepare_tools` with its metadata.
        return CapabilityOrdering(wraps=[SetToolMetadata])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return PreparedToolset(toolset, self._prepare_tools)

    async def _prepare_tools(
        self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        self._prepared_modes.clear()
        # Core skips a deferred capability's hooks until it is loaded, but not its wrapper toolset.
        if self.defer_loading is True and self.id not in ctx.active_capability_ids:
            return tool_defs
        prepared: list[ToolDefinition] = []
        for tool_def in tool_defs:
            mode = await self._background_mode(ctx, tool_def)
            self._prepared_modes[tool_def.name] = mode
            prepared.append(_with_run_in_background(tool_def) if mode == 'optional' else tool_def)
        return prepared

    async def _prepared_mode(
        self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition
    ) -> Literal['always', 'optional'] | None:
        if tool_def.name in self._prepared_modes:
            return self._prepared_modes[tool_def.name]
        return await self._background_mode(ctx, tool_def)

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
    ) -> RawToolArgs:
        if await self._prepared_mode(ctx, tool_def) != 'optional':
            return args
        parsed: Any = args
        if isinstance(args, str):
            # The parser `ToolCallPart.args_as_dict` uses, so `wrap_tool_execute` reads the same flag back.
            try:
                parsed = pydantic_core.from_json(args)
            except ValueError:
                return args  # Core turns malformed or too deeply nested JSON into a retry.
        if not isinstance(parsed, dict):
            return args
        # The tool's validator rejects unknown arguments, so the flag is removed and checked here.
        stripped: dict[str, Any] = {**parsed}
        flag = stripped.pop(_RUN_IN_BACKGROUND, False)
        if flag is not None and not isinstance(flag, bool):
            raise ModelRetry(f'`{_RUN_IN_BACKGROUND}` must be true or false.')
        return stripped

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        mode = await self._prepared_mode(ctx, tool_def)
        # The flag was removed before validation, so it is read from the call as the model sent it.
        if (
            mode is None
            or _dispatched_by_tool(ctx)
            or _backgrounded_call.get() is call
            or (mode == 'optional' and call.args_as_dict().get(_RUN_IN_BACKGROUND) is not True)
        ):
            return await handler(args)

        task_id = call.tool_call_id
        tool_name = call.tool_name

        def failure(error: _SignalledError, *, retry: Exception | None = None) -> _Completion:
            content = (_format_background_failure(tool_name, task_id, error),)
            return _Completion(tool_name, ctx.max_retries, content, retry=retry)

        async def _run() -> None:
            outcome: _Outcome
            try:
                # A task the run cancelled before it got to start must not run the tool.
                await anyio.lowlevel.checkpoint_if_cancelled()
                # Set inside the task, so only this call's own handler chain sees it.
                _backgrounded_call.set(call)
                try:
                    result = await handler(args)
                except ToolRetryError as e:
                    # Core charged the tool body's `ModelRetry` to the issuing step, whose count the next step may
                    # already have carried over. `_settle` charges it on delivery instead, so it counts exactly once.
                    assert ctx.tool_manager is not None  # Set for every tool call.
                    ctx.tool_manager.failed_tools.discard(tool_name)
                    outcome = failure(e, retry=e)
                except (ModelRetry, ValidationError) as e:
                    # Raised by an execute hook such as `before_tool_execute`. Core counts it outside
                    # `wrap_tool_execute`, so `_settle` does.
                    outcome = failure(e, retry=e)
                except (ApprovalRequired, CallDeferred, ToolFailedError, ToolFailed) as e:
                    outcome = failure(e)
                except anyio.get_cancelled_exc_class() as e:
                    # Propagate cancellation delivered by any enclosing scope. If none is pending,
                    # the tool raised this itself and it ends the run like a sequential tool.
                    await anyio.lowlevel.checkpoint_if_cancelled()
                    outcome = e
                except BaseException as e:
                    # Unexpected failures, including an exhausted retry budget, end the run as they
                    # would for a sequential tool.
                    outcome = e
                else:
                    outcome = _Completion(
                        tool_name,
                        ctx.max_retries,
                        _format_background_result(tool_name, task_id, result),
                        succeeded=True,
                    )
                self._send.send_nowait(outcome)
            finally:
                self._live -= 1
                # Core counts a successful call when the tool body returns, replacing this reservation.
                ctx.usage.tool_calls -= 1

        ctx.usage.tool_calls += 1
        self._live += 1
        self._receipts.add(tool_name)
        self._task_group.start_soon(_run, name=f'background tool {tool_name} ({task_id})')
        return (
            f"Tool '{tool_name}' is running in background (task {task_id}). This call is pending. "
            'Do not repeat or poll it; its result will arrive automatically. '
            'Continue independent work, or end your response.'
        )

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        from pydantic_graph import End

        if isinstance(result, End) and isinstance(result.data.output, DeferredToolRequests):
            # Finished tasks are left unswept so that a deferred-tool pause is not turned into
            # another model request by the end-of-run drain.
            return result

        for outcome in self._arrived():
            self._deliver(ctx, outcome)
        if isinstance(result, End) and self._live and not ctx.pending_messages:
            # The model is ending the run while tasks are still live: wait for the next outcome, so
            # that the end-of-run drain turns its follow-up into another model request.
            self._deliver(ctx, await self._outcomes.receive())
        return result

    async def before_node_run(
        self, ctx: RunContext[AgentDepsT], *, node: AgentNode[AgentDepsT]
    ) -> AgentNode[AgentDepsT]:
        self._settle(ctx)
        return node

    def _deliver(self, ctx: RunContext[AgentDepsT], outcome: _Outcome) -> None:
        if isinstance(outcome, BaseException):
            raise outcome
        ctx.enqueue(*outcome.content)
        self._unsettled.append(outcome)

    def _settle(self, ctx: RunContext[AgentDepsT]) -> None:
        """Charge delivered calls to the tool manager whose retry counts the next step carries over.

        A background call finishes after the step that issued it, so this settles it as core settles a
        sequential call at the end of its step: a retry counts toward `max_retries`, a result resets the count,
        and the started message counts as neither.
        """
        if not self._receipts and not self._unsettled:
            return
        # Before a node runs, the context holds the run's current tool manager.
        tool_manager = ctx.tool_manager
        assert tool_manager is not None and tool_manager.ctx is not None
        tool_manager.succeeded_tools.difference_update(self._receipts)
        self._receipts.clear()
        unsettled, self._unsettled = self._unsettled, []
        for completion in unsettled:
            if completion.retry is not None:
                if tool_manager.ctx.retries.get(completion.tool_name, 0) >= completion.max_retries:
                    raise UnexpectedModelBehavior(
                        f'Tool {completion.tool_name!r} exceeded max retries count of {completion.max_retries}. '
                        'Consider raising the retry limit, or see the docs on tool retries: '
                        'https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-retries'
                    ) from completion.retry
                tool_manager.failed_tools.add(completion.tool_name)
            elif completion.succeeded:
                tool_manager.succeeded_tools.add(completion.tool_name)

    def _arrived(self) -> Iterator[_Outcome]:
        """Outcomes that have arrived so far, in completion order."""
        with suppress(anyio.WouldBlock):
            while True:
                yield self._outcomes.receive_nowait()

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        self._send, self._outcomes = anyio.create_memory_object_stream[_Outcome](math.inf)
        result: AgentRunResult[Any] | None = None
        with self._send, self._outcomes:
            async with anyio.create_task_group() as self._task_group:
                result = await handler()
                # Tasks still live after a deferred-tool pause or `run_stream()` are dropped.
                self._task_group.cancel_scope.cancel()
            # An error from a task that finished after the last node boundary still ends the run.
            for outcome in self._arrived():
                if isinstance(outcome, BaseException):
                    raise outcome
        # `result` is bound: the group re-raises the run's own exception, and no task cancels the group.
        assert result is not None
        return result
