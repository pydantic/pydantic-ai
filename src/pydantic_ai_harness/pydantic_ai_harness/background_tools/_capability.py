"""Background tools capability that runs selected tools concurrently."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import anyio
import anyio.abc
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult, RawToolArgs
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailedError,
    ToolRetryError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import ToolCallPart, ToolReturn, ToolReturnPart, UserContent
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolRequests,
    RunContext,
    ToolDefinition,
    ToolSelector,
    matches_tool_selector,
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


def _instructions(ctx: RunContext[Any]) -> str | None:
    # Realtime sessions run tools concurrently already; see `_background_mode`.
    return None if ctx.realtime else _INSTRUCTIONS


def _with_run_in_background(tool_def: ToolDefinition) -> ToolDefinition:
    """Add the optional `run_in_background` argument to a tool's schema."""
    properties = tool_def.parameters_json_schema.get('properties', {})
    if _RUN_IN_BACKGROUND in properties:
        raise UserError(
            f"Tool '{tool_def.name}' already has a '{_RUN_IN_BACKGROUND}' parameter, "
            'so it cannot be an optional background tool.'
        )
    flag = {
        'type': 'boolean',
        'description': 'Set to true to keep working and get the result later as a follow-up message.',
    }
    schema = {**tool_def.parameters_json_schema, 'properties': {**properties, _RUN_IN_BACKGROUND: flag}}
    return replace(tool_def, parameters_json_schema=schema)


_Outcome = tuple[UserContent, ...] | BaseException
"""What a finished background task hands to the run: the follow-up to deliver, or the error that ends the run."""


def _deliver(ctx: RunContext[Any], outcome: _Outcome) -> None:
    if isinstance(outcome, BaseException):
        raise outcome
    ctx.enqueue(*outcome)


def _format_background_error(error: ApprovalRequired | CallDeferred | ToolRetryError | ToolFailedError) -> str:
    """Describe a tool-signalled failure to the model."""
    if isinstance(error, (ApprovalRequired, CallDeferred)):
        return f'{type(error).__name__} was raised; background tools cannot defer a running task.'
    content = error.tool_retry.content if isinstance(error, ToolRetryError) else error.tool_failed.content
    return content if isinstance(content, str) and content else type(error).__name__


def _format_background_result(tool_name: str, task_id: str, result: Any) -> tuple[UserContent, ...]:
    """Format a tool result as model-visible user content without application metadata."""
    if isinstance(result, ToolReturn):
        return_value: object = result.return_value
        extra_content = result.content
    else:
        return_value = result
        extra_content = None

    return_part = ToolReturnPart(tool_name=tool_name, tool_call_id=task_id, content=return_value)
    return_text = return_part.model_response_str()
    content: list[UserContent] = [return_text, *return_part.files]
    if isinstance(extra_content, str):
        content.append(extra_content)
    elif extra_content is not None:
        content.extend(extra_content)

    prefix = f"Background tool '{tool_name}' (task {task_id}) completed.\nResult:"
    content[0] = f'{prefix} {content[0]}'

    if all(isinstance(item, str) for item in content):
        return ('\n'.join(item for item in content if isinstance(item, str)),)
    return tuple(content)


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
    unless this selector chooses it. Sequential tools and realtime sessions run normally.
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
    _prepared_modes: dict[str, Literal['always', 'optional'] | None] = field(
        default_factory=dict[str, Literal['always', 'optional'] | None], init=False, repr=False, compare=False
    )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return _instructions

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

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        self._prepared_modes.clear()
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
            try:
                parsed = json.loads(args)
            except ValueError:
                return args  # Core turns malformed JSON into a retry.
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
        if mode is None or (mode == 'optional' and call.args_as_dict().get(_RUN_IN_BACKGROUND) is not True):
            return await handler(args)

        task_id = call.tool_call_id
        tool_name = call.tool_name

        async def _run() -> None:
            outcome: _Outcome
            try:
                # A task the run cancelled before it got to start must not run the tool.
                await anyio.lowlevel.checkpoint_if_cancelled()
                try:
                    result = await handler(args)
                except (ApprovalRequired, CallDeferred, ToolRetryError, ToolFailedError) as e:
                    outcome = (f"Background tool '{tool_name}' (task {task_id}) failed: {_format_background_error(e)}",)
                except anyio.get_cancelled_exc_class() as e:
                    # Propagate cancellation delivered by any enclosing scope. If none is pending,
                    # the tool raised this itself and it ends the run like a sequential tool.
                    await anyio.lowlevel.checkpoint_if_cancelled()
                    outcome = e
                except UnexpectedModelBehavior as e:
                    # The retry budget ran out: it ends the run, as it would for a sequential tool.
                    outcome = e
                except Exception as e:
                    # Exception messages can contain private details, so the model only learns the type.
                    outcome = (f"Background tool '{tool_name}' (task {task_id}) failed: {type(e).__name__}",)
                except BaseException as e:
                    outcome = e
                else:
                    outcome = _format_background_result(tool_name, task_id, result)
                self._send.send_nowait(outcome)
            finally:
                self._live -= 1
                # Core counts a successful call when the tool body returns, replacing this reservation.
                ctx.usage.tool_calls -= 1

        ctx.usage.tool_calls += 1
        self._live += 1
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
            _deliver(ctx, outcome)
        if isinstance(result, End) and self._live and not ctx.pending_messages:
            # The model is ending the run while tasks are still live: wait for the next outcome, so
            # that the end-of-run drain turns its follow-up into another model request.
            _deliver(ctx, await self._outcomes.receive())
        return result

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
