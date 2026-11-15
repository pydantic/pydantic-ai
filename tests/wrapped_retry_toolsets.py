"""Toolsets for the tests of the retry count a tool's `prepare` sees under name-changing wrappers.

Shared by the plain, Temporal, Prefect and Render tests of https://github.com/pydantic/pydantic-ai/issues/9741. The
Temporal tests import this module into the workflow sandbox, so it imports nothing but `pydantic_ai`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

DepsT = TypeVar('DepsT')


def failing_toolset(
    id: str | None = None, *, seen: list[int] | None = None, retries: int | None = None
) -> FunctionToolset[object]:
    """A toolset whose `fail` tool always fails, with a `prepare` that hides it once it has.

    The `prepare` appends each `ctx.retry` it sees to `seen`, when given.
    """
    toolset = FunctionToolset[object](id=id)

    async def hide_after_failure(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition | None:
        if seen is not None:
            seen.append(ctx.retry)
        return None if ctx.retry else tool_def

    @toolset.tool(prepare=hide_after_failure, retries=retries)
    async def fail(ctx: RunContext[object]) -> str:
        raise ModelRetry('Choose another approach.')

    return toolset


def snapshot_probe_toolset(
    record: Callable[[str, RunContext[DepsT]], None],
) -> Callable[[RunContext[DepsT]], AbstractToolset[DepsT]]:
    """A dynamic toolset factory that calls `record` from each site a durable unit hands a context to.

    The sites are the factory, the toolset's instructions, and the `probe` tool's `prepare`, argument validator and
    body. The factory prefixes and then renames the toolset so `probe` keeps its name, which makes each wrapper copy
    the context inside the durable unit before the tool runs.
    """

    def factory(ctx: RunContext[DepsT]) -> AbstractToolset[DepsT]:
        record('factory', ctx)
        toolset = FunctionToolset[DepsT](id='snapshot_inner')

        @toolset.instructions
        def instructions(ctx: RunContext[DepsT]) -> str:
            record('instructions', ctx)
            return 'Call the probe once.'

        async def prepare(ctx: RunContext[DepsT], tool_def: ToolDefinition) -> ToolDefinition:
            record('prepare', ctx)
            return tool_def

        def validate(ctx: RunContext[DepsT]) -> None:
            record('validator', ctx)

        @toolset.tool(prepare=prepare, args_validator=validate)
        async def probe(ctx: RunContext[DepsT]) -> str:
            record('body', ctx)
            return 'Probed.'

        return toolset.prefixed('inner').renamed({'probe': 'inner_probe'})

    return factory


def call_while_listed(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model function that calls the first listed tool, and answers `'No tools left.'` once none is."""
    if info.function_tools:
        return ModelResponse(parts=[ToolCallPart(info.function_tools[0].name, {})])
    return ModelResponse(parts=[TextPart('No tools left.')])


def call_probe_once(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model function that calls `demo_probe` once, then answers `'Done.'`."""
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('Done.')])
    return ModelResponse(parts=[ToolCallPart('demo_probe', {})])
