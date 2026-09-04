from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from pydantic_ai._utils import await_maybe
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.capabilities._native_resolution import resolve_native_tool
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.native_tools import XSearchTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool

XSearchFallbackModelFunc = Callable[
    [RunContext[Any]],
    Awaitable[Model | KnownModelName | str] | Model | KnownModelName | str,
]
"""Callable that resolves a fallback model dynamically per-run.

May return a `Model` instance or a model name string (e.g. `'xai:grok-4-1-fast-non-reasoning'`);
strings are resolved to a model at call time.
"""

XSearchFallbackModel = Model | KnownModelName | str | XSearchFallbackModelFunc | None
"""Type for the fallback model: a model, model name, factory callable, or None."""

XSearchNativeTool: TypeAlias = XSearchTool | Callable[[RunContext[AgentDepsT]], Awaitable[XSearchTool] | XSearchTool]
"""Type for the native tool: an `XSearchTool` instance, or a callable resolving one from the run context.

The callable resolves once per fallback subagent invocation, from that tool call's
[`RunContext`][pydantic_ai.tools.RunContext]. It belongs to the same run and carries the same
`deps` as the resolution on the native path, but it is not the same context: `tool_call_id` and
`tool_name` name the fallback tool call rather than being `None`, and `messages` holds the run so
far. Read `ctx.deps` for configuration that has to match across both.

Unlike the capability-level `native=` parameter, this callable may not return `None`: omitting the
tool is meaningless once the subagent has been invoked, so returning `None` anyway raises
[`UserError`][pydantic_ai.exceptions.UserError] rather than enabling a default `XSearchTool`.
"""

__all__ = (
    'XSearchFallbackModel',
    'XSearchFallbackModelFunc',
    'XSearchNativeTool',
    'XSearchSubagentTool',
    'x_search_tool',
)


@dataclass(kw_only=True)
class XSearchSubagentTool:
    """Local X search tool that delegates to a subagent.

    Uses a subagent with the specified xAI model and `XSearchTool` native tool
    to search X/Twitter when the outer agent's model doesn't support
    X search natively.
    """

    model: Model | KnownModelName | str | XSearchFallbackModelFunc
    """The model to use for X search, or a callable that returns one."""

    native_tool: XSearchNativeTool[Any]
    """The X search tool configuration or outer-run factory to pass to the subagent."""

    instructions: str = 'Search X/Twitter based on the user query. Return a comprehensive summary of the results.'
    """Instructions for the subagent that performs the X search."""

    async def __call__(self, ctx: RunContext[Any], query: str) -> str:
        """Search X/Twitter using a subagent.

        Args:
            ctx: The run context from the outer agent.
            query: The search query to run on X/Twitter.
        """
        model = self.model
        if callable(model):
            model = await await_maybe(model(ctx))

        native_tool = await resolve_native_tool(XSearchTool, self.native_tool, ctx)

        agent = Agent(
            model,
            output_type=str,
            capabilities=[NativeTool(native_tool)],
            instructions=self.instructions,
        )
        try:
            result = await agent.run(query)
        except UnexpectedModelBehavior as e:
            raise ModelRetry(str(e)) from e
        return result.output


def x_search_tool(
    model: Model | KnownModelName | str | XSearchFallbackModelFunc,
    native_tool: XSearchNativeTool[Any],
    *,
    instructions: str = 'Search X/Twitter based on the user query. Return a comprehensive summary of the results.',
) -> Tool[Any]:
    """Creates an X search tool backed by a subagent.

    Args:
        model: The model to use for X search. Must be an xAI model that natively
            supports the `XSearchTool` native tool, e.g. `'xai:grok-4.3'`.
            Can also be a callable taking `RunContext` that returns such a model.
        native_tool: The X search tool configuration, or a callable that resolves it from the outer run context.
        instructions: Instructions for the subagent that performs the X search.
    """
    return Tool[Any](
        XSearchSubagentTool(model=model, native_tool=native_tool, instructions=instructions).__call__,
        name='x_search',
        description='Search X/Twitter for posts and content based on the given query.',
    )
