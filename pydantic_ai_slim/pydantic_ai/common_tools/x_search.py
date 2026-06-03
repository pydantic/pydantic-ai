from __future__ import annotations

import inspect
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai._utils import install_deprecated_kwarg_alias
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.native_tools import XSearchTool
from pydantic_ai.tools import RunContext, Tool

XSearchFallbackModelFunc = Callable[
    [RunContext[Any]],
    Awaitable[Model] | Model,
]
"""Callable that resolves a fallback model dynamically per-run."""

XSearchFallbackModel = Model | KnownModelName | str | XSearchFallbackModelFunc | None
"""Type for the fallback model: a model, model name, factory callable, or None."""

__all__ = (
    'XSearchFallbackModel',
    'XSearchFallbackModelFunc',
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

    native_tool: XSearchTool
    """The X search tool configuration to pass to the subagent."""

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
            result = model(ctx)
            if inspect.isawaitable(result):
                result = await result
            model = result

        agent = Agent(
            model,
            output_type=str,
            capabilities=[NativeTool(self.native_tool)],
            instructions=self.instructions,
        )
        try:
            result = await agent.run(query)
        except UnexpectedModelBehavior as e:
            raise ModelRetry(str(e)) from e
        return result.output

    def __getattr__(self, name: str) -> Any:
        # Deprecated alias for read access to the renamed `builtin_tool` field.
        if name == 'builtin_tool':
            warnings.warn(
                '`XSearchSubagentTool.builtin_tool` is deprecated, use `.native_tool` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            return self.native_tool
        raise AttributeError(name)


install_deprecated_kwarg_alias(XSearchSubagentTool, old='builtin_tool', new='native_tool')


def x_search_tool(
    model: Model | KnownModelName | str | XSearchFallbackModelFunc,
    native_tool: XSearchTool | None = None,
    *,
    instructions: str = 'Search X/Twitter based on the user query. Return a comprehensive summary of the results.',
    **_deprecated_kwargs: Any,
) -> Tool[Any]:
    """Creates an X search tool backed by a subagent.

    Args:
        model: The model to use for X search. Must be an xAI model that natively
            supports the `XSearchTool` native tool, e.g. `'xai:grok-4.3'`.
            Can also be a callable taking `RunContext` that returns such a model.
        native_tool: The X search tool configuration to pass to the subagent.
        instructions: Instructions for the subagent that performs the X search.
    """
    if 'builtin_tool' in _deprecated_kwargs:
        warnings.warn(
            '`x_search_tool(builtin_tool=...)` is deprecated, use `native_tool=` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        legacy_native_tool = _deprecated_kwargs.pop('builtin_tool')
        if native_tool is None:
            native_tool = legacy_native_tool
    if _deprecated_kwargs:
        unknown = ', '.join(f'`{k}`' for k in _deprecated_kwargs)
        raise TypeError(f'x_search_tool() got unexpected keyword arguments: {unknown}')
    if native_tool is None:
        raise TypeError("x_search_tool() missing required argument: 'native_tool'")
    return Tool[Any](
        XSearchSubagentTool(model=model, native_tool=native_tool, instructions=instructions).__call__,
        name='x_search',
        description='Search X/Twitter for posts and content based on the given query.',
    )
