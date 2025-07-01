from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Protocol

from .._run_context import AgentDepsT, RunContext
from .run import RunToolset
from .wrapper import WrapperToolset


class CallToolFunc(Protocol):
    """A function protocol that represents a tool call."""

    def __call__(self, name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any) -> Awaitable[Any]: ...


ToolProcessFunc = Callable[
    [
        RunContext[AgentDepsT],
        CallToolFunc,
        str,
        dict[str, Any],
    ],
    Awaitable[Any],
]


@dataclass
class ProcessedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that lets the tool call arguments and return value be customized using a process function."""

    process: ToolProcessFunc[AgentDepsT]

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)
        processed = ProcessedToolset(wrapped_for_run, self.process)
        return RunToolset(processed, ctx)

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await self.process(ctx, partial(self.wrapped.call_tool, ctx), name, tool_args, *args, **kwargs)
