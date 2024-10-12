from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Callable, Generic, Union

from . import _utils, retrievers as _r
from .retrievers import AgentDeps

# This is basically a function that may or maybe not take `CallInfo` as an argument, and may or may not be async.
# Usage `SystemPromptFunc[AgentDependencies]`
SystemPromptFunc = Union[
    Callable[[_r.CallContext[AgentDeps]], str],
    Callable[[_r.CallContext[AgentDeps]], Awaitable[str]],
    Callable[[], str],
    Callable[[], Awaitable[str]],
]


@dataclass
class SystemPromptRunner(Generic[AgentDeps]):
    function: SystemPromptFunc[AgentDeps]
    takes_ctx: bool = False
    is_async: bool = False

    def __post_init__(self):
        self.takes_ctx = len(inspect.signature(self.function).parameters) > 0
        self.is_async = inspect.iscoroutinefunction(self.function)

    async def run(self, deps: AgentDeps) -> str:
        if self.takes_ctx:
            args = (_r.CallContext(deps, 0),)
        else:
            args = ()

        if self.is_async:
            return await self.function(*args)  # pyright: ignore[reportGeneralTypeIssues,reportUnknownVariableType]
        else:
            return await _utils.run_in_executor(
                self.function,  # pyright: ignore[reportArgumentType,reportReturnType]
                *args,
            )
