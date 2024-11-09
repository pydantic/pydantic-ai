from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Union, cast

from . import _utils
from .call_typing import AgentDeps, CallContext

# A function that may or maybe not take `CallContext` as an argument, and may or may not be async.
# Usage `SystemPromptFunc[AgentDeps]`
SystemPromptFunc = Union[
    Callable[[CallContext[AgentDeps]], str],
    Callable[[CallContext[AgentDeps]], Awaitable[str]],
    Callable[[], str],
    Callable[[], Awaitable[str]],
]


@dataclass
class SystemPromptRunner(Generic[AgentDeps]):
    function: SystemPromptFunc[AgentDeps]
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 0
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def run(self, deps: AgentDeps) -> str:
        if self._takes_ctx:
            args = (CallContext(deps, 0, None),)
        else:
            args = ()

        if self._is_async:
            function = cast(Callable[[Any], Awaitable[str]], self.function)
            return await function(*args)
        else:
            function = cast(Callable[[Any], str], self.function)
            return await _utils.run_in_executor(function, *args)
