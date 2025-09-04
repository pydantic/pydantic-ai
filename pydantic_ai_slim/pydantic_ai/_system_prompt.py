from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, cast

from . import _utils
from ._run_context import AgentDepsT, RunContext
from .tools import SystemPromptFunc


@dataclass
class SystemPromptRunner(Generic[AgentDepsT]):
    function: SystemPromptFunc[AgentDepsT]
    dynamic: bool = False
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 0
        self._is_async = _utils.is_async_callable(self.function)

    async def run(self, run_context: RunContext[AgentDepsT]) -> str:
        if self._takes_ctx:
            args = (run_context,)
        else:
            args = ()

        if self._is_async:
            function = cast(Callable[[Any], Awaitable[str]], self.function)
            return await function(*args)
        else:
            function = cast(Callable[[Any], str], self.function)
            return await _utils.run_in_executor(function, *args)
