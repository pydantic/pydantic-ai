from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Union, cast

from . import _utils
from .tools import AgentDepsT, RunContext, SystemPromptFunc


@dataclass
class SystemPromptRunner(Generic[AgentDepsT]):
    function: SystemPromptFunc[AgentDepsT]
    dynamic: bool = False
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 0
        self._is_async = inspect.iscoroutinefunction(self.function)

    async def run(self, run_context: RunContext[AgentDepsT]) -> Union[str, None]:  # noqa UP007
        if self._takes_ctx:
            args = (run_context,)
        else:
            args = ()

        if self._is_async:
            function = cast(Callable[[Any], Awaitable[Union[str, None]]], self.function)
            return await function(*args)
        else:
            function = cast(Callable[[Any], Union[str, None]], self.function)
            return await _utils.run_in_executor(function, *args)
