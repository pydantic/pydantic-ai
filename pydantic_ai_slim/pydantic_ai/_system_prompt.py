from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, cast

from . import _utils
from ._run_context import AgentDepsT, RunContext
from .messages import SystemPromptPart
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

    async def run(self, run_context: RunContext[AgentDepsT]) -> str | None:
        if self._takes_ctx:
            args = (run_context,)
        else:
            args = ()

        if self._is_async:
            function = cast(Callable[[Any], Awaitable[str | None]], self.function)
            return await function(*args)
        else:
            function = cast(Callable[[Any], str | None], self.function)
            return await _utils.run_in_executor(function, *args)


async def resolve_system_prompts(
    static_prompts: Sequence[str],
    runners: Sequence[SystemPromptRunner[AgentDepsT]],
    run_context: RunContext[AgentDepsT],
) -> list[SystemPromptPart]:
    """Resolve configured static strings and runner functions into `SystemPromptPart`s.

    Dynamic runners produce parts with `dynamic_ref` set so they can be re-evaluated on
    subsequent turns by the standard agent graph path. Non-dynamic runners are evaluated
    once and stored with their static content; empty results are skipped.
    """
    parts: list[SystemPromptPart] = [SystemPromptPart(p) for p in static_prompts]
    for runner in runners:
        prompt = await runner.run(run_context)
        if runner.dynamic:
            parts.append(SystemPromptPart(prompt or '', dynamic_ref=runner.function.__qualname__))
        elif prompt:
            parts.append(SystemPromptPart(prompt))
    return parts
