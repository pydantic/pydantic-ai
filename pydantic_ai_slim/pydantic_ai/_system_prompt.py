from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeAlias, cast

from typing_extensions import TypeVar

from . import _utils
from ._run_context import AgentDepsT, RunContext
from .messages import ModelMessage, ModelRequest, ModelRequestPart, SystemPromptPart

PromptT = TypeVar('PromptT', default=str)
"""What a prompt function produces: text for a system prompt, or also an `InstructionPart` for an instruction."""

PromptFunc: TypeAlias = (
    Callable[[RunContext[AgentDepsT]], PromptT | None]
    | Callable[[RunContext[AgentDepsT]], Awaitable[PromptT | None]]
    | Callable[[], PromptT | None]
    | Callable[[], Awaitable[PromptT | None]]
)
"""A function that may or may not take `RunContext`, may or may not be async, and returns a `PromptT` or `None`."""


@dataclass
class SystemPromptRunner(Generic[AgentDepsT, PromptT]):
    function: PromptFunc[AgentDepsT, PromptT]
    dynamic: bool = False
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 0
        self._is_async = _utils.is_async_callable(self.function)

    async def run(self, run_context: RunContext[AgentDepsT]) -> PromptT | None:
        if self._takes_ctx:
            args = (run_context,)
        else:
            args = ()

        if self._is_async:
            function = cast(Callable[[Any], Awaitable[PromptT | None]], self.function)
            return await function(*args)
        else:
            # A plain `def` may still return an awaitable, which `run_in_executor` would leave un-awaited.
            function = cast(Callable[[Any], PromptT | None | Awaitable[PromptT | None]], self.function)
            return await _utils.await_maybe(await _utils.run_in_executor(function, *args))


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


async def reevaluate_dynamic_system_prompts(
    messages: list[ModelMessage],
    dynamic_runners: Mapping[str, SystemPromptRunner[AgentDepsT]],
    run_context: RunContext[AgentDepsT],
) -> None:
    """Re-run the function behind each `SystemPromptPart` in `messages` that has a `dynamic_ref`, in place.

    Each changed message is replaced in the list rather than mutated, so caller-owned message history
    objects are never changed.
    """
    # Only proceed if there's at least one dynamic runner.
    if not dynamic_runners:
        return
    for index, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            reevaluated_message_parts: list[ModelRequestPart] = []
            for part in msg.parts:
                if isinstance(part, SystemPromptPart) and part.dynamic_ref:
                    # Look up the runner by its ref
                    if runner := dynamic_runners.get(part.dynamic_ref):  # pragma: lax no cover
                        # To enable dynamic system prompt refs in future runs, use a placeholder string
                        updated_part_content = await runner.run(run_context)
                        part = SystemPromptPart(
                            updated_part_content or '', dynamic_ref=part.dynamic_ref, metadata=part.metadata
                        )

                reevaluated_message_parts.append(part)

            if reevaluated_message_parts != msg.parts:
                messages[index] = replace(msg, parts=reevaluated_message_parts)
