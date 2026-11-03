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
    context_id: str | None = None
    """For a dynamic system prompt with `on_change='append'`, the id of the `<context>` element its statements are sent as."""
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
            parts.append(
                SystemPromptPart(prompt or '', dynamic_ref=runner.function.__qualname__, context_id=runner.context_id)
            )
        elif prompt:
            parts.append(SystemPromptPart(prompt))
    return parts


async def reevaluate_dynamic_system_prompts(
    messages: list[ModelMessage],
    dynamic_runners: Mapping[str, SystemPromptRunner[AgentDepsT]],
    run_context: RunContext[AgentDepsT],
    *,
    append: bool = True,
) -> list[SystemPromptPart]:
    """Re-run the function behind each `SystemPromptPart` in `messages` that has a `dynamic_ref`.

    A part is replaced in place, unless its function was registered with `on_change='append'` and `append`
    is set. Each changed message is replaced in the list rather than mutated, so caller-owned message
    history objects are never changed.

    An appending function runs once, and when its value differs from the latest part with its ref, the
    new statement is returned for the caller to put at the start of the upcoming request. Its earlier parts
    stay as they were, so the start of the conversation does too.
    """
    # Only proceed if there's at least one dynamic runner.
    if not dynamic_runners:
        return []
    latest: dict[str, SystemPromptPart] = {}
    for index, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            reevaluated_message_parts: list[ModelRequestPart] = []
            for part in msg.parts:
                if isinstance(part, SystemPromptPart) and part.dynamic_ref:
                    # Look up the runner by its ref
                    if runner := dynamic_runners.get(part.dynamic_ref):  # pragma: lax no cover
                        if append and runner.context_id is not None:
                            latest[part.dynamic_ref] = part
                        else:
                            # To enable dynamic system prompt refs in future runs, use a placeholder string
                            updated_part_content = await runner.run(run_context)
                            part = SystemPromptPart(
                                updated_part_content or '', dynamic_ref=part.dynamic_ref, metadata=part.metadata
                            )

                reevaluated_message_parts.append(part)

            if reevaluated_message_parts != msg.parts:
                messages[index] = replace(msg, parts=reevaluated_message_parts)

    statements: list[SystemPromptPart] = []
    for ref, part in latest.items():
        runner = dynamic_runners[ref]
        content = await runner.run(run_context) or ''
        if content != part.content:
            statements.append(SystemPromptPart(content, dynamic_ref=ref, context_id=runner.context_id))
    return statements
