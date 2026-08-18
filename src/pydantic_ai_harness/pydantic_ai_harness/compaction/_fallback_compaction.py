"""`FallbackCompaction` -- try compaction strategies until one succeeds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Generic

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._shared import CompactionStrategy, SupportsFocus


@dataclass
class FallbackCompaction(Generic[AgentDepsT]):
    """Try compaction strategies in order, falling back when one raises.

    Each attempt receives a fresh list containing the original message objects.
    `fallback_on` defaults to model API errors, including an exhausted `FallbackModel`, so
    programming errors pass through. The last matching exception is re-raised if every strategy
    fails. Entries must derive from `Exception`, so cancellation and other `BaseException`
    subclasses are never caught.
    """

    fallback_chain: Sequence[CompactionStrategy[AgentDepsT]]
    fallback_on: tuple[type[Exception], ...] = (ModelAPIError, FallbackExceptionGroup)

    def __post_init__(self) -> None:
        if not self.fallback_chain:
            raise ValueError('fallback_chain must not be empty.')
        if not self.fallback_on:
            raise ValueError('fallback_on must not be empty.')
        if not all(_is_exception_type(exception) for exception in self.fallback_on):
            raise ValueError('fallback_on must contain only Exception subclasses.')

    def with_focus(self, focus: str) -> FallbackCompaction[AgentDepsT]:
        """Return a copy that forwards focus to strategies that support it."""
        return replace(
            self,
            fallback_chain=[
                strategy.with_focus(focus) if isinstance(strategy, SupportsFocus) else strategy
                for strategy in self.fallback_chain
            ],
        )

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Return the first successful compaction result."""
        last_error: Exception | None = None
        for strategy in self.fallback_chain:
            try:
                return await strategy.compact(list(messages), ctx)
            except self.fallback_on as error:
                last_error = error
        assert last_error is not None
        raise last_error


def _is_exception_type(value: object) -> bool:
    return isinstance(value, type) and issubclass(value, Exception)
