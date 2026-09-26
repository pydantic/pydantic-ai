"""`FallbackCompaction` -- try compaction strategies until one succeeds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._shared import (
    CompactionStrategy,
    SupportsFocus,
    compact_with_span,
    context_for_request,
    estimate_context_tokens,
    estimate_token_count,
    record_compaction_reclaim,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class FallbackCompaction(AbstractCapability[AgentDepsT]):
    """Try compaction strategies in order, falling back when one raises.

    Each attempt receives a fresh list containing the original message objects.
    `fallback_on` defaults to model API errors, including an exhausted `FallbackModel`, so
    programming errors pass through. The last matching exception is re-raised if every strategy
    fails. Entries must derive from `Exception`, so cancellation and other `BaseException`
    subclasses are never caught.
    """

    fallback_chain: Sequence[CompactionStrategy[AgentDepsT]]
    fallback_on: tuple[type[Exception], ...] = (ModelAPIError, FallbackExceptionGroup)

    max_tokens: int | None = field(default=None, kw_only=True)
    """Trigger when estimated context tokens exceed this value. `None` disables."""

    max_fraction: float | None = field(default=None, kw_only=True)
    """Trigger above this fraction of the request model's window, exclusive with `max_tokens`."""

    context_window: int | None = field(default=None, kw_only=True)
    """Window override in tokens, consulted only alongside `max_fraction`."""

    fallback_context_window: int = field(default=DEFAULT_CONTEXT_WINDOW, kw_only=True)
    """Window assumed when the request model cannot be resolved, alongside `max_fraction`."""

    tokenizer: Callable[[str], int] | None = field(default=None, kw_only=True)
    """Optional token counter; defaults to the characters-per-token heuristic."""

    def __post_init__(self) -> None:
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)
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

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Run the chain above its threshold; without one, leave automatic requests alone."""
        request_ctx = context_for_request(ctx, request_context)
        trigger = resolve_token_trigger(
            self.max_tokens, self.max_fraction, request_ctx.model, self.fallback_context_window, self.context_window
        )
        messages = list(request_context.messages)
        if (
            trigger is None
            or estimate_context_tokens(
                messages, self.tokenizer, model_request_parameters=request_context.model_request_parameters
            )
            <= trigger
        ):
            return request_context
        compacted = await compact_with_span(
            request_ctx,
            strategy='FallbackCompaction',
            messages=messages,
            compact=lambda: self._compact_pinned(messages, request_ctx),
            tokenizer=self.tokenizer,
        )
        record_compaction_reclaim(
            request_context,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        request_context.messages = compacted
        return request_context

    async def _compact_pinned(self, messages: list[ModelMessage], ctx: RunContext[AgentDepsT]) -> list[ModelMessage]:
        return reinject_pinned(messages, await self.compact(messages, ctx))


def _is_exception_type(value: object) -> bool:
    return isinstance(value, type) and issubclass(value, Exception)
