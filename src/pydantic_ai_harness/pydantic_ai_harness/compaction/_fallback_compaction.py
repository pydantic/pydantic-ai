"""`FallbackCompaction` -- try compaction strategies until one succeeds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction._background import BackgroundCompaction, BackgroundOutcome
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
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.run import AgentRunResult


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

    background_start: float | None = field(default=None, kw_only=True)
    """Experimental: start the first strategy in the background above this fraction of the trigger.

    `None` (the default) disables it. With `0.85`, the first strategy starts compacting a copy of the
    history once the context passes 85% of the `max_tokens` or `max_fraction` trigger, checked before each
    model request and at the end of each run, without blocking either. Its result is held, not applied,
    until the trigger fires (or `compact()` is called); it then replaces the history it was made from,
    keeping every message added since verbatim. A result that has gone stale is dropped, and the chain
    runs as it would without this option. The work outlives the run that started it, so it can finish
    between runs; call `aclose()` when the agent is done with this instance.
    """

    _background: dict[str | None, BackgroundCompaction] = field(
        default_factory=dict[str | None, BackgroundCompaction], init=False, repr=False, compare=False
    )
    """The background compaction of each conversation, by `RunContext.conversation_id`."""

    def __post_init__(self) -> None:
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)
        if self.background_start is not None:
            if not 0 < self.background_start < 1:
                raise ValueError('background_start must be greater than 0 and less than 1.')
            if self.max_tokens is None and self.max_fraction is None:
                raise ValueError('background_start needs a max_tokens or max_fraction trigger.')
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

    def has_background_compaction(self, conversation_id: str | None) -> bool:
        """Whether a background compaction is running or held for the conversation with this ID."""
        return conversation_id in self._background

    async def aclose(self) -> None:
        """Cancel every background compaction this instance is running, and drop the held results."""
        jobs = list(self._background.values())
        self._background.clear()
        for job in jobs:
            await job.cancel()

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Return a held background compaction that still applies, else the first successful compaction result."""
        held, _ = await self._take_background(messages, ctx)
        if held is not None:
            return held
        return await self._compact_chain(messages, ctx)

    async def _compact_chain(self, messages: list[ModelMessage], ctx: RunContext[AgentDepsT]) -> list[ModelMessage]:
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
        messages: list[ModelMessage] = list(ctx.messages)
        if trigger is None:
            return request_context
        parameters = request_context.model_request_parameters
        tokens = estimate_context_tokens(messages, self.tokenizer, model_request_parameters=parameters)
        if tokens <= trigger:
            await self._check_background(messages, tokens, trigger, request_ctx)
            return request_context
        held, outcome = await self._take_background(messages, request_ctx, parameters)
        compacted = await compact_with_span(
            request_ctx,
            strategy='FallbackCompaction',
            messages=messages,
            compact=lambda: self._compact_pinned(messages, request_ctx, held),
            tokenizer=self.tokenizer,
            background=outcome,
        )
        ctx.messages[:] = compacted
        record_compaction_reclaim(
            ctx,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        return replace(request_context, messages=compacted)

    async def after_run(self, ctx: RunContext[AgentDepsT], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        """Start a background compaction for the finished history, so it can finish before the next run."""
        if self.background_start is not None:
            trigger = resolve_token_trigger(
                self.max_tokens, self.max_fraction, ctx.model, self.fallback_context_window, self.context_window
            )
            messages = result.all_messages()
            tokens = estimate_context_tokens(messages, self.tokenizer)
            if trigger is not None and tokens <= trigger:
                await self._check_background(messages, tokens, trigger, ctx)
        return result

    async def _compact_pinned(
        self, messages: list[ModelMessage], ctx: RunContext[AgentDepsT], held: list[ModelMessage] | None
    ) -> list[ModelMessage]:
        compacted = held if held is not None else await self._compact_chain(messages, ctx)
        return reinject_pinned(messages, compacted)

    async def _check_background(
        self, messages: list[ModelMessage], tokens: int, trigger: int, ctx: RunContext[AgentDepsT]
    ) -> None:
        """Below the trigger: start a background compaction above `background_start`, and drop a stale one.

        A held compaction whose conversation was rewritten (same first message, different history) or
        moved to another model is dropped at once. One for a history that starts elsewhere is left for
        its owner: a side run, such as one started with an empty `message_history`, can share the
        conversation ID. Above `background_start` any mismatch is replaced by a fresh start.
        """
        if self.background_start is None:
            return
        job = self._background.get(ctx.conversation_id)
        if job is not None and job.matches(messages, ctx.model):
            return
        above = tokens > trigger * self.background_start
        if job is not None and (above or job.same_start(messages)):
            del self._background[ctx.conversation_id]
            await self._drop(job, ctx)
        if above:
            self._background[ctx.conversation_id] = BackgroundCompaction.start(self.fallback_chain[0], messages, ctx)

    async def _take_background(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        parameters: ModelRequestParameters | None = None,
    ) -> tuple[list[ModelMessage] | None, BackgroundOutcome | None]:
        """Take the conversation's background compaction, waiting for it, and apply it when it still holds.

        It holds when the history only grew since it started, on the same model, and the compacted
        history, the messages added since included, fits under `background_start` of the trigger, so
        too much new content leaves it unused. Its usage is added to this run's either way.
        """
        job = self._background.pop(ctx.conversation_id, None)
        if job is None:
            return None, None
        if not job.matches(messages, ctx.model):
            await self._drop(job, ctx)
            return None, 'stale'
        try:
            compacted = await job.result()
        except BaseException:
            # A caller cancelled while waiting leaves the work to finish for a later request.
            self._background.setdefault(ctx.conversation_id, job)
            raise
        await self._drop(job, ctx)
        if compacted is None:
            return None, 'failed'
        held = job.applied_to(compacted, messages)
        trigger = resolve_token_trigger(
            self.max_tokens, self.max_fraction, ctx.model, self.fallback_context_window, self.context_window
        )
        assert trigger is not None and self.background_start is not None, 'a background compaction needs both'
        reclaimed = estimate_token_count(messages, self.tokenizer) - estimate_token_count(held, self.tokenizer)
        tokens = estimate_context_tokens(messages, self.tokenizer, model_request_parameters=parameters) - reclaimed
        if reclaimed <= 0 or tokens > trigger * self.background_start:
            return None, 'stale'
        return held, 'used'

    async def _drop(self, job: BackgroundCompaction, ctx: RunContext[AgentDepsT]) -> None:
        """Cancel *job* if it is still running, and add what it spent to this run's usage."""
        await job.cancel()
        ctx.usage.incr(job.usage)


def _is_exception_type(value: object) -> bool:
    return isinstance(value, type) and issubclass(value, Exception)
