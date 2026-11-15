"""`FallbackCompaction` -- try compaction strategies until one succeeds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import NoOpTracer

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import AbstractModel, Model
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness._durable import in_durable_context
from pydantic_ai_harness.compaction._background import BackgroundCompaction
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._shared import (
    BackgroundOutcome,
    CompactionStrategy,
    SupportsFocus,
    compact_if_smaller,
    compact_with_span,
    context_for_request,
    estimate_context_tokens,
    estimate_token_count,
    record_compaction_reclaim,
    resolve_token_trigger,
    validate_token_trigger,
)
from pydantic_ai_harness.compaction._summarizing_compaction import SummarizingCompaction

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.run import AgentRunResult

MAX_BACKGROUND_COMPACTIONS = 8
"""How many conversations one `FallbackCompaction` holds a background compaction for; starting another cancels the oldest.

An application that never calls `discard_background_compaction` for a conversation that ended, such as a
long-lived agent serving one-shot runs, then holds a bounded number of history copies.
"""


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
    model request and at the end of each run, without blocking either; a run that ends already over the
    trigger starts none, and its next request compacts as usual. Its result is held, not applied,
    until the trigger fires (or `take_background_compaction()` is called); it then replaces the history it
    was made from, keeping every message added since verbatim. A result that no longer applies is dropped,
    and the chain runs as it would without this option. The work can outlive the run that started it, so
    call `discard_background_compaction()` when a conversation ends and `aclose()` when the agent is done
    with this instance. Not supported under durable execution.
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

    @property
    def background_conversations(self) -> frozenset[str | None]:
        """The conversations a background compaction is running or held for, by `RunContext.conversation_id`."""
        return frozenset(self._background)

    async def take_background_compaction(
        self,
        messages: list[ModelMessage],
        *,
        model: Model | str,
        conversation_id: str | None,
        tracer: Tracer | None = None,
    ) -> list[ModelMessage] | None:
        """Outside a run, *messages* with the conversation's background compaction applied, or `None`.

        Waits for a compaction still running. Returns `None` when there is none, or when it no longer
        applies to *messages* and *model*, in which case it is dropped; the caller then compacts as it
        would have. Like `compact_now`, a compaction that applies emits a `compact_messages` span on
        *tracer*. The conversation holds no background compaction afterwards.
        """
        from pydantic_ai.models import infer_model

        resolved = infer_model(model) if isinstance(model, str) else model
        held, outcome = await self._take_background(messages, resolved, conversation_id)
        if held is None:
            return None
        ctx = RunContext[None](
            deps=None,
            model=resolved,
            usage=RunUsage(),
            conversation_id=conversation_id,
            tracer=tracer if tracer is not None else NoOpTracer(),
        )

        async def applied() -> list[ModelMessage]:
            return held

        return await compact_with_span(
            ctx,
            strategy='FallbackCompaction',
            messages=messages,
            compact=applied,
            tokenizer=self.tokenizer,
            background=outcome,
        )

    async def discard_background_compaction(self, conversation_id: str | None) -> None:
        """Cancel the conversation's background compaction, if any, and drop its result."""
        job = self._background.pop(conversation_id, None)
        if job is not None:
            await job.cancel()

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
        """Return the first successful compaction result."""
        return await self._compact(messages, ctx)

    async def _compact(
        self, messages: list[ModelMessage], ctx: RunContext[AgentDepsT], *, smaller_only: bool = False
    ) -> list[ModelMessage]:
        """The first successful compaction result; with *smaller_only*, a summary that isn't smaller falls through.

        When no strategy succeeds and one of them was such a summary, *messages* is returned unchanged:
        that strategy completed, it just didn't help, so an error from another one isn't the outcome.
        """
        last_error: Exception | None = None
        rejected = False
        for strategy in self.fallback_chain:
            try:
                if not (smaller_only and isinstance(strategy, SummarizingCompaction)):
                    return await strategy.compact(list(messages), ctx)
                compacted = await compact_if_smaller(
                    ctx,
                    strategy='SummarizingCompaction',
                    messages=messages,
                    compact=partial(strategy.compact, list(messages), ctx),
                    tokenizer=self.tokenizer,
                )
                if compacted is not None:
                    return compacted
                rejected = True
            except self.fallback_on as error:
                last_error = error
        if rejected:
            return messages
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
        held, outcome = await self._take_background(
            messages, request_ctx.model, request_ctx.conversation_id, parameters
        )
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

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reject background compaction inside a durable workflow or flow, before any request."""
        if self.background_start is not None and in_durable_context(ctx):
            raise UserError(
                '`FallbackCompaction(background_start=...)` cannot be used inside a durable workflow or flow: '
                'the background compaction outlives the step that starts it, so it is not recorded and would '
                'repeat on replay. Leave `background_start` unset under durable execution.'
            )

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
        compacted = held if held is not None else await self._compact(messages, ctx, smaller_only=True)
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
        if tokens <= trigger * self.background_start:
            if job is not None and job.same_start(messages):
                await self.discard_background_compaction(ctx.conversation_id)
            return
        # The job this one replaces frees its own slot.
        while len(self._background) - (ctx.conversation_id in self._background) >= MAX_BACKGROUND_COMPACTIONS:
            await self.discard_background_compaction(next(iter(self._background)))
        # Replace and register without awaiting in between, so a concurrent run on the same conversation
        # replaces this one in turn instead of being overwritten untracked.
        replaced = self._background.pop(ctx.conversation_id, None)
        self._background[ctx.conversation_id] = BackgroundCompaction.start(self.fallback_chain[0], messages, ctx)
        if replaced is not None:
            await replaced.cancel()

    async def _take_background(
        self,
        messages: list[ModelMessage],
        model: AbstractModel,
        conversation_id: str | None,
        parameters: ModelRequestParameters | None = None,
    ) -> tuple[list[ModelMessage] | None, BackgroundOutcome | None]:
        """Take the conversation's background compaction, waiting for it, and apply it when it still holds.

        It holds when the history only grew since it started, on the same model, and the compacted
        history, the messages added since included, fits under the trigger.
        """
        job = self._background.get(conversation_id)
        if job is None:
            return None, None
        if not job.matches(messages, model):
            await self.discard_background_compaction(conversation_id)
            return None, 'stale'
        # Still tracked while it runs, so `aclose()` can cancel it; a cancelled caller leaves it for a later request.
        compacted = await job.result()
        if self._background.get(conversation_id) is job:
            del self._background[conversation_id]
        if compacted is None:
            return None, 'failed'
        held = job.applied_to(compacted, messages)
        trigger = resolve_token_trigger(
            self.max_tokens, self.max_fraction, model, self.fallback_context_window, self.context_window
        )
        assert trigger is not None, 'a background compaction needs a trigger'
        reclaimed = estimate_token_count(messages, self.tokenizer) - estimate_token_count(held, self.tokenizer)
        tokens = estimate_context_tokens(messages, self.tokenizer, model_request_parameters=parameters) - reclaimed
        if reclaimed <= 0 or tokens > trigger:
            return None, 'stale'
        return held, 'used'


def _is_exception_type(value: object) -> bool:
    return isinstance(value, type) and issubclass(value, Exception)
