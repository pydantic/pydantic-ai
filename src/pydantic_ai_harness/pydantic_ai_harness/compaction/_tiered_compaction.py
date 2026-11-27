"""`TieredCompaction` -- escalation orchestrator over a sequence of strategies."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._receipts import (
    drain_receipts,
    open_receipt_scope,
    record_receipt,
    reset_receipt_scope,
)
from pydantic_ai_harness.compaction._shared import (
    CompactionStrategy,
    SupportsFocus,
    compact_with_span,
    context_for_request,
    estimate_context_tokens,
    estimate_token_count,
    has_durable_operations,
    record_compaction_reclaim,
    record_failed_compaction,
    record_span_attributes,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import AbstractModel, ModelRequestContext, ModelRequestParameters

_logger = logging.getLogger(__name__)
_escalating: ContextVar[bool] = ContextVar('pydantic_ai_harness.compaction.escalating', default=False)


@dataclass
class TieredCompaction(AbstractCapability[AgentDepsT]):
    """Escalation orchestrator over a sequence of compaction strategies.

    Runs each tier in order, re-measuring the token count after each, and stops as soon as
    the conversation fits `target_tokens`.  Order tiers cheap-to-expensive (e.g. clear
    tool results, deduplicate reads, then summarize) so the expensive summarization tier is
    only reached when the cheap passes cannot reclaim enough.

    Each tier's own trigger is bypassed -- `TieredCompaction` drives the tiers directly via
    their `compact` method and decides when to stop -- so tiers need none.

    On its own trigger, a tier that raises a `skip_on` exception is skipped and escalation
    moves on to the next tier; other exceptions propagate. A direct `compact` call lets
    every exception propagate.

    Under durable execution, when a tier has durable operations of its own (the summary call of
    `SummarizingCompaction`), the tiers run as this capability's `compact` operation, because a
    tier is not registered on the agent and its own operations do not bind. Under Temporal that
    is an activity, whose context omits `tracer` and `capabilities`, so a tier that reads them
    (`RelevanceCompaction`, or `receipts=True` on a summarizing or sliding-window tier) fails
    there. Tiers without durable operations run in workflow code, as before.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import (
            ClearToolResults,
            SummarizingCompaction,
            TieredCompaction,
        )

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[TieredCompaction(
                tiers=[
                    ClearToolResults(),
                    SummarizingCompaction(model='openai:gpt-4o-mini'),
                ],
                target_tokens=100_000,
            )],
        )
        ```
    """

    tiers: Sequence[CompactionStrategy[AgentDepsT]]
    """Strategies to apply in order, cheap-to-expensive.  The last is typically a summarizer."""

    target_tokens: int | None = None
    """Stop escalating once the estimated token count is at or below this value.

    Mutually exclusive with `target_fraction`; exactly one of the two must be set.
    """

    target_fraction: float | None = field(default=None, kw_only=True)
    """Target expressed as a fraction of the model's context window, resolved per request.

    Use this instead of `target_tokens` when the same agent runs on models with
    different windows. Mutually exclusive with `target_tokens`.
    """

    context_window: int | None = field(default=None, kw_only=True)
    """Window override in tokens. `None` resolves it from the request's model.

    Unlike `fallback_context_window`, this applies whether or not resolution succeeds. Reach
    for it when the registry is confidently wrong: a beta- or tier-gated window it records as
    the maximum, or a self-hosted endpoint whose model id describes someone else's
    deployment. Only consulted alongside `target_fraction`."""

    fallback_context_window: int = field(default=DEFAULT_CONTEXT_WINDOW, kw_only=True)
    """Window assumed when the model is not in the pricing registry.

    Only consulted alongside `target_fraction`. Supply the real number for a deployment the
    registry cannot resolve."""

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When `None`, uses the built-in text estimate, which counts dense text such as non-ASCII
    scripts, base64 and compact JSON higher than prose.
    """

    skip_on: tuple[type[Exception], ...] = field(
        default=(ModelAPIError, FallbackExceptionGroup, UnexpectedModelBehavior), kw_only=True
    )
    """Exceptions that skip a failing tier on this capability's own trigger, rather than failing the run.

    Defaults to model failures, so programming errors and usage limits propagate. Pass `()` to
    propagate every exception. Unlike `FallbackCompaction.fallback_on`, this includes
    `UnexpectedModelBehavior`, which a summary tier raises when it exhausts its retries: the next
    tier can still shrink the history without that tier's model.
    """

    # Override the inherited default ID because durable-operation recovery needs a stable identity.
    id: str | None = field(default='tiered_compaction', kw_only=True)

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError('tiers must not be empty.')
        if not all(isinstance(error, type) and issubclass(error, Exception) for error in self.skip_on):
            raise ValueError('skip_on must contain only Exception subclasses.')
        if self.target_tokens is None and self.target_fraction is None:
            raise ValueError('One of target_tokens or target_fraction must be set.')
        validate_token_trigger(
            self.target_tokens,
            self.target_fraction,
            self.fallback_context_window,
            self.context_window,
            tokens_name='target_tokens',
            fraction_name='target_fraction',
        )

    def with_focus(self, focus: str) -> TieredCompaction[AgentDepsT]:
        """Return a copy whose focus-capable tiers prioritize `focus`.

        A tiered strategy is focusable when any of its tiers is: the summarizing tier writes
        the prose, so the hint has to reach it rather than stopping at this wrapper. Tiers that
        cannot honour a focus are passed through unchanged.
        """
        return replace(
            self,
            tiers=[tier.with_focus(focus) if isinstance(tier, SupportsFocus) else tier for tier in self.tiers],
        )

    def _target(self, model: AbstractModel | str) -> int | None:
        """Absolute token target, resolved against *model* when expressed as a fraction.

        Returns `None` when *model* is realtime (an `AbstractModel` that is not a request-response
        `Model`): a realtime session never compacts, so this only guards the widened
        `RunContext.model` type. `__post_init__` requires one of `target_tokens` /
        `target_fraction`, so a request-response `Model` never resolves to `None`. #585
        """
        return resolve_token_trigger(
            self.target_tokens, self.target_fraction, model, self.fallback_context_window, self.context_window
        )

    async def _escalate(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        target: int,
        model_request_parameters: ModelRequestParameters | None = None,
        skip_on: tuple[type[Exception], ...] = (),
    ) -> list[ModelMessage]:
        """Apply tiers in order until the history fits *target* or tiers run out, skipping any raising *skip_on*."""
        # The usage anchor describes the request as it was sent, so a tier's rewrite of older
        # messages is invisible to re-anchoring. Subtract the estimated tokens of the text a
        # tier removed from the anchored baseline instead: the reclaim is measured only on
        # message text, so the baseline's fixed overhead (tool definitions, instructions) is
        # never treated as compacted away. On content the estimator undercounts, this
        # understates the reclaim and escalates a tier early -- the cheap direction.
        baseline = estimate_context_tokens(messages, self.tokenizer, model_request_parameters=model_request_parameters)
        nested = _escalating.get()
        if any(has_durable_operations(tier) for tier in self.tiers):
            messages, estimate = await self._run_tiers_durably(ctx, messages, target, baseline, bool(skip_on))
        else:
            messages, estimate = await self._run_tiers(messages, ctx, target, baseline, skip_on)
        if estimate > target and not nested:
            # Every tier ran and the history is still over target: what the tiers keep (a `keep_tokens`
            # floor, the first user message, the current turn) costs more than the target allows.
            record_span_attributes(
                {
                    'compaction.target_unreached': True,
                    'compaction.target_tokens': target,
                    'compaction.estimate': estimate,
                }
            )
            _logger.warning(
                '`TieredCompaction` ran every tier and left the history at an estimated %d tokens, above its '
                'target of %d. The messages its tiers keep cost more than the target.',
                estimate,
                target,
            )
        return messages

    @durable_operation('compact')
    async def _run_tiers_durably(
        self, ctx: RunContext[AgentDepsT], messages: list[ModelMessage], target: int, baseline: int, skip: bool
    ) -> tuple[list[ModelMessage], int]:
        """The tiers' result and its estimate, as a durable operation of this capability.

        A tier is not registered on the agent, so a durable operation of its own (the summary call
        of `SummarizingCompaction`) does not bind; this one does, and the tiers run inline within
        it, so a replay reuses the recorded history instead of calling a model again. *skip*
        selects `skip_on`, which is not serializable, as the exceptions to skip a tier on.
        """
        return await self._run_tiers(messages, ctx, target, baseline, self.skip_on if skip else ())

    async def _run_tiers(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        target: int,
        baseline: int,
        skip_on: tuple[type[Exception], ...],
    ) -> tuple[list[ModelMessage], int]:
        """Apply tiers in order until the estimate fits *target*, returning the history and its estimate."""
        original = messages
        heuristic_baseline = estimate_token_count(messages, self.tokenizer)
        estimate = baseline
        token = _escalating.set(True)
        try:
            for tier in self.tiers:
                if estimate <= target:
                    break
                compacted = await self._run_tier(tier, messages, ctx, skip_on)
                if compacted is None:
                    continue
                # Before the next stop decision, so escalation measures the history it would return.
                messages = reinject_pinned(original, compacted)
                after = estimate_token_count(messages, self.tokenizer)
                reclaimed = heuristic_baseline - after
                # The two counts are on different bases, and the anchor's can be the smaller one: content a
                # provider's tokenizer merges costs it far fewer tokens than the heuristic charges it. Crediting
                # that text's removal against the smaller baseline subtracts more than the anchor ever counted;
                # flooring the result at 0 would then report the history the tier just rewrote as fitting while
                # it still holds `after` tokens. Keep the surviving text's own count as the floor instead.
                estimate = max(baseline - reclaimed, after)
        finally:
            _escalating.reset(token)
        return messages, estimate

    async def _run_tier(
        self,
        tier: CompactionStrategy[AgentDepsT],
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        skip_on: tuple[type[Exception], ...],
    ) -> list[ModelMessage] | None:
        """*tier*'s result, or `None` when it raised *skip_on*, recorded and with its receipts discarded."""
        token = open_receipt_scope()
        try:
            compacted = await tier.compact(messages, ctx)
            receipts = drain_receipts()
        except skip_on as error:
            strategy = type(tier).__name__
            record_failed_compaction(
                ctx, strategy=strategy, skipped='tier_failed', error=error, messages=messages, tokenizer=self.tokenizer
            )
            _logger.warning('`TieredCompaction` skipped its `%s` tier, which raised %s', strategy, type(error).__name__)
            return None
        finally:
            reset_receipt_scope(token)
        for receipt in receipts:
            record_receipt(receipt)
        return compacted

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Apply tiers in order until the history fits the target or tiers run out."""
        target = self._target(ctx.model)
        if target is None:
            # A realtime model has no token target to escalate toward; leave the history as-is.
            return messages
        return await self._escalate(messages, ctx, target)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Escalate through the tiers when the conversation exceeds the target."""
        messages: list[ModelMessage] = list(ctx.messages)
        # The tiers get the request's context, not the run's: a tier that resolves a model --
        # a summarizing one, or a `TieredCompaction` nested inside this one -- has to reach the
        # same conclusion this gate did.
        request_ctx = context_for_request(ctx, request_context)
        # Resolved once, so the gate and the escalation loop cannot disagree about the target.
        target = self._target(request_ctx.model)
        if target is None or (
            estimate_context_tokens(
                messages,
                self.tokenizer,
                model_request_parameters=request_context.model_request_parameters,
            )
            <= target
        ):
            return request_context
        compacted = await compact_with_span(
            request_ctx,
            strategy='TieredCompaction',
            messages=messages,
            compact=lambda: self._escalate(
                messages, request_ctx, target, request_context.model_request_parameters, self.skip_on
            ),
            tokenizer=self.tokenizer,
        )
        ctx.messages[:] = compacted
        record_compaction_reclaim(
            ctx,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        return replace(request_context, messages=compacted)
