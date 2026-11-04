"""`WarnNearLimits` -- injects warnings as the run approaches configured limits."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    SystemPromptPart,
    TextContent,
    UserPromptPart,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._shared import (
    estimate_context_tokens,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

WarningKind = Literal['iterations', 'context_window', 'total_tokens']
"""Categories of limits that can trigger warnings."""

_WARNING_ORDER: tuple[WarningKind, ...] = ('iterations', 'context_window', 'total_tokens')
_WARNING_METADATA = 'pydantic-ai-harness.warn-near-limits.v1'
"""Metadata on the user-prompt warnings older versions sent, which is how they're recognized and stripped."""


@dataclass(frozen=True)
class _Warning:
    kind: WarningKind
    severity: Literal['URGENT', 'CRITICAL']
    details: str


@dataclass
class WarnNearLimits(AbstractCapability[AgentDepsT]):
    """Warns the model when the run approaches configured limits.

    Each warning is enqueued as a [request-scoped][pydantic_ai.messages.SystemPromptPart.scope]
    system prompt: the model sees it for the request it's part of only, and it stays in the
    message history, so the cached prefix and the model's earlier reasoning stay valid.

    Older versions sent the warning as a user prompt marked with metadata instead. Those are
    stripped from the request when a history still holds them; a user prompt that quotes a
    warning is left alone, since they're recognized by that metadata, never by their text.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import WarnNearLimits

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[WarnNearLimits(
                max_iterations=40,
                max_context_tokens=100_000,
            )],
        )
        ```
    """

    max_iterations: int | None = None
    """Maximum allowed requests for the run."""

    max_context_tokens: int | None = None
    """Maximum context-window size to warn against."""

    max_context_fraction: float | None = field(default=None, kw_only=True)
    """Context limit as a fraction of the model's real context window, resolved per request.

    Use this instead of `max_context_tokens` when the same agent runs on models with
    different windows. Mutually exclusive with `max_context_tokens`.
    """

    context_window: int | None = field(default=None, kw_only=True)
    """Window override in tokens. `None` resolves it from the request's model.

    Unlike `fallback_context_window`, this applies whether or not resolution succeeds. Reach
    for it when the registry is confidently wrong: a beta- or tier-gated window it records as
    the maximum, or a self-hosted endpoint whose model id describes someone else's
    deployment. Only consulted alongside `max_context_fraction`."""

    fallback_context_window: int = field(default=DEFAULT_CONTEXT_WINDOW, kw_only=True)
    """Window assumed when the request's model is not in the pricing registry.

    Only consulted alongside `max_context_fraction`. Supply the real number for a deployment
    the registry cannot resolve."""

    max_total_tokens: int | None = None
    """Maximum cumulative run token budget to warn against."""

    warn_on: list[WarningKind] | None = None
    """Which limits should emit warnings.  Defaults to all configured limits."""

    warning_threshold: float = 0.7
    """Fraction of a limit at which warnings begin (between 0 and 1)."""

    critical_remaining_iterations: int = 3
    """Remaining request count at which iteration warnings become CRITICAL."""

    _active_kinds: tuple[WarningKind, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError('max_iterations must be positive.')
        validate_token_trigger(
            self.max_context_tokens,
            self.max_context_fraction,
            self.fallback_context_window,
            self.context_window,
            tokens_name='max_context_tokens',
            fraction_name='max_context_fraction',
        )
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError('max_total_tokens must be positive.')
        if not 0 < self.warning_threshold <= 1:
            raise ValueError('warning_threshold must be between 0 (exclusive) and 1 (inclusive).')
        if self.critical_remaining_iterations < 0:
            raise ValueError('critical_remaining_iterations must be non-negative.')

        # The two context limits are mutually exclusive (checked above), so either one marks
        # the kind as configured.
        configured: dict[WarningKind, bool] = {
            'iterations': self.max_iterations is not None,
            'context_window': self.max_context_tokens is not None or self.max_context_fraction is not None,
            'total_tokens': self.max_total_tokens is not None,
        }
        if not any(configured.values()):
            raise ValueError(
                'At least one of max_iterations, max_context_tokens, max_context_fraction, '
                'or max_total_tokens must be set.'
            )

        if self.warn_on is None:
            self._active_kinds = tuple(k for k in _WARNING_ORDER if configured[k])
        else:
            if not self.warn_on:
                raise ValueError('warn_on must not be empty.')
            for kind in self.warn_on:
                if not configured[kind]:
                    raise ValueError(f'{kind!r} requires its corresponding max_* limit to be configured.')
            self._active_kinds = tuple(dict.fromkeys(self.warn_on))

    # -- internal helpers --

    @staticmethod
    def _is_warning_part(part: ModelRequestPart) -> bool:
        return (
            isinstance(part, UserPromptPart)
            and not isinstance(part.content, str)
            and any(isinstance(item, TextContent) and item.metadata == _WARNING_METADATA for item in part.content)
        )

    def _strip_old_warnings(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        cleaned: list[ModelMessage] = []
        for msg in messages:
            if not isinstance(msg, ModelRequest):
                cleaned.append(msg)
                continue
            # An already-empty request has no warnings to strip; keep it. pydantic-ai's
            # empty-response retry appends a `ModelRequest` with no parts, and dropping it
            # would leave history ending on a `ModelResponse`, which fails the next
            # request's "must end with a ModelRequest" precondition.
            if not msg.parts:
                cleaned.append(msg)
                continue
            parts = [p for p in msg.parts if not self._is_warning_part(p)]
            if not parts:
                continue
            if len(parts) == len(msg.parts):
                cleaned.append(msg)
            else:
                cleaned.append(replace(msg, parts=parts))
        return cleaned

    def _build_iteration_warning(self, ctx: RunContext[AgentDepsT]) -> _Warning | None:
        if self.max_iterations is None or 'iterations' not in self._active_kinds:
            return None
        usage_frac = ctx.usage.requests / self.max_iterations
        if usage_frac < self.warning_threshold:
            return None
        remaining = max(0, self.max_iterations - ctx.usage.requests)
        severity: Literal['URGENT', 'CRITICAL'] = (
            'CRITICAL' if remaining <= self.critical_remaining_iterations else 'URGENT'
        )
        details = f'Iterations: {ctx.usage.requests}/{self.max_iterations} requests used ({usage_frac:.0%}); {remaining} remaining.'
        return _Warning(kind='iterations', severity=severity, details=details)

    def _build_context_warning(self, context_tokens: int, limit: int) -> _Warning | None:
        usage_frac = context_tokens / limit
        if usage_frac < self.warning_threshold:
            return None
        remaining = max(0, limit - context_tokens)
        severity: Literal['URGENT', 'CRITICAL'] = 'CRITICAL' if usage_frac >= 1 else 'URGENT'
        details = f'Context window: {context_tokens}/{limit} tokens used ({usage_frac:.0%}); {remaining} remaining.'
        return _Warning(kind='context_window', severity=severity, details=details)

    def _build_total_tokens_warning(self, ctx: RunContext[AgentDepsT]) -> _Warning | None:
        if self.max_total_tokens is None or 'total_tokens' not in self._active_kinds:
            return None
        total = ctx.usage.total_tokens
        usage_frac = total / self.max_total_tokens
        if usage_frac < self.warning_threshold:
            return None
        remaining = max(0, self.max_total_tokens - total)
        severity: Literal['URGENT', 'CRITICAL'] = 'CRITICAL' if usage_frac >= 1 else 'URGENT'
        details = f'Total tokens: {total}/{self.max_total_tokens} used ({usage_frac:.0%}); {remaining} remaining.'
        return _Warning(kind='total_tokens', severity=severity, details=details)

    @staticmethod
    def _format_warning(warnings: list[_Warning]) -> str:
        severity: Literal['URGENT', 'CRITICAL'] = (
            'URGENT' if all(w.severity == 'URGENT' for w in warnings) else 'CRITICAL'
        )
        guidance = (
            'Complete the current task efficiently and avoid unnecessary tool calls.'
            if severity == 'URGENT'
            else 'Complete the current task immediately and avoid unnecessary tool calls.'
        )
        lines = [f'{severity}: Configured run limits are approaching.']
        lines.extend(f'- {w.details}' for w in warnings)
        lines.append(guidance)
        return '\n'.join(lines)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Strip user-prompt warnings older versions sent, then enqueue a warning if a threshold is crossed.

        Enqueued from `before_model_request`, the warning goes into this request and into the run's
        history, as a `ModelRequest` of its own that's sent merged into the request it follows.
        """
        messages = self._strip_old_warnings(list(request_context.messages))
        request_context.messages = messages
        # A provider-resume turn (`_prepare_resume_request`) ends in a suspended `ModelResponse` and
        # continues the paused request, so the core holds anything enqueued for the next request. That
        # request gets a warning of its own, so warning here would only send two.
        if not messages or not isinstance(messages[-1], ModelRequest):
            return request_context

        active: list[_Warning] = []

        w = self._build_iteration_warning(ctx)
        if w is not None:
            active.append(w)

        if 'context_window' in self._active_kinds:
            context_limit = resolve_token_trigger(
                self.max_context_tokens,
                self.max_context_fraction,
                request_context.model,
                self.fallback_context_window,
                self.context_window,
            )
            if context_limit is not None:  # pragma: no branch -- the kind is only active when one is set
                w = self._build_context_warning(
                    estimate_context_tokens(
                        messages,
                        model_request_parameters=request_context.model_request_parameters,
                    ),
                    context_limit,
                )
                if w is not None:
                    active.append(w)

        w = self._build_total_tokens_warning(ctx)
        if w is not None:
            active.append(w)

        if active:
            order = {k: i for i, k in enumerate(_WARNING_ORDER)}
            active.sort(key=lambda w: order[w.kind])
            ctx.enqueue(SystemPromptPart(self._format_warning(active), scope='request'))
        return request_context
