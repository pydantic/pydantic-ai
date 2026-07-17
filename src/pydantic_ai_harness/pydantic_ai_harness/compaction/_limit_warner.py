"""`LimitWarner` -- injects warnings as the run approaches configured limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._shared import estimate_token_count

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

WarningKind = Literal['iterations', 'context_window', 'total_tokens']
"""Categories of limits that can trigger warnings."""

_WARNING_ORDER: tuple[WarningKind, ...] = ('iterations', 'context_window', 'total_tokens')
_MARKER = '[LimitWarner]'


@dataclass(frozen=True)
class _Warning:
    kind: WarningKind
    severity: Literal['URGENT', 'CRITICAL']
    details: str


@dataclass
class LimitWarner(AbstractCapability[AgentDepsT]):
    """Injects a warning message when the agent approaches configured limits.

    The warning is appended as a trailing ``ModelRequest`` with a
    ``UserPromptPart`` so that the model treats it as a distinct user turn
    (models tend to pay more attention to user messages than system messages).

    Previous warnings injected by this capability are stripped before deciding
    whether to inject a new one.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import LimitWarner

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[LimitWarner(
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
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError('max_context_tokens must be positive.')
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError('max_total_tokens must be positive.')
        if not 0 < self.warning_threshold <= 1:
            raise ValueError('warning_threshold must be between 0 (exclusive) and 1 (inclusive).')
        if self.critical_remaining_iterations < 0:
            raise ValueError('critical_remaining_iterations must be non-negative.')

        configured: dict[WarningKind, int | None] = {
            'iterations': self.max_iterations,
            'context_window': self.max_context_tokens,
            'total_tokens': self.max_total_tokens,
        }
        if all(v is None for v in configured.values()):
            raise ValueError('At least one of max_iterations, max_context_tokens, or max_total_tokens must be set.')

        if self.warn_on is None:
            self._active_kinds = tuple(k for k in _WARNING_ORDER if configured[k] is not None)
        else:
            if not self.warn_on:
                raise ValueError('warn_on must not be empty.')
            for kind in self.warn_on:
                if configured[kind] is None:
                    raise ValueError(f'{kind!r} requires its corresponding max_* limit to be configured.')
            self._active_kinds = tuple(dict.fromkeys(self.warn_on))

    # -- internal helpers --

    @staticmethod
    def _is_marker_part(part: Any) -> bool:
        if isinstance(part, SystemPromptPart):
            return _MARKER in part.content
        if isinstance(part, UserPromptPart) and isinstance(part.content, str):
            return _MARKER in part.content
        return False

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
            parts = [p for p in msg.parts if not self._is_marker_part(p)]
            if not parts:
                continue
            if len(parts) == len(msg.parts):
                cleaned.append(msg)
            else:
                cleaned.append(ModelRequest(parts=parts))
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

    def _build_context_warning(self, context_tokens: int) -> _Warning | None:
        if self.max_context_tokens is None or 'context_window' not in self._active_kinds:
            return None  # pragma: no cover
        usage_frac = context_tokens / self.max_context_tokens
        if usage_frac < self.warning_threshold:
            return None
        remaining = max(0, self.max_context_tokens - context_tokens)
        severity: Literal['URGENT', 'CRITICAL'] = 'CRITICAL' if usage_frac >= 1 else 'URGENT'
        details = f'Context window: {context_tokens}/{self.max_context_tokens} tokens used ({usage_frac:.0%}); {remaining} remaining.'
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
        lines = [_MARKER, f'{severity}: Configured run limits are approaching.']
        lines.extend(f'- {w.details}' for w in warnings)
        lines.append(guidance)
        return '\n'.join(lines)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Strip old warnings, then inject a new one if thresholds are exceeded."""
        messages = self._strip_old_warnings(list(request_context.messages))

        active: list[_Warning] = []

        w = self._build_iteration_warning(ctx)
        if w is not None:
            active.append(w)

        if self.max_context_tokens is not None and 'context_window' in self._active_kinds:
            context_tokens = estimate_token_count(messages)
            w = self._build_context_warning(context_tokens)
            if w is not None:
                active.append(w)

        w = self._build_total_tokens_warning(ctx)
        if w is not None:
            active.append(w)

        if not active:
            request_context.messages = messages
            return request_context

        order = {k: i for i, k in enumerate(_WARNING_ORDER)}
        active.sort(key=lambda w: order[w.kind])
        warning_text = self._format_warning(active)
        messages.append(ModelRequest(parts=[UserPromptPart(content=warning_text)]))

        request_context.messages = messages
        return request_context
