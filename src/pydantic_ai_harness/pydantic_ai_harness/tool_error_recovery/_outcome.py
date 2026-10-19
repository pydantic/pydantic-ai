"""The recovery verdict (`RecoveryOutcome`) and the default error classifications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, assert_never

import httpx

from pydantic_ai.exceptions import UserError

DEFAULT_BUG_TYPES: tuple[type[BaseException], ...] = (
    TypeError,
    AttributeError,
    NameError,
    KeyError,
    IndexError,
    AssertionError,
    ImportError,
)
"""Programming errors: always propagate, never silently recovered."""

DEFAULT_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    httpx.TimeoutException,
    httpx.ConnectError,
)
"""Errors retried by default: only failures where the request never executed.

Deliberately narrow -- no `OSError` (`FileNotFoundError`/`PermissionError` are not
transient), no mid-flight `ReadError` (could double-execute a non-idempotent tool),
no `HTTPStatusError` (retry semantics are a product decision). httpx ships with
pydantic-ai-slim, so its types cost no dependency.
"""


@dataclass(frozen=True, kw_only=True)
class RecoveryOutcome:
    """What to do with a failed tool call.

    Construct one with the classmethods -- `RecoveryOutcome.retry()`,
    `RecoveryOutcome.inform()`, `RecoveryOutcome.fallback()`,
    `RecoveryOutcome.propagate()` -- rather than the raw fields.

    `inform` reports a terminal failure to the model (via `ToolFailed`, `outcome='failed'`);
    `fallback` returns a substitute value as a normal (successful) tool result. Both let the
    model continue; neither is a `ModelRetry` (no retry, no retry-budget cost).
    """

    action: Literal['retry', 'inform', 'fallback', 'propagate']
    """How the capability should react to the failure."""

    max_attempts: int = 1
    """For `retry`, total attempts including the first."""

    backoff: float | Callable[[int], float] = 0.0
    """For `retry`, base delay in seconds (`base * 2 ** attempt`) or a callable of the 0-based attempt."""

    on_exhausted: RecoveryOutcome | None = None
    """For `retry`, the terminal outcome once attempts run out (default: `inform`)."""

    label: str | None = None
    """Curated short text: shown instead of the raw error, and the trace/metric dimension."""

    value: Any = None
    """For `fallback`, the substitute tool result (`None` is a valid substitute)."""

    expose_message: bool = False
    """For `inform`, append the capped `str(error)` to the model-facing text.

    The explicit per-case opt-out of the leak-free default, for exceptions whose text
    is meant for the model. `label` stays the low-cardinality metric dimension instead
    of carrying the dynamic message.
    """

    def __post_init__(self) -> None:
        """Reject field combinations the four-outcome contract does not allow."""
        if self.expose_message and self.action != 'inform':
            raise UserError("RecoveryOutcome.expose_message is only valid for action='inform'.")
        match self.action:
            case 'retry':
                if self.max_attempts < 1:
                    raise UserError(f'RecoveryOutcome.retry() requires max_attempts >= 1, got {self.max_attempts}.')
                if self.on_exhausted is not None and self.on_exhausted.action == 'retry':
                    raise UserError('RecoveryOutcome.retry() cannot use another retry as `on_exhausted`.')
            case 'inform':
                if self.value is not None:
                    raise UserError("RecoveryOutcome(action='inform') must not set `value`.")
            case 'propagate':
                if self.label is not None or self.value is not None:
                    raise UserError("RecoveryOutcome(action='propagate') must not set `label` or `value`.")
            case 'fallback':
                # Any `value` is valid -- `None` is a legitimate substitute result.
                pass
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(self.action)

    @classmethod
    def retry(
        cls,
        max_attempts: int = 3,
        *,
        backoff: float | Callable[[int], float] = 0.0,
        on_exhausted: RecoveryOutcome | None = None,
    ) -> RecoveryOutcome:
        """Retry the tool transparently up to `max_attempts` times (total, incl. the first).

        `classify` decides again on every attempt, so a changed error can end the retrying
        early; spent attempts fall through to `on_exhausted` (default: `inform`). `backoff`
        is a base delay in seconds (`base * 2 ** attempt`) or a callable of the 0-based
        attempt index.
        """
        return cls(action='retry', max_attempts=max_attempts, backoff=backoff, on_exhausted=on_exhausted)

    @classmethod
    def inform(cls, *, label: str | None = None, expose_message: bool = False) -> RecoveryOutcome:
        """Report a terminal failure to the model (via `ToolFailed`, `outcome='failed'`) so it adapts.

        No retry and no retry-budget cost. With `label`, the label is sent instead of the raw
        exception (and is the trace/metric dimension). With `expose_message`, the capped
        `str(error)` is appended.
        """
        return cls(action='inform', label=label, expose_message=expose_message)

    @classmethod
    def fallback(cls, value: Any = None, *, label: str | None = None) -> RecoveryOutcome:
        """Return `value` as the tool result instead of the error."""
        return cls(action='fallback', value=value, label=label)

    @classmethod
    def propagate(cls) -> RecoveryOutcome:
        """Do not recover -- re-raise (the error stays loud for the operator)."""
        return cls(action='propagate')
