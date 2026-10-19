"""How tool failures are classified, rendered for the model, and logged."""

from __future__ import annotations

import inspect
import logging
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import HookTimeoutError
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ToolCallPart

from ._outcome import DEFAULT_BUG_TYPES, DEFAULT_TRANSIENT_TYPES, RecoveryOutcome

_LOGGER = logging.getLogger('pydantic_ai_harness.tool_error_recovery')

RecoveryClassifier = (
    Callable[[ToolCallPart, BaseException], 'RecoveryOutcome | None | Awaitable[RecoveryOutcome | None]']
    | Callable[
        [RunContext[Any], ToolCallPart, BaseException],
        'RecoveryOutcome | None | Awaitable[RecoveryOutcome | None]',
    ]
)
"""Signature of the classifier passed to `RecoveryPolicy`.

The callable receives the tool call and the exception and returns a
`RecoveryOutcome` (`None` = propagate). It may optionally take a `RunContext`
as a first argument -- for `deps`, message history, or other run state -- and may
be sync or async, matching pydantic-ai's optional-`ctx` convention. It is
re-invoked on every retry attempt, so keep it cheap and idempotent.
"""

ErrorFormatter = Callable[[ToolCallPart, BaseException, 'str | None'], str]
"""Signature of a custom `(call, error, label) -> str` renderer for the model-facing error text.

A custom renderer takes over entirely: neither the `max_message_len` cap nor an
outcome's `expose_message` applies -- the callable sees the raw error and decides.
"""


def make_default_classify(
    *,
    transient: tuple[type[BaseException], ...] = DEFAULT_TRANSIENT_TYPES,
    bugs: tuple[type[BaseException], ...] = DEFAULT_BUG_TYPES,
) -> RecoveryClassifier:
    """Build a conservative classifier: bugs propagate, transient errors retry, deadlines and the rest inform.

    The default transient set covers builtin `TimeoutError`/`ConnectionError` plus httpx
    timeouts and connect errors -- HTTP-based tools retry out of the box. Other client
    libraries (qdrant, database drivers) define their own exception roots; pass them in:
    `make_default_classify(transient=(*DEFAULT_TRANSIENT_TYPES, qdrant_client.QdrantException))`.
    """

    def classify(ctx: RunContext[Any], call: ToolCallPart, error: BaseException) -> RecoveryOutcome:
        if isinstance(error, bugs):
            return RecoveryOutcome.propagate()
        if isinstance(error, HookTimeoutError):
            # A deadline is wall-clock and may have fired after the call reached the
            # server, breaking the "never executed" premise `transient` rests on.
            return RecoveryOutcome.inform()
        if isinstance(error, transient):
            return RecoveryOutcome.retry(3)
        return RecoveryOutcome.inform()

    return classify


default_classify: RecoveryClassifier = make_default_classify()


def required_positionals(func: Callable[..., object]) -> int | None:
    """How many positional parameters a call must fill, or `None` when the signature says nothing.

    Counted rather than annotated, because the documented `lambda ctx, call, error` form carries none.
    Only parameters the call fills count: an optional or keyword-only one would shift the payload by one.
    `*args` absorbs any count, so it yields `None` rather than a misleading zero.
    """
    try:
        parameters = list(inspect.signature(func).parameters.values())
    except ValueError:  # pragma: no cover - callable without an introspectable signature
        return None
    if any(p.kind is p.VAR_POSITIONAL for p in parameters):
        return None
    return sum(1 for p in parameters if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))


@dataclass
class RecoveryPolicy:
    """How errors are classified, rendered for the model, and logged.

    Holds no per-run state; the counters live on the capability so a fresh run resets them.
    Logging uses the standard library `logging` module; where records land (Logfire, OTLP,
    stderr) is configured by the application.
    """

    classify: RecoveryClassifier = default_classify
    """The `(call, error) -> RecoveryOutcome | None` policy; may take a leading `RunContext`."""

    format_error: ErrorFormatter | None = None
    """Custom renderer for the model-facing error text; `None` uses the built-in capped renderer."""

    max_message_len: int = 300
    """Cap on the built-in renderer's uncurated text; a pure `label` is exempt (see `render`)."""

    include_traceback: bool = False
    """Whether the built-in renderer appends the traceback (debugging aid; costs tokens)."""

    logger: logging.Logger = field(default=_LOGGER)
    """Where recovery events are logged."""

    def __post_init__(self) -> None:
        """Reject configuration the contract does not allow."""
        if self.max_message_len < 1:
            raise UserError(f'RecoveryPolicy.max_message_len must be positive, got {self.max_message_len}.')
        required = required_positionals(self.classify)
        if required is not None and required not in (2, 3):
            # Caught here rather than as a TypeError inside the recovery path, at the first tool failure.
            raise UserError(
                f'RecoveryPolicy.classify must take (call, error) or (ctx, call, error), got {required} parameters.'
            )

    def render(
        self, call: ToolCallPart, error: BaseException, label: str | None, *, expose_message: bool = False
    ) -> str:
        """Build the model-facing text for an `inform` recovery.

        Never includes `str(error)` by default: exception text can carry secrets and would reach
        the user through the model. `label`, `expose_message`, `format_error` and
        `include_traceback` each opt out of that. `max_message_len` caps uncurated content;
        a pure `label` is exempt.
        """
        if self.format_error is not None:
            return self.format_error(call, error, label)
        if label and not expose_message:
            return label
        base = label or f'Tool {call.tool_name!r} failed ({type(error).__name__})'
        text = f'{base}: {error}' if expose_message else f'{base}.'
        if self.include_traceback:
            text += '\n' + ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        if len(text) > self.max_message_len:
            text = text[: self.max_message_len - 1] + '…'
        return text

    async def run_classify(
        self, ctx: RunContext[Any], call: ToolCallPart, error: BaseException
    ) -> RecoveryOutcome | None:
        """Call the classifier (passing `ctx` when declared) and await it if async."""
        classify: Callable[..., RecoveryOutcome | None | Awaitable[RecoveryOutcome | None]] = self.classify
        result = classify(ctx, call, error) if required_positionals(classify) == 3 else classify(call, error)
        if inspect.isawaitable(result):
            return await result
        return result
