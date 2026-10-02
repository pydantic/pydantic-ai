"""Retry policy capability: retry a tool's transient failures with exponential backoff."""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio

from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.tools import AgentDepsT, ToolDefinition

if TYPE_CHECKING:
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import RunContext

logger = logging.getLogger(__name__)

_RETRYABLE_ERROR_TYPES = frozenset({'rate_limit', 'timeout', 'server_error'})

_TOOL_OVERRIDE_KEYS = frozenset(
    {
        'backoff_factor',
        'idempotent',
        'max_backoff',
        'max_retries',
        'on_failure',
        'on_retry',
        'retryable_exceptions',
        'retryable_status_codes',
    }
)


def _is_retryable_http_error(exc: Exception, status_codes: tuple[int, ...]) -> bool:
    """Whether the exception carries a status code (its own or on `.response`) that is retryable."""
    status_code = getattr(exc, 'status_code', None)
    if status_code is not None:
        return status_code in status_codes

    response = getattr(exc, 'response', None)
    if response is not None:
        response_status = getattr(response, 'status_code', None)
        if response_status is not None:
            return response_status in status_codes

    return False


def _validate_status_codes(value: tuple[int, ...], *, prefix: str) -> None:
    """A bad member would never match, but a non-tuple would fail obscurely at classification time."""
    if not isinstance(value, tuple) or any(not isinstance(code, int) or isinstance(code, bool) for code in value):
        raise ValueError(f'{prefix} must be a tuple of ints, got {value!r}')


def _validate_exception_types(value: tuple[type[Exception], ...], *, prefix: str) -> None:
    """A non-`Exception` member would raise `TypeError` from `isinstance` while classifying a failure."""
    if not isinstance(value, tuple) or any(
        not isinstance(exc_type, type) or not issubclass(exc_type, Exception) for exc_type in value
    ):
        raise ValueError(f'{prefix} must be a tuple of Exception types, got {value!r}')


def _validate_tool_overrides(tool_overrides: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject unknown keys and malformed values at construction, not at the first tool failure."""
    for tool_name, config in tool_overrides.items():
        prefix = f'tool_overrides[{tool_name!r}]'
        unknown = sorted(set(config) - _TOOL_OVERRIDE_KEYS)
        if unknown:
            raise ValueError(f'{prefix} has unknown keys {unknown}; allowed keys: {sorted(_TOOL_OVERRIDE_KEYS)}')
        if 'max_retries' in config:
            retries = config['max_retries']
            if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
                raise ValueError(f"{prefix}['max_retries'] must be an int >= 0, got {retries!r}")
        for name in ('backoff_factor', 'max_backoff'):
            if name in config:
                value = config[name]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f'{prefix}[{name!r}] must be a finite number > 0, got {value!r}')
        if 'retryable_status_codes' in config:
            _validate_status_codes(config['retryable_status_codes'], prefix=f"{prefix}['retryable_status_codes']")
        if 'retryable_exceptions' in config:
            _validate_exception_types(config['retryable_exceptions'], prefix=f"{prefix}['retryable_exceptions']")
        for name in ('on_retry', 'on_failure'):
            if name in config and not callable(config[name]):
                raise ValueError(f'{prefix}[{name!r}] must be callable, got {config[name]!r}')
        if 'idempotent' in config and not isinstance(config['idempotent'], bool):
            raise ValueError(f"{prefix}['idempotent'] must be a bool, got {config['idempotent']!r}")


@dataclass
class RetryPolicy(AbstractCapability[AgentDepsT]):
    """Retry a tool's transient failures with exponential backoff.

    Rate limits, timeouts, and connection drops are often gone on the second try. `RetryPolicy`
    re-runs a tool call that failed this way, sleeping between attempts, so a flaky dependency
    slows a run down instead of failing it. Model requests are not touched: core retries those
    at the transport and provider layers (see the retries guide).

    Retrying re-runs the tool's function, and a function that already wrote a file, sent a
    message, or charged a card must not simply run again. The capability therefore retries
    nothing by default. Mark a tool safe to re-run and switch the gate on:

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import RetryPolicy

    agent = Agent(
        'anthropic:claude-fable-5',
        capabilities=[
            RetryPolicy(
                allow_idempotent_retries=True,
                idempotent_tools=frozenset({'web_search'}),
            )
        ],
    )
    ```

    A failure counts as transient when the exception is an instance of `retryable_exceptions`,
    carries a `status_code` attribute (or one on its `.response`) listed in
    `retryable_status_codes`, or carries an `error_type` attribute of `rate_limit`, `timeout`,
    or `server_error`. Anything else propagates immediately.
    """

    max_retries: int = 3
    """Retry attempts after the first; `0` disables retries."""

    backoff_factor: float = 0.5
    """Base delay in seconds; each attempt waits twice the previous one."""

    max_backoff: float = 30.0
    """Upper bound in seconds on the delay between attempts."""

    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
    """HTTP status codes that count as a transient failure."""

    retryable_exceptions: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)
    """Exception types that count as a transient failure.

    `ConnectionError` covers the connection-level failures of `OSError` without also retrying
    permanent OS errors such as `FileNotFoundError` or `PermissionError`. Widen it explicitly
    for tools whose failures you know to be transient.
    """

    tool_overrides: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    """Per-tool configuration; keys are tool names, values are dicts of the fields above.

    Recognized keys: `max_retries`, `backoff_factor`, `max_backoff`, `retryable_status_codes`,
    `retryable_exceptions`, `idempotent`, `on_retry`, `on_failure`. An override dict replaces
    the top-level value of each field it names, for that tool alone. Unknown keys raise
    `ValueError` at construction.
    """

    allow_idempotent_retries: bool = False
    """The gate for retries: with the default `False`, no tool call is ever retried."""

    idempotent_tools: frozenset[str] = frozenset()
    """Tools whose handler may run again after a failure, once the gate above is open.

    A tool can also be marked with `idempotent: True` in its `tool_overrides` entry; the two
    spellings are equivalent. Tools not marked this way surface their first transient failure.
    """

    on_retry: Callable[[str, int, Exception], None] | None = None
    """Called synchronously as `on_retry(tool_name, attempt, exc)` before each retry."""

    on_failure: Callable[[str, Exception], None] | None = None
    """Called synchronously as `on_failure(tool_name, exc)` when a transient failure is surfaced.

    This happens when the tool is not marked idempotent, or when the attempts run out.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool) or self.max_retries < 0:
            raise ValueError(f'max_retries must be an int >= 0, got {self.max_retries!r}')
        if not math.isfinite(self.backoff_factor) or self.backoff_factor <= 0:
            raise ValueError(f'backoff_factor must be a finite number > 0, got {self.backoff_factor!r}')
        if not math.isfinite(self.max_backoff) or self.max_backoff <= 0:
            raise ValueError(f'max_backoff must be a finite number > 0, got {self.max_backoff!r}')
        if not isinstance(self.allow_idempotent_retries, bool):
            raise ValueError(
                f'allow_idempotent_retries must be a bool, got {self.allow_idempotent_retries!r}; '
                "a truthy non-bool like 'false' would otherwise open the retry gate"
            )
        _validate_status_codes(self.retryable_status_codes, prefix='retryable_status_codes')
        _validate_exception_types(self.retryable_exceptions, prefix='retryable_exceptions')
        if self.max_backoff < self.backoff_factor:
            raise ValueError(
                f'max_backoff must be >= backoff_factor, got {self.max_backoff!r} < {self.backoff_factor!r}; '
                'a cap below the base delay means every wait is the cap, so set backoff_factor to it instead'
            )
        _validate_tool_overrides(self.tool_overrides)

    def get_tool_config(self, tool_name: str) -> dict[str, Any]:
        """The override dict for `tool_name`, or an empty dict when it has none."""
        return self.tool_overrides.get(tool_name, {})

    def get_max_retries(self, tool_name: str) -> int:
        """The retry budget for `tool_name`, after per-tool overrides."""
        config = self.get_tool_config(tool_name)
        return config.get('max_retries', self.max_retries)

    def _get_retryable_status_codes(self, tool_name: str) -> tuple[int, ...]:
        config = self.get_tool_config(tool_name)
        return config.get('retryable_status_codes', self.retryable_status_codes)

    def _get_retryable_exceptions(self, tool_name: str) -> tuple[type[Exception], ...]:
        config = self.get_tool_config(tool_name)
        return config.get('retryable_exceptions', self.retryable_exceptions)

    def _get_on_retry(self, tool_name: str) -> Callable[[str, int, Exception], None] | None:
        config = self.get_tool_config(tool_name)
        return config.get('on_retry', self.on_retry)

    def _get_on_failure(self, tool_name: str) -> Callable[[str, Exception], None] | None:
        config = self.get_tool_config(tool_name)
        return config.get('on_failure', self.on_failure)

    def _is_idempotent(self, tool_name: str) -> bool:
        """Whether the tool's handler may run again after a failure."""
        if not self.allow_idempotent_retries:
            return False
        config = self.get_tool_config(tool_name)
        return bool(config.get('idempotent', tool_name in self.idempotent_tools))

    def should_retry(self, exc: Exception, tool_name: str) -> bool:
        """Whether `exc` is a transient failure for `tool_name`, after per-tool overrides."""
        if isinstance(exc, self._get_retryable_exceptions(tool_name)):
            return True
        if _is_retryable_http_error(exc, self._get_retryable_status_codes(tool_name)):
            return True
        return getattr(exc, 'error_type', None) in _RETRYABLE_ERROR_TYPES

    def calculate_delay(self, attempt: int, tool_name: str) -> float:
        """Delay in seconds before the retry after failed `attempt`: backoff plus jitter, capped.

        The delay is `backoff_factor * 2 ** attempt` (per-tool overrides apply), plus jitter of
        up to 25% in either direction, clamped into `[0.01, max_backoff]`.
        """
        config = self.get_tool_config(tool_name)
        backoff_factor = config.get('backoff_factor', self.backoff_factor)
        max_backoff = config.get('max_backoff', self.max_backoff)

        # `backoff_factor * 2 ** attempt` overflows to `OverflowError` for large `attempt`, so compare
        # in log space and scale with `ldexp`, which stays below `max_backoff` whenever the comparison
        # says the raw product would.
        if math.log2(backoff_factor) + attempt >= math.log2(max_backoff):
            delay = max_backoff
        else:
            delay = min(math.ldexp(float(backoff_factor), attempt), max_backoff)

        jitter = delay * 0.25 * (2.0 * random.random() - 1.0)
        return min(max_backoff, max(0.01, delay + jitter))

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Run the tool, retrying while the failure is transient and the tool is marked idempotent.

        The handler is one opaque call: the capability cannot observe anything before or after it,
        so every failure happens after the handler was entered. That is why the idempotency mark is
        the tool author's statement that the handler may run again, and why the default surface is
        no retries at all.
        """
        tool_name = call.tool_name
        max_retries = self.get_max_retries(tool_name)
        on_retry = self._get_on_retry(tool_name)
        on_failure = self._get_on_failure(tool_name)

        for attempt in range(max_retries + 1):
            try:
                return await handler(args)
            except Exception as exc:
                if not self.should_retry(exc, tool_name):
                    raise
                if not self._is_idempotent(tool_name) or attempt == max_retries:
                    if on_failure:
                        on_failure(tool_name, exc)
                    raise
                delay = self.calculate_delay(attempt, tool_name)
                if on_retry:
                    on_retry(tool_name, attempt + 1, exc)
                logger.warning(
                    'Retry %d/%d for tool %r after %.2fs: %s', attempt + 1, max_retries, tool_name, delay, exc
                )
                await anyio.sleep(delay)
        raise AssertionError('unreachable: every attempt returns or raises')  # pragma: no cover
