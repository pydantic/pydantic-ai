"""Budget windows, and the store keys they accumulate under.

A window decides the key a budget counts against and nothing else, so a window
rolls over by producing a different key rather than by anyone resetting a
counter. A new day is a new key.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Generic, Literal, NamedTuple, TypeVar, assert_never

from typing_extensions import TypeAliasType, TypedDict

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext

_CeilingT = TypeVar('_CeilingT', Decimal, int)

Ceiling = TypeAliasType(
    'Ceiling',
    _CeilingT | Callable[[RunContext[AgentDepsT]], _CeilingT | None],
    type_params=(_CeilingT, AgentDepsT),
)
"""A `Budget` ceiling: a fixed amount, or a callable resolving one from the run (`None` for no ceiling)."""

Window = Literal['run', 'conversation', 'day', 'month', 'total']
"""The period a budget counts over."""

SEPARATOR = '|'
"""Joins name, scope, and bucket into a store key.

`|` rather than `:`, which appears inside model references and tenant
identifiers.
"""

_ANY_SCOPE = '*'


def delimited(*parts: str) -> str:
    """Join parts so that no two different sequences of them produce the same string.

    Each part is prefixed with its length, so a `SEPARATOR` inside one cannot be read as
    the boundary between two. Plain joining is enough where every part is checked for the
    separator first, as `store_key`'s are; it is not enough for a provider's response id
    or a caller's run id, where a collision would let one response be mistaken for another
    and dropped as a replay.
    """
    return SEPARATOR.join(f'{len(part)}:{part}' for part in parts)


WINDOWS: frozenset[str] = frozenset({'run', 'conversation', 'day', 'month', 'total'})
"""The accepted `window` values, for validating one that arrived as plain data."""

_RETAIN_POLICIES = frozenset({'window default', 'forever'})
"""The `retain` values that are not a duration, for validating one that arrived as plain data."""

# A time window may expire freely: its bucket has already rolled over, so the
# counter is obsolete anyway. `run` and `conversation` buckets never roll over,
# so expiry there hands back the ceiling rather than starting a new period --
# but a per-conversation budget mints a key per conversation, so never expiring
# grows the store without bound. The compromise is a horizon long enough that a
# conversation reaching it cannot practically be resumed.
_TTLS: dict[Window, timedelta | None] = {
    'run': timedelta(hours=24),
    'conversation': timedelta(days=30),
    'day': timedelta(hours=48),
    'month': timedelta(days=62),
    'total': None,
}


@dataclass(frozen=True, kw_only=True)
class Budget(Generic[AgentDepsT]):
    """One spend window: what it limits, over what period, for whom.

    A budget with neither `usd` nor `tokens` is a pure counter: it accumulates
    and reports, and never stops a run. That is how per-tenant accounting with
    no cap is expressed.

    Generic in the agent's dependency type so `scope` and the ceiling callables are
    checked against it: the parameter comes from the `Agent` the capability is passed
    to, so a callable reaching for a field the deps do not have is a type error rather
    than an `AttributeError` on the first request.

    ```python
    from decimal import Decimal

    from pydantic_ai_harness.spend_limits import Budget

    Budget(usd=Decimal('100'), window='day')
    Budget(usd=Decimal('10'), window='day', scope=lambda ctx: ctx.deps.tenant_id)
    Budget(window='month', name='accounting')  # counts, never blocks
    Budget(usd=lambda ctx: ctx.deps.allowance_usd, window='month', scope=lambda ctx: ctx.deps.account_id)
    ```

    A callable ceiling is resolved from the run alongside `scope` whenever the budget is read, so
    each partition can carry its own allowance. `None` means no ceiling on that partition, and
    zero or less means nothing is allowed. Under durable execution it must return the same value
    when replayed with the same run context: it is read in orchestration, so a changed value can
    flip a replayed refusal.
    """

    usd: Ceiling[Decimal, AgentDepsT] | None = None
    """Ceiling in US dollars, or a callable resolving it from the run. `None` means this budget does not limit spend."""

    tokens: Ceiling[int, AgentDepsT] | None = None
    """Ceiling in total tokens, or a callable resolving it from the run. `None` means this budget does not limit tokens."""

    window: Window = 'day'
    """The period the ceiling applies to."""

    scope: Callable[[RunContext[AgentDepsT]], str] | None = None
    """Partitions the counter -- per tenant, per user, per agent. `None` counts globally.

    Under durable execution this callable must return the same value when replayed
    with the same run context, because its result is part of the store key.
    """

    warn_at: float | None = None
    """Fraction of the ceiling past which `BudgetStatus.warning` is set. Never blocks."""

    name: str = 'default'
    """Distinguishes budgets sharing a window and scope. Part of the store key."""

    retain: timedelta | Literal['window default', 'forever'] = 'window default'
    """How long a store may keep this window's counter after its last write.

    `'window default'` takes the horizon from `window` (see `_TTLS`). A time window
    may expire freely once it has rolled over, but `run` and `conversation` buckets
    never roll over, so their defaults are a compromise between never expiring and
    growing the store without bound -- and a conversation resumed past the horizon
    starts again from zero. Set `'forever'` where that matters and the keys are
    cleaned up some other way, or a `timedelta` to pick the horizon outright.
    """

    def __post_init__(self) -> None:
        """Reject configurations that would misbehave rather than fail.

        A ceiling of zero or less makes a budget exhausted before anything is
        spent, so the first request is refused with no way to tell that from a
        real overspend -- and `usd: 0` in a spec is far more likely to mean "no
        limit", which is what `None` says. A `warn_at` on a budget with no ceiling has nothing to be a
        fraction of, so it can never fire. Both read as configuration and behave
        as breakage, which is why they are errors here rather than surprises
        later.
        """
        if self.window not in WINDOWS:
            raise UserError(f'Budget.window must be one of {sorted(WINDOWS)}; got {self.window!r}.')
        if not self.name or SEPARATOR in self.name:
            raise UserError(f'Budget.name must be non-empty and must not contain {SEPARATOR!r}; got {self.name!r}.')
        self._check_ceilings()
        self._check_retain()
        if self.warn_at is not None:
            if not 0 < self.warn_at <= 1:
                raise UserError(f'Budget.warn_at must be a fraction in (0, 1]; got {self.warn_at!r}.')
            if not self.enforces:
                raise UserError(
                    f'Budget {self.name!r} sets warn_at but no `usd` or `tokens` ceiling, so the warning could '
                    'never fire. Add a ceiling, or drop warn_at to leave it a plain counter.'
                )

    def _check_ceilings(self) -> None:
        """Validate plain ceilings; a callable's are validated once `resolve_ceilings` has its value."""
        for kind, ceiling in (('usd', self.usd), ('tokens', self.tokens)):
            if ceiling is not None and not callable(ceiling):
                _check_ceiling(kind, ceiling)

    def _check_retain(self) -> None:
        if isinstance(self.retain, timedelta):
            if self.retain <= timedelta(0):
                raise UserError(
                    f'Budget.retain must be a positive duration; got {self.retain}. '
                    "Use 'forever' to keep the counter until something else removes it."
                )
        elif self.retain not in _RETAIN_POLICIES:
            # The annotation is a `Literal`, which nothing enforces at run time. A misspelt
            # `'forevr'` would reach `ttl` and be returned as a string, and the store would
            # fail on `clock() + retain` at the first recorded response instead of here.
            raise UserError(
                f'Budget.retain must be a timedelta or one of {sorted(_RETAIN_POLICIES)}; got {self.retain!r}.'
            )

    @property
    def enforces(self) -> bool:
        """Whether this budget can refuse a request, rather than only counting."""
        return self.usd is not None or self.tokens is not None

    @property
    def ttl(self) -> timedelta | None:
        """How long a store may keep this window's counter after its last write."""
        if self.retain == 'forever':
            return None
        if self.retain == 'window default':
            return _TTLS[self.window]
        return self.retain


class BudgetSpec(TypedDict, total=False):
    """The part of a `Budget` an agent spec can express.

    Declared as a `TypedDict` rather than reusing `Budget` because `scope` is a
    callable, which has no JSON schema: core's schema builder strips a callable from
    the top level of a union but not from a dataclass field nested inside a
    `Sequence`, and generation fails on it outright.

    `usd` accepts a string so a price does not round through a YAML float. `retain`
    takes only the two policies -- a `timedelta` has no spec form, and
    `Budget.__post_init__` already refuses anything else.
    """

    usd: str | int | float
    tokens: int
    window: Window
    warn_at: float
    name: str
    retain: Literal['window default', 'forever']


def bucket(window: Window, ctx: RunContext[Any] | None, now: datetime) -> str | None:
    """The period identifier for `window`, or `None` when it needs a run and none was given.

    `run` and `conversation` are meaningless outside an agent run, which is what
    the `None` reports back to the caller.
    """
    match window:
        case 'run':
            return None if ctx is None else _run_identity(ctx.run_id, 'run')
        case 'conversation':
            return None if ctx is None else _run_identity(ctx.conversation_id, 'conversation')
        case 'day':
            return now.date().isoformat()
        case 'month':
            return f'{now.year:04d}-{now.month:02d}'
        case 'total':
            return 'total'
        case _:  # pragma: no cover - assert_never exhaustiveness guard
            assert_never(window)


def _run_identity(identity: str | None, window: Window) -> str:
    """The run or conversation id, refusing to share a bucket when it is absent."""
    if identity is None:
        raise UserError(
            f"A Budget with window='{window}' needs the run's {window} id, but this run reports none. "
            f'Use a time window, or run the agent through `Agent.run`, which always sets one.'
        )
    return identity


def scope_key(budget: Budget[Any], ctx: RunContext[Any] | None, explicit: str | None) -> str:
    """The scope segment of a budget's store key.

    `explicit` is what `SpendLimits.status` was given, for use outside a run. It
    is ignored by a budget that declares no `scope`, since such a budget counts
    globally.
    """
    if budget.scope is None:
        return _ANY_SCOPE
    resolved = budget.scope(ctx) if ctx is not None else explicit
    if resolved is None:  # pragma: no cover - callers filter these budgets out first
        raise UserError(f'Budget {budget.name!r} declares a scope, which cannot be resolved without a run.')
    if not isinstance(resolved, str):
        # `scope` is annotated `-> str`, but it is supplied by the caller and a tenant id
        # is often an int or a UUID. Checked rather than coerced: `str()` on an object
        # with no `__str__` produces a repr carrying a memory address, which mints a new
        # counter on every run. Without this the failure is `TypeError: argument of type
        # 'int' is not iterable`, from the separator check below.
        raise UserError(
            f'The scope of budget {budget.name!r} must return a string; got {type(resolved).__name__}. '
            'Convert it in the scope callable, so the key it produces is the one you intend.'
        )
    if not resolved or SEPARATOR in resolved or resolved == _ANY_SCOPE:
        raise UserError(
            f'The scope of budget {budget.name!r} must be non-empty and must not be {_ANY_SCOPE!r} or '
            f'contain {SEPARATOR!r}; got {resolved!r}. {_ANY_SCOPE!r} is how an unscoped budget is keyed, '
            "so returning it would share that budget's counter."
        )
    return resolved


class Ceilings(NamedTuple):
    """The ceilings a budget applies to one reading, with any callable resolved for the run."""

    usd: Decimal | None
    tokens: int | None

    @property
    def enforces(self) -> bool:
        """Whether either ceiling is set, so the reading can refuse a request."""
        return self.usd is not None or self.tokens is not None


def plain_ceilings(budget: Budget[Any]) -> Ceilings | None:
    """`budget`'s ceilings, or `None` when either is a callable, which only a run context can resolve."""
    usd, tokens = budget.usd, budget.tokens
    if callable(usd) or callable(tokens):
        return None
    return Ceilings(usd, tokens)


def resolve_ceilings(budget: Budget[AgentDepsT], ctx: RunContext[AgentDepsT]) -> Ceilings:
    """The ceilings `budget` applies in this run, with a callable one resolved.

    A resolved zero or less is kept rather than refused: unlike a configured `usd: 0`, which more
    likely means "no limit", an allowance of zero read from `deps` means nothing is allowed, and
    `None` already says "no limit". It reads as exhausted.
    """
    return Ceilings(
        _resolve_ceiling(budget.name, 'usd', budget.usd, ctx, Decimal),
        _resolve_ceiling(budget.name, 'tokens', budget.tokens, ctx, int),
    )


def _resolve_ceiling(
    name: str,
    kind: str,
    ceiling: Ceiling[_CeilingT, AgentDepsT] | None,
    ctx: RunContext[AgentDepsT],
    expected: type[_CeilingT],
) -> _CeilingT | None:
    if not callable(ceiling):
        return ceiling
    resolved: object = ceiling(ctx)
    if resolved is None:
        return None
    # Checked rather than coerced, as `scope` is: an allowance read from deps is often a float.
    if isinstance(resolved, bool) or not isinstance(resolved, expected):
        raise UserError(
            f'The `{kind}` ceiling of budget {name!r} must return {expected.__name__} or None; '
            f'got {type(resolved).__name__}.'
        )
    if isinstance(resolved, Decimal) and not resolved.is_finite():
        raise UserError(f'The `{kind}` ceiling of budget {name!r} must return a finite amount or None; got {resolved}.')
    return resolved


def _check_ceiling(kind: str, ceiling: Decimal | int) -> None:
    """Reject a ceiling of zero or less, and a USD one that is not finite."""
    # Checked before the comparison, which raises `InvalidOperation` on a NaN
    # rather than returning False. An infinity passes it and reads as a ceiling
    # that can never be reached, which `usd=None` already says outright.
    if isinstance(ceiling, Decimal) and not ceiling.is_finite():
        raise UserError(f'Budget.{kind} must be a finite amount; got {ceiling}. Use `{kind}=None` for no ceiling.')
    if ceiling <= 0:
        raise UserError(f'Budget.{kind} must be positive; got {ceiling}. Use `{kind}=None` for no ceiling.')


def store_key(budget: Budget[Any], bucket_id: str, scope: str) -> str:
    """Join the parts of a budget's store key.

    The window is part of the key because bucket values are not drawn from
    disjoint sets: a run whose id happens to be `total` would otherwise share a
    counter with a `total` budget, and a run and a conversation collide whenever
    their ids match.

    `name`, `window`, and `scope` are all free of `SEPARATOR` -- the first two
    are checked in `Budget.__post_init__`, the third in `scope_key` -- so the
    first three separators delimit unambiguously and whatever follows is the
    bucket. That is why `bucket_id` needs no check of its own even though a
    caller-supplied conversation id may contain anything.
    """
    return f'{budget.name}{SEPARATOR}{budget.window}{SEPARATOR}{scope}{SEPARATOR}{bucket_id}'
