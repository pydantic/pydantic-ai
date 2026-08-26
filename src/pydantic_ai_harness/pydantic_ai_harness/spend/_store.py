"""Where spend counters live.

`BatchSpendStore` is the seam between the gate and its counter. The default keeps
counters in the process, which catches a runaway loop inside one worker; a
shared store is what makes a budget hold across the workers of a queue.

The seam takes every window of a response at once. One response counts against
every configured window, so applying them one at a time leaves a failure between
two of them counted in one window and not the other, which reads as less spend
than really happened.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Lock
from typing import Protocol, runtime_checkable

from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness.spend._snapshot import Spent, money_precision

_Entries = dict[str, tuple[Spent, 'datetime | None']]
"""Each key's counter and the moment it stops counting, if it ever does."""

DEFAULT_DEDUP_RETAIN = timedelta(hours=1)
"""How long a store remembers a `SpendEntry.token` by default.

Long enough for a durable engine to re-execute an accrual it had already committed:
DBOS recovers a workflow when the process that owned it comes back, and a Prefect
flow retry replays from its cache. Short enough that the markers are bounded by an
hour of traffic rather than a day of it.
"""


def utc_now() -> datetime:
    """Current UTC time. The default clock for windows and expiry."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True, kw_only=True)
class SpendEntry:
    """One window's share of one response.

    Everything except `key` defaults to nothing, so a reconciler correcting drift
    against an external source can post a `usd` delta on its own without inflating
    the request count.
    """

    key: str
    """The window's store key."""

    usd: Decimal = Decimal(0)
    """Priced cost to add. May be negative, which is how a reconciler corrects drift."""

    tokens: int = 0
    """Total tokens to add."""

    requests: int = 0
    """Model requests to add. Explicit rather than an implied `+= 1` so a correction
    can move the money without moving the count."""

    unpriced: int = 0
    """How many of `requests` had no resolvable price."""

    ttl: timedelta | None = None
    """How long the key may be kept after this write. `None` means indefinitely."""

    token: str | None = None
    """Identifies the response this entry came from, so it is applied at most once.

    A capability hook runs in orchestration code, which a durable engine re-executes:
    DBOS recovers a workflow by running it again, and a Prefect flow retry replays the
    model request from its cache. Both hand the same response back to the accrual, and
    without a token the window counts it twice. A store that recognises a token it has
    already applied to `key` returns the current total instead of adding again.

    `None` means "apply unconditionally", which is what a reconciler posting a delta
    wants: two corrections of the same size are two corrections.
    """


@runtime_checkable
class SpendStore(Protocol):
    """Reads and accumulates the counter behind one budget window at a time.

    Deprecated, and removed in 0.28.0. Implement
    [`BatchSpendStore`][pydantic_ai_harness.spend.BatchSpendStore] instead: it takes
    every window of a response in one call, which is what lets a backend apply them
    together, and it carries the replay token that keeps a re-executed accrual from
    counting twice. `SpendLimits` still accepts a store of this shape and drives it
    through an adapter, one window per call, warning once about what that costs.

    The two protocols are separate names rather than two versions of one, because
    `runtime_checkable` tests method presence and not signatures: reusing `add` and
    `get` would leave nothing able to tell a store that batches from one that cannot.
    """

    async def get(self, key: str) -> Spent:
        """What `key` has accumulated. A key that was never written reads as zero."""
        ...  # pragma: no cover

    async def add(
        self,
        key: str,
        *,
        usd: Decimal,
        tokens: int,
        requests: int,
        unpriced: int,
        ttl: timedelta | None,
    ) -> Spent:
        """Add to `key` and return the result. `ttl` is how long the key may be kept."""
        ...  # pragma: no cover


@runtime_checkable
class BatchSpendStore(Protocol):
    """Reads and accumulates the counters behind every window of one response.

    `add_many` returns the state **after** the increment so an atomic backend can
    answer without a second round trip, keyed by `SpendEntry.key` rather than
    positionally so entries sharing a key collapse the way the caller expects.

    Both methods take a sequence rather than a single key so a backend that can read
    or apply the whole set as one unit does, and one that cannot still sees the whole
    set and can say so.
    """

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, Spent]:
        """What each key has accumulated. A key that was never written reads as zero."""
        ...  # pragma: no cover

    async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
        """Apply every entry and return each key's new total."""
        ...  # pragma: no cover


@dataclass
class _LegacyStoreAdapter:
    """Drives a `SpendStore` one window per call, because that is all it can do."""

    store: SpendStore

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, Spent]:
        """Read each key on its own."""
        return {key: await self.store.get(key) for key in keys}

    async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
        """Apply each entry on its own, dropping `SpendEntry.token` the store cannot use."""
        return {
            entry.key: await self.store.add(
                entry.key,
                usd=entry.usd,
                tokens=entry.tokens,
                requests=entry.requests,
                unpriced=entry.unpriced,
                ttl=entry.ttl,
            )
            for entry in entries
        }


_SINGLE_KEY = frozenset({'get', 'add'})
_BATCH = frozenset({'get_many', 'add_many'})


def warn_unreachable_overrides(store: object, base: type[object]) -> None:
    """Warn when a subclass overrode the single-key pair that no longer drives anything.

    `SpendLimits` calls `get_many` and `add_many`. A subclass of a concrete store that
    overrode only `get` and `add` still satisfies `BatchSpendStore` through what it
    inherited, so `as_batch_store` hands it back unwrapped and the override never runs.
    Before the batch pair existed that override was the only path, which makes this a
    change of behavior with nothing to show for it -- an audit or a mirrored write bolted
    onto `add` simply stops happening.

    Checked against the class dictionaries between `type(store)` and `base` rather than by
    comparing attributes, so a subclass that redefines the batch pair as well is silent:
    it has already moved.
    """
    mro = type(store).__mro__
    redefined = {name for klass in mro[: mro.index(base)] for name in klass.__dict__}
    if redefined & _SINGLE_KEY and not redefined & _BATCH:
        warnings.warn(
            f'{type(store).__name__} overrides {sorted(redefined & _SINGLE_KEY)} but not `get_many` or '
            '`add_many`. `SpendLimits` drives the batch pair, so those overrides are never called and '
            'whatever they add -- an audit, a mirrored write -- stops happening. Move them onto '
            '`get_many` and `add_many`.',
            HarnessDeprecationWarning,
            stacklevel=4,
        )


def as_batch_store(store: SpendStore | BatchSpendStore) -> BatchSpendStore:
    """The store as a batch store, wrapping a legacy one and saying what that costs.

    Warned about here rather than on every request: this runs once per `SpendLimits`,
    at construction, so the message lands next to the line that chose the store.
    """
    if isinstance(store, BatchSpendStore):
        return store
    warnings.warn(
        f'{type(store).__name__} implements the deprecated `SpendStore` protocol, so each response is applied '
        'one window at a time: a response counting against a day and a month budget is two writes, and a failure '
        'between them leaves the day counted and the month not. `SpendEntry.token` is dropped too, so a durable '
        'engine that re-executes the accrual (DBOS recovery, a Prefect flow retry) counts the response twice. '
        'Implement `get_many` and `add_many` (`BatchSpendStore`) to get both. `SpendStore` is removed in 0.28.0.',
        HarnessDeprecationWarning,
        stacklevel=4,
    )
    return _LegacyStoreAdapter(store)


@dataclass
class InMemorySpendStore:
    """Counters for the lifetime of one process.

    Catches a runaway loop inside the worker it runs in. It does not enforce a
    budget across processes: every worker of a queue would keep its own count,
    which is what a shared store such as
    [`RedisSpendStore`][pydantic_ai_harness.spend.RedisSpendStore] is for.
    """

    clock: Callable[[], datetime] = utc_now
    """Supplies the time expiry is measured against."""

    sweep_every: int = 256
    """Writes between expiry sweeps.

    Expiry cannot wait for the next read of a key: a day window produces a new key each day,
    so yesterday's is never asked for again. The scan is linear in resident keys, so it is
    amortised over this many writes rather than run on each one, which bounds dead entries to
    roughly that many. Lower it where scopes are high-cardinality and memory matters more than
    the scan.
    """

    dedup_retain: timedelta | None = DEFAULT_DEDUP_RETAIN
    """How long an applied `SpendEntry.token` is remembered, or `None` to apply every entry.

    This is the window a replay is recognised in, not the counter's lifetime: a response
    replayed later than this is counted again. A remembered token costs one small entry
    per response per window until it is swept.
    """

    _entries: _Entries = field(default_factory=_Entries, init=False, repr=False)
    _applied: dict[tuple[str, str], datetime] = field(
        default_factory=dict[tuple[str, str], datetime], init=False, repr=False
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _writes_since_sweep: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Report a subclass whose single-key overrides the batch pair has left unreachable."""
        warn_unreachable_overrides(self, InMemorySpendStore)

    def __len__(self) -> int:
        """How many windows are still live.

        Rolled-over entries are excluded whether or not the amortised sweep has reached them
        yet, so this counts what is being tracked rather than what happens to be resident.
        That is one entry per budget, scope and period -- and for a `run` or `conversation`
        budget the period is an id, so the count grows with traffic until those entries reach
        their horizon. Worth watching there and wherever scopes are high-cardinality.

        Defining `__len__` makes an empty store falsy, so write `if store is not None`.
        """
        now = self.clock()
        with self._lock:
            return sum(1 for _, expires_at in self._entries.values() if expires_at is None or now < expires_at)

    async def get(self, key: str) -> Spent:
        """What `key` has accumulated. Deprecated in favour of `get_many`, removed in 0.28.0."""
        return (await self.get_many([key]))[key]

    async def add(
        self,
        key: str,
        *,
        usd: Decimal,
        tokens: int,
        requests: int,
        unpriced: int,
        ttl: timedelta | None,
    ) -> Spent:
        """Add to `key` and return the result. Deprecated in favour of `add_many`, removed in 0.28.0."""
        entry = SpendEntry(key=key, usd=usd, tokens=tokens, requests=requests, unpriced=unpriced, ttl=ttl)
        return (await self.add_many([entry]))[key]

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, Spent]:
        """What each key has accumulated, treating an expired key as absent.

        Under the lock, because `_live` deletes the key it finds expired: unlocked, that
        `del` races the `_sweep` iteration inside `add_many` (`RuntimeError: dictionary
        changed size during iteration`) and a second concurrent reader (`KeyError`).
        Reachable whenever the guard is shared across threads -- `run_sync` from a pool,
        a sync endpoint -- and any key is read past its horizon.
        """
        with self._lock:
            now = self.clock()
            return {key: self._live(key, now) for key in keys}

    async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
        """Apply every entry and return each key's new total.

        The mutation spans no `await`, so concurrent runs on one event loop cannot
        interleave halfway through it and no reader sees some of a response applied and
        not the rest. The lock covers the case that is not free: `run_sync` called from a
        thread pool, or a free-threaded interpreter, where a read-modify-write loses
        updates in the direction that under-counts spend.

        Every entry is worked out before any of them is stored, so a response that fails
        part-way through -- an amount whose arithmetic raises, say -- leaves none of its
        windows applied rather than the ones before the failure. Applying the whole set
        together is what this method exists for.

        The clock is read once, before anything is applied, and a token is remembered
        only once the counter it stands for has moved. A token remembered ahead of that
        write would be consumed by a call that then failed, and the retry that could have
        recorded the response would be skipped as a replay of it.
        """
        with self._lock:
            now = self.clock()
            self._writes_since_sweep += len(entries)
            if self._writes_since_sweep >= self.sweep_every:
                self._sweep(now)
            pending: dict[str, tuple[Spent, datetime | None]] = {}
            claimed: dict[tuple[str, str], datetime] = {}
            totals: dict[str, Spent] = {}
            for entry in entries:
                held = pending.get(entry.key)
                current = held[0] if held is not None else self._live(entry.key, now)
                if self._already_applied(entry, now, claimed):
                    totals[entry.key] = current
                    continue
                with money_precision():
                    updated = Spent(
                        usd=current.usd + entry.usd,
                        tokens=current.tokens + entry.tokens,
                        requests=current.requests + entry.requests,
                        unpriced_requests=current.unpriced_requests + entry.unpriced,
                    )
                pending[entry.key] = (updated, None if entry.ttl is None else now + entry.ttl)
                if entry.token is not None and self.dedup_retain is not None:
                    claimed[(entry.key, entry.token)] = now + self._remembered_for(entry)
                totals[entry.key] = updated
            self._entries.update(pending)
            self._applied.update(claimed)
            return totals

    def _already_applied(
        self,
        entry: SpendEntry,
        now: datetime,
        claimed: Mapping[tuple[str, str], datetime],
    ) -> bool:
        """Whether this entry's token has reached its key, in an earlier call or earlier in this one."""
        if entry.token is None or self.dedup_retain is None:
            return False
        marker = (entry.key, entry.token)
        if marker in claimed:
            return True
        seen = self._applied.get(marker)
        return seen is not None and now < seen

    def _remembered_for(self, entry: SpendEntry) -> timedelta:
        """How long this entry's token is remembered, never past the counter it guards.

        A token outliving its window would skip the replay of a response against a counter
        that has since rolled over, and the window would read as zero rather than as the
        response it should hold. What the horizon does and does not promise, and which way
        it errs, is settled in `RedisSpendStore._marker_seconds`.
        """
        retain = self.dedup_retain or timedelta(0)
        return retain if entry.ttl is None else min(retain, entry.ttl)

    def _sweep(self, now: datetime) -> None:
        """Drop every entry whose window has rolled over, and every token past its horizon.

        Amortised over `sweep_every` writes rather than run on each one. The scan is linear in
        resident keys, and `run` and `conversation` budgets hold one key per run and per
        conversation for a day and a month respectively -- a worker at ten runs a second
        carries most of a million of them after a day, and a full scan under a
        `threading.Lock` inside an `async def` would block the event loop for milliseconds on
        every model request.
        """
        self._writes_since_sweep = 0
        stale = [key for key, (_, expires_at) in self._entries.items() if expires_at is not None and now >= expires_at]
        for key in stale:
            del self._entries[key]
        expired = [marker for marker, seen in self._applied.items() if now >= seen]
        for marker in expired:
            del self._applied[marker]

    def _live(self, key: str, now: datetime) -> Spent:
        """The entry at `key`, dropping it first if its window has rolled over.

        A read also expires the key it touches, so a rolled-over window reads as
        zero even between writes.
        """
        entry = self._entries.get(key)
        if entry is None:
            return Spent()
        spent, expires_at = entry
        if expires_at is not None and now >= expires_at:
            del self._entries[key]
            return Spent()
        return spent
