"""Where spend counters live.

`SpendStore` is the seam between the gate and its counter. The default keeps
counters in the process, which catches a runaway loop inside one worker; a
shared store is what makes a budget hold across the workers of a queue.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Lock
from typing import Protocol, runtime_checkable

from pydantic_ai_harness.spend._snapshot import Spent

_Entries = dict[str, tuple[Spent, 'datetime | None']]
"""Each key's counter and the moment it stops counting, if it ever does."""


def utc_now() -> datetime:
    """Current UTC time. The default clock for windows and expiry."""
    return datetime.now(timezone.utc)


@runtime_checkable
class SpendStore(Protocol):
    """Reads and accumulates the counter behind each budget window.

    `add` returns the state **after** the increment so an atomic backend can
    answer without a second round trip. `requests` is an explicit parameter
    rather than an implied `+= 1` so a reconciler correcting drift against an
    external source can post a delta (including a negative `usd`) without
    inflating the request count, and without the protocol growing a second
    method for it.
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

    _entries: _Entries = field(default_factory=_Entries, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _writes_since_sweep: int = field(default=0, init=False, repr=False)

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
        """What `key` has accumulated, treating an expired key as absent.

        Under the lock, because `_live` deletes the key it finds expired: unlocked, that
        `del` races the `_sweep` iteration inside `add` (`RuntimeError: dictionary changed
        size during iteration`) and a second concurrent reader (`KeyError`). Reachable
        whenever the guard is shared across threads -- `run_sync` from a pool, a sync
        endpoint -- and any key is read past its horizon.
        """
        with self._lock:
            return self._live(key)

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
        """Add to `key` and return the result.

        The mutation spans no `await`, so concurrent runs on one event loop
        cannot interleave halfway through it. The lock covers the case that is
        not free: `run_sync` called from a thread pool, or a free-threaded
        interpreter, where a read-modify-write loses updates in the direction
        that under-counts spend.
        """
        with self._lock:
            self._writes_since_sweep += 1
            if self._writes_since_sweep >= self.sweep_every:
                self._sweep()
            current = self._live(key)
            updated = Spent(
                usd=current.usd + usd,
                tokens=current.tokens + tokens,
                requests=current.requests + requests,
                unpriced_requests=current.unpriced_requests + unpriced,
            )
            self._entries[key] = (updated, None if ttl is None else self.clock() + ttl)
            return updated

    def _sweep(self) -> None:
        """Drop every entry whose window has rolled over.

        Amortised over `sweep_every` writes rather than run on each one. The scan is linear in
        resident keys, and `run` and `conversation` budgets hold one key per run and per
        conversation for a day and a month respectively -- a worker at ten runs a second
        carries most of a million of them after a day, and a full scan under a
        `threading.Lock` inside an `async def` would block the event loop for milliseconds on
        every model request.
        """
        self._writes_since_sweep = 0
        now = self.clock()
        stale = [key for key, (_, expires_at) in self._entries.items() if expires_at is not None and now >= expires_at]
        for key in stale:
            del self._entries[key]

    def _live(self, key: str) -> Spent:
        """The entry at `key`, dropping it first if its window has rolled over.

        A read also expires the key it touches, so a rolled-over window reads as
        zero even between writes.
        """
        entry = self._entries.get(key)
        if entry is None:
            return Spent()
        spent, expires_at = entry
        if expires_at is not None and self.clock() >= expires_at:
            del self._entries[key]
            return Spent()
        return spent
