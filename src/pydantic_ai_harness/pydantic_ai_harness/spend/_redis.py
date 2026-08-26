"""A spend counter shared across processes, backed by Redis.

Takes a client protocol rather than a Redis dependency, the way
`pydantic_ai_harness.memory` takes a Postgres connection protocol: any client
exposing these two coroutines works, and installing the harness pulls in
nothing extra.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol, runtime_checkable

from pydantic_ai.exceptions import UserError

from pydantic_ai_harness.spend._budget import SEPARATOR, delimited
from pydantic_ai_harness.spend._snapshot import Spent, money_precision
from pydantic_ai_harness.spend._store import DEFAULT_DEDUP_RETAIN, SpendEntry, warn_unreachable_overrides

_SCALE = Decimal(10) ** 9

_MARKER = f'{SEPARATOR}dedup'
"""First segment of a dedup marker's name.

Markers and counters share one namespace so a script can take both and stay in one Redis
Cluster slot, which means the two must not be able to name each other: a counter under a
marker's name would meet `HINCRBY` against the string a marker holds, and Redis answers
`WRONGTYPE` and aborts the whole script.

The leading `SEPARATOR` is what keeps them apart, and it costs nothing to carry. A budget
key is `store_key`'s `name|window|scope|bucket` and `Budget.name` is refused empty, so a
budget key never starts with the separator and no configuration reaches this namespace.
"""

_USD_FIELD = 'usd_nanos'
_TOKENS_FIELD = 'tokens'
_REQUESTS_FIELD = 'requests'
_UNPRICED_FIELD = 'unpriced'

_ADD_SCRIPT = f"""
local n = tonumber(ARGV[1])
local out = {{}}
for i = 1, n do
  local key = KEYS[i]
  local marker = KEYS[n + i]
  local base = 1 + (i - 1) * 6
  local marker_ttl = tonumber(ARGV[base + 6])
  if marker_ttl == 0 or redis.call('EXISTS', marker) == 0 then
    redis.call('HINCRBY', key, '{_USD_FIELD}', ARGV[base + 1])
    redis.call('HINCRBY', key, '{_TOKENS_FIELD}', ARGV[base + 2])
    redis.call('HINCRBY', key, '{_REQUESTS_FIELD}', ARGV[base + 3])
    redis.call('HINCRBY', key, '{_UNPRICED_FIELD}', ARGV[base + 4])
    local ttl = tonumber(ARGV[base + 5])
    if ttl > 0 then
      redis.call('EXPIRE', key, ttl)
    else
      redis.call('PERSIST', key)
    end
    if marker_ttl > 0 then
      redis.call('SET', marker, '1', 'EX', marker_ttl)
    end
  end
  local totals = redis.call('HMGET', key, '{_USD_FIELD}', '{_TOKENS_FIELD}', '{_REQUESTS_FIELD}', '{_UNPRICED_FIELD}')
  for j = 1, 4 do
    if not totals[j] then totals[j] = '0' end
  end
  out[i] = totals
end
return out
"""
"""Applies one response to every window it counts against, and returns each window's totals.

KEYS is one key per entry followed by one marker key per entry; ARGV is the entry count
followed by six values per entry: the four increments, the key's horizon in seconds, and
the marker's, where a zero horizon means "keep this key" and "claim no marker".

Redis runs a script to completion without interleaving another command, so no other client
observes a response part-applied -- neither across the four counters of one window nor across
the windows themselves. Issued as separate commands instead, a failure between them leaves a
window counting some of a response and not the rest, which reads as a smaller number than was
really spent and so releases the brake later than it should.

The totals come back as bulk strings, read with `HMGET` after the increments rather than taken
from what `HINCRBY` returns. `HINCRBY` is exact 64-bit integer arithmetic and the increment
arrives as a string, so the stored counter never rounds; its *reply* is an integer that becomes
a Lua number, and Lua 5.1 numbers are doubles, so a total past 2**53 would come back rounded
even though Redis holds it exactly. A Lua string is returned as a bulk string with no double in
the path.

What is left is `HINCRBY`'s own range: a counter passing the signed 64-bit range, around $9.22
billion or 9.2 quintillion tokens against a single key, which Redis refuses before writing that
field. It aborts the script there, and Redis does not roll back what earlier commands in the
script already did, so a multi-window response would keep the windows applied before the one
that overflowed.

A per-entry marker is how an entry carrying a `SpendEntry.token` is applied at most once: the
marker is read before the increments and written after them, rather than claimed with a single
`SET ... NX` up front. Up front, an increment that overflowed would abort the script with the
marker already committed, and the retry that could have completed the window would be skipped
as a replay instead. Read-then-write is safe here without `NX` because Redis runs the whole
script without interleaving another client's commands.

A zero `ttl` means "keep this key", and says so with `PERSIST` rather than by doing nothing.
`HINCRBY` leaves an existing expiry in place, so a budget moved from a finite `retain` to
`'forever'` would otherwise keep expiring on the old horizon -- handing back the ceiling on a
schedule nothing in the configuration mentions any more, and diverging from
`InMemorySpendStore`, which drops the expiry on the next write.

"""


@runtime_checkable
class RedisClient(Protocol):
    """The part of a Redis client `RedisSpendStore` uses.

    `redis.asyncio.Redis` satisfies this. So does any wrapper or fake exposing the
    same two methods, `async def` included.

    Declared as returning an `Awaitable` rather than as `async def`, which would
    narrow the requirement to a `Coroutine` and is what an implementation actually
    has to hand back. `redis.asyncio.Redis` types these as `Awaitable`, so an
    `async def` protocol refused the one client this exists to accept.
    """

    def hgetall(self, name: str) -> Awaitable[Mapping[str | bytes, str | bytes]]:
        """Every field of a hash. An absent hash reads as empty."""
        ...  # pragma: no cover

    def eval(self, script: str, numkeys: int, *keys_and_args: str | int) -> Awaitable[Sequence[Sequence[str | bytes]]]:
        """Run a Lua script server-side and return what it returns.

        Rows of strings, because that is what `_ADD_SCRIPT` returns and what keeps a
        total past 2**53 exact. `str` or `bytes` depending on the client's
        `decode_responses`.
        """
        ...  # pragma: no cover


def _to_nanos(usd: Decimal) -> int:
    """US dollars as whole billionths.

    Counters are integers because `INCRBYFLOAT` accumulates binary rounding
    error over the tens of thousands of requests a busy day produces, while
    `HINCRBY` on integers is both exact and atomic.

    Billionths rather than millionths because the residue does not average out:
    an agent repeats requests of near-identical shape, so the same fraction is
    rounded the same way every time. At a cheap model's per-request price,
    rounding to a millionth drifts by tens of percent over a day; a billionth
    keeps that under a part in ten thousand.

    The local context is pinned so an application that lowered `Decimal`
    precision for its own arithmetic cannot silently truncate money here.
    """
    with money_precision():
        return int((usd * _SCALE).to_integral_value(rounding=ROUND_HALF_UP))


def _from_nanos(nanos: int) -> Decimal:
    """Whole billionths back to US dollars."""
    with money_precision():
        return Decimal(nanos) / _SCALE


def _expiry_seconds(ttl: timedelta) -> int:
    """A positive `timedelta` as whole seconds, rounded up.

    `EXPIRE` takes seconds, and the script reads zero as "keep this key". Rounding down
    would turn any horizon under a second -- which `Budget.retain` accepts -- into no
    expiry at all, the opposite of what was asked for and a silent divergence from
    `InMemorySpendStore`, which honours the horizon exactly.

    Ceiling division on the `timedelta` itself rather than on `total_seconds()`, which is
    a float: `timedelta` divides in whole microseconds, so a horizon with a sub-millisecond
    remainder rounds up like any other instead of being truncated away before the ceiling
    is taken.
    """
    return max(1, -(-ttl // timedelta(seconds=1)))


def _text(value: str | bytes) -> str:
    """One value of a reply, whichever way the client decodes."""
    return value.decode() if isinstance(value, bytes) else value


def _field(fields: Mapping[str | bytes, str | bytes], name: str) -> int:
    """One integer field of a hash, treating an absent field as zero."""
    for key, value in fields.items():
        if _text(key) == name:
            return int(_text(value))
    return 0


def _spent(fields: Mapping[str | bytes, str | bytes]) -> Spent:
    """A window's counters, read from its hash."""
    return Spent(
        usd=_from_nanos(_field(fields, _USD_FIELD)),
        tokens=_field(fields, _TOKENS_FIELD),
        requests=_field(fields, _REQUESTS_FIELD),
        unpriced_requests=_field(fields, _UNPRICED_FIELD),
    )


def _merged(current: Spent, previous: Spent) -> Spent:
    """One window's counters across the two key names it may be spread over.

    Under `money_precision` because both totals reach here exact, and plain addition would
    take the application's `Decimal` precision rather than the one the counters are held at.
    """
    with money_precision():
        return Spent(
            usd=current.usd + previous.usd,
            tokens=current.tokens + previous.tokens,
            requests=current.requests + previous.requests,
            unpriced_requests=current.unpriced_requests + previous.unpriced_requests,
        )


@dataclass
class RedisSpendStore:
    """Spend counters in Redis, so every worker enforces one budget.

    One hash per window, holding the four counters as integers.

    ```python
    from redis.asyncio import Redis

    from pydantic_ai_harness.spend import RedisSpendStore

    store = RedisSpendStore(Redis.from_url('redis://localhost'))
    ```

    A read and the increment that follows it are separate round trips, so
    concurrent runs can each observe a budget as unexhausted and push past it
    together. That is the same overshoot the in-process store has, widened by
    the number of workers; see the README on what the gate does and does not
    guarantee.
    """

    client: RedisClient
    """Any client exposing `hgetall` and `eval`."""

    prefix: str = 'pydantic-ai-harness:spend'
    """Namespace for the keys, so a shared Redis stays tidy.

    Every key this store writes is `{prefix}:...` with the braces literal, which is a
    Redis Cluster hash tag: the slot is computed from the prefix alone, so all of a
    store's keys land in one slot and a script may take several of them at once.
    Applying one response to a day and a month window in one script is what that buys,
    and the cost is that a cluster cannot spread this store's keys across its nodes.

    A prefix carrying a brace of its own is refused at construction, since it would move
    the tag and put two windows of one budget in different slots.
    """

    dedup_retain: timedelta | None = DEFAULT_DEDUP_RETAIN
    """How long an applied `SpendEntry.token` is remembered, or `None` to apply every entry.

    This is the window a replay is recognised in, not the counter's lifetime: a response
    replayed later than this is counted again, because the marker it would have matched
    has expired. The counter usually lives far longer, since every write extends it.

    Each remembered token is one small key per response per window. Raise it where a
    durable engine may recover long after the fact; lower it where the write rate makes
    that memory matter more.
    """

    def __post_init__(self) -> None:
        """Reject a prefix that would break the hash tag it is wrapped in, and report dead overrides.

        A brace inside the prefix moves or truncates the tag, so two windows of one
        budget would hash to different slots and a cluster would refuse the script that
        applies them together. Checked here for the same reason `Budget.name` is checked
        against its separator: the failure otherwise arrives as a `CROSSSLOT` error on a
        model request.
        """
        if not self.prefix or '{' in self.prefix or '}' in self.prefix:
            raise UserError(
                f'RedisSpendStore.prefix must be non-empty and must not contain braces; got {self.prefix!r}. '
                'The prefix is wrapped in a Redis Cluster hash tag: a brace inside it would change which slot '
                'the keys hash to, and an empty one leaves `{}`, which Redis Cluster does not read as a tag at '
                'all -- it would hash each key whole and refuse a script spanning two windows with `CROSSSLOT`.'
            )
        warn_unreachable_overrides(self, RedisSpendStore)

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
        """Add to `key` and return the result. Deprecated in favour of `add_many`, removed in 0.28.0.

        One window per call, so a response counting against a day and a month budget is
        two calls and a failure between them leaves the day counted and the month not.
        `add_many` is one script over every window, which is what closes that.
        """
        entry = SpendEntry(key=key, usd=usd, tokens=tokens, requests=requests, unpriced=unpriced, ttl=ttl)
        return (await self.add_many([entry]))[key]

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, Spent]:
        """What each key has accumulated. An absent hash reads as zero.

        Two round trips per key while the pre-hash-tag fallback is in place: the key's own
        hash, and the one an earlier release would have written; see `_before_hash_tags`.
        """
        totals: dict[str, Spent] = {}
        for key in keys:
            current = _spent(await self.client.hgetall(self._name(key)))
            totals[key] = _merged(current, await self._before_hash_tags(key))
        return totals

    async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
        """Apply every entry as one script and return each key's new total.

        One unit of work: every window of the response lands or none does, and the script
        returns each new total, so the totals need no second read. What does cost a read
        is the pre-hash-tag fallback, one per key, until it goes away; see `_before_hash_tags`.

        A failure before the server runs the script -- the client cannot connect, the
        request never lands -- writes nothing. A failure after it does not say which:
        the connection can drop once `EVAL` has already committed, so an error here
        means the outcome is unknown rather than that nothing happened. Retrying
        therefore risks counting the response twice, which is why
        `SpendLimits.wrap_model_request` does not retry and lets the error end the run
        instead. Over-counting a response the provider did bill is the direction a
        brake can survive; under-counting is not.

        A `SpendEntry.token` makes the retry that *is* safe: a durable engine
        re-executing an accrual it already committed finds the marker for it already
        written, so the entry is skipped and the current total is returned.
        """
        if not entries:
            return {}
        applied = await self._apply(entries)
        return {key: _merged(spent, await self._before_hash_tags(key)) for key, spent in applied.items()}

    async def _apply(self, entries: Sequence[SpendEntry]) -> dict[str, Spent]:
        """Run `_ADD_SCRIPT` over the entries, returning each key's new totals.

        Entries sharing a key are applied in order and the last row holds the running
        total, which is what a caller reading the result by key wants.
        """
        keys = [self._name(entry.key) for entry in entries]
        markers = [self._marker_name(entry) for entry in entries]
        arguments: list[str | int] = [len(entries)]
        for entry in entries:
            arguments += [
                _to_nanos(entry.usd),
                entry.tokens,
                entry.requests,
                entry.unpriced,
                0 if entry.ttl is None else _expiry_seconds(entry.ttl),
                self._marker_seconds(entry),
            ]
        rows = await self.client.eval(_ADD_SCRIPT, 2 * len(entries), *keys, *markers, *arguments)
        return {
            entry.key: Spent(
                usd=_from_nanos(int(_text(row[0]))),
                tokens=int(_text(row[1])),
                requests=int(_text(row[2])),
                unpriced_requests=int(_text(row[3])),
            )
            for entry, row in zip(entries, rows)
        }

    # `_before_hash_tags` and `_legacy_name` carry counters written before the keys gained a
    # hash tag. Delete both and their three call sites (`get_many` and the two in `add_many`)
    # in 0.28.0 -- but not as a bare deletion: a `total` window never expires and a
    # `retain='forever'` one does not either, so whatever is still under the old name is
    # subtracted from the enforced total the moment the fallback goes. What that release owes
    # an operator is settled in
    # <https://github.com/pydantic/pydantic-ai-harness/issues/694>.
    async def _before_hash_tags(self, key: str) -> Spent:
        """What this budget key accumulated under the name an earlier release used.

        Added to what the tagged key holds rather than moved into it. Moving it would have
        to decide when the move is complete, and nothing here can know that: a worker
        still running the old version can write to the old name at any time during a
        rolling deploy, and a move that already ran would never pick that up. Summing has
        no such moment -- the old name is read every time until it expires or an operator
        removes it, so a late write to it is counted like any other.

        The read is a separate round trip rather than part of the script because the old
        name has no hash tag, so it lands in a different slot and a cluster would refuse a
        script spanning both.
        """
        return _spent(await self.client.hgetall(self._legacy_name(key)))

    def _legacy_name(self, key: str) -> str:
        """The Redis key an earlier release wrote this budget key under."""
        return f'{self.prefix}:{key}'

    def _name(self, key: str) -> str:
        """The Redis key for a budget key, hash-tagged so one script may take several."""
        return self._tagged(key)

    def _tagged(self, name: str) -> str:
        """`name` under this store's hash tag, whether it is a counter or a marker."""
        return f'{{{self.prefix}}}:{name}'

    def _marker_name(self, entry: SpendEntry) -> str:
        """The key written to record that this entry's response reached this window.

        The key and the token are length-prefixed rather than joined on a separator: both
        can contain anything, so `key='a|b'` with `token='c'` and `key='a'` with
        `token='b|c'` would otherwise name the same marker, and the second response would
        be dropped as a replay of the first.

        An entry with no token still needs a name here, because the script reads a marker
        per entry: it gets one nothing writes to, since its horizon is zero.
        """
        if entry.token is None:
            return self._tagged(_MARKER)
        return self._tagged(f'{_MARKER}{SEPARATOR}{delimited(entry.key, entry.token)}')

    def _marker_seconds(self, entry: SpendEntry) -> int:
        """How long this entry's marker is held, or zero to apply the entry unconditionally.

        `dedup_retain` is the whole of the guarantee: a response replayed within it is
        applied once, and a response replayed after it is counted again. A marker has to
        expire for the memory it costs to be bounded, and the counter it guards usually
        outlives it by a long way, since every write to a window extends that window's
        horizon by `Budget.retain`.

        Capped at this entry's horizon on top of that, so a marker cannot outlive the
        counter. One that did would skip the replay of a response against a counter that
        had already rolled over, and the window would read as zero rather than as the
        response it should hold. The cap only bites where `Budget.retain` is shorter than
        `dedup_retain`, and it can only shorten the window a replay is recognised in.

        Both err towards counting a billed response twice rather than not at all, which is
        the preference `add` states: over-counting is a brake that trips early, and
        under-counting is one that releases late.
        """
        if entry.token is None or self.dedup_retain is None:
            return 0
        retain = self.dedup_retain if entry.ttl is None else min(self.dedup_retain, entry.ttl)
        return _expiry_seconds(retain)
