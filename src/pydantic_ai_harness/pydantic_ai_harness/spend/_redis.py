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
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Protocol, runtime_checkable

from pydantic_ai_harness.spend._snapshot import Spent

_SCALE = Decimal(10) ** 9
_PRECISION = 40

_USD_FIELD = 'usd_nanos'
_TOKENS_FIELD = 'tokens'
_REQUESTS_FIELD = 'requests'
_UNPRICED_FIELD = 'unpriced'

_ADD_SCRIPT = f"""
local current = tonumber(redis.call('HGET', KEYS[1], '{_USD_FIELD}') or '0')
if current + tonumber(ARGV[1]) >= {2**53} then
  return redis.error_reply('spend usd counter would pass the exact-integer range')
end
local usd = redis.call('HINCRBY', KEYS[1], '{_USD_FIELD}', ARGV[1])
local tokens = redis.call('HINCRBY', KEYS[1], '{_TOKENS_FIELD}', ARGV[2])
local requests = redis.call('HINCRBY', KEYS[1], '{_REQUESTS_FIELD}', ARGV[3])
local unpriced = redis.call('HINCRBY', KEYS[1], '{_UNPRICED_FIELD}', ARGV[4])
local ttl = tonumber(ARGV[5])
if ttl > 0 then
  redis.call('EXPIRE', KEYS[1], ttl)
else
  redis.call('PERSIST', KEYS[1])
end
return {{usd, tokens, requests, unpriced}}
"""
"""Applies one response to a window and returns the four totals.

Redis runs a script to completion without interleaving another command, so no other
client observes the four counters part-applied, and the ceiling below is checked
against the running total before anything is written. Issued as separate commands
instead, a failure between them leaves a window counting some of a response and not
the rest, which reads as a smaller number than was really spent and so releases the
brake later than it should.

The ceiling is checked against the running total rather than the increment. Checking
the increment alone in Python would not hold: increments that each pass can still
carry the counter past the range where a Lua number is an exact integer, and past
that the totals this returns are rounded.

Not a transaction, though: Redis does not roll back a script that fails partway, so
a command erroring after an earlier one succeeded would leave the hash half-applied.
What could error there is a counter passing the 64-bit signed range -- around 9.2
quintillion tokens or requests against one key -- which is not a number any
deployment produces. Guarding it would mean reading three more fields on the hot
path of every model request to prevent something unreachable.

A zero `ttl` means "keep this key", and says so with `PERSIST` rather than by doing
nothing. `HINCRBY` leaves an existing expiry in place, so a budget moved from a
finite `retain` to `'forever'` would otherwise keep expiring on the old horizon --
handing back the ceiling on a schedule nothing in the configuration mentions any
more, and diverging from `InMemorySpendStore`, which drops the expiry on the next
write.
"""

_LUA_EXACT_INTEGER = 2**53
"""Above this a Lua 5.1 number stops being an exact integer.

The script's counters pass through Lua, so a window that accumulates more than
this many billionths of a dollar -- about $9,007,199 against a *single* key --
would start rounding. `_ADD_SCRIPT` refuses the write that would cross it rather
than letting the rounding happen unannounced.
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

    def eval(self, script: str, numkeys: int, *keys_and_args: str | int) -> Awaitable[Sequence[int]]:
        """Run a Lua script server-side and return what it returns."""
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
    with localcontext() as context:
        context.prec = _PRECISION
        return int((usd * _SCALE).to_integral_value(rounding=ROUND_HALF_UP))


def _from_nanos(nanos: int) -> Decimal:
    """Whole billionths back to US dollars."""
    with localcontext() as context:
        context.prec = _PRECISION
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


def _field(fields: Mapping[str | bytes, str | bytes], name: str) -> int:
    """One integer field of a hash, treating an absent field as zero."""
    for key, value in fields.items():
        decoded = key.decode() if isinstance(key, bytes) else key
        if decoded == name:
            return int(value.decode() if isinstance(value, bytes) else value)
    return 0


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
    """Namespace for the keys, so a shared Redis stays tidy."""

    async def get(self, key: str) -> Spent:
        """What `key` has accumulated. An absent hash reads as zero."""
        fields = await self.client.hgetall(self._name(key))
        return Spent(
            usd=_from_nanos(_field(fields, _USD_FIELD)),
            tokens=_field(fields, _TOKENS_FIELD),
            requests=_field(fields, _REQUESTS_FIELD),
            unpriced_requests=_field(fields, _UNPRICED_FIELD),
        )

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

        One round trip, and one window. The script returns each new total, so the
        result needs no second read.

        A failure before the server runs the script -- the client cannot connect, the
        request never lands -- writes nothing. A failure after it does not say which:
        the connection can drop once `EVAL` has already committed, so an error here
        means the outcome is unknown rather than that nothing happened. Retrying an
        `add` therefore risks counting the response twice, which is why
        `SpendLimits.wrap_model_request` does not retry and lets the error end the run
        instead. Over-counting a response the provider did bill is the direction a
        brake can survive; under-counting is not.

        One window, because the key is one window. A response counting against a day and
        a month budget is two calls, so a failure between them leaves the day counted and
        the month not; see `SpendLimits.wrap_model_request` and
        <https://github.com/pydantic/pydantic-ai-harness/issues/536>.
        """
        nanos, total_tokens, total_requests, total_unpriced = await self.client.eval(
            _ADD_SCRIPT,
            1,
            self._name(key),
            _to_nanos(usd),
            tokens,
            requests,
            unpriced,
            0 if ttl is None else _expiry_seconds(ttl),
        )
        return Spent(
            usd=_from_nanos(nanos),
            tokens=total_tokens,
            requests=total_requests,
            unpriced_requests=total_unpriced,
        )

    def _name(self, key: str) -> str:
        """The Redis key for a budget key, namespaced so a shared server stays tidy."""
        return f'{self.prefix}:{key}'
