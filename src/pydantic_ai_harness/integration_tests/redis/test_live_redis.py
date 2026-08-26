"""Live Redis conformance tests for `RedisSpendStore`.

The unit suite in `tests/spend` drives the store through a fake that reproduces the
result `_ADD_SCRIPT` is meant to produce. It never runs the Lua, so what it cannot
answer decides whether the store is correct:

- that the script is valid Lua the server accepts at all;
- that a total past `2**53` billionths comes back exact, which is the whole reason
  the totals are read with `HMGET` instead of taken from what `HINCRBY` returns;
- that one script applies every window of a response, each with its own horizon;
- that a counter passing the signed 64-bit range is refused before that field is
  written, rather than rounded;
- that a repeated `SpendEntry.token` claims nothing and adds nothing;
- that a zero `ttl` clears an expiry an earlier finite `retain` set, which is
  `PERSIST` doing work `HINCRBY` would not do on its own;
- that a counter written under the untagged key an earlier release used is still read
  and carried forward.

This file covers exactly those. It is not a second copy of the unit suite --
API-shape coverage belongs there, where it runs on every matrix leg.

Run against a local server with `make integration-redis` after starting one, e.g.
`docker run -d -p 6379:6379 redis:8`. Without a reachable server the tests skip,
unless `REDIS_REQUIRE_LIVE` is set (CI does), where an unreachable server fails
instead -- a service container that never came up must not pass as a silent skip.

External assumptions, verified 2026-08-05 against Redis 8 (`redis:8`):

- `HINCRBY` is 64-bit integer arithmetic and takes its increment as a string, so a
  stored counter is exact whatever Lua could hold; it errors with `increment or
  decrement would overflow` past the signed range, without writing. Source:
  <https://redis.io/docs/latest/commands/hincrby/>.
- `HINCRBY` does not alter a key's TTL. Source:
  <https://redis.io/docs/latest/commands/hincrby/>.
- `PERSIST` removes an expiry and returns 0 when the key has none, so calling it
  unconditionally on the keep-forever path is safe. Source:
  <https://redis.io/docs/latest/commands/persist/>.
- `EVAL` runs the script to completion without interleaving another client's
  command, but does not roll back on a mid-script error. Source:
  <https://redis.io/docs/latest/develop/programmability/eval-intro/>.
- A Lua string returned from a script becomes a bulk string, and a Lua number
  becomes an integer via a double. Source:
  <https://redis.io/docs/latest/develop/programmability/lua-api/>.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from datetime import timedelta
from decimal import Decimal
from typing import NoReturn
from urllib.parse import urlsplit

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from pydantic_ai_harness.spend import RedisSpendStore, SpendEntry, Spent

pytestmark = pytest.mark.anyio

_NANOS = Decimal(10) ** 9
"""Billionths of a dollar to the dollar, which is how the store keeps money."""

_ONE_NANO = Decimal(1) / _NANOS
"""The smallest amount the store can hold, so a total can be moved by exactly one unit."""


@pytest.fixture
def anyio_backend() -> str:
    """Run live server tests once under asyncio."""
    return 'asyncio'


def _requires_live() -> bool:
    return os.environ.get('REDIS_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}


def _redis_url() -> str:
    return os.environ.get('REDIS_TEST_URL', 'redis://127.0.0.1:6379')


def _redis_target() -> str:
    """Host and port, without the credentials a Redis URL may carry.

    `REDIS_TEST_URL` accepts `redis://user:password@host:6379`, and the only place
    this string is used is a skip or failure message -- which CI keeps. Naming the
    variable rather than echoing an unparsable value keeps that true for a URL
    `urlsplit` cannot make sense of.
    """
    host = urlsplit(_redis_url()).hostname
    if host is None:
        return 'the server named by REDIS_TEST_URL'
    port = urlsplit(_redis_url()).port
    return host if port is None else f'{host}:{port}'


def _unavailable(message: str) -> NoReturn:
    if _requires_live():
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(params=[True, False], ids=['decoded', 'bytes'])
async def store(request: pytest.FixtureRequest) -> AsyncGenerator[RedisSpendStore, None]:
    """A store on a reachable server, namespaced per test so runs cannot collide.

    Run both ways round: `decode_responses` decides whether the script's rows arrive as
    `str` or as `bytes`, and the store has to read a total out of either. That is a
    property of redis-py's parser rather than of the fake, so it belongs here.
    """
    client = Redis.from_url(_redis_url(), decode_responses=request.param)
    try:
        await client.ping()
    except (RedisError, OSError) as error:
        await client.aclose()
        _unavailable(f'no reachable Redis at {_redis_target()}: {error}')

    prefix = f'harness-test:{uuid.uuid4().hex}'
    try:
        yield RedisSpendStore(client, prefix=prefix)
    finally:
        # Closing sits in its own `finally` so a server that went away mid-test takes the
        # key cleanup down without also leaking the connection into the rest of the suite.
        try:
            keys = [key async for key in client.scan_iter(match=f'*{prefix}*')]
            if keys:
                await client.delete(*keys)
        finally:
            await client.aclose()


async def _ttl(store: RedisSpendStore, key: str) -> int:
    """Seconds left on a budget key. -1 means no expiry, -2 means the key is gone."""
    return await store.client.ttl(f'{{{store.prefix}}}:{key}')  # pyright: ignore[reportAttributeAccessIssue]


class TestLiveScript:
    """What the fake reproduces rather than executes."""

    async def test_the_script_runs_and_returns_the_four_totals(self, store: RedisSpendStore):
        """The whole point of the Lua: valid on the server, and no second read to get the result."""
        first = await store.add('k', usd=Decimal('0.000123456'), tokens=7, requests=1, unpriced=1, ttl=None)
        assert first == Spent(usd=Decimal('0.000123456'), tokens=7, requests=1, unpriced_requests=1)

        second = await store.add('k', usd=Decimal('0.000000675'), tokens=3, requests=1, unpriced=0, ttl=None)
        assert second == Spent(usd=Decimal('0.000124131'), tokens=10, requests=2, unpriced_requests=1)
        assert await store.get('k') == second

    async def test_a_finite_horizon_sets_an_expiry(self, store: RedisSpendStore):
        """The baseline the `'forever'` case below is measured against."""
        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=1))

        assert 0 < await _ttl(store, 'k') <= 3600

    async def test_moving_a_budget_to_forever_clears_the_old_horizon(self, store: RedisSpendStore):
        """`HINCRBY` leaves an expiry alone, so skipping `EXPIRE` is not the same as no expiry.

        Reconfiguring a counter from a finite `retain` to `'forever'` left it expiring on
        a horizon the configuration no longer mentions, handing the ceiling back on a
        schedule. This is the assertion the fake cannot make on its own.
        """
        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=1))
        assert await _ttl(store, 'k') > 0

        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=None)

        assert await _ttl(store, 'k') == -1

    async def test_a_sub_second_horizon_still_expires(self, store: RedisSpendStore):
        """`EXPIRE` takes whole seconds, and rounding down would leave the key forever."""
        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(milliseconds=500))

        assert await _ttl(store, 'k') == 1

    async def test_a_total_past_lua_s_exact_integer_range_is_exact(self, store: RedisSpendStore):
        """The reason the totals are read back with `HMGET` rather than taken from `HINCRBY`.

        `HINCRBY` returns an integer reply, which becomes a Lua number, and Lua 5.1
        numbers are doubles: past `2**53` the reply rounds even though Redis holds the
        counter exactly. A bulk string has no double in its path.
        """
        below = Decimal(2**53 - 1) / _NANOS
        added = await store.add('k', usd=below, tokens=1, requests=1, unpriced=0, ttl=None)
        assert added.usd == below

        crossed = await store.add('k', usd=_ONE_NANO, tokens=1, requests=1, unpriced=0, ttl=None)

        assert crossed.usd == Decimal(2**53) / _NANOS
        assert (await store.get('k')).usd == crossed.usd

    async def test_a_counter_past_the_64_bit_range_is_refused_before_it_is_written(self, store: RedisSpendStore):
        """`HINCRBY`'s own ceiling, around $9.22 billion against one key, reported by Redis.

        It refuses the field rather than wrapping it, so the counter is left holding what
        it held rather than a number that went round the houses.
        """
        near = Decimal(2**63 - 1) / _NANOS
        before = await store.add('k', usd=near, tokens=1, requests=1, unpriced=0, ttl=None)

        with pytest.raises(ResponseError, match='overflow'):
            await store.add('k', usd=_ONE_NANO, tokens=1, requests=1, unpriced=0, ttl=None)

        assert await store.get('k') == before

    async def test_every_window_of_a_response_is_one_script(self, store: RedisSpendStore):
        """The #536 fix: no failure between two windows, and each keeps its own horizon."""
        totals = await store.add_many(
            [
                SpendEntry(key='day', usd=Decimal('0.5'), tokens=5, requests=1, ttl=timedelta(hours=48)),
                SpendEntry(key='month', usd=Decimal('0.5'), tokens=5, requests=1, ttl=timedelta(days=62)),
                SpendEntry(key='total', usd=Decimal('0.5'), tokens=5, requests=1, ttl=None),
            ]
        )

        assert totals == {
            'day': Spent(usd=Decimal('0.5'), tokens=5, requests=1),
            'month': Spent(usd=Decimal('0.5'), tokens=5, requests=1),
            'total': Spent(usd=Decimal('0.5'), tokens=5, requests=1),
        }
        assert 0 < await _ttl(store, 'day') <= 172_800
        assert 172_800 < await _ttl(store, 'month') <= 5_356_800
        assert await _ttl(store, 'total') == -1

    async def test_a_repeated_token_adds_nothing(self, store: RedisSpendStore):
        """The marker is read and written inside the same script, so a replay is one round trip."""
        entry = SpendEntry(key='k', usd=Decimal('1'), tokens=5, requests=1, token='resp-1')

        first = await store.add_many([entry])
        second = await store.add_many([entry])

        assert first == second == {'k': Spent(usd=Decimal('1'), tokens=5, requests=1)}
        assert await store.get('k') == first['k']

    async def test_an_overflow_leaves_no_marker_to_skip_the_retry(self, store: RedisSpendStore):
        """The marker is written after the increments, not claimed before them.

        Redis does not roll back a script, so a marker claimed up front would survive the
        overflow that aborted the script and the retry that could have completed the
        window would be skipped as a replay of a response that never fully landed.
        """
        await store.add('k', usd=Decimal(2**63 - 1) / _NANOS, tokens=0, requests=0, unpriced=0, ttl=None)
        entry = SpendEntry(key='k', usd=_ONE_NANO, tokens=1, requests=1, token='resp-1')

        with pytest.raises(ResponseError, match='overflow'):
            await store.add_many([entry])

        recovered = await store.add_many([SpendEntry(key='k', tokens=1, requests=1, token='resp-1')])

        assert recovered['k'].requests == 1

    async def test_a_counter_written_before_the_hash_tag_is_added_to_the_one_after_it(self, store: RedisSpendStore):
        """Keys gained a hash tag, so an upgrade must not strand what an earlier release counted.

        Written here the way an earlier release wrote it: the same name without the tag.
        The second write models a worker still running the old version during a rolling
        deploy, which is why the old name is read every time rather than moved once.
        """
        await store.client.hset(  # pyright: ignore[reportAttributeAccessIssue]
            f'{store.prefix}:k', mapping={'usd_nanos': 3_000_000_000, 'tokens': 8, 'requests': 2}
        )
        assert await store.get('k') == Spent(usd=Decimal('3'), tokens=8, requests=2)

        totals = await store.add_many([SpendEntry(key='k', usd=Decimal('1'), tokens=1, requests=1)])
        assert totals == {'k': Spent(usd=Decimal('4'), tokens=9, requests=3)}

        await store.client.hincrby(f'{store.prefix}:k', 'requests', 1)  # pyright: ignore[reportAttributeAccessIssue]

        assert await store.get('k') == Spent(usd=Decimal('4'), tokens=9, requests=4)
