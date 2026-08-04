"""Live Redis conformance tests for `RedisSpendStore`.

The unit suite in `tests/spend` drives the store through a fake that reproduces the
result `_ADD_SCRIPT` is meant to produce. It never runs the Lua, so three things it
cannot answer decide whether the store is correct:

- that the script is valid Lua the server accepts at all;
- that a zero `ttl` clears an expiry an earlier finite `retain` set, which is
  `PERSIST` doing work `HINCRBY` would not do on its own;
- that the exact-integer guard refuses the write before any field is incremented,
  rather than after some of them.

This file covers exactly those. It is not a second copy of the unit suite --
API-shape coverage belongs there, where it runs on every matrix leg.

Run against a local server with `make integration-redis` after starting one, e.g.
`docker run -d -p 6379:6379 redis:8`. Without a reachable server the tests skip,
unless `REDIS_REQUIRE_LIVE` is set (CI does), where an unreachable server fails
instead -- a service container that never came up must not pass as a silent skip.

External assumptions, verified 2026-08-04 against Redis 8 (`redis:8`):

- `HINCRBY` does not alter a key's TTL. Source:
  <https://redis.io/docs/latest/commands/hincrby/>.
- `PERSIST` removes an expiry and returns 0 when the key has none, so calling it
  unconditionally on the keep-forever path is safe. Source:
  <https://redis.io/docs/latest/commands/persist/>.
- `EVAL` runs the script to completion without interleaving another client's
  command, but does not roll back on a mid-script error. Source:
  <https://redis.io/docs/latest/develop/programmability/eval-intro/>.
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

from pydantic_ai_harness.spend import RedisSpendStore, Spent

pytestmark = pytest.mark.anyio


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


@pytest.fixture
async def store() -> AsyncGenerator[RedisSpendStore, None]:
    """A store on a reachable server, namespaced per test so runs cannot collide."""
    client = Redis.from_url(_redis_url(), decode_responses=True)
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
            keys = [key async for key in client.scan_iter(match=f'{prefix}:*')]
            if keys:
                await client.delete(*keys)
        finally:
            await client.aclose()


async def _ttl(store: RedisSpendStore, key: str) -> int:
    """Seconds left on a budget key. -1 means no expiry, -2 means the key is gone."""
    return await store.client.ttl(f'{store.prefix}:{key}')  # pyright: ignore[reportAttributeAccessIssue]


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

    async def test_the_exact_integer_guard_refuses_before_writing_anything(self, store: RedisSpendStore):
        """Checked against the running total inside the script, ahead of the first `HINCRBY`.

        Past `2**53` billionths a Lua number stops being an exact integer, so the totals
        the script returns would start rounding. The refusal has to leave the counter
        untouched, or the window would hold part of the response it just rejected.
        """
        half = Decimal('4600000')
        before = await store.add('k', usd=half, tokens=1, requests=1, unpriced=0, ttl=None)

        with pytest.raises(ResponseError, match='exact-integer range'):
            await store.add('k', usd=half, tokens=1, requests=1, unpriced=0, ttl=None)

        assert await store.get('k') == before
