"""`plugin_keys.on_loop`: a settings menu's worker thread running an async flow on the event loop."""

import asyncio

import pytest
from anyio import to_thread

from pydantic_clai2.plugin_keys import on_loop

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'  # `on_loop` bridges to an asyncio loop, as CLAI runs on one.


async def test_a_flows_own_timeout_reaches_the_menu_instead_of_being_polled_forever() -> None:
    # `concurrent.futures.TimeoutError` is the builtin since Python 3.11, so it must not double as "still waiting".
    async def timed_out() -> str:
        raise TimeoutError('the service did not answer')

    loop = asyncio.get_running_loop()
    with pytest.raises(TimeoutError, match='the service did not answer'):
        await to_thread.run_sync(on_loop, timed_out, loop)


async def test_the_flows_result_is_returned() -> None:
    async def answered() -> str:
        await asyncio.sleep(0.1)  # Past one polling interval.
        return 'done'

    assert await to_thread.run_sync(on_loop, answered, asyncio.get_running_loop()) == 'done'
