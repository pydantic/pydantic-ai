from __future__ import annotations

import os
from collections.abc import AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import cached_async_http_client

try:
    from dbos import DBOS, DBOSConfig

    from pydantic_ai.durable_exec.dbos import DBOSAgent
except ImportError:  # pragma: lax no cover
    import pytest

    pytest.skip('DBOS is not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    import pytest

    pytest.skip('openai not installed', allow_module_level=True)

import pytest
from inline_snapshot import snapshot

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='dbos'),
]


# We need to use a custom cached HTTP client here as the default one created for OpenAIProvider will be closed automatically
# at the end of each test, but we need this one to live longer.
http_client = cached_async_http_client(provider='dbos')


@pytest.fixture(autouse=True, scope='module')
async def close_cached_httpx_client(anyio_backend: str) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await http_client.aclose()


# `LogfirePlugin` calls `logfire.instrument_pydantic_ai()`, so we need to make sure this doesn't bleed into other tests.
@pytest.fixture(autouse=True, scope='module')
def uninstrument_pydantic_ai() -> Iterator[None]:
    try:
        yield
    finally:
        Agent.instrument_all(False)


@contextmanager
def workflow_raises(exc_type: type[Exception], exc_message: str) -> Iterator[None]:
    """Helper for asserting that a DBOS workflow fails with the expected error."""
    with pytest.raises(Exception) as exc_info:
        yield
    assert isinstance(exc_info.value.__cause__, Exception)
    # TODO (Qian): check if this is the right way to check the cause
    assert exc_info.value.__cause__.__class__.__name__ == exc_type.__name__
    assert exc_info.value.__cause__ == exc_message


DBOS_DATABASE_URL = os.environ.get('DBOS_DATABASE_URL', 'postgresql://postgres:dbos@localhost:5432/postgres')
DBOS_CONFIG: DBOSConfig = {
    'name': 'pydantic_dbos_tests',
    'database_url': DBOS_DATABASE_URL,
    'system_database_url': DBOS_DATABASE_URL,
    'run_admin_server': False,
}


@pytest.fixture()
def dbos() -> Generator[DBOS, Any, None]:
    dbos = DBOS(config=DBOS_CONFIG, conductor_key=os.environ.get('DBOS_CONDUCTOR_KEY'))
    DBOS.launch()
    yield dbos
    DBOS.destroy()


async def test_simple_agent_run_in_workflow(allow_model_requests: None, dbos: DBOS, openai_api_key: str) -> None:
    """Test that a simple agent can run in a DBOS workflow."""

    model = OpenAIModel(
        'gpt-4o',
        provider=OpenAIProvider(
            api_key=openai_api_key,
            http_client=http_client,
        ),
    )

    simple_agent = Agent(model, name='simple_agent')
    simple_dbos_agent = DBOSAgent(simple_agent)

    # Add 20 second sleep to let test hang
    # import asyncio
    # await asyncio.sleep(20)

    result = await simple_dbos_agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')
