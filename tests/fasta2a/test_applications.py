from __future__ import annotations as _annotations

from contextlib import asynccontextmanager

import httpx
import pytest
from asgi_lifespan import LifespanManager
from inline_snapshot import snapshot

from ..conftest import try_import

with try_import() as imports_successful:
    from fasta2a.applications import FastA2A
    from fasta2a.broker import InMemoryBroker
    from fasta2a.storage import InMemoryStorage


pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='fasta2a not installed')]


@asynccontextmanager
async def create_test_client(app: FastA2A):
    async with LifespanManager(app=app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url='http://testclient') as client:
            yield client


async def test_agent_card():
    app = FastA2A(storage=InMemoryStorage(), broker=InMemoryBroker())
    async with create_test_client(app) as client:
        response = await client.get('/.well-known/agent.json')
        assert response.status_code == 200
        assert response.json() == snapshot(
            {
                'name': 'Agent',
                'url': 'http://localhost:8000',
                'version': '1.0.0',
                'skills': [],
                'defaultInputModes': ['application/json'],
                'defaultOutputModes': ['application/json'],
                'capabilities': {'streaming': False, 'pushNotifications': False, 'stateTransitionHistory': False},
                'authentication': {'schemes': []},
            }
        )
