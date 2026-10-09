"""Which runs reuse, and which close, the HTTP clients of models an agent builds from a model name.

These tests replace the transport's `handle_async_request` instead of replaying a cassette, and build
every provider-owned client fresh instead of through the per-test client cache in `conftest.py`, so
that what they count and check is each client the provider itself creates.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai._http import create_async_httpx2_client
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.models import Model, ModelResolutionContext

from .conftest import try_import

with try_import() as imports_successful:
    import pydantic_ai.providers._openai_compatible as openai_compatible
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]

_OPENAI_RESPONSE: dict[str, Any] = {
    'id': 'chatcmpl-1',
    'object': 'chat.completion',
    'created': 0,
    'model': 'gpt-5.2',
    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Paris'}, 'finish_reason': 'stop'}],
    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
}


@pytest.fixture
async def provider_clients(
    monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
) -> AsyncIterator[list[httpx2.AsyncClient]]:
    """Every HTTP client an OpenAI provider creates for itself, each one new, answered without a network."""
    clients: list[httpx2.AsyncClient] = []

    def create_client() -> httpx2.AsyncClient:
        client = create_async_httpx2_client()
        clients.append(client)
        return client

    async def handle(self: httpx2.AsyncHTTPTransport, request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_OPENAI_RESPONSE)

    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setattr(openai_compatible, 'create_async_httpx2_client', create_client)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, 'handle_async_request', handle)
    yield clients
    for client in clients:
        await client.aclose()


async def test_run_closes_client_of_model_built_from_name(provider_clients: list[httpx2.AsyncClient]):
    """A model a run builds from a name belongs to that run, which closes its client when it ends."""
    agent = Agent()

    result = await agent.run('What is the capital of France?', model='openai-chat:gpt-5.2')
    assert result.output == 'Paris'
    assert [client.is_closed for client in provider_clients] == [True]

    async with agent.iter('What is the capital of France?', model='openai-chat:gpt-5.2') as agent_run:
        async for _ in agent_run:
            pass
        assert [client.is_closed for client in provider_clients] == [True, False]
    assert [client.is_closed for client in provider_clients] == [True, True]


async def test_entered_agent_reuses_model_built_from_name(provider_clients: list[httpx2.AsyncClient]):
    """Inside `async with agent:`, runs share one model per name, which the agent closes on exit."""
    agent = Agent()

    async with agent:
        await agent.run('What is the capital of France?', model='openai-chat:gpt-5.2')
        await agent.run('What is the capital of France?', model='openai-chat:gpt-5.2')
        # Concurrent runs that are the first to use a name still build a single model for it.
        await asyncio.gather(
            agent.run('What is the capital of France?', model='openai-chat:gpt-5-mini'),
            agent.run('What is the capital of France?', model='openai-chat:gpt-5-mini'),
        )
        assert [client.is_closed for client in provider_clients] == [False, False]
    assert [client.is_closed for client in provider_clients] == [True, True]

    # Entering the agent again starts over with new models.
    async with agent:
        await agent.run('What is the capital of France?', model='openai-chat:gpt-5.2')
    assert [client.is_closed for client in provider_clients] == [True, True, True]


async def test_entered_agent_reuses_its_deferred_model_name(provider_clients: list[httpx2.AsyncClient]):
    """An agent's model name whose check is deferred is built per run, or once while the agent is entered."""
    agent = Agent('openai-chat:gpt-5.2', defer_model_check=True)

    await agent.run('What is the capital of France?')
    assert [client.is_closed for client in provider_clients] == [True]

    async with agent:
        await agent.run('What is the capital of France?')
        await agent.run('What is the capital of France?', model='openai-chat:gpt-5.2')
        assert [client.is_closed for client in provider_clients] == [True, False]
    assert [client.is_closed for client in provider_clients] == [True, True]


async def test_run_leaves_client_of_passed_model_open(provider_clients: list[httpx2.AsyncClient]):
    """A `Model` instance passed to a run is the caller's, so the run doesn't close its client."""
    model = OpenAIChatModel('gpt-5.2', provider=OpenAIProvider())
    agent = Agent()

    await agent.run('What is the capital of France?', model=model)
    await agent.run('What is the capital of France?', model=model)
    assert [client.is_closed for client in provider_clients] == [False]


async def test_names_resolved_through_capability_are_not_reused(provider_clients: list[httpx2.AsyncClient]):
    """With a `resolve_model_id` capability, a name can resolve differently per run, so it's resolved every run."""
    seen: list[str] = []

    def resolve(ctx: ModelResolutionContext[Any], model_id: str) -> Model | None:
        seen.append(model_id)
        return None

    agent = Agent(capabilities=[ResolveModelId(resolve)])

    async with agent:
        await agent.run('What is the capital of France?', model='openai-chat:gpt-5.2')
        await agent.run('What is the capital of France?', model='openai-chat:gpt-5.2')
    assert seen == ['openai-chat:gpt-5.2', 'openai-chat:gpt-5.2']
    assert len(provider_clients) == 2
