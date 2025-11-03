import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import suppress
from typing import Any, cast

import pytest
import pytest_asyncio

from ..conftest import try_import

with try_import() as imports_successful:
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from openai import AsyncOpenAI, DefaultAioHttpClient

    from pydantic_ai import Agent
    from pydantic_ai.fastapi.agent_router import AgentAPIRouter
    from pydantic_ai.fastapi.registry import AgentRegistry
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed or FastAPI not installed'),
    pytest.mark.anyio,
]


@pytest.fixture(scope='session')
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Provide an asyncio event loop for pytest-asyncio."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app configured with an AgentAPIRouter backed by a small test
    AgentRegistry. This fixture also disables the `response_model` on the /v1/responses
    route so tests can return simple dicts without matching the full complex Pydantic
    Responses model.
    """
    registry = AgentRegistry()
    registry.chat_completions_agents['test-model'] = cast(Any, object())
    registry.responses_agents['test-model'] = cast(Any, object())

    router = AgentAPIRouter(agent_registry=registry)

    for route in list(getattr(router, 'routes', [])):
        if getattr(route, 'path', None) == '/v1/responses':
            with suppress(Exception):
                route.response_model = None

    app = FastAPI()
    app.include_router(router)

    app.state.agent_router = router

    return app


@pytest.fixture
def agent_router(app: FastAPI) -> AgentAPIRouter:
    """Return the AgentAPIRouter instance attached to the app by the `app` fixture.
    Tests can use this to stub `completions_api` and `responses_api` coroutine methods.
    """
    return app.state.agent_router


@pytest_asyncio.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Provide an httpx.AsyncClient configured to talk to the FastAPI app in-process.
    Use this in async tests to make HTTP requests against the test app.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://testserver') as client:
        yield client


@pytest_asyncio.fixture
async def registry_with_openai_clients() -> AsyncGenerator[AgentRegistry, None]:
    """Build an AgentRegistry wired to real pydantic-ai Agents backed by AsyncOpenAI
    clients pointed at a test base URL.
    """
    fake_openai_base = 'https://api.openai.test/v1'

    openai_client_for_chat = AsyncOpenAI(
        base_url=fake_openai_base,
        api_key='test-key',
        http_client=DefaultAioHttpClient(),
    )
    openai_client_for_responses = AsyncOpenAI(
        base_url=fake_openai_base,
        api_key='test-key',
        http_client=DefaultAioHttpClient(),
    )

    chat_model = OpenAIChatModel(
        model_name='test-model',
        provider=OpenAIProvider(openai_client=openai_client_for_chat),
    )
    responses_model = OpenAIResponsesModel(
        model_name='test-model',
        provider=OpenAIProvider(openai_client=openai_client_for_responses),
    )

    agent_chat = Agent(model=chat_model, system_prompt='You are a helpful assistant')
    agent_responses = Agent(model=responses_model, system_prompt='You are a helpful assistant')

    registry = AgentRegistry()
    registry.chat_completions_agents['test-model'] = agent_chat
    registry.responses_agents['test-model'] = agent_responses

    registry.chat_completions_agents['test-model-only-completions'] = agent_chat
    registry.responses_agents['test-model-only-responses'] = agent_responses

    try:
        yield registry
    finally:
        await openai_client_for_chat.close()
        await openai_client_for_responses.close()
