"""Which runs reuse, close or replace the HTTP clients of the models an agent uses.

These tests replace the transport's `handle_async_request` instead of replaying a cassette, and build
every provider-owned client fresh instead of through the per-test client cache in `conftest.py`, so
that what they count and check is each client the provider itself creates.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import httpx
import httpx2
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai._http import create_async_httpx2_client as create_uncached_client
from pydantic_ai.capabilities import AbstractCapability, ResolveModelId, SelectModel
from pydantic_ai.models import Model, ModelResolutionContext, ModelSelectionContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel

from .conftest import try_import

with try_import() as imports_successful:
    import pydantic_ai.providers._openai_compatible as openai_compatible
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as huggingface_imports_successful:
    from huggingface_hub import AsyncInferenceClient
    from huggingface_hub.utils import _http as huggingface_http  # pyright: ignore[reportPrivateUsage]

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

_OPENAI_RESPONSE: dict[str, Any] = {
    'id': 'chatcmpl-1',
    'object': 'chat.completion',
    'created': 0,
    'model': 'gpt-5.2',
    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Paris'}, 'finish_reason': 'stop'}],
    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
}


@pytest.fixture
def transport_loops() -> dict[httpx2.AsyncHTTPTransport, set[asyncio.AbstractEventLoop]]:
    """The event loops each HTTP transport answered requests on. A real one can only use its connections on one."""
    return {}


@pytest.fixture
def new_provider_clients(
    monkeypatch: pytest.MonkeyPatch,
    allow_model_requests: None,
    transport_loops: dict[httpx2.AsyncHTTPTransport, set[asyncio.AbstractEventLoop]],
) -> list[httpx2.AsyncClient]:
    """Every HTTP client an OpenAI provider creates for itself, each one new, answered without a network.

    A client used on an event loop that has since closed can't be closed, so unlike `provider_clients`, this
    leaves them open.
    """
    clients: list[httpx2.AsyncClient] = []

    def create_client() -> httpx2.AsyncClient:
        # Imported under another name, so the autouse `track_httpx_clients` fixture doesn't swap in its client cache.
        client = create_uncached_client()
        clients.append(client)
        return client

    async def handle(self: httpx2.AsyncHTTPTransport, request: httpx2.Request) -> httpx2.Response:
        transport_loops.setdefault(self, set()).add(asyncio.get_running_loop())
        return httpx2.Response(200, json=_OPENAI_RESPONSE)

    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setattr(openai_compatible, 'create_async_httpx2_client', create_client)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, 'handle_async_request', handle)
    return clients


@pytest.fixture
async def provider_clients(new_provider_clients: list[httpx2.AsyncClient]) -> AsyncIterator[list[httpx2.AsyncClient]]:
    """The clients of `new_provider_clients`, closed when the test ends."""
    yield new_provider_clients
    for client in new_provider_clients:
        await client.aclose()


def _run_on_new_event_loops(agent: Agent[Any, str], *, count: int) -> list[str]:
    """Run the agent with `asyncio.run()` `count` times, each on a new event loop that closes when it ends."""

    async def run() -> str:
        return (await agent.run('What is the capital of France?')).output

    return [asyncio.run(run()) for _ in range(count)]


def _openai_http_client(model: Model) -> object:
    """The HTTP client the OpenAI SDK client of the innermost wrapped model sends requests with."""
    while isinstance(model, WrapperModel):
        model = model.wrapped
    if isinstance(model, FallbackModel):
        return _openai_http_client(model.models[0])
    assert isinstance(model, OpenAIChatModel)
    return model.client._client  # pyright: ignore[reportPrivateUsage]


def _unwrapped(model: Model) -> Model:
    return model


@pytest.mark.parametrize(
    'wrap',
    [
        pytest.param(_unwrapped, id='model'),
        pytest.param(WrapperModel, id='wrapper'),
        pytest.param(FallbackModel, id='fallback'),
    ],
)
def test_unentered_agent_gets_new_client_on_new_event_loop(
    new_provider_clients: list[httpx2.AsyncClient],
    transport_loops: dict[httpx2.AsyncHTTPTransport, set[asyncio.AbstractEventLoop]],
    wrap: Callable[[Model], Model],
):
    """An agent that isn't entered keeps its model's HTTP client between runs, and replaces it on a new event loop."""
    model = wrap(OpenAIChatModel('gpt-5.2', provider=OpenAIProvider()))
    agent = Agent(model)

    loop = asyncio.new_event_loop()
    try:
        for _ in range(2):
            result = loop.run_until_complete(agent.run('What is the capital of France?'))
            assert result.output == 'Paris'
    finally:
        loop.close()
    # Runs on one event loop share the client.
    assert len(new_provider_clients) == 1

    assert _run_on_new_event_loops(agent, count=2) == ['Paris', 'Paris']
    assert len(new_provider_clients) == 3
    assert _openai_http_client(model) is new_provider_clients[2]
    assert [len(loops) for loops in transport_loops.values()] == [1, 1, 1]
    # Nothing closes a client whose event loop is gone.
    assert [client.is_closed for client in new_provider_clients] == [False, False, False]


class _CountingOpenAIServer(ThreadingHTTPServer):
    """A local HTTP/1.1 server with keep-alive that answers every chat completion, counting the connections it accepts."""

    daemon_threads = True

    def __init__(self) -> None:
        self.connections = 0
        owner = self
        body = json.dumps(_OPENAI_RESPONSE).encode()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def setup(self) -> None:
                super().setup()
                owner.connections += 1

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        super().__init__(('127.0.0.1', 0), Handler)


class _TransportTrackingEventLoop(asyncio.SelectorEventLoop):
    """An event loop that keeps the transport of each connection it opens."""

    def __init__(self, transports: list[asyncio.Transport]) -> None:
        super().__init__()
        self.transports = transports

    async def create_connection(self, *args: Any, **kwargs: Any) -> tuple[asyncio.Transport, asyncio.BaseProtocol]:
        transport, protocol = await super().create_connection(*args, **kwargs)
        self.transports.append(transport)
        return transport, protocol


def _finalize_transport_of_closed_loop(transport: asyncio.Transport) -> None:
    """Close a connection's socket after its event loop has closed, which `transport.close()` can no longer do.

    `close()` and `abort()` schedule this same finalizer on the event loop, which raises once it's closed. Left
    open, the transport warns about it whenever it's garbage collected, which can be during a later test.
    """
    cast(Any, transport)._call_connection_lost(None)


def test_unentered_agent_runs_on_new_event_loop_over_real_connections(
    monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
):
    """A second run on a new event loop used to reuse a pooled connection of the closed one: `Event loop is closed`."""
    # Imported under another name, so the autouse `track_httpx_clients` fixture doesn't swap in its client cache.
    monkeypatch.setattr(openai_compatible, 'create_async_httpx2_client', create_uncached_client)
    server = _CountingOpenAIServer()
    server_thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    provider = OpenAIProvider(base_url=f'http://127.0.0.1:{server.server_port}/v1', api_key='test-key')
    agent = Agent(OpenAIChatModel('gpt-5.2', provider=provider))
    transports: list[asyncio.Transport] = []
    server_thread.start()
    try:
        for _ in range(2):
            # Like `asyncio.run()`: each run gets an event loop of its own, closed when the run ends.
            loop = _TransportTrackingEventLoop(transports)
            try:
                result = loop.run_until_complete(agent.run('What is the capital of France?'))
                assert result.output == 'Paris'
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
        assert server.connections == 2
    finally:
        # No client can be closed once its event loop is, so close their connections here.
        for transport in transports:
            _finalize_transport_of_closed_loop(transport)
        server.shutdown()
        server.server_close()
        server_thread.join()
    assert len(transports) == 2


def test_unentered_agent_keeps_client_while_its_event_loop_runs(new_provider_clients: list[httpx2.AsyncClient]):
    """While the event loop a client was last used on still runs, as in another thread, the client isn't replaced."""
    model = OpenAIChatModel('gpt-5.2', provider=OpenAIProvider())
    agent = Agent(model)
    other_loop = asyncio.new_event_loop()
    other_thread = threading.Thread(target=other_loop.run_forever)
    other_thread.start()
    try:
        result = asyncio.run_coroutine_threadsafe(agent.run('What is the capital of France?'), other_loop).result()
        assert result.output == 'Paris'
        assert _run_on_new_event_loops(agent, count=1) == ['Paris']
        assert len(new_provider_clients) == 1
    finally:
        other_loop.call_soon_threadsafe(other_loop.stop)
        other_thread.join()
        other_loop.close()

    # Once the loop the client was last used on has stopped, a run on another one gets a new client.
    assert _run_on_new_event_loops(agent, count=1) == ['Paris']
    assert len(new_provider_clients) == 2


def test_unentered_agent_keeps_passed_in_client_on_new_event_loop(
    new_provider_clients: list[httpx2.AsyncClient],
    transport_loops: dict[httpx2.AsyncHTTPTransport, set[asyncio.AbstractEventLoop]],
):
    """An HTTP client passed to the provider is the caller's, so it's never replaced, even on another event loop."""
    http_client = create_uncached_client()
    model = OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(http_client=http_client))
    agent = Agent(model)

    assert _run_on_new_event_loops(agent, count=2) == ['Paris', 'Paris']
    assert new_provider_clients == []
    assert _openai_http_client(model) is http_client
    assert [len(loops) for loops in transport_loops.values()] == [2]


async def test_unentered_run_after_agent_exit_gets_new_client(provider_clients: list[httpx2.AsyncClient]):
    """A run outside `async with agent:` replaces the client the agent closed on exit."""
    agent = Agent(OpenAIChatModel('gpt-5.2', provider=OpenAIProvider()))

    async with agent:
        await agent.run('What is the capital of France?')
    assert [client.is_closed for client in provider_clients] == [True]

    result = await agent.run('What is the capital of France?')
    assert result.output == 'Paris'
    assert [client.is_closed for client in provider_clients] == [True, False]


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_provider_entered_without_asyncio_keeps_client(provider_clients: list[httpx2.AsyncClient]):
    """Without an asyncio event loop there's no loop to compare, so entering a provider keeps its client."""
    provider = OpenAIProvider()

    async with provider:
        pass
    async with provider:
        pass
    assert len(provider_clients) == 2


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
        assert provider_clients[0] is not provider_clients[1]
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


@dataclass
class _SwitchToTestModel(AbstractCapability[Any]):
    """Contributes no model until `for_run` replaces it with one that contributes a `TestModel`."""

    switched: bool = False

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return _SwitchToTestModel(switched=True)

    def get_model(self) -> Model | None:
        return TestModel() if self.switched else None


async def test_run_closes_client_of_model_built_from_name_it_replaced(provider_clients: list[httpx2.AsyncClient]):
    """A model the run built from a name and a capability then replaced still has its client closed."""
    agent = Agent('openai-chat:gpt-5.2', defer_model_check=True, capabilities=[_SwitchToTestModel()])

    result = await agent.run('What is the capital of France?')
    assert result.output == 'success (no tool calls)'
    assert [client.is_closed for client in provider_clients] == [True]


async def test_run_leaves_client_of_passed_model_open(provider_clients: list[httpx2.AsyncClient]):
    """A `Model` instance passed to a run is the caller's, so the run doesn't close its client."""
    model = OpenAIChatModel('gpt-5.2', provider=OpenAIProvider())
    agent = Agent()

    await agent.run('What is the capital of France?', model=model)
    await agent.run('What is the capital of France?', model=model)
    assert [client.is_closed for client in provider_clients] == [False]


async def test_entered_agent_keeps_model_its_selector_returns_again(provider_clients: list[httpx2.AsyncClient]):
    """Inside `async with agent:`, a `Model` instance a selector returns again stays open until the agent exits."""
    model = OpenAIChatModel('gpt-5.2', provider=OpenAIProvider())

    def select(ctx: ModelSelectionContext[Any]) -> Model:
        return model

    agent = Agent(capabilities=[SelectModel(select)])

    # Without the block, each run enters the model and closes its client when it ends.
    await agent.run('What is the capital of France?')
    assert [client.is_closed for client in provider_clients] == [True]

    async with agent:
        # The first time it's selected, the run enters it, as it may never be selected again.
        await agent.run('What is the capital of France?')
        assert [client.is_closed for client in provider_clients] == [True, True]
        # The next time, the agent enters it once, for these and later runs.
        await asyncio.gather(
            agent.run('What is the capital of France?'),
            agent.run('What is the capital of France?'),
        )
        await agent.run('What is the capital of France?')
        assert [client.is_closed for client in provider_clients] == [True, True, False]
    assert [client.is_closed for client in provider_clients] == [True, True, True]


async def test_entered_agent_leaves_new_model_its_selector_returns_to_the_run(
    provider_clients: list[httpx2.AsyncClient],
):
    """A selector that builds a new `Model` for each run gets each one closed when its run ends, not on agent exit."""

    def select(ctx: ModelSelectionContext[Any]) -> Model:
        return OpenAIChatModel('gpt-5.2', provider=OpenAIProvider())

    agent = Agent(capabilities=[SelectModel(select)])

    async with agent:
        await agent.run('What is the capital of France?')
        assert [client.is_closed for client in provider_clients] == [True]
        await agent.run('What is the capital of France?')
        assert [client.is_closed for client in provider_clients] == [True, True]


@pytest.mark.parametrize('select_by_name', [False, True])
async def test_entered_agent_keeps_model_selected_after_first_step(
    provider_clients: list[httpx2.AsyncClient], select_by_name: bool
):
    """A model a selector returns again for a later step of a run stays open until the agent exits, too."""
    model = None if select_by_name else OpenAIChatModel('gpt-5.2', provider=OpenAIProvider())

    def select(ctx: ModelSelectionContext[Any]) -> Model | str:
        if ctx.run_step == 1:
            return 'test'
        return model or 'openai-chat:gpt-5.2'

    agent = Agent(capabilities=[SelectModel(select)])

    @agent.tool_plain
    def capital() -> str:
        return 'Paris'

    async with agent:
        await agent.run('What is the capital of France?')
        await agent.run('What is the capital of France?')
        await agent.run('What is the capital of France?')
        # A name is entered on the agent the first time; an instance, the first time it's selected again.
        expected = [False] if select_by_name else [True, False]
        assert [client.is_closed for client in provider_clients] == expected
    assert all(client.is_closed for client in provider_clients)


async def test_model_selected_with_resolve_model_id_capability_is_entered_per_run(
    provider_clients: list[httpx2.AsyncClient],
):
    """With a `resolve_model_id` capability, a `Model` instance a selector returns is entered by each run."""
    model = OpenAIChatModel('gpt-5.2', provider=OpenAIProvider())

    def select(ctx: ModelSelectionContext[Any]) -> Model:
        return model

    def resolve(ctx: ModelResolutionContext[Any], model_id: str) -> Model | None:
        return None  # pragma: no cover

    agent = Agent(capabilities=[SelectModel(select), ResolveModelId(resolve)])

    async with agent:
        await agent.run('What is the capital of France?')
        await agent.run('What is the capital of France?')
    assert [client.is_closed for client in provider_clients] == [True, True]


async def test_names_resolved_through_capability_are_not_reused(provider_clients: list[httpx2.AsyncClient]):
    """With a `resolve_model_id` capability, a name is resolved every run, and a model built for it closed when the run ends."""
    seen: list[str] = []

    def resolve(ctx: ModelResolutionContext[Any], model_id: str) -> Model | None:
        seen.append(model_id)
        return None

    agent = Agent(capabilities=[ResolveModelId(resolve)])

    async with agent:
        await agent.run('What is the capital of France?', model='openai-chat:gpt-5.2')
        await agent.run('What is the capital of France?', model='openai-chat:gpt-5.2')
    assert seen == ['openai-chat:gpt-5.2', 'openai-chat:gpt-5.2']
    assert [client.is_closed for client in provider_clients] == [True, True]


@pytest.mark.skipif(not huggingface_imports_successful(), reason='huggingface_hub not installed')
def test_unentered_huggingface_model_gets_new_client_on_new_event_loop(
    monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
):
    """The Hugging Face client opens its HTTP client on first use, so the provider builds a new one on a new event loop."""
    transports: list[httpx.AsyncHTTPTransport] = []

    async def handle(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        transports.append(self)
        return httpx.Response(200, json={**_OPENAI_RESPONSE, 'system_fingerprint': 'test'})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', handle)
    provider = HuggingFaceProvider(base_url='https://hf.example.com/v1', api_key='test-key')
    hf_client = provider.client
    agent = Agent(HuggingFaceModel('Qwen/Qwen3-8B', provider=provider))

    assert _run_on_new_event_loops(agent, count=2) == ['Paris', 'Paris']
    assert provider.client is not hf_client
    assert transports[0] is not transports[1]

    # While the agent, and with it the provider, is entered, the client stays.
    async def run_entered() -> None:
        async with agent:
            entered_client = provider.client
            await agent.run('What is the capital of France?')
            assert provider.client is entered_client

    asyncio.run(run_entered())

    # A client that was passed in is never replaced.
    provider = HuggingFaceProvider(hf_client=hf_client, api_key='test-key')
    agent = Agent(HuggingFaceModel('Qwen/Qwen3-8B', provider=provider))
    assert _run_on_new_event_loops(agent, count=1) == ['Paris']
    assert provider.client is hf_client


@pytest.fixture
def huggingface_http_clients(monkeypatch: pytest.MonkeyPatch, allow_model_requests: None) -> list[httpx.AsyncClient]:
    """Every HTTP client a Hugging Face `AsyncInferenceClient` opens, answered without a network."""
    clients: list[httpx.AsyncClient] = []
    create_client = huggingface_http._GLOBAL_ASYNC_CLIENT_FACTORY  # pyright: ignore[reportPrivateUsage]

    def create_tracked_client() -> httpx.AsyncClient:
        client = create_client()
        clients.append(client)
        return client

    async def handle(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**_OPENAI_RESPONSE, 'system_fingerprint': 'test'})

    monkeypatch.setattr(huggingface_http, '_GLOBAL_ASYNC_CLIENT_FACTORY', create_tracked_client)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', handle)
    return clients


@pytest.mark.skipif(not huggingface_imports_successful(), reason='huggingface_hub not installed')
async def test_huggingface_provider_closes_its_client_on_exit(huggingface_http_clients: list[httpx.AsyncClient]):
    provider = HuggingFaceProvider(base_url='https://hf.example.com/v1', api_key='test-key')
    agent = Agent(HuggingFaceModel('Qwen/Qwen3-8B', provider=provider))

    async with agent:
        assert (await agent.run('What is the capital of France?')).output == 'Paris'
        assert (await agent.run('What is the capital of France?')).output == 'Paris'
        assert [client.is_closed for client in huggingface_http_clients] == [False]
    assert [client.is_closed for client in huggingface_http_clients] == [True]

    # The closed client can't be used again, so a run after the agent exits gets a new one.
    assert (await agent.run('What is the capital of France?')).output == 'Paris'
    assert [client.is_closed for client in huggingface_http_clients] == [True, False]
    async with provider:
        pass
    assert [client.is_closed for client in huggingface_http_clients] == [True, True]

    # An exit without a matching enter does nothing.
    await provider.__aexit__(None, None, None)
    assert (await agent.run('What is the capital of France?')).output == 'Paris'
    assert [client.is_closed for client in huggingface_http_clients] == [True, True, False]


@pytest.mark.skipif(not huggingface_imports_successful(), reason='huggingface_hub not installed')
async def test_huggingface_provider_leaves_passed_in_client_open(huggingface_http_clients: list[httpx.AsyncClient]):
    hf_client = AsyncInferenceClient(base_url='https://hf.example.com/v1', api_key='test-key')
    provider = HuggingFaceProvider(hf_client=hf_client, api_key='test-key')
    agent = Agent(HuggingFaceModel('Qwen/Qwen3-8B', provider=provider))

    async with agent:
        assert (await agent.run('What is the capital of France?')).output == 'Paris'
    assert provider.client is hf_client
    assert [client.is_closed for client in huggingface_http_clients] == [False]

    await hf_client.close()
    assert [client.is_closed for client in huggingface_http_clients] == [True]
