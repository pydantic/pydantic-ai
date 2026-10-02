"""The `system_one` plugin: a speculation-only `rank_relevance` tool over the SystemOne decisions API."""

import json
from collections.abc import AsyncIterator, Callable

import anyio
import httpx2
import pytest
from pydantic import SecretStr
from rich.console import Console

from pydantic_ai import Agent, ModelRetry, models
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins import system_one
from pydantic_clai2.builtin_plugins.system_one import (
    GUIDANCE,
    JEV_KEY,
    MAX_CANDIDATES,
    SystemOneContext,
    SystemOnePlugin,
    SystemOneSettings,
)
from pydantic_clai2.plugins import PluginHost, load_plugin
from pydantic_clai2.runtime.speculation import SPECULATION_ID

pytestmark = pytest.mark.anyio

Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], list[httpx2.Request]]:
    """Route the tool's HTTP client to `handler`, recording every request."""
    real = httpx2.AsyncClient
    monkeypatch.setattr(models, 'ALLOW_MODEL_REQUESTS', True)

    def route(handler: Handler) -> list[httpx2.Request]:
        seen: list[httpx2.Request] = []

        def record(request: httpx2.Request) -> httpx2.Response:
            seen.append(request)
            return handler(request)

        def client(**kwargs: object) -> httpx2.AsyncClient:
            return real(transport=httpx2.MockTransport(record))

        monkeypatch.setattr(httpx2, 'AsyncClient', client)
        return seen

    return route


@pytest.fixture(autouse=True)
def no_saved_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(JEV_KEY, raising=False)
    monkeypatch.setattr(system_one, 'load_keys', dict[str, SecretStr])


def answer(request: httpx2.Request) -> httpx2.Response:
    """A SystemOne answer: yes for any state that mentions retries."""
    body = json.loads(request.content)
    yes = 0.9 if 'retry' in json.dumps(body['state']) else 0.1
    return httpx2.Response(
        200,
        json={
            'model': body['model'],
            'answers': {name: {'type': 'noul', 'noul': yes} for name in body['questions']},
            'usage': {'input_tokens': 10, 'output_tokens': 1},
        },
    )


def offered(capabilities: list[AbstractCapability[None]]) -> tuple[list[str], str]:
    seen: list[AgentInfo] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart('done')])

    Agent(FunctionModel(respond), deps_type=type(None), capabilities=capabilities).run_sync('hi')
    [info] = seen
    return [tool.name for tool in info.function_tools], info.instructions or ''


def test_tool_and_guidance_follow_speculation() -> None:
    assert offered([SystemOneContext()]) == ([], '')
    # The speculative bundle runs under `SPECULATION_ID`; any capability with that id stands in for it.
    tools, instructions = offered([SystemOneContext(), Hooks(id=SPECULATION_ID)])
    assert tools == ['rank_relevance']
    assert GUIDANCE.strip() in instructions


def test_builtin_is_enabled_by_default() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'system_one']
    assert declaration.enabled
    plugin = load_plugin(SystemOnePlugin, PluginHost(name='system_one', console=Console(), settings={}))
    [capability] = plugin.capabilities
    assert isinstance(capability, SystemOneContext)


async def test_ranks_with_local_ollama_without_a_key(
    requests: Callable[[Handler], list[httpx2.Request]],
) -> None:
    seen = requests(answer)
    ranked = await SystemOneContext().rank_relevance(
        'Does this code retry failed requests?', {'a.py': 'def add(a, b): ...', 'b.py': 'def retry(): ...', 'c.py': ' '}
    )
    assert ranked == {'b.py': 0.9, 'a.py': 0.1, 'c.py': 0.0}
    assert list(ranked) == ['b.py', 'a.py', 'c.py']
    assert {str(request.url) for request in seen} == {'http://localhost:11434/v1/systemone'}
    assert all('authorization' not in request.headers for request in seen)
    assert {json.loads(request.content)['model'] for request in seen} == {'nimble'}


async def test_saved_jev_key_selects_jev(
    requests: Callable[[Handler], list[httpx2.Request]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(system_one, 'load_keys', lambda: {JEV_KEY: SecretStr('jev-secret')})
    seen = requests(answer)
    assert await SystemOneContext().rank_relevance('Does it retry?', {'x': 'retry loop'}) == {'x': 0.9}
    [request] = seen
    assert str(request.url) == 'https://api.typesafe.ai/v1/systemone'
    assert request.headers['authorization'] == 'Bearer jev-secret'
    assert json.loads(request.content)['model'] == 'jev-latest'


async def test_environment_key_selects_jev(
    requests: Callable[[Handler], list[httpx2.Request]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(JEV_KEY, 'from-env')
    seen = requests(answer)
    await SystemOneContext(settings=SystemOneSettings(jev_url='https://jev.test/v1')).rank_relevance('q?', {'x': 'y'})
    [request] = seen
    assert str(request.url) == 'https://jev.test/v1/systemone'
    assert request.headers['authorization'] == 'Bearer from-env'


async def test_no_model_tells_the_agent_how_to_enable_one(
    requests: Callable[[Handler], list[httpx2.Request]],
) -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('connection refused', request=request)

    requests(refuse)
    message = await SystemOneContext().rank_relevance(
        'q?', {'a': 'one', 'b': 'two', 'c': 'three', 'd': 'four', 'e': 'five'}
    )
    assert isinstance(message, str)
    assert JEV_KEY in message and '/keys' in message and 'ollama pull nimble' in message


async def test_rejected_jev_key_points_at_keys(
    requests: Callable[[Handler], list[httpx2.Request]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(system_one, 'load_keys', lambda: {JEV_KEY: SecretStr('stale')})
    requests(lambda request: httpx2.Response(403, json={'detail': 'Must supply an API key!'}))
    message = await SystemOneContext().rank_relevance('q?', {'a': 'one'})
    assert isinstance(message, str)
    assert message.startswith('Jev (jev-latest at https://api.typesafe.ai) failed')
    assert f'check the {JEV_KEY} key saved in /keys' in message


@pytest.mark.parametrize(
    ('jev_key', 'ollama_url', 'launches'),
    [
        pytest.param(None, 'http://localhost:11434', 1, id='local-nimble'),
        pytest.param('jev-secret', 'http://localhost:11434', 0, id='jev'),
        pytest.param(None, 'http://gpu-box:11434', 0, id='remote-ollama'),
    ],
)
async def test_speculates_only_with_a_local_model(
    requests: Callable[[Handler], list[httpx2.Request]],
    monkeypatch: pytest.MonkeyPatch,
    jev_key: str | None,
    ollama_url: str,
    launches: int,
) -> None:
    """A launch for an untaken branch must not send text to a remote API or spend its quota."""
    pytest.importorskip('pydantic_monty')
    from pydantic_clai2.runtime.speculation import SpeculationCounters
    from pydantic_clai2.runtime.speculative_mode import speculative_capabilities

    if jev_key is not None:
        monkeypatch.setattr(system_one, 'load_keys', lambda: {JEV_KEY: SecretStr(jev_key)})
    requests(answer)
    snippet = 'if False:\n    await rank_relevance(question="q?", candidates={"a": "retry"})\n'

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) > 1:
            yield 'done'
            return
        args = json.dumps({'code': snippet})
        yield {0: DeltaToolCall(name='run_code')}
        for offset in range(0, len(args), 8):
            yield {0: DeltaToolCall(json_args=args[offset : offset + 8])}
            await anyio.sleep(0)

    counters = SpeculationCounters()
    context = SystemOneContext[None](settings=SystemOneSettings(ollama_url=ollama_url))
    agent = Agent(
        FunctionModel(stream_function=stream),
        deps_type=type(None),
        capabilities=[context, *speculative_capabilities(counters, [context])],
    )
    with anyio.fail_after(10):
        await agent.run('go')
    assert (counters.hits, counters.wasted) == (0, launches)


async def test_candidate_limits() -> None:
    assert await SystemOneContext().rank_relevance('q?', {}) == {}
    with pytest.raises(ModelRetry, match='at most'):
        await SystemOneContext().rank_relevance('q?', {str(n): 'x' for n in range(MAX_CANDIDATES + 1)})
