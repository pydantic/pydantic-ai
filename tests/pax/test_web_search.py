"""The `web_search` built-in: off by default, and once enabled, harness `KeenableSearch` as a deferred capability."""

import functools
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx2
import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.keenable import HttpKeenableClient, KeenableSearch
from pydantic_ai_pax import DEFAULT_PLUGINS
from pydantic_ai_pax.builtin_plugins.web_search import DESCRIPTION
from pydantic_ai_pax.commands import Commands
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import SessionStart
from pydantic_ai_pax.plugins.loader import PluginLoader

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'web_search')
# The shape of a recorded keyless Keenable search (tests/harness/keenable/cassettes), trimmed to one result.
SEARCH_RESPONSE = {
    'query': 'Pydantic AI agent framework',
    'mode': 'pro',
    'results': [
        {
            'title': 'Pydantic AI',
            'url': 'https://pydantic.dev/docs/ai/',
            'description': '',
            'snippet': 'Pydantic AI is a Python agent framework.',
        }
    ],
}


@pytest.fixture(autouse=True)
def keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keyless public endpoint, whatever the developer's environment holds."""
    monkeypatch.delenv('KEENABLE_API_KEY', raising=False)
    monkeypatch.delenv('KEENABLE_API_URL', raising=False)


@pytest.fixture
async def sent(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[httpx2.Request]]:
    """Answer every Keenable request from `SEARCH_RESPONSE` in process, recording what was sent.

    The default client the plugin builds gets an `http_client` on a mock transport, so nothing leaves the process.
    """
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=SEARCH_RESPONSE)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http_client:
        monkeypatch.setattr(
            HttpKeenableClient,
            '__init__',
            functools.partialmethod(HttpKeenableClient.__init__, http_client=http_client),
        )
        yield requests


def loader(tmp_path: Path) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'config.db')
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(BUILTIN,),
    )


def tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]


def load_then_search(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Load the deferred capability from its catalog line, search, then answer with the result."""
    names = {tool.name for tool in info.function_tools}
    returns = tool_returns(messages)
    if not returns:
        [load] = info.function_tools
        assert load.name == 'load_capability'
        assert f'\n- keenable_search: {DESCRIPTION}\n' in (info.instructions or '')
        return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'keenable_search'})])
    if returns[0].tool_name == 'load_capability':
        assert {'web_search', 'get_page'} <= names
        return ModelResponse(parts=[ToolCallPart('web_search', {'query': 'Pydantic AI agent framework'})])
    return ModelResponse(parts=[TextPart(str(returns[0].content))])


async def test_enabling_gives_the_model_keenable_search_and_its_result_comes_back_as_tool_output(
    tmp_path: Path, sent: list[httpx2.Request]
) -> None:
    plugins = loader(tmp_path)
    await plugins.load_all()
    await plugins.enable('web_search')
    [capability] = plugins.capabilities()
    assert isinstance(capability, KeenableSearch)
    agent = Agent(FunctionModel(load_then_search), deps_type=type(None), capabilities=plugins.capabilities())

    result = await agent.run('What is Pydantic AI?')

    assert result.output == (
        '1. Pydantic AI\n   https://pydantic.dev/docs/ai/\n   Pydantic AI is a Python agent framework.'
    )
    [request] = sent
    assert str(request.url) == 'https://api.keenable.ai/v1/search/public'
    assert json.loads(request.content) == {'query': 'Pydantic AI agent framework', 'max_results': 5}
    [search] = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'web_search'
    ]
    assert search.metadata == {'sources': [{'url': 'https://pydantic.dev/docs/ai/', 'title': 'Pydantic AI'}]}


async def test_disabled_by_default_it_adds_nothing_to_requests(tmp_path: Path, sent: list[httpx2.Request]) -> None:
    plugins = loader(tmp_path)
    await plugins.load_all()
    assert [(entry.name, entry.state) for entry in plugins.entries()] == [('web_search', 'disabled')]
    assert plugins.capabilities() == []

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        assert 'Keenable' not in (info.instructions or '')
        return ModelResponse(parts=[TextPart('no search')])

    agent = Agent(FunctionModel(answer), deps_type=type(None), capabilities=plugins.capabilities())
    assert (await agent.run('What is Pydantic AI?')).output == 'no search'
    assert sent == []
