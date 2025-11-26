"""Tests for the web chat UI module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent

from .conftest import try_import

with try_import() as starlette_import_successful:
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from pydantic_ai.builtin_tools import WebSearchTool
    from pydantic_ai.ui.web import AIModel, create_web_app


pytestmark = [
    pytest.mark.skipif(not starlette_import_successful(), reason='starlette not installed'),
]


def test_create_chat_app_basic():
    """Test creating a basic chat app."""
    agent = Agent('test')
    app = create_web_app(agent)

    assert isinstance(app, Starlette)
    assert app.state.agent is agent


def test_agent_to_web():
    """Test the Agent.to_web() method."""
    agent = Agent('test')
    app = agent.to_web()

    assert isinstance(app, Starlette)
    assert app.state.agent is agent


def test_chat_app_health_endpoint():
    """Test the /api/health endpoint."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/api/health')
        assert response.status_code == 200
        assert response.json() == {'ok': True}


def test_chat_app_configure_endpoint():
    """Test the /api/configure endpoint with explicit models and tools."""
    agent = Agent('test')
    app = create_web_app(
        agent,
        models=[AIModel(id='openai:gpt-4o', name='GPT-4o', builtin_tools=['web_search'])],
        builtin_tools=[WebSearchTool()],
    )

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        data = response.json()
        assert data == {
            'models': [
                {
                    'id': 'openai:gpt-4o',
                    'name': 'GPT-4o',
                    'builtinTools': ['web_search'],
                },
            ],
            'builtinTools': [
                {'id': 'web_search', 'name': 'Web Search'},
            ],
        }


def test_chat_app_configure_endpoint_empty():
    """Test the /api/configure endpoint with no models or tools."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        data = response.json()
        assert data == {'models': [], 'builtinTools': []}


def test_chat_app_index_endpoint():
    """Test that the index endpoint serves the UI from CDN."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/html; charset=utf-8'
        assert 'cache-control' in response.headers
        assert response.headers['cache-control'] == 'public, max-age=31536000, immutable'
        assert len(response.content) > 0


@pytest.mark.anyio
async def test_get_ui_html_cdn_fetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html fetches from CDN when cache misses."""
    import pydantic_ai.ui.web.app as app_module

    app_module._memory_cache.clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)

    test_content = b'<html>Test UI</html>'

    class MockResponse:
        content = test_content

        def raise_for_status(self) -> None:
            pass

    class MockAsyncClient:
        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def get(self, url: str) -> MockResponse:
            return MockResponse()

    monkeypatch.setattr(app_module.httpx, 'AsyncClient', MockAsyncClient)

    from pydantic_ai.ui.web.app import _get_ui_html  # pyright: ignore[reportPrivateUsage]

    result = await _get_ui_html('test-version')

    assert result == test_content
    cache_file: Path = tmp_path / 'test-version.html'
    assert cache_file.exists()
    assert cache_file.read_bytes() == test_content
    assert app_module._memory_cache['test-version'] == test_content  # pyright: ignore[reportPrivateUsage]


def test_chat_app_index_caching():
    """Test that the UI HTML is cached after first fetch."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response1 = client.get('/')
        response2 = client.get('/')

        assert response1.content == response2.content
        assert response1.status_code == 200
        assert response2.status_code == 200


def test_get_agent_missing():
    """Test that _get_agent raises RuntimeError when agent is not configured."""
    from pydantic_ai.ui.web.api import _get_agent  # pyright: ignore[reportPrivateUsage]

    app = Starlette()

    class FakeRequest:
        def __init__(self, app: Starlette):
            self.app = app

    request = FakeRequest(app)

    with pytest.raises(RuntimeError, match='No agent configured'):
        _get_agent(request)  # pyright: ignore[reportArgumentType]


@pytest.mark.anyio
async def test_post_chat_endpoint():
    """Test the POST /api/chat endpoint."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(custom_output_text='Hello from test!'))
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.post(
            '/api/chat',
            json={
                'trigger': 'submit-message',
                'id': 'test-message-id',
                'messages': [
                    {
                        'id': 'msg-1',
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': 'Hello'}],
                    }
                ],
                'model': 'test',
                'builtinTools': [],
            },
        )

        assert response.status_code == 200


def test_chat_app_options_endpoint():
    """Test the OPTIONS /api/chat endpoint (CORS preflight)."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.options('/api/chat')
        assert response.status_code == 200


def test_expand_env_vars_simple(monkeypatch: pytest.MonkeyPatch):
    """Test _expand_env_vars with a simple environment variable."""
    from pydantic_ai.ui.web._mcp import _expand_env_vars  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv('TEST_VAR', 'test_value')
    result = _expand_env_vars('${TEST_VAR}')
    assert result == 'test_value'


def test_expand_env_vars_with_default():
    """Test _expand_env_vars uses default value when env var is not set."""
    from pydantic_ai.ui.web._mcp import _expand_env_vars  # pyright: ignore[reportPrivateUsage]

    result = _expand_env_vars('${UNDEFINED_VAR:-default_value}')
    assert result == 'default_value'


def test_expand_env_vars_empty_default():
    """Test _expand_env_vars with empty default value."""
    from pydantic_ai.ui.web._mcp import _expand_env_vars  # pyright: ignore[reportPrivateUsage]

    result = _expand_env_vars('${UNDEFINED_VAR:-}')
    assert result == ''


def test_expand_env_vars_missing_raises():
    """Test _expand_env_vars raises ValueError for undefined env var without default."""
    from pydantic_ai.ui.web._mcp import _expand_env_vars  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ValueError, match='Environment variable .* is not defined'):
        _expand_env_vars('${UNDEFINED_VAR_NO_DEFAULT}')


def test_expand_env_vars_nested_dict(monkeypatch: pytest.MonkeyPatch):
    """Test _expand_env_vars recursively expands in dicts."""
    from pydantic_ai.ui.web._mcp import _expand_env_vars  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv('TOKEN', 'secret123')
    result = _expand_env_vars({'url': 'https://example.com', 'token': '${TOKEN}'})
    assert result == {'url': 'https://example.com', 'token': 'secret123'}


def test_expand_env_vars_nested_list(monkeypatch: pytest.MonkeyPatch):
    """Test _expand_env_vars recursively expands in lists."""
    from pydantic_ai.ui.web._mcp import _expand_env_vars  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv('ITEM', 'value')
    result = _expand_env_vars(['${ITEM}', 'static'])
    assert result == ['value', 'static']


def test_expand_env_vars_passthrough():
    """Test _expand_env_vars passes through non-string/dict/list values."""
    from pydantic_ai.ui.web._mcp import _expand_env_vars  # pyright: ignore[reportPrivateUsage]

    assert _expand_env_vars(123) == 123
    assert _expand_env_vars(None) is None
    assert _expand_env_vars(True) is True


def test_load_mcp_server_tools_basic(tmp_path: Path):
    """Test loading MCP server tools from a config file."""
    from pydantic_ai.builtin_tools import MCPServerTool
    from pydantic_ai.ui.web._mcp import load_mcp_server_tools

    config = {
        'mcpServers': {
            'test-server': {
                'url': 'https://example.com/mcp',
            }
        }
    }
    config_file: Path = tmp_path / 'mcp.json'
    config_file.write_text(json.dumps(config), encoding='utf-8')

    tools = load_mcp_server_tools(str(config_file))
    assert len(tools) == 1
    assert isinstance(tools[0], MCPServerTool)
    assert tools[0].id == 'test-server'
    assert tools[0].url == 'https://example.com/mcp'


def test_load_mcp_server_tools_with_all_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test loading MCP server tools with all optional fields."""
    from pydantic_ai.ui.web._mcp import load_mcp_server_tools

    monkeypatch.setenv('MCP_TOKEN', 'my-secret-token')

    config = {
        'mcpServers': {
            'full-server': {
                'url': 'https://example.com/mcp',
                'authorizationToken': '${MCP_TOKEN}',
                'description': 'A test MCP server',
                'allowedTools': ['tool1', 'tool2'],
                'headers': {'X-Custom': 'header-value'},
            }
        }
    }
    config_file: Path = tmp_path / 'mcp.json'
    config_file.write_text(json.dumps(config), encoding='utf-8')

    tools = load_mcp_server_tools(str(config_file))
    assert len(tools) == 1
    assert tools[0].id == 'full-server'
    assert tools[0].url == 'https://example.com/mcp'
    assert tools[0].authorization_token == 'my-secret-token'
    assert tools[0].description == 'A test MCP server'
    assert tools[0].allowed_tools == ['tool1', 'tool2']
    assert tools[0].headers == {'X-Custom': 'header-value'}


def test_load_mcp_server_tools_file_not_found():
    """Test load_mcp_server_tools raises FileNotFoundError for missing file."""
    from pydantic_ai.ui.web._mcp import load_mcp_server_tools

    with pytest.raises(FileNotFoundError, match='MCP config file not found'):
        load_mcp_server_tools('/nonexistent/path/mcp.json')


def test_load_mcp_server_tools_multiple_servers(tmp_path: Path):
    """Test loading multiple MCP servers from config."""
    from pydantic_ai.ui.web._mcp import load_mcp_server_tools

    config = {
        'mcpServers': {
            'server-a': {'url': 'https://a.example.com/mcp'},
            'server-b': {'url': 'https://b.example.com/mcp'},
        }
    }
    config_file: Path = tmp_path / 'mcp.json'
    config_file.write_text(json.dumps(config), encoding='utf-8')

    tools = load_mcp_server_tools(str(config_file))
    assert len(tools) == 2
    ids = {t.id for t in tools}
    assert ids == {'server-a', 'server-b'}
