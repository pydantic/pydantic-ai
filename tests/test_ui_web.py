"""Tests for the web chat UI module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent

from .conftest import try_import

with try_import() as starlette_import_successful:
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from pydantic_ai.builtin_tools import WebSearchTool
    from pydantic_ai.ui._web import create_web_app


pytestmark = [
    pytest.mark.skipif(not starlette_import_successful(), reason='starlette not installed'),
]


def test_agent_to_web():
    """Test the Agent.to_web() method."""
    agent = Agent('test')
    app = agent.to_web()

    assert isinstance(app, Starlette)


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
        models=['test'],
        builtin_tools=[WebSearchTool()],
    )

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        assert response.json() == snapshot(
            {
                'models': [{'id': 'test:test', 'name': 'Test', 'builtinTools': ['web_search']}],
                'builtinTools': [{'id': 'web_search', 'name': 'Web Search'}],
            }
        )


def test_chat_app_configure_endpoint_empty():
    """Test the /api/configure endpoint with no models or tools."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        assert response.json() == snapshot({'models': [], 'builtinTools': []})


def test_chat_app_index_endpoint():
    """Test that the index endpoint serves HTML with proper caching headers."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/html; charset=utf-8'
        assert 'cache-control' in response.headers
        assert response.headers['cache-control'] == 'public, max-age=3600'
        assert len(response.content) > 0


@pytest.mark.anyio
async def test_get_ui_html_cdn_fetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html fetches from CDN when filesystem cache misses."""
    import pydantic_ai.ui._web.app as app_module

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

    from pydantic_ai.ui._web.app import _get_ui_html  # pyright: ignore[reportPrivateUsage]

    result = await _get_ui_html('test-version')

    assert result == test_content
    cache_file: Path = tmp_path / 'test-version.html'
    assert cache_file.exists()
    assert cache_file.read_bytes() == test_content


@pytest.mark.anyio
async def test_get_ui_html_filesystem_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html returns cached content from filesystem."""
    import pydantic_ai.ui._web.app as app_module

    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)

    test_content = b'<html>Cached UI</html>'
    cache_file = tmp_path / 'cached-version.html'
    cache_file.write_bytes(test_content)

    from pydantic_ai.ui._web.app import _get_ui_html  # pyright: ignore[reportPrivateUsage]

    result = await _get_ui_html('cached-version')

    assert result == test_content


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


def test_mcp_server_tool_label():
    """Test MCPServerTool.label property."""
    from pydantic_ai.builtin_tools import MCPServerTool

    tool = MCPServerTool(id='test-server', url='https://example.com')
    assert tool.label == 'MCP: test-server'


def test_model_profile():
    """Test Model.profile cached property."""
    from pydantic_ai.models.test import TestModel

    model = TestModel()
    assert model.profile is not None


@pytest.mark.parametrize('profile_name', ['base', 'openai', 'google', 'groq'])
def test_supported_builtin_tools(profile_name: str):
    """Test profile.supported_builtin_tools returns proper tool types."""
    from pydantic_ai.builtin_tools import AbstractBuiltinTool
    from pydantic_ai.profiles import ModelProfile

    if profile_name == 'base':
        profile: ModelProfile = ModelProfile()
    elif profile_name == 'openai':
        from pydantic_ai.profiles.openai import OpenAIModelProfile

        profile = OpenAIModelProfile()
    elif profile_name == 'google':
        from pydantic_ai.profiles.google import GoogleModelProfile

        profile = GoogleModelProfile()
    else:
        from pydantic_ai.profiles.groq import GroqModelProfile

        profile = GroqModelProfile()

    result = profile.supported_builtin_tools
    assert isinstance(result, frozenset)
    assert all(issubclass(t, AbstractBuiltinTool) for t in result)


def test_post_chat_invalid_model():
    """Test POST /api/chat returns 400 when model is not in allowed list."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(custom_output_text='Hello'))
    # Use 'test' as the allowed model, then send a different model in the request
    app = create_web_app(agent, models=['test'])

    with TestClient(app) as client:
        response = client.post(
            '/api/chat',
            json={
                'trigger': 'submit-message',
                'id': 'test-id',
                'messages': [
                    {
                        'id': 'msg-1',
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': 'Hello'}],
                    }
                ],
                'model': 'test:different_model',
                'builtinTools': [],
            },
        )

        assert response.status_code == 400
        assert response.json() == snapshot({'error': 'Model "test:different_model" is not in the allowed models list'})


def test_model_label_openrouter():
    """Test Model.label handles OpenRouter-style names with /."""
    from pydantic_ai.models.test import TestModel

    model = TestModel(model_name='meta-llama/llama-3-70b')
    assert model.label == snapshot('Llama 3 70b')
