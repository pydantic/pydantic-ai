"""Tests for the web chat UI module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent

from .conftest import try_import

with try_import() as imports_successful:
    # Used on a test in this file.
    import openai  # pyright: ignore[reportUnusedImport] # noqa: F401
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from pydantic_ai.builtin_tools import WebSearchTool
    from pydantic_ai.ui._web import create_web_app


pytestmark = [pytest.mark.skipif(not imports_successful(), reason='starlette not installed')]


def test_agent_to_web():
    """Test the Agent.to_web() method."""
    agent = Agent('test')
    app = agent.to_web()

    assert isinstance(app, Starlette)


def test_agent_to_web_with_model_instances():
    """Test to_web() accepts model instances, not just strings."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel())
    model_instance = TestModel()

    # List with instances
    app = agent.to_web(models=[model_instance, 'test'])
    assert isinstance(app, Starlette)

    # Dict with instances
    app = agent.to_web(models={'Custom': model_instance, 'Test': 'test'})
    assert isinstance(app, Starlette)


@pytest.mark.anyio
async def test_model_instance_preserved_in_dispatch(monkeypatch: pytest.MonkeyPatch):
    """Test that model instances are preserved and used in dispatch, not reconstructed from string."""
    from unittest.mock import AsyncMock

    from starlette.responses import Response

    from pydantic_ai.models.test import TestModel
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter

    model_instance = TestModel(custom_output_text='Custom output')
    agent: Agent[None, str] = Agent()
    app = create_web_app(agent, models=[model_instance])

    # Mock dispatch_request to capture the model parameter
    mock_dispatch = AsyncMock(return_value=Response(content=b'', status_code=200))
    monkeypatch.setattr(VercelAIAdapter, 'dispatch_request', mock_dispatch)

    with TestClient(app) as client:
        client.post(
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
                'model': 'test:test',
                'builtinTools': [],
            },
        )

    # Verify dispatch_request was called with the original model instance
    mock_dispatch.assert_called_once()
    call_kwargs = mock_dispatch.call_args.kwargs
    assert call_kwargs['model'] is model_instance, 'Model instance should be preserved, not reconstructed from string'


def test_agent_to_web_with_deps():
    """Test to_web() accepts deps parameter."""
    from dataclasses import dataclass

    from pydantic_ai.models.test import TestModel

    @dataclass
    class MyDeps:
        api_key: str

    agent: Agent[MyDeps, str] = Agent(TestModel(), deps_type=MyDeps)
    deps = MyDeps(api_key='test-key')

    app = agent.to_web(deps=deps)
    assert isinstance(app, Starlette)


def test_agent_to_web_with_model_settings():
    """Test to_web() accepts model_settings parameter."""
    from pydantic_ai import ModelSettings
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel())
    settings = ModelSettings(temperature=0.5, max_tokens=100)

    app = agent.to_web(model_settings=settings)
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
                'models': [
                    {'id': 'test:test', 'name': 'Test', 'builtinTools': ['web_search']},
                    {'id': 'test', 'name': 'Test', 'builtinTools': ['web_search']},
                ],
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
        assert response.json() == snapshot(
            {'models': [{'id': 'test:test', 'name': 'Test', 'builtinTools': []}], 'builtinTools': []}
        )


def test_chat_app_configure_preserves_chat_vs_responses(monkeypatch: pytest.MonkeyPatch):
    """Test that openai-chat: and openai-responses: models are kept as separate entries."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    agent = Agent('test')
    app = create_web_app(
        agent,
        models=['openai-chat:gpt-4o', 'openai-responses:gpt-4o'],
    )

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        data = response.json()
        model_ids = [m['id'] for m in data['models']]
        assert 'openai-chat:gpt-4o' in model_ids
        assert 'openai-responses:gpt-4o' in model_ids
        assert len([m for m in model_ids if 'gpt-4o' in m]) == 2


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


def test_chat_app_index_invalid_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that index endpoint returns 502 for invalid UI version."""
    import httpx

    import pydantic_ai.ui._web.app as app_module

    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)

    class MockResponse:
        status_code = 404

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError('Not Found', request=None, response=self)  # type: ignore

    class MockAsyncClient:
        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def get(self, url: str) -> MockResponse:
            return MockResponse()

    monkeypatch.setattr(app_module.httpx, 'AsyncClient', MockAsyncClient)

    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/?version=nonexistent-version')
        assert response.status_code == 502
        assert 'nonexistent-version' in response.text
        assert '404' in response.text


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
                'model': 'test:test',
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


def test_post_chat_invalid_builtin_tool():
    """Test POST /api/chat returns 400 when builtin tool is not in allowed list."""
    from pydantic_ai.builtin_tools import WebSearchTool
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(custom_output_text='Hello'))
    app = create_web_app(agent, builtin_tools=[WebSearchTool()])

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
                'model': 'test:test',
                'builtinTools': ['code_execution'],  # Not in allowed list
            },
        )

        assert response.status_code == 400
        assert response.json() == snapshot(
            {'error': "Builtin tool(s) ['code_execution'] not in the allowed tools list"}
        )


def test_model_label_openrouter():
    """Test Model.label handles OpenRouter-style names with /."""
    from pydantic_ai.models.test import TestModel

    model = TestModel(model_name='meta-llama/llama-3-70b')
    assert model.label == snapshot('Llama 3 70b')


def test_agent_to_web_with_instructions():
    """Test to_web() accepts instructions parameter."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel())
    app = agent.to_web(instructions='Always respond in Spanish')
    assert isinstance(app, Starlette)


@pytest.mark.anyio
async def test_instructions_passed_to_dispatch(monkeypatch: pytest.MonkeyPatch):
    """Test that instructions from create_web_app are passed to dispatch_request."""
    from unittest.mock import AsyncMock

    from starlette.responses import Response

    from pydantic_ai.models.test import TestModel
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter

    agent = Agent(TestModel(custom_output_text='Hello'))
    app = create_web_app(agent, instructions='Always respond in Spanish')

    # Mock dispatch_request to capture the instructions parameter
    mock_dispatch = AsyncMock(return_value=Response(content=b'', status_code=200))
    monkeypatch.setattr(VercelAIAdapter, 'dispatch_request', mock_dispatch)

    with TestClient(app) as client:
        client.post(
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
                'model': 'test:test',
                'builtinTools': [],
            },
        )

    # Verify dispatch_request was called with instructions
    mock_dispatch.assert_called_once()
    call_kwargs = mock_dispatch.call_args.kwargs
    assert call_kwargs['instructions'] == 'Always respond in Spanish'
