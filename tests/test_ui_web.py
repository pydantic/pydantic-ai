"""Tests for the web chat UI module."""

from __future__ import annotations

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
