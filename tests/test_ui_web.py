"""Tests for the web chat UI module."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent

from .conftest import try_import

with try_import() as fastapi_import_successful:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pydantic_ai.ui.web import builtin_tool_definitions, create_web_app, models

pytestmark = [
    pytest.mark.skipif(not fastapi_import_successful(), reason='fastapi not installed'),
]


def test_create_chat_app_basic():
    """Test creating a basic chat app."""
    agent = Agent('test')
    app = create_web_app(agent)

    assert isinstance(app, FastAPI)
    assert app.state.agent is agent


def test_agent_to_web():
    """Test the Agent.to_web() method."""
    agent = Agent('test')
    app = agent.to_web()

    assert isinstance(app, FastAPI)
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
    """Test the /api/configure endpoint."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        data = response.json()
        assert data == snapshot(
            {
                'models': [
                    {
                        'id': 'anthropic:claude-sonnet-4-5',
                        'name': 'Claude Sonnet 4.5',
                        'builtinTools': ['web_search', 'code_execution'],
                    },
                    {
                        'id': 'openai-responses:gpt-5',
                        'name': 'GPT 5',
                        'builtinTools': ['web_search', 'code_execution', 'image_generation'],
                    },
                    {
                        'id': 'google-gla:gemini-2.5-pro',
                        'name': 'Gemini 2.5 Pro',
                        'builtinTools': ['web_search', 'code_execution'],
                    },
                ],
                'builtinToolDefs': [
                    {'id': 'web_search', 'name': 'Web Search'},
                    {'id': 'code_execution', 'name': 'Code Execution'},
                    {'id': 'image_generation', 'name': 'Image Generation'},
                ],
            }
        )


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


def test_ai_models_configuration():
    """Test that AI models are configured correctly."""
    assert len(models) == 3

    model_ids = {model.id for model in models}
    assert 'anthropic:claude-sonnet-4-5' in model_ids
    assert 'openai-responses:gpt-5' in model_ids
    assert 'google-gla:gemini-2.5-pro' in model_ids


def test_builtin_tools_configuration():
    """Test that builtin tool definitions are configured correctly."""
    assert len(builtin_tool_definitions) == 3

    tool_ids = {tool_def.id for tool_def in builtin_tool_definitions}
    assert 'web_search' in tool_ids
    assert 'code_execution' in tool_ids
    assert 'image_generation' in tool_ids

    from pydantic_ai.builtin_tools import CodeExecutionTool, ImageGenerationTool, WebSearchTool

    tools_by_id = {tool_def.id: tool_def.tool for tool_def in builtin_tool_definitions}
    assert isinstance(tools_by_id['web_search'], WebSearchTool)
    assert isinstance(tools_by_id['code_execution'], CodeExecutionTool)
    assert isinstance(tools_by_id['image_generation'], ImageGenerationTool)
