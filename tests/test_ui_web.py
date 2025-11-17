"""Tests for the web chat UI module."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.ui.web import AI_MODELS, BUILTIN_TOOLS, create_chat_app

from .conftest import try_import

with try_import() as fastapi_import_successful:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

pytestmark = [
    pytest.mark.skipif(not fastapi_import_successful, reason='fastapi not installed'),
]


def test_create_chat_app_basic():
    """Test creating a basic chat app."""
    agent = Agent('test')
    app = create_chat_app(agent)

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
    app = create_chat_app(agent)

    with TestClient(app) as client:
        response = client.get('/api/health')
        assert response.status_code == 200
        assert response.json() == {'ok': True}


def test_chat_app_configure_endpoint():
    """Test the /api/configure endpoint."""
    agent = Agent('test')
    app = create_chat_app(agent)

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        data = response.json()
        assert 'models' in data
        assert 'builtinTools' in data  # camelCase due to alias generator

        # no snapshot bc we're checking against the actual model/tool definitions
        assert len(data['models']) == len(AI_MODELS)
        assert len(data['builtinTools']) == len(BUILTIN_TOOLS)


def test_chat_app_index_endpoint():
    """Test that the index endpoint serves the UI from CDN."""
    agent = Agent('test')
    app = create_chat_app(agent)

    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/html; charset=utf-8'
        assert len(response.content) > 0


def test_ai_models_configuration():
    """Test that AI models are configured correctly."""
    assert len(AI_MODELS) == 3

    model_ids = {model.id for model in AI_MODELS}
    assert 'anthropic:claude-sonnet-4-5' in model_ids
    assert 'openai-responses:gpt-5' in model_ids
    assert 'google-gla:gemini-2.5-pro' in model_ids


def test_builtin_tools_configuration():
    """Test that builtin tools are configured correctly."""
    assert len(BUILTIN_TOOLS) == 3

    assert 'web_search' in BUILTIN_TOOLS
    assert 'code_execution' in BUILTIN_TOOLS
    assert 'image_generation' in BUILTIN_TOOLS

    from pydantic_ai.builtin_tools import CodeExecutionTool, ImageGenerationTool, WebSearchTool

    assert isinstance(BUILTIN_TOOLS['web_search'], WebSearchTool)
    assert isinstance(BUILTIN_TOOLS['code_execution'], CodeExecutionTool)
    assert isinstance(BUILTIN_TOOLS['image_generation'], ImageGenerationTool)
