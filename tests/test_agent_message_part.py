"""Tests for AgentMessagePart across all provider adapters and UI adapters."""

from __future__ import annotations

import pytest

from pydantic_ai import AgentMessagePart, ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from .conftest import try_import

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as groq_imports:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as mistral_imports:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as cohere_imports:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as huggingface_imports:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

with try_import() as xai_imports:
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as bedrock_imports:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider


def _imports_ok(check_fn):
    return check_fn()

AGENT_MSG = AgentMessagePart(agent_name='researcher', content='Analysis complete.')
EXPECTED_PREFIX = "[Agent 'researcher']"
EXPECTED_CONTENT = 'Analysis complete.'


def _make_request() -> ModelRequest:
    return ModelRequest(parts=[AGENT_MSG])


# --- OpenAI Chat Completions ---


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_openai_chat_maps_agent_message_part():
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    messages = await model._map_messages([_make_request()], ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    # Find the user message with agent content
    user_msgs = [m for m in messages if m.get('role') == 'user']
    assert len(user_msgs) >= 1
    assert any(EXPECTED_PREFIX in str(m.get('content', '')) for m in user_msgs)
    assert any(EXPECTED_CONTENT in str(m.get('content', '')) for m in user_msgs)


# --- OpenAI Responses API ---


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_openai_responses_maps_agent_message_part():
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    _, messages = await model._map_messages([_make_request()], {}, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage,reportArgumentType]
    # The agent message should appear as a user message with the agent-prefixed content
    all_content = str(messages)
    assert EXPECTED_PREFIX in all_content
    assert EXPECTED_CONTENT in all_content


# --- Anthropic ---


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
async def test_anthropic_maps_agent_message_part():
    model = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key='test'))
    system, messages = await model._map_message([_make_request()], ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage]
    # Agent message should be in a user message as a text block
    all_content = str(messages)
    assert EXPECTED_PREFIX in all_content
    assert EXPECTED_CONTENT in all_content


# --- Google ---


@pytest.mark.skipif(not google_imports(), reason='google not installed')
async def test_google_maps_agent_message_part():
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(api_key='test'))
    _, contents = await model._map_messages([_make_request()], ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    all_content = str(contents)
    assert EXPECTED_PREFIX in all_content
    assert EXPECTED_CONTENT in all_content


# --- Bedrock ---


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
async def test_bedrock_maps_agent_message_part():
    model = BedrockConverseModel('anthropic.claude-3-5-sonnet-20241022-v2:0', provider=BedrockProvider(region_name='us-east-1'))
    _, messages = await model._map_messages([_make_request()], ModelRequestParameters(), None)  # pyright: ignore[reportPrivateUsage]
    all_content = str(messages)
    assert EXPECTED_PREFIX in all_content
    assert EXPECTED_CONTENT in all_content


# --- Groq ---


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
async def test_groq_maps_agent_message_part():
    model = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key='test'))
    messages = await model._map_messages([_make_request()], ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    user_msgs = [m for m in messages if m.get('role') == 'user']
    assert len(user_msgs) >= 1
    assert any(EXPECTED_PREFIX in str(m.get('content', '')) for m in user_msgs)


# --- Mistral ---


@pytest.mark.skipif(not mistral_imports(), reason='mistral not installed')
async def test_mistral_maps_agent_message_part():
    model = MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test'))
    messages = await model._map_messages([_make_request()], ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    all_content = str(messages)
    assert EXPECTED_PREFIX in all_content
    assert EXPECTED_CONTENT in all_content


# --- Cohere ---


@pytest.mark.skipif(not cohere_imports(), reason='cohere not installed')
async def test_cohere_maps_agent_message_part():
    model = CohereModel('command-r-plus', provider=CohereProvider(api_key='test'))
    messages = model._map_messages([_make_request()], ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    all_content = str(messages)
    assert EXPECTED_PREFIX in all_content
    assert EXPECTED_CONTENT in all_content


# --- HuggingFace ---


@pytest.mark.skipif(not huggingface_imports(), reason='huggingface not installed')
async def test_huggingface_maps_agent_message_part():
    model = HuggingFaceModel('meta-llama/Llama-3.1-8B-Instruct', provider=HuggingFaceProvider(api_key='test'))
    messages = await model._map_messages([_make_request()], ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    all_content = str(messages)
    assert EXPECTED_PREFIX in all_content
    assert EXPECTED_CONTENT in all_content


# --- xAI ---


@pytest.mark.skipif(not xai_imports(), reason='xai not installed')
async def test_xai_maps_agent_message_part():
    model = XaiModel('grok-2', provider=XaiProvider(api_key='test'))
    messages = await model._map_messages([_make_request()], ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    all_content = str(messages)
    # xAI uses protobuf format which escapes single quotes
    assert 'researcher' in all_content
    assert EXPECTED_CONTENT in all_content


# --- TestModel ---


async def test_test_model_handles_agent_message_part():
    """TestModel should not crash when message history contains AgentMessagePart."""
    model = TestModel()
    # TestModel._process_request should handle the message without crashing
    response = await model.request([_make_request()], None, ModelRequestParameters())  # pyright: ignore[reportArgumentType]
    assert response is not None


# --- Vercel AI UI adapter ---


async def test_vercel_ai_adapter_skips_agent_message_part():
    """Vercel AI adapter should skip AgentMessagePart (not crash, not display as user input)."""
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter

    # _dump_request_message is a static method
    system_parts, user_parts = VercelAIAdapter._dump_request_message(_make_request())  # pyright: ignore[reportPrivateUsage]
    # AgentMessagePart should not appear in either system or user UI parts
    assert len(system_parts) == 0
    assert len(user_parts) == 0


# --- AG-UI adapter ---


async def test_ag_ui_adapter_skips_agent_message_part():
    """AG-UI adapter should skip AgentMessagePart (not crash, not display as user input)."""
    from pydantic_ai.ui.ag_ui import AGUIAdapter

    # dump_messages is a classmethod
    messages = AGUIAdapter.dump_messages([_make_request()])  # pyright: ignore[reportAttributeAccessIssue]
    # AgentMessagePart should not appear in the dumped messages as user content
    all_content = str(messages)
    assert EXPECTED_CONTENT not in all_content or EXPECTED_PREFIX not in all_content