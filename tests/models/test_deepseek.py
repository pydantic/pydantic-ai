from __future__ import annotations as _annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient as AsyncHTTPClient
from inline_snapshot import snapshot
from openai import AsyncOpenAI, OpenAIError

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)
from pydantic_ai.result import Usage

from ..conftest import IsNow, try_import

with try_import() as imports_successful:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.deepseek import DeepSeekModel

    from .test_openai import MockOpenAI, completion_message


@contextmanager
def modified_env(name: str, value: str | None) -> Iterator[None]:
    """Temporarily modify an environment variable."""
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_init():
    # Test with explicit API key
    m = DeepSeekModel('deepseek-chat', api_key='test-key')
    assert m.openai_model.client.api_key == 'test-key'
    assert m.openai_model.client.base_url == 'https://api.deepseek.com/v1/'
    assert m.name() == 'deepseek-chat'

    # Test with openai_client
    client = AsyncOpenAI(api_key='test-key-2', base_url='https://api.deepseek.com/v1/')
    m = DeepSeekModel('deepseek-chat', openai_client=client)
    assert m.openai_model.client.api_key == 'test-key-2'
    assert m.openai_model.client.base_url == 'https://api.deepseek.com/v1/'
    assert m.name() == 'deepseek-chat'

    # Test with http_client
    client = AsyncHTTPClient(base_url='https://api.deepseek.com/v1')
    m = DeepSeekModel('deepseek-chat', api_key='test-key-3', http_client=client)
    assert m.openai_model.client.api_key == 'test-key-3'
    assert m.openai_model.client.base_url == 'https://api.deepseek.com/v1/'
    assert m.name() == 'deepseek-chat'

    # Test without API key should raise error
    with pytest.raises(OpenAIError, match='DeepSeek API key not found'):
        with modified_env('DEEPSEEK_API_KEY', None):
            DeepSeekModel('deepseek-chat')


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = DeepSeekModel('deepseek-chat', openai_client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )
