from __future__ import annotations as _annotations

from datetime import datetime, timezone

import pytest
from inline_snapshot import snapshot

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

    from pydantic_ai.models.ollama import OllamaModel

    from .test_openai import MockOpenAI, completion_message

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_init():
    m = OllamaModel('llama3.2', base_url='foobar/')
    assert m.openai_model.client.api_key == 'ollama'
    assert m.openai_model.client.base_url == 'foobar/'
    assert m.name() == 'ollama:llama3.2'


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    print('here')
    m = OllamaModel('llama3.2', openai_client=mock_client, base_url=None)
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
