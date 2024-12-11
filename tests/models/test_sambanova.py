from __future__ import annotations as _annotations

from datetime import datetime, timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelTextResponse,
    UserPrompt,
)
from pydantic_ai.result import Cost

from ..conftest import IsNow, try_import

with try_import() as imports_successful:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.sambanova import SambaNovaModel

    from .test_openai import MockOpenAI, completion_message

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_init():
    m = SambaNovaModel('Meta-Llama-3.1-8B-Instruct', api_key='foobar')
    assert m.client.api_key == 'foobar'
    assert str(m.client.base_url) == 'https://api.sambanova.ai/v1/'
    assert m.name() == 'sambanova:Meta-Llama-3.1-8B-Instruct'


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = SambaNovaModel('Meta-Llama-3.1-8B-Instruct', openai_client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.cost() == snapshot(Cost())

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.data == 'world'
    assert result.cost() == snapshot(Cost())
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
            UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )
