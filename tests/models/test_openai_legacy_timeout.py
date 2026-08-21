"""Tests for the OpenAI paths' legacy-timeout normalization.

Not VCR tests: `ModelSettings.timeout` still takes a legacy `httpx.Timeout`, which the OpenAI SDK's
HTTPX2 client rejects, so both request paths convert it to an equivalent `httpx2.Timeout` before the
SDK call — and the SDK-bound `timeout` never reaches the wire, so only the mock kwargs show the
conversion.
"""

import httpx
import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import
from .mock_openai import (
    MockOpenAI,
    MockOpenAIResponses,
    completion_message,
    get_mock_chat_completion_kwargs,
    get_mock_responses_kwargs,
    response_message,
)

with try_import() as imports_successful:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]

LEGACY_TIMEOUT = ModelSettings(timeout=httpx.Timeout(connect=1, read=2, write=3, pool=4))


def _assert_normalized(timeout: object) -> None:
    assert isinstance(timeout, httpx2.Timeout)
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (1, 2, 3, 4)


def test_chat_legacy_timeout_normalized(allow_model_requests: None) -> None:
    mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='hello', role='assistant')))
    agent = Agent(OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client)))

    agent.run_sync('hello', model_settings=LEGACY_TIMEOUT)

    _assert_normalized(get_mock_chat_completion_kwargs(mock_client)[0]['timeout'])


async def test_responses_legacy_timeout_normalized(allow_model_requests: None) -> None:
    content: list[Content] = [ResponseOutputText(text='done', type='output_text', annotations=[])]
    c = response_message(
        [
            ResponseOutputMessage(
                id='output-1',
                content=content,
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model).run('hello', model_settings=LEGACY_TIMEOUT)

    _assert_normalized(get_mock_responses_kwargs(mock_client)[0]['timeout'])
