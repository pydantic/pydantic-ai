"""An empty response a provider ended with an error falls back in `ModelFallback`, whatever the provider calls it.

Each provider has its own way to end a response with an error, which its adapter maps to `finish_reason='error'`:
Mistral's and OpenRouter's `"error"`, Bedrock's `malformed_model_output` and `malformed_tool_use`, and Gemini's
`MALFORMED_FUNCTION_CALL`. Not VCR tests: providers don't send these responses on demand, so each case replays the
response's shape through the provider SDK with a mocked client.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from botocore.hooks import HierarchicalEmitter
    from google.genai.types import FinishReason as GoogleFinishReason, GenerateContentResponse
    from mistralai.client.models import AssistantMessage as MistralAssistantMessage
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.mistral import MistralProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from .mock_openai import MockOpenAI
    from .test_mistral import MockMistralAI, completion_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='provider SDKs not installed')


def _mistral(monkeypatch: pytest.MonkeyPatch) -> Model:
    response = completion_message(MistralAssistantMessage(content=''))
    response.choices[0].finish_reason = 'error'
    return MistralModel(
        'mistral-large-latest', provider=MistralProvider(mistral_client=MockMistralAI.create_mock(response))
    )


class _BedrockClient:
    """A Bedrock Runtime client whose `converse` returns `response`."""

    def __init__(self, response: dict[str, Any]):
        self._response = response
        self.meta = SimpleNamespace(endpoint_url='https://bedrock.stub', events=HierarchicalEmitter())

    def converse(self, **_: Any) -> dict[str, Any]:
        return self._response


def _bedrock(stop_reason: str) -> Callable[[pytest.MonkeyPatch], Model]:
    def build(monkeypatch: pytest.MonkeyPatch) -> Model:
        client = _BedrockClient(
            {
                'output': {'message': {'role': 'assistant', 'content': []}},
                'stopReason': stop_reason,
                'usage': {'inputTokens': 10, 'outputTokens': 0, 'totalTokens': 10},
                'metrics': {'latencyMs': 1},
            }
        )
        return BedrockConverseModel(
            'us.amazon.nova-micro-v1:0', provider=BedrockProvider(bedrock_client=cast(Any, client))
        )

    return build


def _openrouter(monkeypatch: pytest.MonkeyPatch) -> Model:
    # Built without validation, as the OpenAI SDK's types don't allow `finish_reason: "error"`. There's no top-level
    # `error`, which OpenRouter raises as an HTTP error instead.
    choice = Choice.model_construct(
        index=0,
        message=ChatCompletionMessage.model_construct(role='assistant', content=None),
        finish_reason='error',
        native_finish_reason='error',
    )
    completion = ChatCompletion.model_construct(
        id='gen-123',
        choices=[choice],
        created=1704067200,
        model='google/gemini-3.5-flash',
        object='chat.completion',
        provider='Google',
    )
    return OpenRouterModel(
        'google/gemini-3.5-flash', provider=OpenRouterProvider(openai_client=MockOpenAI.create_mock(completion))
    )


def _google(monkeypatch: pytest.MonkeyPatch) -> Model:
    model = GoogleModel('gemini-3.5-flash', provider=GoogleProvider(api_key='test'))
    response = GenerateContentResponse.model_validate(
        {
            'response_id': 'resp-123',
            'model_version': 'gemini-3.5-flash',
            'candidates': [{'finish_reason': GoogleFinishReason.MALFORMED_FUNCTION_CALL, 'content': None}],
        }
    )
    monkeypatch.setattr(model.client.aio.models, 'generate_content', AsyncMock(return_value=response))
    return model


@dataclass
class Case:
    id: str
    build: Callable[[pytest.MonkeyPatch], Model]
    """The provider's model, with a mocked client that answers with an empty response it ended with an error."""
    expected_attempts: list[tuple[str, str]]
    """The `(model_name, outcome)` of each attempt the answering response lists."""


CASES = [
    Case('mistral', _mistral, expected_attempts=snapshot([('mistral-large-123', 'rejected')])),
    Case(
        'bedrock-malformed-model-output',
        _bedrock('malformed_model_output'),
        expected_attempts=snapshot([('us.amazon.nova-micro-v1:0', 'rejected')]),
    ),
    Case(
        'bedrock-malformed-tool-use',
        _bedrock('malformed_tool_use'),
        expected_attempts=snapshot([('us.amazon.nova-micro-v1:0', 'rejected')]),
    ),
    Case('openrouter', _openrouter, expected_attempts=snapshot([('google/gemini-3.5-flash', 'rejected')])),
    Case('google-malformed-function-call', _google, expected_attempts=snapshot([('gemini-3.5-flash', 'rejected')])),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_empty_error_response_falls_back(
    case: Case, allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = Agent(case.build(monkeypatch), capabilities=[ModelFallback(TestModel(custom_output_text='fallback used'))])

    result = await agent.run('hello')

    assert result.output == 'fallback used'
    assert [(a.model_name, a.outcome) for a in result.response.failed_attempts or []] == case.expected_attempts
