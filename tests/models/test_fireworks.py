from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelSettings, NativeOutput, PromptedOutput

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.fireworks import FireworksProvider

    from .mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


class City(BaseModel):
    city: str
    country: str


@dataclass(frozen=True)
class NativeCase:
    id: str
    model_name: str
    response_format: dict[str, object]


NATIVE_CASES = [
    NativeCase(
        id='moonshotai-family',
        model_name='kimi-k3',
        response_format=snapshot(
            {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'City',
                    'schema': {
                        'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                        'required': ['city', 'country'],
                        'type': 'object',
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ),
    ),
    NativeCase(
        id='qwen-family',
        model_name='qwen3p8-max',
        response_format=snapshot(
            {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'City',
                    'schema': {
                        'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                        'required': ['city', 'country'],
                        'title': 'City',
                        'type': 'object',
                    },
                    'strict': True,
                },
            }
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in NATIVE_CASES])
async def test_fireworks_native_output(
    allow_model_requests: None, request_capture: RequestCapture, fireworks_api_key: str, case: NativeCase
):
    """Fireworks enforces `NativeOutput` schemas on models whose family profile doesn't claim support.

    Neither `moonshotai_model_profile` nor `qwen_model_profile` sets `supports_json_schema_output`, so this
    would raise `UserError: Native structured output is not supported by this model` before any request
    if `FireworksProvider` didn't set the flag. Although Fireworks' docs say `json_schema` disables reasoning
    output, both models return `reasoning_content` alongside the schema-constrained answer.
    """
    provider = FireworksProvider(api_key=fireworks_api_key, http_client=request_capture.client)
    model = OpenAIChatModel(f'accounts/fireworks/models/{case.model_name}', provider=provider)
    agent = Agent(model, output_type=NativeOutput(City))
    result = await agent.run('Where is the Eiffel Tower?')
    assert result.output == City(city='Paris', country='France')
    assert [part.part_kind for part in result.response.parts] == ['thinking', 'text']
    assert request_capture.body('/chat/completions')['response_format'] == case.response_format


async def test_fireworks_prompted_output(
    allow_model_requests: None, request_capture: RequestCapture, fireworks_api_key: str
):
    """`PromptedOutput` sends no `response_format` to a model whose profile doesn't claim JSON mode.

    `FireworksProvider` sets `supports_json_schema_output` only, so prompted requests keep their wire shape.
    """
    provider = FireworksProvider(api_key=fireworks_api_key, http_client=request_capture.client)
    model = OpenAIChatModel('accounts/fireworks/models/kimi-k3', provider=provider)
    agent = Agent(model, output_type=PromptedOutput(City))
    result = await agent.run('Where is the Eiffel Tower?')
    assert result.output == City(city='Paris', country='France')
    assert 'response_format' not in request_capture.body('/chat/completions')


async def test_fireworks_kimi_thinking(allow_model_requests: None):
    """`thinking` reaches a Kimi request in Fireworks' `p` spelling, and the model name is sent as given.

    Not recorded (no Fireworks key at hand), so it pins what the request carries, not what Fireworks accepts.
    """
    mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='OK', role='assistant')))
    model = OpenAIChatModel(
        'accounts/fireworks/models/kimi-k2p6', provider=FireworksProvider(openai_client=mock_client)
    )
    agent = Agent(model, model_settings=ModelSettings(thinking=True))
    await agent.run('hello')
    [kwargs] = get_mock_chat_completion_kwargs(mock_client)
    assert (kwargs['model'], kwargs.get('reasoning_effort')) == snapshot(
        ('accounts/fireworks/models/kimi-k2p6', 'medium')
    )
