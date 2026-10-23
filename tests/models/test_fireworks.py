from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, NativeOutput, PromptedOutput

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.fireworks import FireworksProvider


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
        id='unprofiled',
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

    Neither the unprofiled `kimi-k3` nor `qwen_model_profile` sets `supports_json_schema_output`, so this
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
