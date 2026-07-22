"""Real-provider coverage for failed function-tool results.

Regression coverage for https://github.com/pydantic/pydantic-ai/issues/2586 and
https://github.com/pydantic/pydantic-ai/pull/5585.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.models import Model

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]

ProviderName = Literal['anthropic', 'bedrock', 'google', 'openai_responses']


@dataclass(frozen=True)
class Case:
    provider: ProviderName
    model_name: str
    expected_output: str
    expected_wire: object
    marks: tuple[pytest.MarkDecorator, ...] = ()


CASES = [
    Case(
        provider='anthropic',
        model_name='claude-haiku-4-5',
        expected_output=snapshot('RECOVERED'),
        expected_wire=snapshot(
            {
                'tool_use_id': 'toolu_01TPSnLV2mMXtrhmUcLnz8LV',
                'type': 'tool_result',
                'content': [{'text': 'missing is unavailable', 'type': 'text'}],
                'is_error': True,
            }
        ),
        marks=(pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),),
    ),
    Case(
        provider='bedrock',
        model_name='us.amazon.nova-micro-v1:0',
        expected_output=snapshot('RECOVERED'),
        expected_wire=snapshot(
            {
                'toolUseId': 'tooluse_5FbuTjrYDTe5pl1ew70ebl',
                'content': [{'text': 'missing is unavailable'}],
                'status': 'error',
            }
        ),
        marks=(pytest.mark.skipif(not bedrock_available(), reason='boto3 not installed'),),
    ),
    Case(
        provider='google',
        model_name='gemini-2.5-flash',
        expected_output=snapshot('The lookup for "missing" returned an error: "missing is unavailable".'),
        expected_wire=snapshot(
            {
                'id': 'pyd_ai_c6fb2c3e541f401dbb2236e0188b18a9',
                'name': 'lookup',
                'response': {'error': 'missing is unavailable'},
            }
        ),
        marks=(pytest.mark.skipif(not google_available(), reason='google-genai not installed'),),
    ),
    Case(
        provider='openai_responses',
        model_name='gpt-5-mini',
        expected_output=snapshot('RECOVERED'),
        expected_wire=snapshot(
            {
                'type': 'function_call_output',
                'call_id': 'call_mh1NrXDAcVVM82dXOBnHXULj',
                'output': '{"error":"missing is unavailable"}',
            }
        ),
        marks=(pytest.mark.skipif(not openai_available(), reason='openai not installed'),),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.provider, marks=case.marks) for case in CASES])
async def test_tool_failed_reaches_model_as_failed_tool_result(
    case: Case,
    allow_model_requests: None,
    anthropic_api_key: str,
    bedrock_provider: BedrockProvider,
    gemini_api_key: str,
    openai_api_key: str,
    vcr: Cassette,
) -> None:
    """A real provider accepts `ToolFailed` and completes after receiving its error channel."""
    if case.provider == 'anthropic':
        model: Model = AnthropicModel(case.model_name, provider=AnthropicProvider(api_key=anthropic_api_key))
    elif case.provider == 'bedrock':
        model = BedrockConverseModel(case.model_name, provider=bedrock_provider)
    elif case.provider == 'google':
        model = GoogleModel(case.model_name, provider=GoogleProvider(api_key=gemini_api_key))
    else:
        model = OpenAIResponsesModel(case.model_name, provider=OpenAIProvider(api_key=openai_api_key))

    agent = Agent(model)

    @agent.tool_plain
    def lookup(item: str) -> str:
        """Look up an item."""
        raise ToolFailed(f'{item} is unavailable')

    result = await agent.run(
        'Call lookup exactly once with item "missing". If it fails, reply with exactly RECOVERED and nothing else.'
    )

    assert result.output == case.expected_output

    request_body = json.loads(
        vcr.requests[-1].body  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    )
    if case.provider == 'anthropic':
        actual_wire = request_body['messages'][-1]['content'][0]
    elif case.provider == 'bedrock':
        actual_wire = request_body['messages'][-1]['content'][0]['toolResult']
    elif case.provider == 'google':
        actual_wire = request_body['contents'][-1]['parts'][0]['functionResponse']
    else:
        actual_wire = request_body['input'][-1]
    assert actual_wire == case.expected_wire
