"""GLM-5.3 on Bedrock requires `$defs`/`$ref` inlined into tool and output schemas, and keeps the strict `contains` rewrite."""

from __future__ import annotations as _annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from pydantic_ai import Agent, ToolOutput

from ...conftest import try_import

with try_import() as imports_successful:
    from botocore.hooks import HierarchicalEmitter
    from botocore.loaders import Loader
    from botocore.model import ServiceModel

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')]

MODEL_NAME = 'zai.glm-5.3'


class Address(BaseModel):
    """Nested model whose schema GLM-5.3 requires to be inlined."""

    street: str


class Result(BaseModel):
    """Output schema that is nested (for inlining) and carries `contains` (for the strict rewrite)."""

    values: list[str] = Field(json_schema_extra={'contains': {'type': 'string', 'pattern': '^A'}})
    address: Address


def get_weather(city: str, address: Address) -> str:
    return f'The weather in {address.street}, {city} is sunny'


def _converse_response(content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
    return {
        'output': {'message': {'role': 'assistant', 'content': content}},
        'stopReason': stop_reason,
        'usage': {'inputTokens': 1, 'outputTokens': 1, 'totalTokens': 2},
    }


class _StubBedrockClient:
    """Captures the Converse request and replays canned responses through the real `BedrockProvider`.

    Not a VCR test: cassette matchers aren't sensitive to the request body, so a recording wouldn't catch a
    schema-inlining regression — asserting the request body directly is what catches it (see tests/AGENTS.md).
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.meta = SimpleNamespace(
            endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com',
            region_name='us-east-1',
            service_model=ServiceModel(Loader().load_service_model('bedrock-runtime', 'service-2')),
            events=HierarchicalEmitter(),
        )

    def converse(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return self._responses.pop(0)


async def test_glm53_tool_schema_inlines_nested_defs(allow_model_requests: None):
    client = _StubBedrockClient(
        [
            _converse_response(
                [
                    {
                        'toolUse': {
                            'toolUseId': 'call_1',
                            'name': 'get_weather',
                            'input': {'city': 'Mexico City', 'address': {'street': 'Main St'}},
                        }
                    }
                ],
                'tool_use',
            ),
            _converse_response([{'text': 'Done'}], 'end_turn'),
        ]
    )
    agent = Agent(
        BedrockConverseModel(MODEL_NAME, provider=BedrockProvider(bedrock_client=client)),
        tools=[get_weather],
        retries=0,
    )
    result = await agent.run('What is the weather in Mexico City?')

    assert result.output == 'Done'
    assert len(client.requests) == 2
    tool_schema = client.requests[0]['toolConfig']['tools'][0]['toolSpec']['inputSchema']['json']
    assert '$defs' not in tool_schema
    assert '$ref' not in json.dumps(tool_schema)
    assert tool_schema['properties']['address']['properties']['street'] == {'type': 'string'}


async def test_glm53_strict_output_schema_inlines_and_rewrites_contains(allow_model_requests: None):
    client = _StubBedrockClient(
        [
            _converse_response(
                [
                    {
                        'toolUse': {
                            'toolUseId': 'call_1',
                            'name': 'final_result',
                            'input': {'values': ['Alpha'], 'address': {'street': 'Main St'}},
                        }
                    }
                ],
                'tool_use',
            )
        ]
    )
    agent = Agent(
        BedrockConverseModel(MODEL_NAME, provider=BedrockProvider(bedrock_client=client)),
        output_type=ToolOutput(Result, strict=True),
        retries=0,
    )
    result = await agent.run('Produce the result')

    assert result.output == Result(values=['Alpha'], address=Address(street='Main St'))
    schema = client.requests[0]['toolConfig']['tools'][0]['toolSpec']['inputSchema']['json']
    assert '$defs' not in schema
    assert '$ref' not in json.dumps(schema)
    assert schema['additionalProperties'] is False
    values_schema = schema['properties']['values']
    assert 'contains' not in values_schema
    assert 'contains=' in values_schema['description']
