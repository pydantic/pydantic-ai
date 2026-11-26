"""Bedrock strict schemas move unsupported contains constraints into descriptions."""

from __future__ import annotations as _annotations

import json
import os
from collections.abc import Iterator
from copy import deepcopy
from typing import Literal

import pytest
from pydantic import BaseModel, Field, JsonValue

from pydantic_ai import Agent, ModelRequest, NativeOutput, ToolOutput, UserPromptPart
from pydantic_ai._json_schema import JsonSchema
from pydantic_ai.direct import model_request
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from ...conftest import try_import

with try_import() as imports_successful:
    from botocore.config import Config
    from botocore.session import Session

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.vcr,
]


@pytest.fixture
def contains_provider() -> Iterator[BedrockProvider]:
    """Bound live schema compilation and disable SDK retries when recording."""
    client = Session().create_client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'AKIA6666666666666666'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', '6666666666666666666666666666666666666666'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        config=Config(connect_timeout=10, read_timeout=60, retries={'total_max_attempts': 1}),
    )
    yield BedrockProvider(bedrock_client=client)
    client.close()


@pytest.mark.parametrize(
    'model_name,schema_type',
    [
        pytest.param('zai.glm-4.7-flash', 'array', id='zai.glm-4.7-flash'),
        pytest.param(
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            'array',
            id='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        ),
        pytest.param('qwen.qwen3-32b-v1:0', 'array', id='qwen.qwen3-32b-v1:0'),
        pytest.param('google.gemma-3-12b-it', 'array', id='google.gemma-3-12b-it'),
        pytest.param('us.anthropic.claude-sonnet-4-5-20250929-v1:0', ['array', 'null'], id='claude-nullable-array'),
    ],
)
@pytest.mark.parametrize('output_mode', ['native', 'tool'])
async def test_bedrock_contains(
    allow_model_requests: None,
    contains_provider: BedrockProvider,
    model_name: str,
    schema_type: str | list[str],
    output_mode: Literal['native', 'tool'],
):
    """Both strict request paths succeed and send the constraint as a description hint."""

    schema_extra: dict[str, JsonValue] = {
        'type': schema_type if isinstance(schema_type, str) else [*schema_type],
        'contains': {'type': 'string'},
    }

    class Result(BaseModel):
        values: list[str] = Field(json_schema_extra=schema_extra)

    requests: list[JsonSchema] = []

    def capture(params: JsonSchema, **_: object) -> None:
        requests.append(deepcopy(params))

    # `request()` streams a Claude request behind the scenes when it sends the model's maximum output as `maxTokens`.
    for operation in ('Converse', 'ConverseStream'):
        contains_provider.client.meta.events.register(f'before-parameter-build.bedrock-runtime.{operation}', capture)
    model = BedrockConverseModel(model_name, provider=contains_provider)
    output_type = NativeOutput(Result) if output_mode == 'native' else ToolOutput(Result, strict=True)
    prompt = 'Return values equal to ["ok"].' if output_mode == 'native' else 'Call final_result with values ["ok"].'
    if model_name == 'google.gemma-3-12b-it' and output_mode == 'tool':
        # Gemma answers in text even with a strict tool. Verify schema acceptance without requiring tool selection.
        response = await model_request(
            model,
            [ModelRequest(parts=[UserPromptPart(prompt)])],
            model_request_parameters=ModelRequestParameters(
                function_tools=[
                    ToolDefinition(
                        name='final_result',
                        description='The final response which ends this conversation',
                        parameters_json_schema=Result.model_json_schema(),
                        strict=True,
                    )
                ]
            ),
        )
        assert response.parts
    else:
        agent = Agent(model, output_type=output_type, retries=0)
        result = await agent.run(prompt)
        assert result.output.values == ['ok']
    assert len(requests) == 1
    schema: JsonSchema
    if output_mode == 'native':
        schema = json.loads(requests[0]['outputConfig']['textFormat']['structure']['jsonSchema']['schema'])
    else:
        tool_spec: JsonSchema = requests[0]['toolConfig']['tools'][0]['toolSpec']
        assert tool_spec['strict'] is True
        schema = tool_spec['inputSchema']['json']
    assert schema['properties']['values'] == {
        'items': {'type': 'string'},
        'type': schema_type,
        'description': "contains={'type': 'string'}",
    }
