"""Which Anthropic SDK clients get `CodeExecutionTool`.

Neither Bedrock API serves Anthropic's code execution tool, so `AnthropicModel.profile` leaves it out of
`supported_native_tools` for `AsyncAnthropicBedrock` and `AsyncAnthropicBedrockMantle`.
https://github.com/pydantic/pydantic-ai/issues/11513
"""

from __future__ import annotations as _annotations

from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import CodeExecutionTool

from ..._inline_snapshot import snapshot
from ...conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from anthropic import (
        APIStatusError,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )

    from pydantic_ai.models.anthropic import AnthropicCodeExecutionToolVersion, AnthropicModel
    from pydantic_ai.models.anthropic._native_tools import map_code_execution_tool
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]


async def test_code_execution_tool_client_gating(allow_model_requests: None):
    """`CodeExecutionTool` is advertised on every client but the two Bedrock ones, where a run asking for it fails.

    The run fails in `prepare_request`, before any request is sent, so the clients carry placeholder credentials.
    """
    clients: dict[str, Any] = {
        'direct': None,
        'bedrock': AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1'),
        'mantle': AsyncAnthropicBedrockMantle(api_key='x', aws_region='us-east-1'),
        'vertex': AsyncAnthropicVertex(project_id='p', region='us-east5', access_token='x'),
        'foundry': AsyncAnthropicFoundry(api_key='x', base_url='https://example.com/anthropic'),
    }
    models = {
        name: AnthropicModel(
            'claude-sonnet-4-5',
            provider=AnthropicProvider(api_key='x') if client is None else AnthropicProvider(anthropic_client=client),
        )
        for name, client in clients.items()
    }

    assert {
        name: CodeExecutionTool in model.profile.get('supported_native_tools', frozenset[Any]())
        for name, model in models.items()
    } == snapshot({'direct': True, 'bedrock': False, 'mantle': False, 'vertex': True, 'foundry': True})

    for name in ('bedrock', 'mantle'):
        agent = Agent(models[name], capabilities=[NativeTool(CodeExecutionTool())])
        with pytest.raises(UserError, match=r"Native tool\(s\) \['CodeExecutionTool'\] not supported by this model"):
            await agent.run('What is 2**100?')


@pytest.mark.vcr()
@pytest.mark.parametrize(
    'client_fixture,model_name,version,error',
    [
        pytest.param(
            'anthropic_bedrock_client',
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            '20250825',
            "Input tag 'code_execution_20250825' found using 'type' does not match any of the expected tags",
            id='bedrock-sonnet-4-5-20250825',
        ),
        pytest.param(
            'anthropic_bedrock_client',
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            '20260120',
            "Input tag 'code_execution_20260120' found using 'type' does not match any of the expected tags",
            id='bedrock-sonnet-4-5-20260120',
        ),
        pytest.param(
            'anthropic_bedrock_client',
            'us.anthropic.claude-opus-4-8',
            '20260120',
            "tool type 'code_execution_20260120' is not supported for this model",
            id='bedrock-opus-4-8-20260120',
        ),
        pytest.param(
            'anthropic_bedrock_mantle_client',
            'anthropic.claude-opus-5-5',
            '20250825',
            "tool type 'code_execution_20250825' is not supported for this model",
            id='mantle-opus-5-5-20250825',
        ),
        pytest.param(
            'anthropic_bedrock_mantle_client',
            'anthropic.claude-opus-5-5',
            '20260120',
            "tool type 'code_execution_20260120' is not supported for this model",
            id='mantle-opus-5-5-20260120',
        ),
    ],
)
async def test_code_execution_tool_rejected_by_bedrock(
    allow_model_requests: None,
    request_capture: RequestCapture,
    request: pytest.FixtureRequest,
    client_fixture: str,
    model_name: str,
    version: AnthropicCodeExecutionToolVersion,
    error: str,
):
    """Both Bedrock APIs reject both code execution tool versions, which is why their clients don't get the tool.

    `AnthropicModel` no longer sends the tool there, so this sends the definition it builds through the SDK.
    That pins the provider behavior `test_code_execution_tool_client_gating` rests on.
    """
    client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle = request.getfixturevalue(client_fixture)

    with pytest.raises(APIStatusError, match=error):
        await client.beta.messages.create(
            model=model_name,
            max_tokens=256,
            messages=[{'role': 'user', 'content': 'Use code execution to compute 2**100.'}],
            tools=[map_code_execution_tool(version)],
        )
