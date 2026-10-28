"""Which Anthropic SDK clients send server tools to `count_tokens`.

Mocked at the HTTP transport rather than recorded, because Bedrock, Vertex AI and Foundry can't be recorded here.
Going through each real SDK client checks the `tools` list survives the client's own request rewriting.
"""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import CodeExecutionTool, MemoryTool
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import ToolDefinition

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.profiles.anthropic import AnthropicModelProfile
    from pydantic_ai.providers.anthropic import AnthropicProvider, AsyncAnthropicClient

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]


SERVER_TOOLS_DROPPED = [('get_exchange_rate', None), ('memory', 'memory_20250818')]
SERVER_TOOLS_KEPT = [
    *SERVER_TOOLS_DROPPED,
    ('tool_search_tool_bm25', 'tool_search_tool_bm25_20251119'),
    ('code_execution', 'code_execution_20260120'),
]


@pytest.mark.parametrize(
    'client,profile,tools',
    [
        pytest.param(lambda: AsyncAnthropic(api_key='x'), None, SERVER_TOOLS_KEPT, id='api'),
        # Unverified on the platform count endpoints, so server tools stay off them. `MemoryTool` is client-side, so
        # it stays on.
        pytest.param(
            lambda: AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1'),
            None,
            SERVER_TOOLS_DROPPED,
            id='bedrock',
        ),
        pytest.param(
            lambda: AsyncAnthropicBedrockMantle(api_key='x', aws_region='us-east-1'),
            None,
            SERVER_TOOLS_DROPPED,
            id='mantle',
        ),
        pytest.param(
            lambda: AsyncAnthropicVertex(project_id='p', region='us-east5', access_token='x'),
            None,
            SERVER_TOOLS_DROPPED,
            id='vertex',
        ),
        pytest.param(
            lambda: AsyncAnthropicFoundry(api_key='x', base_url='https://example.com/anthropic'),
            None,
            SERVER_TOOLS_DROPPED,
            id='foundry',
        ),
        pytest.param(
            lambda: AsyncAnthropicVertex(project_id='p', region='us-east5', access_token='x'),
            {'anthropic_count_tokens_accepts_server_tools': True},
            SERVER_TOOLS_KEPT,
            id='vertex-profile-on',
        ),
    ],
)
async def test_count_tokens_server_tools_by_client(
    allow_model_requests: None,
    client: Callable[[], AsyncAnthropicClient],
    profile: AnthropicModelProfile | None,
    tools: list[tuple[str, str | None]],
):
    """The profile's `anthropic_count_tokens_accepts_server_tools` decides whether `count_tokens` keeps server tools."""
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if 'input' in body:
            # The legacy Bedrock CountTokens endpoint wraps the Messages body.
            body = json.loads(base64.b64decode(body['input']['invokeModel']['body']))
        bodies.append(body)
        return httpx2.Response(200, json={'input_tokens': 10, 'inputTokens': 10})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        model = AnthropicModel(
            'claude-sonnet-5-5',
            provider=AnthropicProvider(anthropic_client=client().copy(http_client=http_client)),
            profile=profile,
        )
        await model.count_tokens(
            [ModelRequest.user_text_prompt('What is the USD to EUR rate?')],
            None,
            ModelRequestParameters(
                function_tools=[
                    ToolDefinition(
                        name='get_exchange_rate',
                        parameters_json_schema={'type': 'object'},
                        defer_loading=True,
                        with_native=ToolSearchTool.kind,
                    ),
                    ToolDefinition(name='memory', parameters_json_schema={'type': 'object'}),
                ],
                native_tools=[MemoryTool(), ToolSearchTool(), CodeExecutionTool()],
            ),
        )

    [body] = bodies
    assert [(tool['name'], tool.get('type')) for tool in body['tools']] == tools


def test_count_tokens_server_tools_provider_profile_wins():
    """A provider's own `model_profile()` value wins over the default `AnthropicModel` sets by client class.

    Checked on the resolved profile rather than a request, since this is profile resolution alone;
    `test_count_tokens_server_tools_by_client` pins what each flag value sends.
    """

    class CountsWithoutServerToolsProvider(AnthropicProvider):
        @staticmethod
        def model_profile(model_name: str) -> ModelProfile | None:
            return AnthropicModelProfile(anthropic_count_tokens_accepts_server_tools=False)

    model = AnthropicModel('claude-sonnet-5-5', provider=CountsWithoutServerToolsProvider(api_key='x'))
    assert model.profile.get('anthropic_count_tokens_accepts_server_tools') is False
