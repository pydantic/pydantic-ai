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
from pydantic_ai.native_tools import CodeExecutionTool
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
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]


@pytest.mark.parametrize(
    'client_name,profile,tool_names',
    [
        pytest.param('api', None, ['get_exchange_rate', 'tool_search_tool_bm25', 'code_execution'], id='api'),
        # Unverified on the platform count endpoints, so server tools stay off them.
        pytest.param('bedrock', None, ['get_exchange_rate'], id='bedrock'),
        pytest.param('mantle', None, ['get_exchange_rate'], id='mantle'),
        pytest.param('vertex', None, ['get_exchange_rate'], id='vertex'),
        pytest.param('foundry', None, ['get_exchange_rate'], id='foundry'),
        pytest.param(
            'vertex',
            {'anthropic_count_tokens_accepts_server_tools': True},
            ['get_exchange_rate', 'tool_search_tool_bm25', 'code_execution'],
            id='vertex-profile-on',
        ),
    ],
)
async def test_count_tokens_server_tools_by_client(
    allow_model_requests: None, client_name: str, profile: AnthropicModelProfile | None, tool_names: list[str]
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
        clients: dict[str, Callable[[], Any]] = {
            'api': lambda: AsyncAnthropic(api_key='x', http_client=http_client),
            'bedrock': lambda: AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1', http_client=http_client),
            'mantle': lambda: AsyncAnthropicBedrockMantle(api_key='x', aws_region='us-east-1', http_client=http_client),
            'vertex': lambda: AsyncAnthropicVertex(
                project_id='p', region='us-east5', access_token='x', http_client=http_client
            ),
            'foundry': lambda: AsyncAnthropicFoundry(
                api_key='x', base_url='https://example.com/anthropic', http_client=http_client
            ),
        }
        model = AnthropicModel(
            'claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=clients[client_name]()), profile=profile
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
                    )
                ],
                native_tools=[ToolSearchTool(), CodeExecutionTool()],
            ),
        )

    [body] = bodies
    assert [tool['name'] for tool in body['tools']] == tool_names


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
