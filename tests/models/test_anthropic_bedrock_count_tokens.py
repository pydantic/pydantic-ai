"""Token counting for Anthropic models served via the legacy `AsyncAnthropicBedrock` client.

The Anthropic SDK blocks its high-level `count_tokens()` on Bedrock, so `_messages_count_tokens`
posts the low-level Bedrock Runtime `/model/{model}/count-tokens` request via
`models/anthropic/_bedrock_count_tokens.py::count_tokens_via_bedrock`. These tests pin that request
and the malformed-response edge case. They mock `client.post` rather than record a cassette: Bedrock
CountTokens rejects every Claude 5-family id.
"""

from __future__ import annotations as _annotations

import base64
import json
from unittest.mock import AsyncMock

import pytest

from pydantic_ai import (
    Agent,
    ModelRequest,
    SystemPromptPart,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import UsageLimits

from .._inline_snapshot import snapshot
from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropicBedrock

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


async def test_anthropic_bedrock_count_tokens_request(allow_model_requests: None):
    """Bedrock token counting posts the request body, betas, `extra_body` and `timeout` to `/count-tokens`."""
    post = AsyncMock(return_value={'inputTokens': 18})
    bedrock_client = AsyncAnthropicBedrock(
        aws_access_key='test-access-key',
        aws_secret_key='test-secret-key',
        aws_region='us-east-1',
    )
    bedrock_client.post = post
    model = AnthropicModel('anthropic.claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=bedrock_client))

    result = await model.count_tokens(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are helpful.'),
                    UserPromptPart(content='How many tokens is this sentence?'),
                ]
            )
        ],
        AnthropicModelSettings(
            anthropic_betas=['token-efficient-tools-2025-02-19'],
            extra_body={'metadata': {'user_id': 'pydantic-ai-test'}},
            timeout=12.5,
        ),
        ModelRequestParameters(),
    )

    assert result.input_tokens == 18
    [call] = post.await_args_list
    assert call.args == snapshot(('/model/anthropic.claude-sonnet-4-6/count-tokens',))
    assert call.kwargs['options']['timeout'] == 12.5
    body = json.loads(base64.b64decode(json.loads(call.kwargs['content'])['input']['invokeModel']['body']))
    assert body == snapshot(
        {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 128000,
            'messages': [{'role': 'user', 'content': [{'text': 'How many tokens is this sentence?', 'type': 'text'}]}],
            'system': 'You are helpful.',
            'anthropic_beta': ['token-efficient-tools-2025-02-19'],
            'metadata': {'user_id': 'pydantic-ai-test'},
        }
    )


async def test_anthropic_bedrock_per_request_count_tokens_before_request(allow_model_requests: None):
    """`per_request_input_tokens_limit` with `count_tokens_before_request=True` raises before any message request."""
    post = AsyncMock(return_value={'inputTokens': 18})
    bedrock_client = AsyncAnthropicBedrock(
        aws_access_key='test-access-key',
        aws_secret_key='test-secret-key',
        aws_region='us-east-1',
    )
    bedrock_client.post = post
    agent = Agent(
        AnthropicModel('anthropic.claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=bedrock_client))
    )

    with pytest.raises(UsageLimitExceeded, match='per_request_input_tokens_limit of 10'):
        await agent.run(
            'hello', usage_limits=UsageLimits(per_request_input_tokens_limit=10, count_tokens_before_request=True)
        )

    assert post.await_count == 1


async def test_anthropic_bedrock_count_tokens_unexpected_response(allow_model_requests: None, env: TestEnv):
    """Pins the defensive `UnexpectedModelBehavior` branch for a malformed Bedrock response.

    Mocks `client.post` to return a body without `inputTokens` — a shape no real Bedrock
    CountTokens endpoint returns, so it can't be exercised through a VCR recording.
    """
    bedrock_client = AsyncAnthropicBedrock(
        aws_access_key='test-access-key',
        aws_secret_key='test-secret-key',
        aws_region='us-east-1',
    )
    bedrock_client.post = AsyncMock(return_value={'outputTokens': 42})
    provider = AnthropicProvider(anthropic_client=bedrock_client)
    model = AnthropicModel('anthropic.claude-3-5-sonnet-20241022-v2:0', provider=provider)

    with pytest.raises(UnexpectedModelBehavior, match='Unexpected Bedrock count tokens response'):
        await model.count_tokens(
            [ModelRequest.user_text_prompt('hello')],
            None,
            ModelRequestParameters(),
        )
