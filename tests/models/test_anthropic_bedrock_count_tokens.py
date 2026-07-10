"""Token counting for Anthropic models served via the legacy `AsyncAnthropicBedrock` client.

The Anthropic SDK blocks its high-level `count_tokens()` on Bedrock, so `_messages_count_tokens`
posts the low-level Bedrock Runtime `/model/{model}/count-tokens` request via
`_anthropic_bedrock_count_tokens.count_tokens_via_bedrock`. These tests pin that path: the happy
path, the user-facing `count_tokens_before_request` path, error mapping, and the server-tool and
malformed-response edge cases.
"""

from __future__ import annotations as _annotations

import base64
import json
import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from pydantic_ai import (
    Agent,
    ModelHTTPError,
    ModelRequest,
    SystemPromptPart,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import UsageLimits

from .._inline_snapshot import snapshot
from ..cassette_utils import single_request_body
from ..conftest import TestEnv, try_import

if TYPE_CHECKING:
    from vcr.cassette import Cassette

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, AsyncAnthropicBedrock, BadRequestError, omit as OMIT
    from anthropic.types.beta import BetaMessageParam, BetaToolUnionParam

    from pydantic_ai.models._anthropic_bedrock_count_tokens import count_tokens_via_bedrock
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture
def bedrock_client() -> AsyncAnthropicBedrock:
    """The env-var-defaulted `AsyncAnthropicBedrock` client shared by the live-recorded Bedrock tests.

    `AsyncAnthropicBedrock`'s SigV4 signer imports `botocore` at request-prep time, which only
    ships under the `bedrock` extra (not in the default `pydantic-ai` install on v2).
    """
    pytest.importorskip('botocore')

    return AsyncAnthropicBedrock(
        aws_access_key=os.environ.get('AWS_ACCESS_KEY_ID', 'test-access-key'),
        aws_secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'test-secret-key'),
        aws_session_token=os.environ.get('AWS_SESSION_TOKEN'),
        aws_region=os.environ.get('AWS_REGION', 'us-east-1'),
    )


async def test_anthropic_bedrock_count_tokens_unexpected_response(env: TestEnv):
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


@pytest.mark.vcr()
async def test_anthropic_bedrock_count_tokens_real_api(
    allow_model_requests: None, bedrock_client: AsyncAnthropicBedrock, vcr: Cassette
):
    """Bedrock token counting hits the low-level `/model/{model}/count-tokens` endpoint.

    The Anthropic SDK blocks the high-level `count_tokens()` on Bedrock, so the request is
    built and posted by `_anthropic_bedrock_count_tokens.count_tokens_via_bedrock`. Uses the
    base model id `anthropic.claude-sonnet-4-...` because the `3.5-sonnet` id in the original
    test is now end-of-life on Bedrock's CountTokens endpoint, and CRIS inference-profile ids
    (`us.`/`eu.`/`global.`) aren't accepted there either.
    """
    model = AnthropicModel(
        'anthropic.claude-sonnet-4-20250514-v1:0',
        provider=AnthropicProvider(anthropic_client=bedrock_client),
    )

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

    assert result.input_tokens == snapshot(18)

    assert vcr.requests[0].path == snapshot('/model/anthropic.claude-sonnet-4-20250514-v1:0/count-tokens')  # pyright: ignore[reportUnknownMemberType]
    envelope = single_request_body(vcr)
    body = json.loads(base64.b64decode(envelope['input']['invokeModel']['body']))
    assert body == snapshot(
        {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 4096,
            'messages': [{'role': 'user', 'content': [{'text': 'How many tokens is this sentence?', 'type': 'text'}]}],
            'system': 'You are helpful.',
            'anthropic_beta': ['token-efficient-tools-2025-02-19'],
            'metadata': {'user_id': 'pydantic-ai-test'},
        }
    )


@pytest.mark.vcr()
async def test_anthropic_bedrock_count_tokens_error(allow_model_requests: None, bedrock_client: AsyncAnthropicBedrock):
    """A cross-region inference-profile (CRIS) id surfaces Bedrock's own error as `ModelHTTPError`.

    Before this fix, `AsyncAnthropicBedrock` raised a `UserError` for every `count_tokens` call. Now
    the request reaches Bedrock's `/count-tokens` endpoint, which only accepts base foundation-model
    ids and 400s on a CRIS profile id; `_map_api_errors` maps that to `ModelHTTPError`.
    """
    model_id = 'us.anthropic.claude-sonnet-4-20250514-v1:0'
    model = AnthropicModel(model_id, provider=AnthropicProvider(anthropic_client=bedrock_client))
    agent = Agent(model)

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == model_id


@pytest.mark.vcr('test_anthropic_bedrock_count_tokens_real_api.yaml')
async def test_anthropic_bedrock_count_tokens_before_request(
    allow_model_requests: None, bedrock_client: AsyncAnthropicBedrock
):
    """The user-facing path: `agent.run(count_tokens_before_request=True)` counts tokens via Bedrock.

    Reuses the `..._real_api` cassette (a single `/count-tokens` interaction returning 18 tokens). With
    a limit below that count, the pre-request check raises `UsageLimitExceeded` before any message
    request, so only the count-tokens call is played back.
    """
    model = AnthropicModel(
        'anthropic.claude-sonnet-4-20250514-v1:0',
        provider=AnthropicProvider(anthropic_client=bedrock_client),
    )
    agent = Agent(model)

    with pytest.raises(UsageLimitExceeded, match='input_tokens_limit of 10'):
        await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=10, count_tokens_before_request=True))


@pytest.mark.vcr('test_anthropic_bedrock_count_tokens_real_api.yaml')
async def test_anthropic_bedrock_per_request_count_tokens_before_request(
    allow_model_requests: None, bedrock_client: AsyncAnthropicBedrock
):
    """The per-request variant of the pre-request guard: `per_request_input_tokens_limit`.

    Reuses the `..._real_api` cassette (a single `/count-tokens` interaction returning 18 tokens).
    With a per-request limit below that count, the pre-request check raises `UsageLimitExceeded`
    before any message request, so only the count-tokens call is played back.
    """
    model = AnthropicModel(
        'anthropic.claude-sonnet-4-20250514-v1:0',
        provider=AnthropicProvider(anthropic_client=bedrock_client),
    )
    agent = Agent(model)

    with pytest.raises(UsageLimitExceeded, match='per_request_input_tokens_limit of 10'):
        await agent.run(
            'hello', usage_limits=UsageLimits(per_request_input_tokens_limit=10, count_tokens_before_request=True)
        )


@pytest.mark.vcr()
async def test_anthropic_bedrock_count_tokens_rejects_server_tools(
    allow_model_requests: None, bedrock_client: AsyncAnthropicBedrock
):
    """Bedrock's count-tokens endpoint rejects server tools, exactly like the standard endpoint.

    `_messages_count_tokens` strips server tools (keeping only client-side tools like `MemoryTool`)
    before counting, on both the standard and Bedrock paths. Posting a `web_search` server tool to the
    low-level Bedrock endpoint directly confirms the strip is needed on Bedrock too: the endpoint 400s
    rather than counting the tool, so Bedrock does not diverge from the standard endpoint and there is
    no Bedrock-specific undercounting to fix.
    """
    messages: list[BetaMessageParam] = [{'role': 'user', 'content': 'What is the weather in Paris today?'}]
    tools: list[BetaToolUnionParam] = [{'type': 'web_search_20250305', 'name': 'web_search'}]

    with pytest.raises(BadRequestError) as exc_info:
        await count_tokens_via_bedrock(
            bedrock_client,
            'anthropic.claude-sonnet-4-20250514-v1:0',
            system=OMIT,
            messages=messages,
            max_tokens=1024,
            tools=tools,
            tool_choice=OMIT,
            mcp_servers=OMIT,
            betas=OMIT,
            output_config=OMIT,
            cache_control=OMIT,
            thinking=OMIT,
            context_management=OMIT,
            timeout=NOT_GIVEN,
            speed=OMIT,
            extra_headers={},
            extra_body=None,
        )

    assert exc_info.value.status_code == 400
    assert 'does not match any of the expected tags' in str(exc_info.value)
