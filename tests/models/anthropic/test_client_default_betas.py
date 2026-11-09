"""A beta in the Anthropic client's `default_headers` goes out with the betas Pydantic AI sends.

Mocked at the HTTP transport rather than captured with `request_capture`, whose client needs a recording to answer:
Bedrock, Vertex AI and Foundry can't be recorded here, and the claim is about the header lines each SDK client sends,
which no response bears on.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider, AsyncAnthropicClient

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')

_MESSAGE = {
    'id': 'msg_1',
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-4-5',
    'content': [{'type': 'text', 'text': 'Hi!'}],
    'stop_reason': 'end_turn',
    'stop_sequence': None,
    'usage': {'input_tokens': 10, 'output_tokens': 5},
}


@dataclass
class SentBetas:
    """The `anthropic-beta` header lines of each request, and the `anthropic_beta` field of its body."""

    header_lines: list[list[str]] = field(default_factory=list[list[str]])
    body_fields: list[list[str] | None] = field(default_factory=list[list[str] | None])

    def model(self, client: AsyncAnthropicClient, default_headers: dict[str, str]) -> AnthropicModel:
        def answer(request: httpx2.Request) -> httpx2.Response:
            self.header_lines.append(
                [value.decode() for name, value in request.headers.raw if name.lower() == b'anthropic-beta']
            )
            self.body_fields.append(json.loads(request.content).get('anthropic_beta'))
            return httpx2.Response(200, json={**_MESSAGE, 'input_tokens': 10})

        http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(answer))
        client = client.copy(http_client=http_client, default_headers=default_headers)
        return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))


CLIENTS = [
    pytest.param(lambda: AsyncAnthropic(api_key='x'), id='api'),
    pytest.param(lambda: AsyncAnthropicVertex(project_id='p', region='us-east5', access_token='x'), id='vertex'),
    pytest.param(lambda: AsyncAnthropicFoundry(api_key='x', base_url='https://example.com/anthropic'), id='foundry'),
    pytest.param(lambda: AsyncAnthropicBedrockMantle(api_key='x', aws_region='us-east-1'), id='mantle'),
]
SPELLINGS = [pytest.param('anthropic-beta', id='lowercase'), pytest.param('Anthropic-Beta', id='title-case')]


@pytest.mark.parametrize('client', CLIENTS)
@pytest.mark.parametrize('header_name', SPELLINGS)
@pytest.mark.parametrize(
    'settings,header_line',
    [
        pytest.param({'max_tokens': 16}, 'b-client', id='no betas of ours'),
        pytest.param(
            {'max_tokens': 16, 'anthropic_betas': ['b-setting']},
            'b-client,b-setting',
            id='betas of ours',
        ),
    ],
)
async def test_client_default_beta_is_sent_with_ours(
    allow_model_requests: None,
    client: Callable[[], AsyncAnthropicClient],
    header_name: str,
    settings: AnthropicModelSettings,
    header_line: str,
):
    """A per-request `anthropic-beta` header replaces the client's, so ours carry the client's betas too."""
    sent = SentBetas()

    await Agent(sent.model(client(), {header_name: 'b-client'})).run('hello', model_settings=settings)

    assert sent.header_lines == [[header_line]]


@pytest.mark.parametrize('header_name', SPELLINGS)
async def test_client_default_beta_is_counted_with_ours(allow_model_requests: None, header_name: str):
    """`count_tokens` sends its own beta besides ours, which also replaces the client's."""
    sent = SentBetas()
    model = sent.model(AsyncAnthropic(api_key='x'), {header_name: 'b-client'})

    await model.count_tokens(
        [ModelRequest.user_text_prompt('hello')],
        AnthropicModelSettings(anthropic_betas=['b-setting']),
        ModelRequestParameters(),
    )

    assert sent.header_lines == [['b-client,b-setting,token-counting-2024-11-01']]


@pytest.mark.parametrize(
    'settings,body_field',
    [
        pytest.param({'max_tokens': 16}, None, id='no betas of ours'),
        pytest.param({'max_tokens': 16, 'anthropic_betas': ['b-setting']}, ['b-setting'], id='betas of ours'),
    ],
)
async def test_bedrock_client_default_beta_stays_out_of_the_body(
    allow_model_requests: None, settings: AnthropicModelSettings, body_field: list[str] | None
):
    """The legacy Bedrock API reads betas from the body alone, where the SDK moves only per-request ones.

    Verified live: a beta Bedrock rejects in the body is ignored in a client's `default_headers`. Sending the client's
    betas in the body would apply betas that never applied on this client.
    """
    sent = SentBetas()
    client = AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1')

    await Agent(sent.model(client, {'anthropic-beta': 'b-client'})).run('hello', model_settings=settings)

    assert sent.body_fields == [body_field]
