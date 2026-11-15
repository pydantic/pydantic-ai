"""Recorded round-trips of stateless signed or encrypted data between providers that share a canonical system.

Each test continues one provider's history on another and pins what the second request carried, as built by the
live code (`request_capture` or a botocore hook), next to the recorded answer that shows the API accepted it.
The offline matrix of which data each model sends back is in `tests/test_canonical_system.py`.
"""

from __future__ import annotations as _annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest
from cassetter import Cassette, RecordMode
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelResponse, ThinkingPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from .conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from google.genai import Client
    from google.genai.types import HttpOptions

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel
    from pydantic_ai.models.openai_codex import OpenAICodexModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexProvider

pytestmark = [
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='model provider SDKs not installed'),
]


def input_shape(body: dict[str, Any]) -> list[tuple[str | None, bool]]:
    """Each Responses API input item's type (or role) and whether it carries encrypted content."""
    return [(item.get('type') or item.get('role'), bool(item.get('encrypted_content'))) for item in body['input']]


@pytest.fixture
def codex_credentials(vcr: Cassette) -> OpenAICodexCredentials:
    if vcr.record_mode != RecordMode.NONE:  # pragma: no cover
        path = Path(os.getenv('CODEX_HOME') or Path.home() / '.codex') / 'auth.json'
        return OpenAICodexCredentials.from_codex_cli_auth(json.loads(path.read_text()))
    return OpenAICodexCredentials(
        access_token='codex-playback-access',
        refresh_token='codex-playback-refresh',
        account_id='codex-playback-account',
    )


async def reject_refresh(request: httpx2.Request) -> None:
    # Recording must not consume the Codex CLI's single-use refresh token.
    assert request.url.host != 'auth.openai.com', 'Run `codex login` before recording to obtain a fresh access token.'


@pytest.fixture
def codex_model(request_capture: RequestCapture, codex_credentials: OpenAICodexCredentials) -> OpenAICodexModel:
    request_capture.client.event_hooks['request'].insert(0, reject_refresh)
    provider = OpenAICodexProvider(credentials=codex_credentials, http_client=request_capture.client)
    return OpenAICodexModel('gpt-6-astra', provider=provider)


@pytest.fixture
def openai_model(request_capture: RequestCapture, openai_api_key: str) -> OpenAIResponsesModel:
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    return OpenAIResponsesModel('gpt-6-astra', provider=provider)


REASONING_SETTINGS: ModelSettings = {
    'thinking': 'medium',
    'openai_reasoning_summary': 'auto',  # pyright: ignore[reportAssignmentType]
    'openai_store': False,
}


async def test_openai_and_codex_share_encrypted_reasoning(
    allow_model_requests: None,
    request_capture: RequestCapture,
    codex_model: OpenAICodexModel,
    openai_model: OpenAIResponsesModel,
):
    """Encrypted reasoning goes from the OpenAI API to the Codex backend and back, which both accept (live-verified)."""
    agent = Agent(openai_model, instructions='Be brief.', model_settings=REASONING_SETTINGS)
    first = await agent.run('How many prime numbers are there between 100 and 200? Reply with just the number.')
    second = await agent.run('And between 200 and 300?', message_history=first.all_messages(), model=codex_model)
    third = await agent.run('And between 300 and 400?', message_history=second.all_messages())
    assert [first.output, second.output, third.output] == snapshot(['21', '16', '16'])

    codex_response = second.new_messages()[-1]
    assert isinstance(codex_response, ModelResponse)
    assert (codex_response.provider_name, codex_response.canonical_system) == ('openai-codex', 'openai')
    assert any(isinstance(part, ThinkingPart) and part.signature for part in codex_response.parts)

    # The OpenAI response's reasoning reaches Codex with its encrypted content, and its text without its stateful ID.
    assert input_shape(request_capture.body('/codex/responses')) == snapshot(
        [('user', False), ('reasoning', True), ('assistant', False), ('user', False)]
    )
    # Both reasoning items go back to the OpenAI API.
    assert input_shape(request_capture.bodies('/v1/responses')[-1]) == snapshot(
        [
            ('user', False),
            ('reasoning', True),
            ('message', False),
            ('user', False),
            ('reasoning', True),
            ('assistant', False),
            ('user', False),
        ]
    )


@pytest.mark.moves_cache_prefix(
    reason='The OpenAI request starts from the compaction item, dropping the earlier turns.'
)
async def test_openai_continues_from_codex_compaction(
    allow_model_requests: None,
    request_capture: RequestCapture,
    codex_model: OpenAICodexModel,
    openai_model: OpenAIResponsesModel,
):
    """A Codex compaction item is a compaction boundary for the OpenAI API too, which uses it (live-verified).

    Without a shared canonical system, the OpenAI request would drop the item and resend the whole history.
    """
    agent = Agent(codex_model, instructions='Be brief.', capabilities=[OpenAICompaction(token_threshold=1000)])
    facts = ' '.join(f'Fact {i}: the number {i} is {"even" if i % 2 == 0 else "odd"}.' for i in range(150))
    first = await agent.run(f'{facts}\nThe secret word is "pelican". Reply with just OK.')
    codex_response = first.new_messages()[-1]
    assert isinstance(codex_response, ModelResponse)
    assert any(isinstance(part, CompactionPart) for part in codex_response.parts)

    result = await Agent(openai_model, instructions='Be brief.').run(
        'What was the secret word? Reply with just the word.', message_history=first.all_messages()
    )
    assert result.output == snapshot('pelican')
    assert input_shape(request_capture.body('/v1/responses')) == snapshot([('compaction', True), ('user', False)])


@dataclass(init=False)
class FailsOnFirstRequest(WrapperModel):
    """Answers the first request with a 429, like the quota error that sent #10905's run to its fallback."""

    failed: bool = False

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if not self.failed:
            self.failed = True
            raise ModelHTTPError(status_code=429, model_name=self.model_name, body='Resource exhausted')
        return await super().request(messages, model_settings, model_request_parameters)


@pytest.fixture
def vertex_client(
    request: pytest.FixtureRequest, vertex_provider_auth: None, request_capture: RequestCapture
) -> Client:  # pragma: lax no cover
    """A Google Cloud client routed through `request_capture`, set up like the `vertex_provider` fixture."""
    record_mode = cast(Any, request.config).getoption('record_mode')
    if not os.getenv('CI', False) and record_mode not in {'all', 'new_episodes', 'rewrite'}:
        pytest.skip('Requires properly configured local google vertex config to pass')
    return Client(
        vertexai=True,
        project=os.getenv('GOOGLE_PROJECT', 'pydantic-ai'),
        location=os.getenv('GOOGLE_LOCATION', 'global'),
        http_options=HttpOptions(httpx_async_client=request_capture.http_client(timeout=60), timeout=60_000),
    )


async def test_gemini_api_signature_is_not_sent_to_vertex_after_fallback(
    allow_model_requests: None, request_capture: RequestCapture, vertex_client: Client, gemini_api_key: str
):
    """#10905: the Gemini API's thought signature isn't replayed to Vertex AI, which rejects it with a 400.

    Both models are on a `GoogleProvider`, so both are named `google`. Vertex answers the first request with a 429,
    the Gemini API calls the tool with a signed function call, and the next request goes to Vertex again, carrying
    the documented `skip_thought_signature_validator` dummy in place of the foreign signature.
    """
    vertex = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(client=vertex_client))
    gemini_api = GoogleModel(
        'gemini-3-flash-preview',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=60)),
    )
    # Only the quota error falls back, so a rejected signature would fail the run, as in #10905.
    model = FallbackModel(
        FailsOnFirstRequest(vertex),
        gemini_api,
        fallback_on=lambda exc: isinstance(exc, ModelHTTPError) and exc.status_code == 429,
    )
    agent = Agent(model, model_settings={'thinking': 'low'})

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather in a city."""
        return 'sunny'

    result = await agent.run('What is the weather in Paris? Use the tool.')
    assert result.output == snapshot('The weather in Paris is sunny.')
    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert [(response.provider_name, response.canonical_system) for response in responses] == snapshot(
        [('google', 'google'), ('google', 'google-cloud')]
    )
    vertex_body = request_capture.bodies(':generateContent')[-1]
    function_call = next(
        part
        for content in cast(list[dict[str, Any]], vertex_body['contents'])
        for part in content['parts']
        if 'functionCall' in part
    )
    assert function_call['thoughtSignature'] == snapshot('c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=')


@contextmanager
def capture_bedrock_bodies(model: BedrockConverseModel) -> Generator[list[dict[str, Any]]]:
    """The Converse request bodies botocore sends, as the live code built them."""
    captured: list[dict[str, Any]] = []

    def capture(request: Any, **_: Any) -> None:
        captured.append(json.loads(request.body))

    events = ['before-send.bedrock-runtime.Converse', 'before-send.bedrock-runtime.ConverseStream']
    for event in events:
        model.client.meta.events.register_last(event, capture)
    try:
        yield captured
    finally:
        for event in events:
            model.client.meta.events.unregister(event, capture)


# Claude only redacts thinking when the prompt contains this string, which Anthropic documents for testing.
REDACTED_THINKING_TRIGGER = (
    'ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB'
)


async def test_anthropic_and_bedrock_share_claude_thinking(
    allow_model_requests: None,
    request_capture: RequestCapture,
    anthropic_api_key: str,
    bedrock_provider: BedrockProvider,
):
    """Claude's thinking signatures and redacted thinking go from the Anthropic API to Bedrock and back.

    Both verify them (live-verified on every pair of the Anthropic API, Bedrock InvokeModel, Converse and Claude
    Platform on AWS). Bedrock records redacted thinking as `redacted_content` and Anthropic as `redacted_thinking`;
    the data is the same.
    """
    anthropic = AnthropicModel(
        'claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key, http_client=request_capture.client)
    )
    bedrock = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(anthropic, model_settings={'thinking': 'low'})

    first = await agent.run(f'{REDACTED_THINKING_TRIGGER}\nWhat is 7 * 8? Answer with just the number.')
    with capture_bedrock_bodies(bedrock) as bedrock_bodies:
        second = await agent.run(
            f'{REDACTED_THINKING_TRIGGER}\nAnd 9 * 6?', message_history=first.all_messages(), model=bedrock
        )
    third = await agent.run('And 2 * 2?', message_history=second.all_messages())
    assert [first.output, second.output, third.output] == snapshot(['56', '54', '4'])

    def thinking_ids(messages: list[ModelMessage]) -> set[str | None]:
        return {
            part.id
            for message in messages
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ThinkingPart) and part.signature
        }

    assert thinking_ids(first.new_messages()) == snapshot({'redacted_thinking'})
    assert thinking_ids(second.new_messages()) == snapshot({'redacted_content'})
    bedrock_response = second.new_messages()[-1]
    assert isinstance(bedrock_response, ModelResponse)
    assert (bedrock_response.provider_name, bedrock_response.canonical_system) == ('bedrock', 'anthropic')

    # Anthropic's redacted thinking reaches Bedrock as `redactedContent`, rather than as text.
    bedrock_assistant = bedrock_bodies[-1]['messages'][1]['content']
    assert {next(iter(block)) for block in bedrock_assistant} == snapshot({'reasoningContent', 'text'})
    assert {next(iter(block['reasoningContent'])) for block in bedrock_assistant if 'reasoningContent' in block} == (
        snapshot({'redactedContent'})
    )
    # And Bedrock's goes back to Anthropic as `redacted_thinking` blocks.
    anthropic_body = request_capture.bodies('/v1/messages')[-1]
    assistant_turns = [
        message for message in cast(list[dict[str, Any]], anthropic_body['messages']) if message['role'] == 'assistant'
    ]
    assert [sorted({block['type'] for block in turn['content']}) for turn in assistant_turns] == snapshot(
        [['redacted_thinking', 'text'], ['redacted_thinking', 'text']]
    )
