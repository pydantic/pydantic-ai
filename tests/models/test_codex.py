from __future__ import annotations as _annotations

from datetime import datetime, timezone

import pytest
from inline_snapshot import snapshot
from pydantic import SecretStr
from vcr.cassette import Cassette
from vcr.record_mode import RecordMode

from pydantic_ai import Agent
from pydantic_ai.auth.codex import CodexAuth, CodexCredentials
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers import Provider
from pydantic_ai.providers.codex import CodexProvider

from ..cassette_utils import single_request_body
from ..conftest import try_import
from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types import responses as resp
    from openai.types.responses.response_output_message import ResponseOutputMessage

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed'),
]


class StaticCodexCredentialSource:
    async def get_credentials(
        self, *, force_refresh: bool = False, rejected_revision: str | None = None
    ) -> CodexCredentials:
        return CodexCredentials(
            access_token=SecretStr('cassette-access-token'),
            refresh_token=SecretStr('cassette-refresh-token'),
            id_token=SecretStr('cassette-id-token'),
            expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            account_id=SecretStr('cassette-account-id'),
            revision='cassette-revision',
        )


class MockCodexProvider(Provider[AsyncOpenAI]):
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return 'codex'

    @property
    def base_url(self) -> str:
        return 'https://chatgpt.com/backend-api/codex'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    model_profile = staticmethod(CodexProvider.model_profile)


def codex_model(vcr: Cassette) -> OpenAIResponsesModel:
    credential_source = StaticCodexCredentialSource() if vcr.record_mode == RecordMode.NONE else CodexAuth()
    return OpenAIResponsesModel('gpt-5.5', provider=CodexProvider(credential_source=credential_source))


async def test_codex_profile_streams_ordinary_requests_and_preserves_provider_identity(
    allow_model_requests: None,
) -> None:
    base_response = resp.Response(
        id='resp_001',
        model='gpt-5.5',
        object='response',
        created_at=1704067200,
        output=[],
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )
    stream: list[resp.ResponseStreamEvent] = [
        resp.ResponseCreatedEvent(response=base_response, type='response.created', sequence_number=0),
        resp.ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id='msg_001',
                content=[],
                role='assistant',
                status='in_progress',
                type='message',
            ),
            output_index=0,
            type='response.output_item.added',
            sequence_number=1,
        ),
        resp.ResponseTextDeltaEvent(
            item_id='msg_001',
            output_index=0,
            content_index=0,
            delta='ok',
            logprobs=[],
            type='response.output_text.delta',
            sequence_number=2,
        ),
        resp.ResponseCompletedEvent(
            response=base_response.model_copy(update={'status': 'completed'}),
            type='response.completed',
            sequence_number=3,
        ),
    ]
    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    model = OpenAIResponsesModel('gpt-5.5', provider=MockCodexProvider(mock_client))

    response = await model.request(
        [ModelRequest.user_text_prompt('hello')],
        OpenAIResponsesModelSettings(openai_store=True),
        ModelRequestParameters(),
    )

    assert response.provider_name == 'codex'
    assert response.text == 'ok'
    assert model.system == 'codex'
    assert model.model_id == 'codex:gpt-5.5'
    assert get_mock_responses_kwargs(mock_client)[0]['store'] is False


@pytest.mark.vcr
async def test_codex_agent_run(allow_model_requests: None, vcr: Cassette) -> None:
    agent = Agent(codex_model(vcr))

    result = await agent.run('Reply with exactly "codex-live-ok" and nothing else.')

    assert result.output == snapshot('codex-live-ok')
    assert single_request_body(vcr)['stream'] is True
    assert single_request_body(vcr)['store'] is False


@pytest.mark.vcr
async def test_codex_agent_run_stream(allow_model_requests: None, vcr: Cassette) -> None:
    agent = Agent(codex_model(vcr))

    async with agent.run_stream('Reply with exactly "codex-stream-ok" and nothing else.') as result:
        chunks = [chunk async for chunk in result.stream_text(delta=True, debounce_by=None)]

    assert ''.join(chunks) == snapshot('codex-stream-ok')
    assert len(chunks) > 1
    assert single_request_body(vcr)['stream'] is True
    assert single_request_body(vcr)['store'] is False
