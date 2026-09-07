from __future__ import annotations

import json
import os
from pathlib import Path

import httpx2
import pytest
from vcr.cassette import Cassette
from vcr.record_mode import RecordMode

from pydantic_ai import Agent

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    import logfire
    from logfire.testing import CaptureLogfire

    from pydantic_ai.models.openai_codex import OpenAICodexModel
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexProvider

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='openai/logfire not installed'),
]


@pytest.fixture
def codex_credentials(vcr: Cassette) -> OpenAICodexCredentials:
    if vcr.record_mode == RecordMode.NONE:
        return OpenAICodexCredentials(
            access_token='codex-playback-access',
            refresh_token='codex-playback-refresh',
            account_id='codex-playback-account',
        )
    path = Path(os.getenv('CODEX_HOME') or Path.home() / '.codex') / 'auth.json'
    return OpenAICodexCredentials.from_codex_cli_auth(json.loads(path.read_text()))


@pytest.mark.parametrize('stream', [False, True])
async def test_codex_tool_roundtrip(
    allow_model_requests: None,
    request_capture: RequestCapture,
    codex_credentials: OpenAICodexCredentials,
    capfire: CaptureLogfire,
    stream: bool,
):
    """Astra calls a local tool and consumes its result over the authenticated Codex SSE API."""

    async def reject_refresh(request: httpx2.Request) -> None:
        # Recording must not consume the CLI's single-use refresh token.
        assert request.url.host != 'auth.openai.com', (
            'Run `codex login` before recording to obtain a fresh access token.'
        )

    request_capture.client.event_hooks['request'].insert(0, reject_refresh)
    logfire.instrument_httpx(request_capture.client)
    logfire.instrument_pydantic_ai()
    provider = OpenAICodexProvider(credentials=codex_credentials, http_client=request_capture.client)
    agent = Agent(
        OpenAICodexModel('gpt-6-astra', provider=provider),
        instructions='Call the moo tool exactly once, then reply with its result verbatim and nothing else.',
    )
    calls: list[str] = []

    @agent.tool_plain
    def moo() -> str:
        """Return a cheerful cow sound."""
        calls.append('moo')
        return 'Moo!'

    if stream:
        async with agent.run_stream('Use your moo tool.') as streamed:
            output = await streamed.get_output()
    else:
        output = (await agent.run('Use your moo tool.')).output

    assert {'output': output, 'tool_calls': calls} == snapshot({'output': 'Moo!', 'tool_calls': ['moo']})
    assert [
        {'stream': body['stream'], 'store': body['store'], 'tools': body['tools']}
        for body in request_capture.bodies('/responses')
    ] == snapshot(
        [
            {
                'stream': True,
                'store': False,
                'tools': [
                    {
                        'name': 'moo',
                        'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                        'type': 'function',
                        'description': 'Return a cheerful cow sound.',
                        'strict': False,
                    }
                ],
            },
            {
                'stream': True,
                'store': False,
                'tools': [
                    {
                        'name': 'moo',
                        'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                        'type': 'function',
                        'description': 'Return a cheerful cow sound.',
                        'strict': False,
                    }
                ],
            },
        ]
    )

    logfire.force_flush()
    spans = capfire.exporter.exported_spans_as_dict()
    exported = json.dumps(spans, default=str)
    secrets: dict[str, str] = {
        'access_token': codex_credentials.access_token,
        'refresh_token': codex_credentials.refresh_token,
        'account_id': codex_credentials.account_id,
    }
    # Report field names rather than allowing pytest's assertion renderer to print a secret.
    assert [name for name, value in secrets.items() if value in exported] == snapshot([])
    assert [
        {
            'name': span['name'],
            'operation': span['attributes'].get('gen_ai.operation.name'),
            'model': span['attributes'].get('gen_ai.request.model'),
        }
        for span in spans
    ] == snapshot(
        [
            {'name': 'POST', 'operation': None, 'model': None},
            {'name': 'chat gpt-6-astra', 'operation': 'chat', 'model': 'gpt-6-astra'},
            {'name': 'execute_tool moo', 'operation': 'execute_tool', 'model': None},
            {'name': 'POST', 'operation': None, 'model': None},
            {'name': 'chat gpt-6-astra', 'operation': 'chat', 'model': 'gpt-6-astra'},
            {'name': 'invoke_agent agent', 'operation': 'invoke_agent', 'model': None},
        ]
    )
