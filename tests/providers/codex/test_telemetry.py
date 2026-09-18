from __future__ import annotations

import json
from functools import partial

import httpx2
import pytest

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as imports_successful:
    import logfire
    from logfire.testing import CaptureLogfire
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from pydantic_ai.providers.openai_codex import (
        OpenAICodexOAuthFlow,
        OpenAICodexProvider,
        _post_token_request,  # pyright: ignore[reportPrivateUsage]
    )

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='openai/logfire not installed')]


async def test_oauth_http_capture_redacts_credentials(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    """Synthetic credentials let us inspect OAuth telemetry without spending a live authorization grant."""
    logfire.configure(
        send_to_logfire=False,
        console=False,
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
        scrubbing=logfire.ScrubbingOptions(
            extra_patterns=[
                'access_token',
                'refresh_token',
                'id_token',
                'code_verifier',
                '^code$',
                'chatgpt-account-id',
                '^account_id$',
                'safety_identifier',
            ]
        ),
    )
    flow = OpenAICodexOAuthFlow()
    secrets: dict[str, str] = {
        'access_token': 'codex-test-value-1',
        'refresh_token': 'codex-test-value-2',
        'id_token': 'codex-test-value-3',
        'account_id': 'codex-test-value-4',
        'code': 'codex-test-value-5',
        'code_verifier': flow.code_verifier,
        'safety_identifier': 'codex-test-value-6',
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == 'auth.openai.com':
            payload = {name: secrets[name] for name in ('access_token', 'refresh_token', 'id_token', 'account_id')}
            return httpx2.Response(
                200,
                headers={'content-type': 'application/json'},
                stream=httpx2.ByteStream(json.dumps(payload).encode()),
            )
        return httpx2.Response(
            200,
            headers={'content-type': 'application/json'},
            stream=httpx2.ByteStream(json.dumps({'safety_identifier': secrets['safety_identifier']}).encode()),
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        logfire.instrument_httpx(client, capture_all=True)
        # Keep the real token exchange and redirect its ephemeral client to the controlled transport.
        monkeypatch.setattr(
            'pydantic_ai.providers.openai_codex._post_token_request',
            partial(_post_token_request, http_client=client),
        )
        with logfire.span('Codex login'):
            credentials = await flow.exchange_code(secrets['code'])
        OpenAICodexProvider(credentials=credentials, http_client=client)
        await client.get('https://chatgpt.com/backend-api/codex/responses')

    logfire.force_flush()
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    exported = json.dumps(spans, default=str)
    assert [name for name, value in secrets.items() if value in exported] == snapshot([])
    assert [
        {
            'name': span['name'],
            'captured': {
                key: value
                for key, value in span['attributes'].items()
                if key.startswith(('http.request.body', 'http.response.body'))
                or key in {'http.request.header.authorization', 'http.request.header.chatgpt-account-id'}
            },
        }
        for span in spans
    ] == snapshot(
        [
            {
                'name': 'POST',
                'captured': {
                    'http.request.body.form': {
                        'grant_type': "[Scrubbed due to 'auth']",
                        'code': "[Scrubbed due to 'code']",
                        'code_verifier': "[Scrubbed due to 'code_verifier']",
                        'redirect_uri': "[Scrubbed due to 'auth']",
                        'client_id': 'app_EMoamEEZ73f0CkXaXp7hrann',
                    }
                },
            },
            {
                'name': 'Reading response body',
                'captured': {
                    'http.response.body.text': {
                        'access_token': "[Scrubbed due to 'access_token']",
                        'refresh_token': "[Scrubbed due to 'refresh_token']",
                        'id_token': "[Scrubbed due to 'id_token']",
                        'account_id': "[Scrubbed due to 'account_id']",
                    }
                },
            },
            {'name': 'Codex login', 'captured': {}},
            {
                'name': 'GET',
                'captured': {
                    'http.request.header.authorization': ("[Scrubbed due to 'auth']",),
                    'http.request.header.chatgpt-account-id': ("[Scrubbed due to 'chatgpt-account-id']",),
                },
            },
            {
                'name': 'Reading response body',
                'captured': {'http.response.body.text': {'safety_identifier': "[Scrubbed due to 'safety_identifier']"}},
            },
        ]
    )
