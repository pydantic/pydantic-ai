from __future__ import annotations as _annotations

import json
from pathlib import Path

import httpx
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import AudioUrl, DocumentUrl, ImageUrl, VideoUrl
from pydantic_ai.models.gemini import GeminiModel

from ..conftest import ClientWithHandler, TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-auth not installed'),
    pytest.mark.anyio,
]


async def mock_refresh_token():
    return 'my-token'


@pytest.mark.parametrize(
    'url',
    [
        pytest.param(AudioUrl(url='https://cdn.openai.com/API/docs/audio/alloy.wav'), id='AudioUrl'),
        pytest.param(
            DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
            id='DocumentUrl',
        ),
        pytest.param(
            ImageUrl(url='https://upload.wikimedia.org/wikipedia/commons/6/6a/Www.wikipedia_screenshot_%282021%29.png'),
            id='ImageUrl',
        ),
        pytest.param(VideoUrl(url='https://data.grepit.app/assets/tiny_video.mp4'), id='VideoUrl'),
    ],
)
async def test_url_inputs_are_sent_via_file_uri_field(
    url: AudioUrl | DocumentUrl | ImageUrl | VideoUrl,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    client_with_handler: ClientWithHandler,
    env: TestEnv,
    allow_model_requests: None,
) -> None:
    service_account_path = tmp_path / 'service_account.json'
    save_service_account(service_account_path, 'my-project-id')

    def handler(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)['contents'][0]['parts'][0]['text']
        assert text == 'What is the main content of this URL?'
        file_data = json.loads(request.content)['contents'][0]['parts'][1]['fileData']
        assert file_data == {
            'fileUri': url.url,
            'mimeType': url.media_type,
        }

        return httpx.Response(
            200,
            content=json.dumps({'candidates': [{'content': {'role': 'user', 'parts': [{'text': '...'}]}}]}),
            headers={'Content-Type': 'application/json'},
        )

    gemini_client = client_with_handler(handler)
    provider = GoogleVertexProvider(http_client=gemini_client, service_account_file=service_account_path)
    monkeypatch.setattr(provider.client.auth, '_refresh_token', mock_refresh_token)

    m = GeminiModel('gemini-1.5-flash', provider=provider)
    agent = Agent(m)
    result = await agent.run(['What is the main content of this URL?', url])

    assert result.output == '...'


def save_service_account(service_account_path: Path, project_id: str) -> None:
    service_account = {
        'type': 'service_account',
        'project_id': project_id,
        'private_key_id': 'abc',
        # this is just a random private key I created with `openssl genpke ...`, it doesn't do anything
        'private_key': (
            '-----BEGIN PRIVATE KEY-----\n'
            'MIICdgIBADANBgkqhkiG9w0BAQEFAASCAmAwggJcAgEAAoGBAMFrZYX4gZ20qv88\n'
            'jD0QCswXgcxgP7Ta06G47QEFprDVcv4WMUBDJVAKofzVcYyhsasWsOSxcpA8LIi9\n'
            '/VS2Otf8CmIK6nPBCD17Qgt8/IQYXOS4U2EBh0yjo0HQ4vFpkqium4lLWxrAZohA\n'
            '8r82clV08iLRUW3J+xvN23iPHyVDAgMBAAECgYBScRJe3iNxMvbHv+kOhe30O/jJ\n'
            'QiUlUzhtcEMk8mGwceqHvrHTcEtRKJcPC3NQvALcp9lSQQhRzjQ1PLXkC6BcfKFd\n'
            '03q5tVPmJiqsHbSyUyHWzdlHP42xWpl/RmX/DfRKGhPOvufZpSTzkmKWtN+7osHu\n'
            '7eiMpg2EDswCvOgf0QJBAPXLYwHbZLaM2KEMDgJSse5ZTE/0VMf+5vSTGUmHkr9c\n'
            'Wx2G1i258kc/JgsXInPbq4BnK9hd0Xj2T5cmEmQtm4UCQQDJc02DFnPnjPnnDUwg\n'
            'BPhrCyW+rnBGUVjehveu4XgbGx7l3wsbORTaKdCX3HIKUupgfFwFcDlMUzUy6fPO\n'
            'IuQnAkA8FhVE/fIX4kSO0hiWnsqafr/2B7+2CG1DOraC0B6ioxwvEqhHE17T5e8R\n'
            '5PzqH7hEMnR4dy7fCC+avpbeYHvVAkA5W58iR+5Qa49r/hlCtKeWsuHYXQqSuu62\n'
            'zW8QWBo+fYZapRsgcSxCwc0msBm4XstlFYON+NoXpUlsabiFZOHZAkEA8Ffq3xoU\n'
            'y0eYGy3MEzxx96F+tkl59lfkwHKWchWZJ95vAKWJaHx9WFxSWiJofbRna8Iim6pY\n'
            'BootYWyTCfjjwA==\n'
            '-----END PRIVATE KEY-----\n'
        ),
        'client_email': 'testing-pydantic-ai@pydantic-ai.iam.gserviceaccount.com',
        'client_id': '123',
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
        'client_x509_cert_url': 'https://www.googleapis.com/...',
        'universe_domain': 'googleapis.com',
    }

    service_account_path.write_text(json.dumps(service_account, indent=2))
