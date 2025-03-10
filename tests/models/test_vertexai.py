# pyright: reportDeprecated=false
from __future__ import annotations as _annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from pydantic_ai import UserError
from pydantic_ai.agent import Agent
from pydantic_ai.messages import DocumentUrl
from pydantic_ai.models.gemini import GeminiModel

from ..conftest import IsNow, try_import

with try_import() as imports_successful:
    from google.oauth2.service_account import Credentials

    from pydantic_ai.models.vertexai import BearerTokenAuth, VertexAIModel


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-auth not installed'),
    pytest.mark.anyio,
    # This ignore is added because we should just remove the `VertexAIModel` class.
    pytest.mark.filterwarnings('ignore::DeprecationWarning'),
]


async def test_init_service_account(tmp_path: Path, allow_model_requests: None):
    service_account_path = tmp_path / 'service_account.json'
    save_service_account(service_account_path, 'my-project-id')

    model = VertexAIModel('gemini-1.5-flash', service_account_file=service_account_path)
    assert model._url is None
    assert model._auth is None

    await model.ainit()

    assert model.base_url == snapshot(
        'https://us-central1-aiplatform.googleapis.com/v1/projects/my-project-id/locations/us-central1/'
        'publishers/google/models/gemini-1.5-flash:'
    )
    assert model.auth is not None
    assert model.model_name == snapshot('gemini-1.5-flash')
    assert model.system == snapshot('google-vertex')


class NoOpCredentials:
    pass


async def test_init_env(mocker: MockerFixture, allow_model_requests: None):
    patch = mocker.patch(
        'pydantic_ai.models.vertexai.google.auth.default',
        return_value=(NoOpCredentials(), 'my-project-id'),
    )
    model = VertexAIModel('gemini-1.5-flash')
    assert model._url is None
    assert model._auth is None

    assert patch.call_count == 0

    await model.ainit()

    assert patch.call_count == 1

    assert model.base_url == snapshot(
        'https://us-central1-aiplatform.googleapis.com/v1/projects/my-project-id/locations/us-central1/'
        'publishers/google/models/gemini-1.5-flash:'
    )
    assert model.auth is not None
    assert model.model_name == snapshot('gemini-1.5-flash')
    assert model.system == snapshot('google-vertex')

    await model.ainit()
    assert model.base_url is not None
    assert model.auth is not None
    assert patch.call_count == 1


async def test_init_right_project_id(tmp_path: Path, allow_model_requests: None):
    service_account_path = tmp_path / 'service_account.json'
    save_service_account(service_account_path, 'my-project-id')

    model = VertexAIModel('gemini-1.5-flash', service_account_file=service_account_path, project_id='my-project-id')
    assert model._url is None
    assert model._auth is None

    await model.ainit()

    assert model.base_url == snapshot(
        'https://us-central1-aiplatform.googleapis.com/v1/projects/my-project-id/locations/us-central1/'
        'publishers/google/models/gemini-1.5-flash:'
    )
    assert model.auth is not None


async def test_init_env_no_project_id(mocker: MockerFixture, allow_model_requests: None):
    mocker.patch(
        'pydantic_ai.models.vertexai.google.auth.default',
        return_value=(NoOpCredentials(), None),
    )
    model = VertexAIModel('gemini-1.5-flash')

    with pytest.raises(UserError) as exc_info:
        await model.ainit()
    assert str(exc_info.value) == snapshot('No project_id provided and none found in `google.auth.default()`')


# pyright: reportPrivateUsage=false
async def test_bearer_token():
    refresh_count = 0

    class MockRefreshCredentials(Credentials):
        def refresh(self, request: Any):
            nonlocal refresh_count
            refresh_count += 1
            self.token = f'custom-token-{refresh_count}'

    # noinspection PyTypeChecker
    creds = MockRefreshCredentials(
        signer=None,
        service_account_email='test@example.com',
        token_uri='https://example.com/token',
        project_id='my-project-id',
    )
    t = BearerTokenAuth(creds)

    assert creds.token is None
    assert t.token_created is None
    assert t._token_expired()
    headers = await t.headers()
    assert headers == snapshot({'Authorization': 'Bearer custom-token-1'})
    assert refresh_count == 1
    assert t.token_created == IsNow()

    assert not t._token_expired()
    assert creds.token == 'custom-token-1'
    headers = await t.headers()
    assert headers == snapshot({'Authorization': 'Bearer custom-token-1'})
    assert refresh_count == 1

    t.token_created = datetime.now() - timedelta(seconds=4000)
    assert t._token_expired()
    headers = await t.headers()
    assert headers == snapshot({'Authorization': 'Bearer custom-token-2'})
    assert t.token_created == IsNow()


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


@pytest.mark.vcr()
async def test_document_url_input(allow_model_requests: None, gemini_api_key: str) -> None:
    m = GeminiModel('gemini-2.0-flash-thinking-exp-01-21', provider='google-vertex')
    agent = Agent(m)

    document_url = DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.data == snapshot("""\
The main content of this document is the introduction and evaluation of **Gemini 1.5 Pro**, a new large language model from Google DeepMind.

**Key takeaways from the document are:**

* **Gemini 1.5 Pro is a highly compute-efficient multimodal mixture-of-experts model.** It can process and reason over vast amounts of information, including millions of tokens of text, hours of video, and audio.
* **Unprecedented Context Window:** Gemini 1.5 Pro boasts an exceptionally large context window, capable of processing up to **10 million tokens**, significantly surpassing previous models like Claude 2.1 (200k) and GPT-4 Turbo (128k).
* **Near-Perfect Recall in Long Context:** The model demonstrates near-perfect recall (>99%) even with context windows up to 10 million tokens across different modalities (text, video, and audio).
* **State-of-the-Art Performance:** Gemini 1.5 Pro achieves or surpasses state-of-the-art performance in long-context tasks like long-document QA, long-video QA, long-context ASR, and general benchmarks, often matching or exceeding Gemini 1.0 Ultra while using less compute.
* **Emergent Capabilities:**  The model exhibits surprising new capabilities like in-context learning of new languages.  As an example, it can learn to translate English to Kalamang (a low-resource language) at a near-human level, solely from a grammar manual and limited linguistic resources provided in context.
* **Rigorous Evaluation:** The document details extensive evaluations of Gemini 1.5 Pro's long-context capabilities, using both synthetic ("needle-in-a-haystack") and real-world tasks. It also covers core capabilities across various modalities, demonstrating overall performance improvements.
* **Emphasis on Responsible Deployment:** The document addresses responsible deployment strategies including impact assessments, safety evaluations, and mitigation efforts for potential harms associated with such powerful models.

In essence, the document highlights Gemini 1.5 Pro as a significant advancement in language models, particularly in its ability to handle and reason over extremely long contexts across multiple modalities, opening up new possibilities for applications requiring understanding of vast and complex information.\
""")
