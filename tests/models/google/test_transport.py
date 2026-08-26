"""Vertex-vs-Gemini-API routing, driven by the client's transport rather than the provider name.

Either provider accepts a pre-built `client=` and stores it as-is, so `name` and transport can
disagree in both directions: a Google Cloud client in `GoogleProvider` keeps `name` `'google'`
(#6792), and a Gemini Developer API client in `GoogleCloudProvider` keeps `name` `'google-cloud'`.
`GoogleModel` reads the transport off the client, so both route by where the bytes actually go.

None of these are VCR tests. The routed fields and headers are decided before the request is built,
the cassette serializer does not persist provider request headers, and the default matchers do not
inspect the body — so a recording would replay green through a regression in either direction.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from pydantic_ai import UploadedFile
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, UploadedFileProviderName, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from ...conftest import try_import

with try_import() as imports_successful:
    from google.genai import Client

    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
]


async def test_google_cloud_client_in_google_provider_uses_cloud_service_tier_headers(
    allow_model_requests: None, vertex_client_google_provider: GoogleProvider
) -> None:
    """A Google Cloud client wrapped in `GoogleProvider` gets Cloud service-tier handling (#6792)."""
    m = GoogleModel('gemini-2.5-flash', provider=vertex_client_google_provider)
    assert m.system == 'google'

    _, config = await m._build_content_and_config(  # pyright: ignore[reportPrivateUsage]
        messages=[ModelRequest(parts=[UserPromptPart(content='Hello')])],
        model_settings=GoogleModelSettings(google_cloud_service_tier='pt_only'),
        model_request_parameters=ModelRequestParameters(),
    )

    config_dict = cast(dict[str, Any], config)
    assert config_dict['http_options']['headers']['X-Vertex-AI-LLM-Request-Type'] == 'dedicated'


async def test_google_cloud_service_tier_is_dropped_on_a_gemini_api_transport(
    allow_model_requests: None, gla_client_google_cloud_provider: GoogleCloudProvider
) -> None:
    """`google_cloud_service_tier` has no Gemini API equivalent, so it is dropped on that transport.

    Newly reachable: a Gemini API client in `GoogleCloudProvider` keeps `system == 'google-cloud'`,
    which used to send Vertex routing headers. Pins that the setting is ignored silently rather than
    raising, and that it does not leak into the Gemini API's own `service_tier` config field.
    """
    m = GoogleModel('gemini-2.5-flash', provider=gla_client_google_cloud_provider)
    assert m.system == 'google-cloud'

    _, config = await m._build_content_and_config(  # pyright: ignore[reportPrivateUsage]
        messages=[ModelRequest(parts=[UserPromptPart(content='Hello')])],
        model_settings=GoogleModelSettings(google_cloud_service_tier='pt_only'),
        model_request_parameters=ModelRequestParameters(),
    )

    config_dict = cast(dict[str, Any], config)
    assert not any(header.startswith('X-Vertex-AI') for header in config_dict['http_options']['headers'])
    assert 'service_tier' not in config_dict


GCS_URI = 'gs://bucket/doc.pdf'
FILES_API_URI = 'https://generativelanguage.googleapis.com/v1beta/files/abc'


@dataclass(frozen=True)
class UploadedFileCase:
    """One construction where the provider name and the client's transport disagree."""

    id: str
    provider_fixture: str
    canonical_name: UploadedFileProviderName
    """The name this construction stamps on its own files — also the one used for the rejection."""
    also_accepted: UploadedFileProviderName
    valid_file_id: str
    """A file id the transport can actually serve."""
    rejected_file_id: str
    rejection_match: str


UPLOADED_FILE_CASES = [
    UploadedFileCase(
        id='google_cloud_client_in_google_provider',
        provider_fixture='vertex_client_google_provider',
        canonical_name='google-cloud',
        also_accepted='google',
        valid_file_id=GCS_URI,
        rejected_file_id=FILES_API_URI,
        rejection_match='must use a GCS URI',
    ),
    UploadedFileCase(
        id='gemini_api_client_in_google_cloud_provider',
        provider_fixture='gla_client_google_cloud_provider',
        canonical_name='google',
        also_accepted='google-cloud',
        valid_file_id=FILES_API_URI,
        rejected_file_id=GCS_URI,
        rejection_match='must use a file URI from the Google Files API',
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in UPLOADED_FILE_CASES])
def test_uploaded_file_validation_follows_the_client_transport(
    request: pytest.FixtureRequest, case: UploadedFileCase
) -> None:
    """`UploadedFile` validation follows the client's transport, not the provider name (#6792).

    Both disagreeing constructions run through one test so the pair cannot drift: each accepts the
    file id its transport can serve and rejects the other's, whichever way `name` points.
    """
    m = GoogleModel('gemini-2.5-flash', provider=request.getfixturevalue(case.provider_fixture))

    for provider_name in (case.canonical_name, case.also_accepted):
        file = UploadedFile(file_id=case.valid_file_id, provider_name=provider_name, media_type='application/pdf')
        assert m._validate_uploaded_file(file) == (case.valid_file_id, 'application/pdf')  # pyright: ignore[reportPrivateUsage]

    wrong_transport_file = UploadedFile(
        file_id=case.rejected_file_id, provider_name=case.canonical_name, media_type='application/pdf'
    )
    with pytest.raises(UserError, match=case.rejection_match):
        m._validate_uploaded_file(wrong_transport_file)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('vertexai', [True, False])
def test_provider_outside_both_name_families_matches_only_itself(vertexai: bool) -> None:
    """A custom `Provider[Client]` keeps its own name as the only accepted one, on either transport.

    `GoogleModel` accepts any `Provider[Client]`, so a third-party provider can carry a name in
    neither Google family. It has no pre-v2 alias to accept, and inheriting a family's aliases would
    replay another provider's thinking signatures and native tool parts. Its transport branches still
    follow its client — that part is what the two directions above also rely on.
    """

    class MyProxyProvider(Provider[Client]):
        def __init__(self, client: Client) -> None:
            self._client = client

        @property
        def name(self) -> str:
            return 'my-google-proxy'

        @property
        def base_url(self) -> str:
            return 'https://proxy.example.invalid'

        @property
        def client(self) -> Client:
            return self._client

    client = (
        Client(vertexai=True, project='test-project', location='us-central1')
        if vertexai
        else Client(vertexai=False, api_key='mock-api-key')
    )
    m = GoogleModel('gemini-2.5-flash', provider=MyProxyProvider(client))

    assert m.system == 'my-google-proxy'
    assert m.base_url == 'https://proxy.example.invalid'
    assert m._matching_provider_names == frozenset({'my-google-proxy'})  # pyright: ignore[reportPrivateUsage]
    assert m._is_google_cloud is vertexai  # pyright: ignore[reportPrivateUsage]
