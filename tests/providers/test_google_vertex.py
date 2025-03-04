import os
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from inline_snapshot import snapshot

from pydantic_ai.agent import Agent

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio(),
    pytest.mark.vcr(),
]


@pytest.fixture()
def ignore_environment_variables():
    with patch.dict(os.environ, {}, clear=True):
        yield


def test_google_vertex_provider(allow_model_requests: None) -> None:
    provider = GoogleVertexProvider()
    assert provider.name == 'google-vertex'
    assert provider.base_url == snapshot(
        'https://us-central1-aiplatform.googleapis.com/v1/projects/None/locations/us-central1/publishers/google/models/'
    )
    assert isinstance(provider.client, httpx.AsyncClient)


async def test_google_vertex_provider_auth(tmp_path: Path, allow_model_requests: None):
    agent = Agent('google-vertex:gemini-1.0-pro')
    result = await agent.run('Hello')
    assert result.data == snapshot('Hello! How can I help you today?')
