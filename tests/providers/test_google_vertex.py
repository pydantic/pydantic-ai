import httpx
import pytest
from inline_snapshot import snapshot

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.vcr(),
    pytest.mark.anyio(),
]


async def test_google_vertex_provider(allow_model_requests: None) -> None:
    provider = GoogleVertexProvider()
    assert provider.name == 'google-vertex'
    assert provider.base_url == snapshot(
        'https://us-central1-aiplatform.googleapis.com/v1/projects/None/locations/us-central1/publishers/google/models/'
    )
    assert isinstance(provider.client, httpx.AsyncClient)
