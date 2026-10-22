"""Recorded Cloudflare embeddings through the dedicated embedding interface."""

import os

import pytest

from pydantic_ai import Embedder
from pydantic_ai.embeddings import infer_embedding_model
from pydantic_ai.providers.cloudflare import CloudflareProvider

from .conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.embeddings.cloudflare import CloudflareEmbeddingModel

pytestmark = [
    pytest.mark.usefixtures('allow_model_requests'),
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK not installed'),
]


@pytest.mark.parametrize('model_name', ['@cf/baai/bge-small-en-v1.5', '@cf/baai/bge-m3'])
async def test_embed(model_name: str, request_capture: RequestCapture):
    provider = CloudflareProvider(
        api_key=os.getenv('CLOUDFLARE_API_TOKEN', 'test-token'),
        account_id=os.getenv('CLOUDFLARE_ACCOUNT_ID', 'test-account'),
        http_client=request_capture.http_client(timeout=120),
    )
    model = infer_embedding_model(f'cloudflare:{model_name}', provider_factory=lambda _: provider)
    assert isinstance(model, CloudflareEmbeddingModel)
    result = await Embedder(model).embed_documents(['An order has shipped.', 'A customer requested a refund.'])
    assert len(result.embeddings) == 2
    assert len(result.embeddings[0]) > 0
    assert result.provider_name == 'cloudflare'
    body = request_capture.body('/embeddings')
    assert body['input'] == ['An order has shipped.', 'A customer requested a refund.']
    assert body['model'] == model_name
