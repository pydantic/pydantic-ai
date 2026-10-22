"""Recorded authenticated catalog metadata and explicit task-filtered discovery."""

import os
from typing import Literal

import pytest

from pydantic_ai.providers.cloudflare import CloudflareProvider

from ..conftest import RequestCapture

pytestmark = pytest.mark.vcr


@pytest.mark.parametrize('task', ['Text Generation', 'Text Embeddings', 'Text-to-Image'])
async def test_discover(
    task: Literal['Text Generation', 'Text Embeddings', 'Text-to-Image'], request_capture: RequestCapture
):
    provider = CloudflareProvider(
        api_key=os.getenv('CLOUDFLARE_API_TOKEN', 'test-token'),
        account_id=os.getenv('CLOUDFLARE_ACCOUNT_ID', 'test-account'),
        http_client=request_capture.http_client(timeout=30),
    )
    names = await provider.discover(task=task)
    assert names
    assert names == sorted(set(names))
    assert '@cf/cloudflare/clef-omni' not in names
    count = len(request_capture.paths)
    assert await provider.discover(task=task) == names
    assert len(request_capture.paths) == count
