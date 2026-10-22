"""Unit tests for CloudflareProvider catalog discovery that VCR cannot reliably reproduce."""

from __future__ import annotations

import httpx2
import pytest

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.providers.cloudflare import CloudflareProvider


def _catalog_page() -> dict[str, object]:
    return {
        'result': [{'name': f'@cf/test/model-{index}', 'task': {'name': 'Text Generation'}} for index in range(100)]
    }


async def test_discover_caps_pagination():
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        requests.append(request)
        return httpx2.Response(200, json=_catalog_page())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    provider = CloudflareProvider(api_key='token', account_id='account', http_client=client)

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await provider.discover('Text Generation')

    assert len(requests) == 20
    assert 'did not complete within 20 pages' in str(exc_info.value)
