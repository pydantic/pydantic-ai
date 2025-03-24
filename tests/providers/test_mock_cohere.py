from __future__ import annotations as _annotations

from unittest.mock import MagicMock

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from cohere import AsyncClientV2

    from pydantic_ai.providers.cohere import CohereProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='cohere not installed')


def test_cohere_provider_with_mock_client():
    # Create a mock AsyncClientV2
    mock_client = MagicMock(spec=AsyncClientV2)

    # This should now work with our fixed overloads
    provider = CohereProvider(cohere_client=mock_client)

    # Verify the provider was correctly initialized with our mock client
    assert provider.client is mock_client
