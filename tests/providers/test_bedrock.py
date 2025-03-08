import os
from unittest.mock import patch

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.bedrock import BedrockProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')


def test_bedrock_provider():
    with patch.dict(os.environ, {'AWS_DEFAULT_REGION': 'us-east-1'}):
        provider = BedrockProvider()
        assert isinstance(provider, BedrockProvider)
        assert provider.name == 'bedrock'
        assert provider.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'
