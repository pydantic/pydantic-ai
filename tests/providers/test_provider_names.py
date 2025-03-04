from typing import Any

import pytest

from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.google_vertex import GoogleVertexProvider

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


@pytest.mark.parametrize(
    ('provider, provider_cls'),
    [
        ('deepseek', DeepSeekProvider),
        ('openai', OpenAIProvider),
        ('google-vertex', GoogleVertexProvider),
        ('google-gla', GoogleGLAProvider),
    ],
)
def test_infer_provider(provider: str, provider_cls: type[Provider[Any]]):
    assert isinstance(infer_provider(provider), provider_cls)
