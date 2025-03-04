from typing import Any

import pytest

from pydantic_ai.providers import Provider, infer_provider

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.google_gla import GoogleGLAProvider
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    test_infer_provider_params = [
        ('deepseek', DeepSeekProvider),
        ('openai', OpenAIProvider),
        ('google-vertex', GoogleVertexProvider),
        ('google-gla', GoogleGLAProvider),
    ]

if not imports_successful():
    test_infer_provider_params = []

pytestmark = pytest.mark.skipif(not imports_successful(), reason='need to install all extra packages')


@pytest.mark.parametrize(('provider, provider_cls'), test_infer_provider_params)
def test_infer_provider(provider: str, provider_cls: type[Provider[Any]]):
    assert isinstance(infer_provider(provider), provider_cls)
