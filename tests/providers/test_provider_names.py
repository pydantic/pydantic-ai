from __future__ import annotations as _annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from pydantic_ai.providers import Provider, infer_provider

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.google_gla import GoogleGLAProvider
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    test_infer_provider_params = [
        ('deepseek', DeepSeekProvider, 'DEEPSEEK_API_KEY'),
        ('openai', OpenAIProvider, None),
        ('google-vertex', GoogleVertexProvider, None),
        ('google-gla', GoogleGLAProvider, 'GEMINI_API_KEY'),
        ('groq', GroqProvider, 'GROQ_API_KEY'),
    ]

if not imports_successful():
    test_infer_provider_params = []

pytestmark = pytest.mark.skipif(not imports_successful(), reason='need to install all extra packages')


@pytest.fixture(scope='module', autouse=True)
def empty_env():
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.mark.parametrize(('provider', 'provider_cls', 'exception_has'), test_infer_provider_params)
def test_infer_provider(provider: str, provider_cls: type[Provider[Any]], exception_has: str | None):
    if exception_has is not None:
        with pytest.raises(ValueError, match=rf'.*{exception_has}.*'):
            infer_provider(provider)
    else:
        assert isinstance(infer_provider(provider), provider_cls)
