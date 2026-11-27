"""Whether a model class on a given provider accepts a request for image output, as its profile says.

Not VCR tests: each case is decided by `Model.prepare_request`'s profile check before anything is sent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from pydantic_ai import Agent, BinaryImage
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncAzureOpenAI

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[], Model]
    supports_image_output: bool


def _azure_chat_model() -> Model:
    client = AsyncAzureOpenAI(
        azure_endpoint='https://example.openai.azure.com', api_version='2024-10-21', api_key='test'
    )
    return OpenAIChatModel('gpt-4o', provider=AzureProvider(openai_client=client))


CASES = [
    # The OpenAI profile allows image output for the Responses API's image generation tool, which OpenAI's Chat
    # Completions doesn't offer.
    Case('openai-chat', lambda: OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test')), False),
    Case('azure-chat', _azure_chat_model, False),
    # OpenRouter returns images on Chat Completions, and its Gemini image routes keep the Google profile's flag.
    Case(
        'openrouter-gemini-image',
        lambda: OpenRouterModel('google/gemini-2.5-flash-image', provider=OpenRouterProvider(api_key='test')),
        True,
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_image_output_follows_the_profile(case: Case, allow_model_requests: None):
    model = case.model()
    assert model.profile.get('supports_image_output') is case.supports_image_output
    params = ModelRequestParameters(allow_image_output=True)
    if case.supports_image_output:
        model.prepare_request(None, params)
    else:
        with pytest.raises(UserError, match='Image output is not supported by this model'):
            await Agent(model, output_type=BinaryImage).run('Draw a cat.')
