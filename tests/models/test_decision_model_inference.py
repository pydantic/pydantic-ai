"""Model-name routing is local behavior and requires no recorded API responses."""

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName, infer_model, known_model_names
from pydantic_ai.providers import infer_provider_class

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openai_decisions import OpenAIDecisionsModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed')


@pytest.mark.parametrize('prefix', ['openai', 'openai-chat', 'openai-responses', 'openai-decisions'])
def test_openai_api_selection(prefix: str, env: TestEnv):
    """Only the Decisions prefix selects a decision model; all four use OpenAI credentials."""
    env.set('OPENAI_API_KEY', 'test-api-key')
    env.remove('OPENAI_BASE_URL')
    expected_class = {
        'openai': OpenAIResponsesModel,
        'openai-chat': OpenAIChatModel,
        'openai-responses': OpenAIResponsesModel,
        'openai-decisions': OpenAIDecisionsModel,
    }[prefix]

    model = infer_model(f'{prefix}:gpt-6-luna')

    assert infer_provider_class(prefix) is OpenAIProvider
    assert isinstance(model, expected_class)
    assert model.model_name == 'gpt-6-luna'
    assert model.system == 'openai'
    assert model.model_id == 'openai:gpt-6-luna'
    assert model.base_url == 'https://api.openai.com/v1/'
    assert infer_model(model) is model


@pytest.mark.parametrize('model_name', ['gpt-6-luna', 'custom-decision-model'])
def test_decisions_custom_provider_factory(model_name: str):
    """The factory receives the API prefix, including for names beyond the known catalog."""
    provider = OpenAIProvider(api_key='test-api-key', base_url='https://example.com/v1')

    def provider_factory(provider_name: str) -> OpenAIProvider:
        assert provider_name == 'openai-decisions'
        return provider

    model = infer_model(f'openai-decisions:{model_name}', provider_factory)

    assert isinstance(model, OpenAIDecisionsModel)
    assert model.model_name == model_name
    assert model.base_url == provider.base_url
    assert model.client is provider.client


def test_decisions_requires_openai_credentials(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    env.remove('OPENAI_BASE_URL')

    with pytest.raises(UserError, match='OPENAI_API_KEY'):
        infer_model('openai-decisions:gpt-6-luna')


def test_decisions_known_model_name():
    model_name: KnownModelName = 'openai-decisions:gpt-6-luna'
    assert model_name in known_model_names()
