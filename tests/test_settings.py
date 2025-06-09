import importlib

import pytest

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


@pytest.fixture(params=['openai_', 'anthropic_', 'bedrock_', 'groq_', 'gemini_', 'mistral_', 'cohere_'])
def settings(request: pytest.FixtureRequest) -> tuple[type[ModelSettings], str]:
    prefix_cls_name = request.param.replace('_', '')
    try:
        module = importlib.import_module(f'pydantic_ai.models.{prefix_cls_name}')
    except ImportError:  # pragma: lax no cover
        pytest.skip(f'{prefix_cls_name} is not installed')
    capitalized_prefix = prefix_cls_name.capitalize().replace('Openai', 'OpenAI')
    cls = getattr(module, capitalized_prefix + 'ModelSettings')
    return cls, request.param


def test_specific_prefix_settings(settings: tuple[type[ModelSettings], str]):
    settings_cls, prefix = settings
    global_settings = set(ModelSettings.__annotations__.keys())
    specific_settings = set(settings_cls.__annotations__.keys()) - global_settings
    assert all(setting.startswith(prefix) for setting in specific_settings), (
        f'{prefix} is not a prefix for {specific_settings}'
    )


@pytest.mark.parametrize(
    'model', ['openai', 'anthropic', 'bedrock', 'mistral', 'groq', 'cohere', 'google'], indirect=True
)
async def test_stop_settings(allow_model_requests: None, model: Model) -> None:
    agent = Agent(model=model, model_settings=ModelSettings(stop_sequences=['Paris']))
    result = await agent.run(
        'What is the capital of France? Give me an answer that contains the word "Paris", but is not the first word.'
    )

    # NOTE: Bedrock has a slightly different behavior. It will include the stop sequence in the response.
    if model.system == 'bedrock':
        assert result.output.endswith('Paris')
    else:
        assert 'Paris' not in result.output
