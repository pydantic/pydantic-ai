import importlib

import pytest

from pydantic_ai.settings import ModelSettings


@pytest.fixture(params=['openai_', 'anthropic_', 'bedrock_', 'groq_', 'gemini_', 'mistral_', 'cohere_'])
def settings(request: pytest.FixtureRequest) -> tuple[type[ModelSettings], str]:
    prefix_cls_name = request.param.replace('_', '')
    try:
        module = importlib.import_module(f'pydantic_ai.models.{prefix_cls_name}')
    except ImportError:
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
