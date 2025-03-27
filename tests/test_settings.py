import pytest

from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModelSettings
    from pydantic_ai.models.bedrock import BedrockModelSettings
    from pydantic_ai.models.cohere import CohereModelSettings
    from pydantic_ai.models.gemini import GeminiModelSettings
    from pydantic_ai.models.groq import GroqModelSettings
    from pydantic_ai.models.mistral import MistralModelSettings
    from pydantic_ai.models.openai import OpenAIModelSettings
    from pydantic_ai.settings import ModelSettings

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='dependencies not installed'),
]


@pytest.mark.parametrize(
    'settings_cls, prefix',
    [
        (OpenAIModelSettings, 'openai_'),
        (AnthropicModelSettings, 'anthropic_'),
        (BedrockModelSettings, 'bedrock_'),
        (GroqModelSettings, 'groq_'),
        (GeminiModelSettings, 'gemini_'),
        (MistralModelSettings, 'mistral_'),
        (CohereModelSettings, 'cohere_'),
    ],
)
def test_specific_prefix_settings(settings_cls: type[ModelSettings], prefix: str):
    global_settings = set(ModelSettings.__annotations__.keys())
    specific_settings = set(settings_cls.__annotations__.keys()) - global_settings
    assert all(setting.startswith(prefix) for setting in specific_settings), (
        f'{prefix} is not a prefix for {specific_settings}'
    )
