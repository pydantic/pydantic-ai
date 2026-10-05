from __future__ import annotations

from datetime import timedelta
from typing import Literal

import pytest

from pydantic_ai.exceptions import PydanticAIDeprecationWarning
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIResponsesModel,
    )
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='provider extras not installed')


def test_model_resolve_cache_retention_defaults_to_none() -> None:
    model: Model = TestModel()

    assert model.resolve_cache_retention(None) is None


@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize(
    ('setting', 'expected'),
    [
        (None, None),
        ('in_memory', None),
        ('24h', timedelta(hours=24)),
    ],
)
def test_openai_resolve_cache_retention(
    api: Literal['chat', 'responses'],
    setting: Literal['in_memory', '24h'] | None,
    expected: timedelta | None,
) -> None:
    model_type = OpenAIChatModel if api == 'chat' else OpenAIResponsesModel
    model = model_type('gpt-5.6', provider=OpenAIProvider(api_key='test-key'))
    settings = OpenAIChatModelSettings(openai_prompt_cache_retention=setting) if setting is not None else None

    assert model.resolve_cache_retention(settings) == expected


@pytest.mark.parametrize(
    ('settings', 'expected'),
    [
        ({'anthropic_cache': True}, timedelta(minutes=5)),
        ({'anthropic_cache': '5m'}, timedelta(minutes=5)),
        ({'anthropic_cache': '1h'}, timedelta(hours=1)),
        ({'anthropic_cache_instructions': True}, timedelta(minutes=5)),
        ({'anthropic_cache_instructions': '1h'}, timedelta(hours=1)),
        ({'anthropic_cache_tool_definitions': True}, timedelta(minutes=5)),
        ({'anthropic_cache_tool_definitions': '1h'}, timedelta(hours=1)),
        ({'anthropic_cache_messages': True}, timedelta(minutes=5)),
        ({'anthropic_cache_messages': '1h'}, timedelta(hours=1)),
    ],
)
def test_anthropic_resolve_cache_retention(settings: AnthropicModelSettings, expected: timedelta) -> None:
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test-key'))

    assert model.resolve_cache_retention(settings) == expected


def test_anthropic_resolve_cache_retention_biases_high() -> None:
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test-key'))
    settings = AnthropicModelSettings(
        anthropic_cache_instructions=True,
        anthropic_cache_tool_definitions='1h',
        anthropic_cache_messages='5m',
    )

    assert model.resolve_cache_retention(settings) == timedelta(hours=1)
    assert model.resolve_cache_retention(None) is None


@pytest.mark.parametrize(
    ('settings', 'profile', 'expected'),
    [
        (
            {'bedrock_cache_instructions': True},
            {'bedrock_supports_prompt_caching': True},
            timedelta(minutes=5),
        ),
        (
            {'bedrock_cache_messages': '1h'},
            {'bedrock_supports_prompt_caching': True},
            timedelta(hours=1),
        ),
        (
            {'bedrock_cache_tool_definitions': '5m'},
            {'bedrock_supports_tool_caching': True},
            timedelta(minutes=5),
        ),
        (
            {'bedrock_cache_instructions': '1h'},
            {'bedrock_supports_prompt_caching': False},
            None,
        ),
        (
            {'bedrock_cache_tool_definitions': '1h'},
            {'bedrock_supports_tool_caching': False},
            None,
        ),
        (None, {'bedrock_supports_prompt_caching': True, 'bedrock_supports_tool_caching': True}, None),
    ],
)
def test_bedrock_resolve_cache_retention(
    bedrock_provider: BedrockProvider,
    settings: BedrockModelSettings | None,
    profile: BedrockModelProfile,
    expected: timedelta | None,
) -> None:
    model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-20250514-v1:0', provider=bedrock_provider, profile=profile
    )

    assert model.resolve_cache_retention(settings) == expected


def test_bedrock_resolve_cache_retention_biases_high(bedrock_provider: BedrockProvider) -> None:
    model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-20250514-v1:0',
        provider=bedrock_provider,
        profile=BedrockModelProfile(
            bedrock_supports_prompt_caching=True,
            bedrock_supports_tool_caching=True,
        ),
    )
    settings = BedrockModelSettings(
        bedrock_cache_instructions=True,
        bedrock_cache_messages='1h',
        bedrock_cache_tool_definitions='5m',
    )

    assert model.resolve_cache_retention(settings) == timedelta(hours=1)


def test_openrouter_resolve_cache_retention() -> None:
    model = OpenRouterModel(
        'anthropic/claude-sonnet-4.6',
        provider=OpenRouterProvider(api_key='test-key'),
        profile=OpenRouterModelProfile(
            openrouter_supports_cache_control=True,
            openrouter_supports_cache_ttl=True,
            openrouter_supports_tool_cache=True,
        ),
    )

    assert model.resolve_cache_retention(OpenRouterModelSettings(openrouter_cache_instructions=True)) == timedelta(
        minutes=5
    )
    assert model.resolve_cache_retention(OpenRouterModelSettings(openrouter_cache_messages='5m')) == timedelta(
        minutes=5
    )
    assert model.resolve_cache_retention(OpenRouterModelSettings(openrouter_cache_tool_definitions='1h')) == timedelta(
        hours=1
    )
    assert model.resolve_cache_retention(None) is None


def test_openrouter_resolve_cache_retention_biases_high() -> None:
    model = OpenRouterModel(
        'anthropic/claude-sonnet-4.6',
        provider=OpenRouterProvider(api_key='test-key'),
        profile=OpenRouterModelProfile(
            openrouter_supports_cache_control=True,
            openrouter_supports_cache_ttl=True,
            openrouter_supports_tool_cache=True,
        ),
    )
    settings = OpenRouterModelSettings(
        openrouter_cache_instructions=True,
        openrouter_cache_messages='1h',
        openrouter_cache_tool_definitions='5m',
    )

    assert model.resolve_cache_retention(settings) == timedelta(hours=1)


@pytest.mark.parametrize(
    'profile',
    [
        {'openrouter_supports_cache_control': True, 'openrouter_supports_tool_cache': True},
        {
            'openrouter_supports_cache_ttl': True,
            'openrouter_supports_cache_control': False,
            'openrouter_supports_tool_cache': False,
        },
    ],
)
def test_openrouter_resolve_cache_retention_ignores_unsupported_settings(
    profile: OpenRouterModelProfile,
) -> None:
    model = OpenRouterModel(
        'google/gemini-3.1-pro-preview',
        provider=OpenRouterProvider(api_key='test-key'),
        profile=profile,
    )
    settings = OpenRouterModelSettings(
        openrouter_cache_instructions='1h',
        openrouter_cache_messages='1h',
        openrouter_cache_tool_definitions='1h',
    )

    assert model.resolve_cache_retention(settings) is None


# `resolve_prompt_cache_retention` was renamed to `resolve_cache_retention`. These are unit tests
# rather than VCR tests: the deprecation shim is pure method dispatch, and no request would exercise it.


def _one_hour() -> AnthropicModelSettings:
    return AnthropicModelSettings(anthropic_cache='1h')


def _anthropic(model_type: type[AnthropicModel] | None = None) -> AnthropicModel:
    return (model_type or AnthropicModel)('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test-key'))


def test_resolve_prompt_cache_retention_is_a_deprecated_alias() -> None:
    model = _anthropic()

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match='`resolve_prompt_cache_retention` is deprecated, use `resolve_cache_retention` instead',
    ):
        assert model.resolve_prompt_cache_retention(_one_hour()) == timedelta(hours=1)  # pyright: ignore[reportDeprecated]


def test_legacy_override_is_honored_under_the_new_name() -> None:
    """A subclass written against the old name keeps working: Pydantic AI calls `resolve_cache_retention`."""
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match='`LegacyModel` overrides `resolve_prompt_cache_retention`, which is deprecated; override `resolve_cache_retention` instead',
    ) as warned:

        class LegacyModel(AnthropicModel):
            def resolve_prompt_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
                return timedelta(hours=2)

    # Attributed to the class statement, so the user can find the override to rename.
    assert warned[0].filename == __file__
    model = _anthropic(LegacyModel)
    assert model.resolve_cache_retention(None) == timedelta(hours=2)
    assert WrapperModel(model).resolve_cache_retention(None) == timedelta(hours=2)


def test_legacy_override_can_extend_its_parent_through_super() -> None:
    """`super()` from a legacy override reaches the parent's renamed implementation instead of recursing."""
    with pytest.warns(PydanticAIDeprecationWarning, match='`LegacyModel` overrides'):

        class LegacyModel(AnthropicModel):
            def resolve_prompt_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
                parent = super().resolve_prompt_cache_retention(model_settings)  # pyright: ignore[reportDeprecated]
                return parent or timedelta(hours=2)

    model = _anthropic(LegacyModel)
    with pytest.warns(PydanticAIDeprecationWarning, match='`resolve_prompt_cache_retention` is deprecated'):
        assert model.resolve_cache_retention(_one_hour()) == timedelta(hours=1)
    with pytest.warns(PydanticAIDeprecationWarning, match='`resolve_prompt_cache_retention` is deprecated'):
        assert model.resolve_cache_retention(None) == timedelta(hours=2)


def test_stacked_legacy_overrides_chain_through_super() -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match='`LegacyModel` overrides'):

        class LegacyModel(AnthropicModel):
            def resolve_prompt_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
                parent = super().resolve_prompt_cache_retention(model_settings)  # pyright: ignore[reportDeprecated]
                return parent or timedelta(hours=2)

    with pytest.warns(PydanticAIDeprecationWarning, match='`LegacierModel` overrides'):

        class LegacierModel(LegacyModel):
            def resolve_prompt_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
                parent = super().resolve_prompt_cache_retention(model_settings)
                assert parent is not None
                return parent * 2

    model = _anthropic(LegacierModel)
    with pytest.warns(PydanticAIDeprecationWarning, match='`resolve_prompt_cache_retention` is deprecated'):
        assert model.resolve_cache_retention(_one_hour()) == timedelta(hours=2)
    with pytest.warns(PydanticAIDeprecationWarning, match='`resolve_prompt_cache_retention` is deprecated'):
        assert model.resolve_cache_retention(None) == timedelta(hours=4)


def test_new_override_below_a_legacy_override_chains_through_it() -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match='`LegacyModel` overrides'):

        class LegacyModel(AnthropicModel):
            def resolve_prompt_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
                parent = super().resolve_prompt_cache_retention(model_settings)  # pyright: ignore[reportDeprecated]
                return parent or timedelta(hours=2)

    class MigratedModel(LegacyModel):
        def resolve_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
            return super().resolve_cache_retention(model_settings) or timedelta(hours=3)

    model = _anthropic(MigratedModel)
    with pytest.warns(PydanticAIDeprecationWarning, match='`resolve_prompt_cache_retention` is deprecated'):
        assert model.resolve_cache_retention(_one_hour()) == timedelta(hours=1)
    with pytest.warns(PydanticAIDeprecationWarning, match='`resolve_prompt_cache_retention` is deprecated'):
        assert model.resolve_cache_retention(None) == timedelta(hours=2)


def test_old_name_reaches_a_new_override() -> None:
    """Callers still using the old name get a subclass's `resolve_cache_retention` override."""

    class NewModel(AnthropicModel):
        def resolve_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
            return timedelta(hours=3)

    model = _anthropic(NewModel)
    with pytest.warns(PydanticAIDeprecationWarning, match='`resolve_prompt_cache_retention` is deprecated'):
        assert model.resolve_prompt_cache_retention(None) == timedelta(hours=3)  # pyright: ignore[reportDeprecated]


def test_overriding_both_names_keeps_the_new_one() -> None:
    class BothModel(AnthropicModel):
        def resolve_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
            return timedelta(hours=3)

        def resolve_prompt_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
            return timedelta(hours=2)

    model = _anthropic(BothModel)
    assert model.resolve_cache_retention(None) == timedelta(hours=3)
    assert model.resolve_prompt_cache_retention(None) == timedelta(hours=2)  # pyright: ignore[reportDeprecated]


def test_wrapper_model_forwards_resolve_cache_retention() -> None:
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test-key'), settings=_one_hour())

    assert WrapperModel(model).resolve_cache_retention(None) == timedelta(hours=1)
