"""Tests for the unified `cache` prompt-caching setting.

Tests the base `Model.prepare_request()` cache resolution and retention snap-down, the
per-provider translation onto provider-specific cache settings, the shared cache-point
budget helper, the retention resolver, and the Google warning path.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._prompt_cache import excess_cache_points, snap_cache_retention
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import CacheRetention, CacheSetting, ModelSettings

from .conftest import try_import

with try_import() as anthropic_imports:
    from anthropic import AsyncAnthropicBedrock

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_imports:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as openai_imports:
    from pydantic_ai.models.openrouter import (
        OpenRouterModelSettings,
        _openrouter_settings_to_openai_settings,
    )
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.anyio,
]


def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='ok')])


def _make_model(
    *,
    supports_cache: bool = False,
    supported_cache_retentions: tuple[CacheRetention, ...] | None = None,
) -> FunctionModel:
    profile = ModelProfile(supports_cache=supports_cache)
    if supported_cache_retentions is not None:
        profile['supported_cache_retentions'] = supported_cache_retentions
    return FunctionModel(_echo, profile=profile)


def _resolve_cache(model: FunctionModel, cache: CacheSetting) -> tuple[ModelSettings | None, CacheSetting | None]:
    settings, params = model.prepare_request(ModelSettings(cache=cache), ModelRequestParameters())
    return settings, params.cache


class TestSnapCacheRetention:
    @pytest.mark.parametrize(
        ('value', 'supported', 'expected'),
        [
            (True, ('5m', '1h'), True),
            (False, ('5m', '1h'), False),
            ('5m', ('5m', '1h'), '5m'),
            ('1h', ('5m', '1h'), '1h'),
            ('30m', ('5m', '1h'), '5m'),
            ('1h', ('5m',), '5m'),
            ('30m', ('5m',), '5m'),
            ('5m', ('30m', '1h'), '30m'),
            ('5m', ('1h',), '1h'),
            ('1h', (), '1h'),
        ],
    )
    def test_snap(self, value: CacheSetting, supported: tuple[CacheRetention, ...], expected: CacheSetting):
        assert snap_cache_retention(value, supported) == expected


class TestPrepareRequestCacheResolution:
    def test_cache_true_with_supports_cache(self):
        model = _make_model(supports_cache=True)
        settings, cache = _resolve_cache(model, True)
        assert cache is True
        assert settings is None

    def test_cache_retention_snapped_to_supported_tier(self):
        model = _make_model(supports_cache=True, supported_cache_retentions=('5m',))
        _, cache = _resolve_cache(model, '1h')
        assert cache == '5m'

    def test_cache_retention_kept_when_supported(self):
        model = _make_model(supports_cache=True, supported_cache_retentions=('5m', '1h'))
        _, cache = _resolve_cache(model, '1h')
        assert cache == '1h'

    def test_cache_false_not_resolved(self):
        model = _make_model(supports_cache=True)
        settings, cache = _resolve_cache(model, False)
        assert cache is None
        assert settings is None

    def test_cache_dropped_without_supports_cache(self):
        model = _make_model(supports_cache=False)
        settings, cache = _resolve_cache(model, True)
        assert cache is None
        assert settings is None

    def test_cache_stripped_but_other_settings_kept(self):
        model = _make_model(supports_cache=True)
        settings, params = model.prepare_request(
            ModelSettings(cache=True, temperature=0.5), ModelRequestParameters()
        )
        assert settings == {'temperature': 0.5}
        assert params.cache is True

    def test_prepare_request_does_not_mutate_model_settings(self):
        model = _make_model(supports_cache=True)
        original = ModelSettings(cache='1h', temperature=0.5)
        model.prepare_request(original, ModelRequestParameters())
        assert original == {'cache': '1h', 'temperature': 0.5}


class TestExcessCachePoints:
    def test_excess_returned_oldest_first_beyond_budget(self):
        blocks_newest_first = [{'cache_control': 1}, {'text': 'x'}, {'cache_control': 2}, {'cache_control': 3}]
        excess = excess_cache_points(
            blocks_newest_first,
            max_points=2,
            reserved=1,
            is_cache_point=lambda b: 'cache_control' in b,
            description='test request',
        )
        assert excess == [{'cache_control': 2}, {'cache_control': 3}]

    def test_no_excess_within_budget(self):
        blocks = [{'cache_control': 1}]
        assert (
            excess_cache_points(
                blocks, max_points=4, reserved=0, is_cache_point=lambda b: 'cache_control' in b, description='x'
            )
            == []
        )

    def test_reserved_exceeding_max_raises(self):
        with pytest.raises(UserError, match='Too many cache points for test request'):
            excess_cache_points(
                [], max_points=4, reserved=5, is_cache_point=lambda b: True, description='test request'
            )


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
class TestAnthropicCacheTranslation:
    def _model(self, client: Any | None = None) -> AnthropicModel:
        provider = AnthropicProvider(api_key='test') if client is None else AnthropicProvider(anthropic_client=client)
        return AnthropicModel('claude-sonnet-4-5', provider=provider)

    def test_cache_true_uses_automatic_caching(self):
        settings, params = self._model().prepare_request(ModelSettings(cache=True), ModelRequestParameters())
        assert settings == {'anthropic_cache': '5m'}
        assert params.cache is True

    def test_cache_retention_forwarded(self):
        settings, _ = self._model().prepare_request(ModelSettings(cache='1h'), ModelRequestParameters())
        assert settings == {'anthropic_cache': '1h'}

    def test_cache_30m_snaps_down_to_5m(self):
        settings, _ = self._model().prepare_request(ModelSettings(cache='30m'), ModelRequestParameters())
        assert settings == {'anthropic_cache': '5m'}

    def test_bedrock_client_uses_stable_boundary_breakpoints(self):
        client = AsyncAnthropicBedrock(aws_access_key='x', aws_secret_key='y', aws_region='us-east-1')
        settings, _ = self._model(client).prepare_request(ModelSettings(cache='1h'), ModelRequestParameters())
        assert settings == {'anthropic_cache_instructions': '1h', 'anthropic_cache_tool_definitions': '1h'}

    def test_explicit_provider_setting_wins(self):
        settings, _ = self._model().prepare_request(
            AnthropicModelSettings(cache=True, anthropic_cache_instructions='1h'), ModelRequestParameters()
        )
        assert settings == {'anthropic_cache_instructions': '1h'}

    def test_explicit_cache_messages_prevents_automatic_caching_conflict(self):
        """`anthropic_cache_messages` cannot be combined with `anthropic_cache`, so the unified
        value must not inject the automatic setting alongside it."""
        settings, _ = self._model().prepare_request(
            AnthropicModelSettings(cache=True, anthropic_cache_messages=True), ModelRequestParameters()
        )
        assert settings == {'anthropic_cache_messages': True}

    def test_provider_profile_flags(self):
        profile = self._model().profile
        assert profile.get('supports_cache') is True
        assert profile.get('supported_cache_retentions') == ('5m', '1h')
        assert profile.get('supports_auto_cache') is True
        assert profile.get('max_cache_points') == 4


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
class TestBedrockCacheTranslation:
    def _model(self) -> BedrockConverseModel:
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        return model

    def test_cache_true_translates_to_stable_boundaries(self):
        settings = self._model()._translate_cache(BedrockModelSettings(), True)
        assert settings == {'bedrock_cache_instructions': '5m', 'bedrock_cache_tool_definitions': '5m'}

    def test_cache_retention_forwarded(self):
        settings = self._model()._translate_cache(BedrockModelSettings(), '1h')
        assert settings == {'bedrock_cache_instructions': '1h', 'bedrock_cache_tool_definitions': '1h'}

    def test_explicit_provider_setting_wins(self):
        settings = self._model()._translate_cache(BedrockModelSettings(bedrock_cache_messages=True), True)
        assert settings == {'bedrock_cache_messages': True}

    def test_provider_profile_flags(self):
        profile = BedrockProvider.model_profile('anthropic.claude-sonnet-4-5-20250929-v1:0')
        assert profile is not None
        assert profile.get('supports_cache') is True
        assert profile.get('supported_cache_retentions') == ('5m', '1h')
        assert profile.get('max_cache_points') == 4
        assert profile.get('supports_auto_cache', False) is False

    def test_unsupported_model_has_no_cache_flags(self):
        profile = BedrockProvider.model_profile('meta.llama3-70b-instruct-v1:0')
        assert profile is not None
        assert profile.get('supports_cache', False) is False


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestOpenRouterCacheTranslation:
    def test_cache_true_translates_to_stable_boundaries(self):
        params = ModelRequestParameters(cache=True)
        result: dict[str, Any] = dict(_openrouter_settings_to_openai_settings(OpenRouterModelSettings(), params))
        assert result.get('openrouter_cache_instructions') == '5m'
        assert result.get('openrouter_cache_tool_definitions') == '5m'

    def test_cache_retention_forwarded(self):
        params = ModelRequestParameters(cache='1h')
        result: dict[str, Any] = dict(_openrouter_settings_to_openai_settings(OpenRouterModelSettings(), params))
        assert result.get('openrouter_cache_instructions') == '1h'
        assert result.get('openrouter_cache_tool_definitions') == '1h'

    def test_explicit_provider_setting_wins(self):
        params = ModelRequestParameters(cache=True)
        settings = OpenRouterModelSettings(openrouter_cache_messages='1h')
        result: dict[str, Any] = dict(_openrouter_settings_to_openai_settings(settings, params))
        assert result.get('openrouter_cache_messages') == '1h'
        assert 'openrouter_cache_instructions' not in result
        assert 'openrouter_cache_tool_definitions' not in result

    @pytest.mark.parametrize(
        ('model_name', 'supports_cache', 'retentions'),
        [
            pytest.param('anthropic/claude-sonnet-4.5', True, ('5m', '1h'), id='anthropic-downstream'),
            pytest.param('google/gemini-2.5-flash', True, ('5m',), id='google-downstream'),
            pytest.param('openai/gpt-5', False, None, id='openai-downstream'),
        ],
    )
    def test_provider_profile_flags(
        self, model_name: str, supports_cache: bool, retentions: tuple[CacheRetention, ...] | None
    ):
        profile = OpenRouterProvider.model_profile(model_name)
        assert profile is not None
        assert profile.get('supports_cache', False) is supports_cache
        if retentions is not None:
            assert profile.get('supported_cache_retentions') == retentions


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestOpenAICacheSupport:
    def test_provider_profile_declares_automatic_caching(self):
        profile = OpenAIProvider.model_profile('gpt-5')
        assert profile is not None
        assert profile.get('supports_cache') is True
        assert profile.get('supports_auto_cache') is True


@pytest.mark.skipif(not google_imports(), reason='google not installed')
class TestGoogleCacheWarning:
    def _model(self) -> GoogleModel:
        return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test'))

    def test_cache_without_cached_content_warns(self):
        with pytest.warns(UserWarning, match='The unified `cache` setting adds nothing to a Google request'):
            self._model().prepare_request(ModelSettings(cache=True), ModelRequestParameters())

    def test_cache_with_cached_content_does_not_warn(self):
        settings: ModelSettings = {'cache': True, 'google_cached_content': 'cachedContents/foo'}
        self._model().prepare_request(settings, ModelRequestParameters())

    def test_no_cache_setting_does_not_warn(self):
        self._model().prepare_request(ModelSettings(), ModelRequestParameters())


class TestResolvePromptCacheRetentionUnified:
    def test_function_model_unified_retention(self):
        model = _make_model(supports_cache=True, supported_cache_retentions=('5m', '1h'))
        assert model.resolve_prompt_cache_retention(ModelSettings(cache='1h')) == timedelta(hours=1)
        assert model.resolve_prompt_cache_retention(ModelSettings(cache=True)) == timedelta(minutes=5)
        assert model.resolve_prompt_cache_retention(ModelSettings(cache=False)) is None
        assert model.resolve_prompt_cache_retention(None) is None

    def test_unified_retention_snapped_before_resolution(self):
        model = _make_model(supports_cache=True, supported_cache_retentions=('5m',))
        assert model.resolve_prompt_cache_retention(ModelSettings(cache='1h')) == timedelta(minutes=5)

    def test_unsupported_model_resolves_none(self):
        model = _make_model(supports_cache=False)
        assert model.resolve_prompt_cache_retention(ModelSettings(cache='1h')) is None

    def test_wrapper_model_delegates_to_wrapped(self):
        from pydantic_ai.models.wrapper import WrapperModel

        model = _make_model(supports_cache=True, supported_cache_retentions=('5m', '1h'))
        assert WrapperModel(model).resolve_prompt_cache_retention(ModelSettings(cache='1h')) == timedelta(hours=1)

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    def test_anthropic_longest_wins_across_unified_and_provider_settings(self):
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test'))
        settings = AnthropicModelSettings(cache='1h', anthropic_cache_instructions='5m')
        assert model.resolve_prompt_cache_retention(settings) == timedelta(hours=1)
        assert model.resolve_prompt_cache_retention(AnthropicModelSettings(cache=True)) == timedelta(minutes=5)
