"""Tests for the unified `cache` prompt-caching setting.

Tests the base `Model.prepare_request()` cache resolution (on by default) and retention snap-down,
the per-provider translation onto provider-specific cache settings, the shared cache-point
budget and lookback helpers, the retention resolver, and the `Caching` capability.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._prompt_cache import (
    excess_cache_points,
    previous_tail_needing_breakpoint,
    snap_cache_retention,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
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
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider


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


async def test_cache_setting_end_to_end_run():
    """The setting survives a full agent run on a supporting model without reaching the model function."""
    result = await Agent(_make_model(supports_cache=True), model_settings={'cache': True}).run('hi')
    assert result.output == 'ok'


class TestSnapCacheRetention:
    @pytest.mark.parametrize(
        ('value', 'supported', 'expected'),
        [
            (True, ('5m', '1h'), True),
            ('5m', ('5m', '1h'), '5m'),
            ('1h', ('5m', '1h'), '1h'),
            ('30m', ('5m', '1h'), '5m'),
            ('1h', ('5m',), '5m'),
            ('30m', ('5m',), '5m'),
            ('5m', ('30m', '1h'), '30m'),
            ('5m', ('1h',), '1h'),
            ('5m', ('30m',), '30m'),
            ('1h', ('30m',), '30m'),
            # No retention tier to request: caching at the provider's default retention.
            ('1h', (), True),
        ],
    )
    def test_snap(
        self,
        value: Literal[True] | CacheRetention,
        supported: tuple[CacheRetention, ...],
        expected: Literal[True] | CacheRetention,
    ):
        assert snap_cache_retention(value, supported) == expected

    def test_unknown_retention_raises_user_error(self):
        """Runtime garbage (e.g. `'24h'`, valid for `openai_prompt_cache_retention` but not here)
        must fail with guidance, not a bare `ValueError` from tuple indexing."""
        with pytest.raises(UserError, match="Unknown `cache` retention '24h'"):
            snap_cache_retention('24h', ('5m',))  # type: ignore[arg-type]
        with pytest.raises(UserError, match='Unknown `cache` retention'):
            snap_cache_retention('24h', ())  # type: ignore[arg-type]


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

    def test_cache_false_resolved_as_false(self):
        model = _make_model(supports_cache=True)
        settings, cache = _resolve_cache(model, False)
        assert cache is False
        assert settings is None

    def test_cache_unset_by_default(self):
        """Without a `cache` setting, caching isn't configured."""
        settings, params = _make_model(supports_cache=True).prepare_request(None, ModelRequestParameters())
        assert params.cache is None
        assert settings is None

    def test_pre_resolved_cache_kept(self):
        """Parameters that already carry a resolved value (a caller that prepared them before) keep it."""
        model = _make_model(supports_cache=True, supported_cache_retentions=('5m', '1h'))
        _, params = model.prepare_request(None, ModelRequestParameters(cache=False))
        assert params.cache is False
        _, params = model.prepare_request(None, ModelRequestParameters(cache='1h'))
        assert params.cache == '1h'

    def test_cache_dropped_without_supports_cache(self):
        model = _make_model(supports_cache=False)
        settings, cache = _resolve_cache(model, True)
        assert cache is None
        assert settings is None

    def test_cache_stripped_but_other_settings_kept(self):
        model = _make_model(supports_cache=True)
        settings, params = model.prepare_request(ModelSettings(cache=True, temperature=0.5), ModelRequestParameters())
        assert settings == {'temperature': 0.5}
        assert params.cache is True

    def test_prepare_request_does_not_mutate_model_settings(self):
        model = _make_model(supports_cache=True)
        original = ModelSettings(cache='1h', temperature=0.5)
        model.prepare_request(original, ModelRequestParameters())
        assert original == {'cache': '1h', 'temperature': 0.5}

    def test_cache_in_model_default_settings(self):
        """`cache` set on the model constructor resolves with no per-request settings at all.

        This pins that resolution happens after `prepare_request` merges the model's own default
        settings; resolving before the merge would silently drop model-level `cache` defaults.
        """
        profile = ModelProfile(supports_cache=True, supported_cache_retentions=('5m', '1h'))
        model = FunctionModel(_echo, profile=profile, settings=ModelSettings(cache='1h'))
        settings, params = model.prepare_request(None, ModelRequestParameters())
        assert params.cache == '1h'
        assert settings is None

    def test_run_level_cache_false_overrides_model_default(self):
        model = FunctionModel(_echo, profile=ModelProfile(supports_cache=True), settings=ModelSettings(cache=True))
        settings, params = model.prepare_request(ModelSettings(cache=False), ModelRequestParameters())
        assert params.cache is False
        assert settings is None


class TestPreviousTailNeedingBreakpoint:
    def test_first_request_needs_none(self):
        assert previous_tail_needing_breakpoint(['user'], [30]) is None

    def test_narrow_turn_needs_none(self):
        assert previous_tail_needing_breakpoint(['user', 'assistant', 'user'], [1, 4, 3]) is None

    def test_wide_turn_marks_previous_tail(self):
        # 1 text + 11 tool calls, then 11 tool results: 23 blocks past the previous breakpoint.
        assert previous_tail_needing_breakpoint(['user', 'assistant', 'user'], [1, 12, 11]) == 0

    def test_history_ending_with_assistant_needs_none(self):
        """With a trailing assistant turn, the previous user turn is already the one that gets the breakpoint."""
        assert previous_tail_needing_breakpoint(['user', 'assistant'], [1, 30]) is None

    def test_assistant_without_preceding_user(self):
        assert previous_tail_needing_breakpoint(['assistant', 'user'], [20, 20]) is None


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
            excess_cache_points([], max_points=4, reserved=5, is_cache_point=lambda b: True, description='test request')


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
class TestAnthropicCacheTranslation:
    def _model(self, client: Any | None = None) -> AnthropicModel:
        provider = AnthropicProvider(api_key='test') if client is None else AnthropicProvider(anthropic_client=client)
        return AnthropicModel('claude-sonnet-4-5', provider=provider)

    def test_cache_true_uses_automatic_caching(self):
        settings, params = self._model().prepare_request(ModelSettings(cache=True), ModelRequestParameters())
        assert settings == {'anthropic_cache': '5m'}
        assert params.cache is True

    def test_cache_unset_sends_nothing(self):
        settings, params = self._model().prepare_request(None, ModelRequestParameters())
        assert settings is None
        assert params.cache is None

    def test_cache_false_sends_nothing(self):
        settings, params = self._model().prepare_request(ModelSettings(cache=False), ModelRequestParameters())
        assert settings is None
        assert params.cache is False

    def test_bedrock_client_uses_breakpoints(self):
        client = AsyncAnthropicBedrock(aws_access_key='x', aws_secret_key='y', aws_region='us-east-1')
        model = self._model(client)
        assert model.profile.get('supports_auto_cache') is False
        settings, _ = model.prepare_request(ModelSettings(cache='1h'), ModelRequestParameters())
        assert settings == {
            'anthropic_cache_instructions': '1h',
            'anthropic_cache_tool_definitions': '1h',
            'anthropic_cache_messages': '1h',
        }

    def test_explicit_provider_setting_wins(self):
        settings, params = self._model().prepare_request(
            AnthropicModelSettings(cache=True, anthropic_cache_instructions='1h'), ModelRequestParameters()
        )
        assert settings == {'anthropic_cache_instructions': '1h'}
        assert params.cache is True

    def test_falsy_explicit_provider_setting_still_wins(self):
        """Precedence is presence-based: `anthropic_cache_instructions=False` is how a user
        disables caching at one seam while an org-wide `cache=True` stands, so it must
        suppress the unified translation entirely, not just override one key."""
        settings, params = self._model().prepare_request(
            AnthropicModelSettings(cache=True, anthropic_cache_instructions=False), ModelRequestParameters()
        )
        assert settings == {'anthropic_cache_instructions': False}
        assert params.cache is True
        model = self._model()
        assert model.resolve_cache_retention(AnthropicModelSettings(cache='1h', anthropic_cache=False)) is None

    def test_explicit_setting_in_model_defaults_wins_over_run_level_unified(self):
        """The presence check runs on the merged settings, so an explicit setting in the model's
        default settings also suppresses a per-run unified `cache`."""
        model = AnthropicModel(
            'claude-sonnet-4-5',
            provider=AnthropicProvider(api_key='test'),
            settings=AnthropicModelSettings(anthropic_cache_instructions='1h'),
        )
        settings, params = model.prepare_request(ModelSettings(cache=True), ModelRequestParameters())
        assert settings == {'anthropic_cache_instructions': '1h'}
        assert params.cache is True

    def test_explicit_cache_messages_prevents_automatic_caching_conflict(self):
        """`anthropic_cache_messages` cannot be combined with `anthropic_cache`, so the unified
        value must not inject the automatic setting alongside it."""
        settings, params = self._model().prepare_request(
            AnthropicModelSettings(cache=True, anthropic_cache_messages=True), ModelRequestParameters()
        )
        assert settings == {'anthropic_cache_messages': True}
        assert params.cache is True

    def test_profile_without_auto_cache_uses_stable_boundaries(self):
        """A profile that disclaims automatic caching gets library-placed breakpoints instead."""
        model = AnthropicModel(
            'claude-sonnet-4-5',
            provider=AnthropicProvider(api_key='test'),
            profile=ModelProfile(supports_auto_cache=False),
        )
        settings, _ = model.prepare_request(ModelSettings(cache=True), ModelRequestParameters())
        assert settings == {
            'anthropic_cache_instructions': '5m',
            'anthropic_cache_tool_definitions': '5m',
            'anthropic_cache_messages': '5m',
        }


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
class TestBedrockCacheTranslation:
    def _model(self, bedrock_provider: BedrockProvider) -> BedrockConverseModel:
        return BedrockConverseModel('anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)

    def test_cache_true_translates_to_breakpoints(self, bedrock_provider: BedrockProvider):
        """`True` stays `True` in the injected settings so no explicit `ttl` reaches the wire."""
        settings, params = self._model(bedrock_provider).prepare_request(
            ModelSettings(cache=True), ModelRequestParameters()
        )
        assert settings == {
            'bedrock_cache_instructions': True,
            'bedrock_cache_tool_definitions': True,
            'bedrock_cache_messages': True,
        }
        assert params.cache is True

    def test_cache_retention_snaps_before_translation(self, bedrock_provider: BedrockProvider):
        """On a model AWS grants only the 5-minute TTL, `'1h'` must snap to `'5m'` before translation:
        Bedrock forwards a string retention as the `cachePoint` `ttl`, so an un-snapped value would
        produce a runtime `ValidationException`."""
        model = BedrockConverseModel('anthropic.claude-3-7-sonnet-20250219-v1:0', provider=bedrock_provider)
        settings, params = model.prepare_request(ModelSettings(cache='1h'), ModelRequestParameters())
        assert settings == {
            'bedrock_cache_instructions': '5m',
            'bedrock_cache_tool_definitions': '5m',
            'bedrock_cache_messages': '5m',
        }
        assert params.cache == '5m'

    def test_one_hour_retention_forwarded_on_supporting_model(self, bedrock_provider: BedrockProvider):
        settings, params = self._model(bedrock_provider).prepare_request(
            ModelSettings(cache='1h'), ModelRequestParameters()
        )
        assert settings == {
            'bedrock_cache_instructions': '1h',
            'bedrock_cache_tool_definitions': '1h',
            'bedrock_cache_messages': '1h',
        }
        assert params.cache == '1h'

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    def test_anthropic_model_on_bedrock_client_matches(self):
        """`AnthropicModel` on an `AsyncAnthropicBedrock` client uses the same per-model list."""
        client = AsyncAnthropicBedrock(aws_access_key='x', aws_secret_key='y', aws_region='us-east-1')
        provider = AnthropicProvider(anthropic_client=client)
        one_hour = AnthropicModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=provider)
        five_minutes = AnthropicModel('anthropic.claude-3-7-sonnet-20250219-v1:0', provider=provider)
        assert one_hour.profile.get('supported_cache_retentions') == ('5m', '1h')
        assert five_minutes.profile.get('supported_cache_retentions') == ('5m',)
        settings, _ = five_minutes.prepare_request(ModelSettings(cache='1h'), ModelRequestParameters())
        assert settings == {
            'anthropic_cache_instructions': '5m',
            'anthropic_cache_tool_definitions': '5m',
            'anthropic_cache_messages': '5m',
        }

    def test_explicit_provider_setting_wins(self, bedrock_provider: BedrockProvider):
        settings, params = self._model(bedrock_provider).prepare_request(
            BedrockModelSettings(cache=True, bedrock_cache_messages=True), ModelRequestParameters()
        )
        assert settings == {'bedrock_cache_messages': True}
        assert params.cache is True

    def test_falsy_explicit_provider_setting_still_wins(self, bedrock_provider: BedrockProvider):
        """Precedence is presence-based: an explicit `False` disables caching entirely rather
        than letting the unified value re-enable it."""
        settings, params = self._model(bedrock_provider).prepare_request(
            BedrockModelSettings(cache=True, bedrock_cache_instructions=False), ModelRequestParameters()
        )
        assert settings == {'bedrock_cache_instructions': False}
        assert params.cache is True


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestOpenRouterCacheTranslation:
    def _model(self) -> OpenRouterModel:
        return OpenRouterModel('anthropic/claude-sonnet-4.5', provider=OpenRouterProvider(api_key='test'))

    def test_cache_true_translates_to_stable_boundaries(self):
        settings, params = self._model().prepare_request(ModelSettings(cache=True), ModelRequestParameters())
        assert settings == snapshot(
            {
                'openrouter_cache_instructions': '5m',
                'openrouter_cache_tool_definitions': '5m',
                'openrouter_cache_messages': '5m',
                'extra_body': {},
            }
        )
        assert params.cache is True

    def test_cache_retention_forwarded(self):
        settings, params = self._model().prepare_request(ModelSettings(cache='1h'), ModelRequestParameters())
        assert settings == snapshot(
            {
                'openrouter_cache_instructions': '1h',
                'openrouter_cache_tool_definitions': '1h',
                'openrouter_cache_messages': '1h',
                'extra_body': {},
            }
        )
        assert params.cache == '1h'

    def test_cache_retention_snapped_on_gemini_downstream(self):
        """Gemini on OpenRouter has no TTL to request, so `'1h'` caches at its default retention."""
        model = OpenRouterModel('google/gemini-2.5-flash', provider=OpenRouterProvider(api_key='test'))
        settings, params = model.prepare_request(ModelSettings(cache='1h'), ModelRequestParameters())
        assert settings == snapshot(
            {
                'openrouter_cache_instructions': '5m',
                'openrouter_cache_tool_definitions': '5m',
                'openrouter_cache_messages': '5m',
                'extra_body': {},
            }
        )
        assert params.cache is True

    def test_explicit_provider_setting_wins(self):
        settings, params = self._model().prepare_request(
            OpenRouterModelSettings(cache=True, openrouter_cache_messages='1h'), ModelRequestParameters()
        )
        result: dict[str, Any] = dict(settings or {})
        assert result.get('openrouter_cache_messages') == '1h'
        assert 'openrouter_cache_instructions' not in result
        assert 'openrouter_cache_tool_definitions' not in result
        assert params.cache is True

    def test_falsy_explicit_provider_setting_still_wins(self):
        """Precedence is presence-based: an explicit `False` disables caching entirely rather
        than letting the unified value re-enable it."""
        settings, params = self._model().prepare_request(
            OpenRouterModelSettings(cache=True, openrouter_cache_instructions=False), ModelRequestParameters()
        )
        result: dict[str, Any] = dict(settings or {})
        assert result.get('openrouter_cache_instructions') is False
        assert 'openrouter_cache_tool_definitions' not in result
        assert params.cache is True


class TestResolveCacheRetentionUnified:
    def test_function_model_unified_retention(self):
        model = _make_model(supports_cache=True, supported_cache_retentions=('5m', '1h'))
        assert model.resolve_cache_retention(ModelSettings(cache='1h')) == timedelta(hours=1)
        assert model.resolve_cache_retention(ModelSettings(cache=True)) == timedelta(minutes=5)
        assert model.resolve_cache_retention(ModelSettings(cache=False)) is None
        assert model.resolve_cache_retention(None) is None

    def test_30m_retention_on_supporting_profile(self):
        model = _make_model(supports_cache=True, supported_cache_retentions=('5m', '30m'))
        assert model.resolve_cache_retention(ModelSettings(cache='30m')) == timedelta(minutes=30)

    def test_unified_retention_snapped_before_resolution(self):
        model = _make_model(supports_cache=True, supported_cache_retentions=('5m',))
        assert model.resolve_cache_retention(ModelSettings(cache='1h')) == timedelta(minutes=5)

    def test_unsupported_model_resolves_none(self):
        model = _make_model(supports_cache=False)
        assert model.resolve_cache_retention(ModelSettings(cache='1h')) is None

    def test_fallback_model_resolves_none(self):
        model = FallbackModel(_make_model(supports_cache=True))
        assert model.resolve_cache_retention(ModelSettings(cache='1h')) is None

    @pytest.mark.skipif(not google_imports(), reason='google not installed')
    def test_google_model_resolves_none(self):
        model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test'))
        assert model.resolve_cache_retention(ModelSettings(cache=True)) is None

    def test_wrapper_model_delegates_to_wrapped(self):
        model = _make_model(supports_cache=True, supported_cache_retentions=('5m', '1h'))
        assert WrapperModel(model).resolve_cache_retention(ModelSettings(cache='1h')) == timedelta(hours=1)

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    def test_anthropic_explicit_settings_shadow_unified_value(self):
        """Retention mirrors translation precedence: an explicit `anthropic_cache*` setting makes
        the unified value contribute nothing, since it also adds nothing to the request."""
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test'))
        settings = AnthropicModelSettings(cache='1h', anthropic_cache_instructions='5m')
        assert model.resolve_cache_retention(settings) == timedelta(minutes=5)
        assert model.resolve_cache_retention(AnthropicModelSettings(cache=True)) == timedelta(minutes=5)
        assert model.resolve_cache_retention(AnthropicModelSettings(cache='1h')) == timedelta(hours=1)


class _RecordingFunctionModel(FunctionModel):
    """Records the `params.cache` each `prepare_request` resolves, to observe what a wrapped
    model inside a fallback chain actually received."""

    recorded_cache: list[CacheSetting | None]

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        settings, params = super().prepare_request(model_settings, model_request_parameters)
        self.recorded_cache.append(params.cache)
        return settings, params


async def test_fallback_model_snaps_cache_per_wrapped_model():
    """`FallbackModel` passes the original settings through, so each wrapped model snaps the
    unified retention against its own profile; a chain with mixed retention support resolves
    per model rather than using the first model's tiers."""

    def _fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=500, model_name='primary')

    primary = _RecordingFunctionModel(
        _fail, profile=ModelProfile(supports_cache=True, supported_cache_retentions=('5m', '1h'))
    )
    primary.recorded_cache = []
    secondary = _RecordingFunctionModel(
        _echo, profile=ModelProfile(supports_cache=True, supported_cache_retentions=('5m',))
    )
    secondary.recorded_cache = []

    result = await Agent(FallbackModel(primary, secondary), model_settings={'cache': '1h'}).run('hi')

    assert result.output == 'ok'
    # `prepare_request` may run more than once per attempt (FallbackModel prepares the winning
    # model again for its span attributes); every resolution must agree per model.
    assert set(primary.recorded_cache) == {'1h'}
    assert set(secondary.recorded_cache) == {'5m'}


class TestCachingCapability:
    @pytest.mark.parametrize('retention', [True, False, '1h'])
    async def test_caching_sets_unified_cache(self, retention: CacheSetting):
        """`Caching` contributes the unified `cache` setting, which each model resolves against its profile."""
        model = _RecordingFunctionModel(
            _echo, profile=ModelProfile(supports_cache=True, supported_cache_retentions=('5m', '1h'))
        )
        model.recorded_cache = []

        await Agent(model, capabilities=[Caching(retention)]).run('hi')

        assert set(model.recorded_cache) == {retention}

    def test_caching_defaults(self):
        caching = Caching()
        assert caching.retention is True
        assert caching.id == 'caching'
        assert caching.get_model_settings() == {'cache': True}

    def test_caching_from_spec(self):
        agent = Agent.from_spec({'model': 'test', 'capabilities': [{'Caching': {'retention': '1h'}}]})
        children = agent._root_capability.capabilities
        assert [child for child in children if isinstance(child, Caching)] == [Caching('1h')]


class TestCachingNotEnabled:
    def test_unset_is_not_enabled(self):
        model = _make_model(supports_cache=True)
        assert model._caching_not_enabled(None)
        assert not model._caching_not_enabled(ModelSettings(cache=False))
        assert not model._caching_not_enabled(ModelSettings(cache=True))

    def test_unsupported_model_never_reported(self):
        assert not _make_model(supports_cache=False)._caching_not_enabled(None)

    @pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
    def test_provider_setting_counts_as_configured(self):
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test'))
        assert model._caching_not_enabled(None)
        assert WrapperModel(model)._caching_not_enabled(None)
        assert not model._caching_not_enabled(AnthropicModelSettings(anthropic_cache_instructions=False))

    @pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
    def test_bedrock_provider_setting_counts_as_configured(self, bedrock_provider: BedrockProvider):
        model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
        assert model._caching_not_enabled(None)
        assert not model._caching_not_enabled(BedrockModelSettings(bedrock_cache_messages=False))

    @pytest.mark.skipif(not openai_imports(), reason='openai not installed')
    def test_openai_chat_provider_setting_counts_as_configured(self):
        model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(api_key='test'))
        assert model._caching_not_enabled(None)
        assert not model._caching_not_enabled(OpenAIChatModelSettings(openai_cache_instructions=False))
        assert model.resolve_cache_retention(OpenAIChatModelSettings(openai_cache_instructions=False)) == timedelta(
            minutes=30
        )

    @pytest.mark.skipif(not openai_imports(), reason='openai not installed')
    def test_openrouter_provider_setting_counts_as_configured(self):
        model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=OpenRouterProvider(api_key='test'))
        assert model._caching_not_enabled(None)
        assert not model._caching_not_enabled(OpenRouterModelSettings(openrouter_cache_messages=False))
