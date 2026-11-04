"""Tests for the unified thinking/reasoning feature.

Tests the base Model.prepare_request() thinking resolution, per-provider translation,
the Thinking capability, and end-to-end integration via FunctionModel.
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false
from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, get_args
from unittest.mock import MagicMock

import pytest

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import CAPABILITY_TYPES, AbstractCapability, SelectModel, Thinking
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.anthropic import AnthropicModelProfile, resolve_anthropic_effort
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.google import GoogleModelProfile, google_model_profile
from pydantic_ai.profiles.grok import GrokModelProfile, grok_model_profile
from pydantic_ai.profiles.groq import GROQ_GPT_OSS_REASONING_EFFORT_MAP, groq_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, OpenAIReasoningEffort, openai_model_profile
from pydantic_ai.settings import ModelSettings, ThinkingEffort, ThinkingLevel
from pydantic_ai.tools import ToolDefinition

from ._inline_snapshot import snapshot
from .conftest import try_import
from .models.test_model_settings_support import (
    PROBE_KEY,
    Probe,
    botocore_probe,
    http_probe,
    legacy_http_probe,
    xai_sdk_probe,
)

with try_import() as anthropic_imports:
    from anthropic import omit as anthropic_omit
    from anthropic.types.beta import BetaThinkingConfigParam

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_imports:
    from openai import AsyncOpenAI, omit as openai_omit
    from openai.types.chat import ChatCompletionMessage

    from pydantic_ai.models.cerebras import (
        CerebrasModel,
        CerebrasModelSettings,
        _cerebras_settings_to_openai_settings,
    )
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.models.openrouter import (
        _OPENROUTER_EFFORT_MAP,
        OpenRouterModel,
        OpenRouterModelSettings,
        _openrouter_settings_to_openai_settings,
    )
    from pydantic_ai.models.snowflake import (
        _REASONING_EFFORT_MAP as _SNOWFLAKE_REASONING_EFFORT_MAP,
        SnowflakeModel,
        SnowflakeModelSettings,
    )
    from pydantic_ai.models.zai import ZaiModel, _zai_settings_to_openai_settings
    from pydantic_ai.profiles.zai import _GLM_5_3_REASONING_EFFORT_MAPPING
    from pydantic_ai.providers.cerebras import CerebrasProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.snowflake import SnowflakeProvider
    from pydantic_ai.providers.zai import ZaiProvider

    from .models.mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as groq_imports:
    from groq import NOT_GIVEN as groq_NOT_GIVEN

    from pydantic_ai.models.groq import GroqModel, GroqModelSettings
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as bedrock_imports:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider

with try_import() as xai_imports:
    from pydantic_ai.models.xai import XaiModel, XaiModelSettings, _map_reasoning_effort
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as mistral_imports:
    from pydantic_ai.models.mistral import _MISTRAL_REASONING_EFFORT_MAP

with try_import() as cohere_imports:
    from pydantic_ai.models.cohere import _THINKING_TOKEN_BUDGETS

pytestmark = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Shared echo function for FunctionModel instances."""
    return ModelResponse(parts=[TextPart(content='ok')])


def _make_model(
    *,
    supports_thinking: bool = False,
    thinking_always_enabled: bool = False,
) -> FunctionModel:
    """Create a FunctionModel with a specific thinking profile."""
    return FunctionModel(
        _echo,
        profile=ModelProfile(
            supports_thinking=supports_thinking,
            thinking_always_enabled=thinking_always_enabled,
        ),
    )


def _resolve_thinking(
    model: FunctionModel,
    thinking: ThinkingLevel | None = None,
) -> ThinkingLevel | None:
    """Call prepare_request and return the resolved params.thinking value."""
    settings: ModelSettings | None = ModelSettings(thinking=thinking) if thinking is not None else None
    params = ModelRequestParameters()
    _settings, resolved = model.prepare_request(settings, params)
    return resolved.thinking


# ---------------------------------------------------------------------------
# 1. Base class thinking resolution tests (in prepare_request())
# ---------------------------------------------------------------------------


class TestPrepareRequestThinkingResolution:
    def test_thinking_true_with_supports_thinking(self):
        model = _make_model(supports_thinking=True)
        assert _resolve_thinking(model, thinking=True) is True

    def test_thinking_effort_level_with_supports_thinking(self):
        model = _make_model(supports_thinking=True)
        assert _resolve_thinking(model, thinking='high') == 'high'

    def test_thinking_true_without_supports_thinking(self):
        """Models that don't support thinking silently ignore the setting."""
        model = _make_model(supports_thinking=False)
        assert _resolve_thinking(model, thinking=True) is None

    def test_thinking_false_with_always_enabled(self):
        """`False` reaches the adapter on always-on models, which sends the least thinking the model accepts."""
        model = _make_model(thinking_always_enabled=True)
        assert _resolve_thinking(model, thinking=False) is False

    def test_thinking_effort_with_always_enabled(self):
        """Effort levels pass through even on always-on models."""
        model = _make_model(thinking_always_enabled=True)
        assert _resolve_thinking(model, thinking='medium') == 'medium'

    def test_no_thinking_in_settings(self):
        """When thinking is not set in settings, params.thinking stays None."""
        model = _make_model(supports_thinking=True)
        assert _resolve_thinking(model, thinking=None) is None

    @pytest.mark.parametrize('effort', ['low', 'medium', 'high'])
    def test_all_effort_levels_pass_through(self, effort: Literal['low', 'medium', 'high']):
        model = _make_model(supports_thinking=True)
        assert _resolve_thinking(model, thinking=effort) == effort

    def test_thinking_true_with_always_enabled(self):
        """thinking=True also passes through on always-on models."""
        model = _make_model(thinking_always_enabled=True)
        assert _resolve_thinking(model, thinking=True) is True

    def test_thinking_false_without_supports_thinking(self):
        """thinking=False on unsupported model -> silently ignored."""
        model = _make_model(supports_thinking=False)
        assert _resolve_thinking(model, thinking=False) is None


# ---------------------------------------------------------------------------
# 2. Per-provider translation tests
# ---------------------------------------------------------------------------


def _anthropic_model(profile: AnthropicModelProfile) -> AnthropicModel:
    return AnthropicModel('claude-opus-4-7', provider=AnthropicProvider(api_key='mock-api-key'), profile=profile)


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
class TestAnthropicThinkingTranslation:
    """Test Anthropic _translate_thinking and _build_output_config translation."""

    @pytest.fixture
    def adaptive_model(self):
        return FunctionModel(
            _echo,
            profile=AnthropicModelProfile(
                supports_thinking=True,
                anthropic_supports_adaptive_thinking=True,
            ),
        )

    @pytest.fixture
    def non_adaptive_model(self):
        return FunctionModel(
            _echo,
            profile=AnthropicModelProfile(
                supports_thinking=True,
                anthropic_supports_adaptive_thinking=False,
            ),
        )

    def test_thinking_true_adaptive(self, adaptive_model: FunctionModel):
        """thinking=True with adaptive model -> {'type': 'adaptive'}."""
        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == snapshot({'type': 'adaptive'})

    def test_thinking_true_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking=True with non-adaptive model -> {'type': 'enabled', 'budget_tokens': 10000}."""
        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 10000})

    def test_thinking_high_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking='high' with non-adaptive -> budget_tokens=16384."""
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 16384})

    def test_thinking_low_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking='low' with non-adaptive -> budget_tokens=2048."""
        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 2048})

    def test_thinking_false_returns_omit(self, adaptive_model: FunctionModel):
        """thinking=False -> OMIT (not sent to API)."""
        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result is anthropic_omit

    def test_thinking_false_disables_thinking_on_by_default(self):
        """thinking=False -> {'type': 'disabled'} on a model that thinks when `thinking` is omitted."""
        model = FunctionModel(
            _echo,
            profile=AnthropicModelProfile(
                supports_thinking=True, anthropic_supports_adaptive_thinking=True, thinking_enabled_by_default=True
            ),
        )
        params = ModelRequestParameters(thinking=False)
        result = AnthropicModel._translate_thinking(model, {}, params)
        assert result == snapshot({'type': 'disabled'})

    def test_thinking_false_between_tools(self):
        """thinking=False -> {'type': 'between_tools'} on a model that turns thinking off that way."""
        model = FunctionModel(
            _echo,
            profile=AnthropicModelProfile(
                supports_thinking=True,
                anthropic_supports_adaptive_thinking=True,
                thinking_enabled_by_default=True,
                anthropic_disables_thinking_with_between_tools=True,
            ),
        )
        params = ModelRequestParameters(thinking=False)
        result = AnthropicModel._translate_thinking(model, {}, params)
        assert result == snapshot({'type': 'between_tools'})

    def test_thinking_false_always_on_uses_lowest_effort(self):
        """thinking=False on a model that can't turn thinking off -> adaptive thinking at `low` effort."""
        model = _anthropic_model(
            AnthropicModelProfile(
                supports_thinking=True,
                thinking_always_enabled=True,
                thinking_enabled_by_default=True,
                anthropic_supports_adaptive_thinking=True,
                anthropic_supports_effort=True,
            ),
        )
        params = ModelRequestParameters(thinking=False)
        assert model._translate_thinking({}, params) == snapshot({'type': 'adaptive'})
        assert model._build_output_config(params, {}) == snapshot({'effort': 'low'})

    def test_thinking_false_always_on_without_effort_sends_nothing(self):
        """thinking=False on a model that can't turn thinking off and has no effort setting leaves its default."""
        model = _anthropic_model(
            AnthropicModelProfile(
                supports_thinking=True,
                thinking_always_enabled=True,
                thinking_enabled_by_default=True,
                anthropic_supports_adaptive_thinking=True,
                anthropic_supports_effort=False,
            ),
        )
        params = ModelRequestParameters(thinking=False)
        assert model._translate_thinking({}, params) is anthropic_omit
        assert model._build_output_config(params, {}) is None

    def test_thinking_none_returns_omit(self, adaptive_model: FunctionModel):
        """thinking=None -> OMIT (not sent to API)."""
        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result is anthropic_omit

    def test_provider_specific_takes_precedence(self, adaptive_model: FunctionModel):
        """anthropic_thinking set -> unified thinking ignored."""
        params = ModelRequestParameters(thinking=True)
        settings = {'anthropic_thinking': {'type': 'disabled'}}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == snapshot({'type': 'disabled'})

    def test_effort_level_on_output_config(self):
        """thinking='high' sets effort on output_config when model supports it."""
        model = _anthropic_model(
            AnthropicModelProfile(
                supports_thinking=True,
                anthropic_supports_effort=True,
            )
        )

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = model._build_output_config(params, settings)
        assert result == snapshot({'effort': 'high'})

    def test_output_config_no_effort_for_bool(self):
        """thinking=True does NOT set effort on output_config (only str values do)."""
        model = _anthropic_model(
            AnthropicModelProfile(
                supports_thinking=True,
                anthropic_supports_effort=True,
            )
        )

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = model._build_output_config(params, settings)
        assert result is None

    def test_adaptive_model_with_effort_level(self):
        """thinking='high' on adaptive+effort model uses adaptive thinking and output_config effort."""
        model = _anthropic_model(
            AnthropicModelProfile(
                supports_thinking=True,
                anthropic_supports_adaptive_thinking=True,
                anthropic_supports_effort=True,
            )
        )

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        # All truthy thinking values use adaptive on adaptive models
        thinking_param = AnthropicModel._translate_thinking(model, settings, params)
        assert thinking_param == snapshot({'type': 'adaptive'})

        # output_config: effort controls depth separately
        output_config = model._build_output_config(params, settings)
        assert output_config == snapshot({'effort': 'high'})

    def test_task_budget_coexists_with_effort(self):
        """Anthropic task budgets share the same output_config object as effort."""
        model = _anthropic_model(
            AnthropicModelProfile(
                supports_thinking=True,
                anthropic_supports_effort=True,
                anthropic_supports_task_budgets=True,
            )
        )

        params = ModelRequestParameters(thinking='high')
        settings = AnthropicModelSettings(anthropic_task_budget={'type': 'tokens', 'total': 2_000})
        output_config = model._build_output_config(params, settings)
        assert output_config == snapshot({'effort': 'high', 'task_budget': {'type': 'tokens', 'total': 2_000}})

    def test_medium_uses_adaptive(self, adaptive_model: FunctionModel):
        """thinking='medium' on adaptive model -> adaptive (not budget)."""
        params = ModelRequestParameters(thinking='medium')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == {'type': 'adaptive'}

    def test_low_uses_adaptive_on_adaptive(self, adaptive_model: FunctionModel):
        """thinking='low' on adaptive model -> adaptive (effort controlled via output_config)."""
        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == {'type': 'adaptive'}

    def test_high_uses_adaptive_on_adaptive(self, adaptive_model: FunctionModel):
        """thinking='high' on adaptive model -> adaptive (effort controlled via output_config)."""
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == {'type': 'adaptive'}


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestOpenAIChatThinkingTranslation:
    """Test OpenAI Chat model _translate_thinking translation."""

    def test_thinking_true(self):
        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}

        # We need a model-like object to call the method; use a FunctionModel with the right profile
        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result == 'medium'

    def test_thinking_high(self):
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result == 'high'

    def test_thinking_false(self):
        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result == 'none'

    def test_thinking_minimal_maps_to_low_for_gpt_5_6(self):
        """Not a VCR test: pins the pre-request fallback for an effort GPT-5.6 does not support."""
        params = ModelRequestParameters(thinking='minimal')
        settings: ModelSettings = {}
        model = FunctionModel(_echo, profile=openai_model_profile('gpt-5.6-sol'))

        result = OpenAIChatModel._translate_thinking(model, settings, params)

        assert result == 'low'

    def test_thinking_minimal_passes_through_when_supported(self):
        """Not a VCR test: pins the capability-on side using a model that supports minimal effort."""
        params = ModelRequestParameters(thinking='minimal')
        settings: ModelSettings = {}
        model = FunctionModel(_echo, profile=openai_model_profile('gpt-5'))

        result = OpenAIChatModel._translate_thinking(model, settings, params)

        assert result == 'minimal'

    def test_thinking_minimal_passes_through_with_sparse_profile(self):
        """Not a VCR test: pins the default for sparse OpenAI-compatible profiles."""
        params = ModelRequestParameters(thinking='minimal')
        settings: ModelSettings = {}
        model = FunctionModel(_echo, profile=OpenAIModelProfile())

        result = OpenAIChatModel._translate_thinking(model, settings, params)

        assert result == 'minimal'

    def test_thinking_none_returns_omit(self):
        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result is openai_omit

    def test_provider_specific_takes_precedence(self):
        params = ModelRequestParameters(thinking=True)
        settings = {'openai_reasoning_effort': 'low'}

        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result == 'low'

    def test_provider_specific_minimal_is_not_clamped(self):
        params = ModelRequestParameters(thinking=True)
        settings = {'openai_reasoning_effort': 'minimal'}
        model = FunctionModel(
            _echo,
            profile=OpenAIModelProfile(openai_supports_minimal_reasoning_effort=False),
        )

        result = OpenAIChatModel._translate_thinking(model, settings, params)

        assert result == 'minimal'

    @pytest.mark.parametrize(
        ('model_name', 'thinking', 'expected'),
        [
            # `thinking=False` on a model that can't stop reasoning sends its lowest effort.
            pytest.param('o3', False, 'low', id='o-series-false'),
            pytest.param('gpt-5', False, 'minimal', id='gpt-5-false'),
            pytest.param('gpt-5-pro', False, 'high', id='gpt-5-pro-false'),
            pytest.param('gpt-5.5-pro', False, 'medium', id='gpt-5.5-pro-false'),
            pytest.param('gpt-6-astra', False, 'low', id='gpt-6-astra-false'),
            pytest.param('gpt-5.3-chat-latest', False, 'medium', id='chat-latest-false'),
            # A model that can turn reasoning off still gets `'none'`.
            pytest.param('gpt-5.6-sol', False, 'none', id='can-disable-false'),
            pytest.param('gpt-6.1-terra', False, 'none', id='gpt-6.1-terra-false'),
            # Other levels resolve to the lowest accepted effort at or above them, else the highest.
            pytest.param('o3', 'minimal', 'low', id='o-series-minimal'),
            pytest.param('o3', 'xhigh', 'high', id='o-series-xhigh'),
            pytest.param('gpt-5-pro', True, 'high', id='gpt-5-pro-true'),
            pytest.param('gpt-5.5-pro', 'low', 'medium', id='gpt-5.5-pro-low'),
            pytest.param('gpt-5.5-pro', 'xhigh', 'xhigh', id='gpt-5.5-pro-xhigh'),
            pytest.param('gpt-6-astra', 'minimal', 'low', id='gpt-6-astra-minimal'),
            pytest.param('gpt-6.1-terra', 'minimal', 'low', id='gpt-6.1-terra-minimal'),
            # `'max'` resolves like `'xhigh'` where the API doesn't take `max` (OpenAI's Chat Completions never does).
            pytest.param('o3', 'max', 'high', id='o-series-max'),
            pytest.param('gpt-6-astra', 'max', 'xhigh', id='gpt-6-astra-max'),
            pytest.param('gpt-6.1-terra', 'max', 'xhigh', id='gpt-6.1-terra-max'),
        ],
    )
    def test_thinking_resolves_to_accepted_effort(self, model_name: str, thinking: ThinkingLevel, expected: str):
        """Not a VCR test: pins the resolution against each model's accepted efforts, which were verified live."""
        params = ModelRequestParameters(thinking=thinking)
        model = FunctionModel(_echo, profile=openai_model_profile(model_name))

        assert OpenAIChatModel._translate_thinking(model, {}, params) == expected

    def test_thinking_max_resolves_to_max_where_accepted(self):
        """An accepted `max` is sent where the API takes it, and skipped where it doesn't."""
        params = ModelRequestParameters(thinking='max')
        efforts = frozenset[OpenAIReasoningEffort](('low', 'high', 'max'))
        with_max = FunctionModel(
            _echo,
            profile=OpenAIModelProfile(
                openai_reasoning_efforts=efforts, openai_chat_supports_max_reasoning_effort=True
            ),
        )
        without_max = FunctionModel(_echo, profile=OpenAIModelProfile(openai_reasoning_efforts=efforts))

        assert OpenAIChatModel._translate_thinking(with_max, {}, params) == 'max'
        assert OpenAIChatModel._translate_thinking(without_max, {}, params) == 'high'

    def test_thinking_false_omitted_without_known_efforts_on_always_on_model(self):
        """An always-on model whose accepted efforts aren't known (e.g. Qwen3 Thinking on vLLM) keeps its default."""
        params = ModelRequestParameters(thinking=False)
        model = FunctionModel(_echo, profile=OpenAIModelProfile(supports_thinking=True, thinking_always_enabled=True))

        assert OpenAIChatModel._translate_thinking(model, {}, params) is openai_omit

    def test_thinking_omitted_without_effort_knob(self):
        """An empty set of accepted efforts means the model has no effort knob, so no level is sent."""
        model = FunctionModel(
            _echo,
            profile=OpenAIModelProfile(
                supports_thinking=True, thinking_always_enabled=True, openai_reasoning_efforts=frozenset()
            ),
        )

        assert OpenAIChatModel._translate_thinking(model, {}, ModelRequestParameters(thinking=False)) is openai_omit
        assert OpenAIChatModel._translate_thinking(model, {}, ModelRequestParameters(thinking='high')) is openai_omit

    def test_thinking_false_keeps_reasoning_active_on_always_on_model(self):
        """`thinking=False` on a model that can't stop reasoning still reasons, so it counts as thinking."""
        params = ModelRequestParameters(thinking=False)
        always_on = FunctionModel(_echo, profile=openai_model_profile('o3'))
        can_disable = FunctionModel(_echo, profile=openai_model_profile('gpt-5.6-sol'))

        assert OpenAIChatModel._request_thinks(always_on, {}, params) is True
        assert OpenAIChatModel._request_thinks(can_disable, {}, params) is False


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestOpenAIResponsesThinkingTranslation:
    """Test OpenAI Responses model _translate_thinking translation."""

    def test_thinking_true(self):
        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._translate_thinking(model, settings, params)
        assert result == snapshot({'effort': 'medium'})

    def test_thinking_high(self):
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._translate_thinking(model, settings, params)
        assert result == snapshot({'effort': 'high'})

    def test_thinking_false(self):
        """thinking=False -> reasoning_effort='none'."""
        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._translate_thinking(model, settings, params)
        # 'none' is falsy for dict truthiness check, but the effort_map maps False -> 'none'
        # which gets set as reasoning_effort. Then `if reasoning_effort:` is truthy for 'none'.
        assert result == snapshot({'effort': 'none'})

    def test_thinking_minimal_maps_to_low_for_gpt_5_6(self):
        """Not a VCR test: pins the pre-request fallback for an effort GPT-5.6 does not support."""
        params = ModelRequestParameters(thinking='minimal')
        settings: ModelSettings = {}
        model = FunctionModel(_echo, profile=openai_model_profile('gpt-5.6-sol'))

        result = OpenAIResponsesModel._translate_thinking(model, settings, params)

        assert result == snapshot({'effort': 'low', 'context': 'all_turns'})

    def test_thinking_minimal_passes_through_when_supported(self):
        """Not a VCR test: pins the capability-on side using a model that supports minimal effort."""
        params = ModelRequestParameters(thinking='minimal')
        settings: ModelSettings = {}
        model = FunctionModel(_echo, profile=openai_model_profile('gpt-5'))

        result = OpenAIResponsesModel._translate_thinking(model, settings, params)

        assert result == snapshot({'effort': 'minimal'})

    def test_thinking_minimal_passes_through_with_sparse_profile(self):
        """Not a VCR test: pins the default for sparse OpenAI-compatible profiles."""
        params = ModelRequestParameters(thinking='minimal')
        settings: ModelSettings = {}
        model = FunctionModel(_echo, profile=OpenAIModelProfile())

        result = OpenAIResponsesModel._translate_thinking(model, settings, params)

        assert result == snapshot({'effort': 'minimal'})

    def test_provider_specific_takes_precedence(self):
        params = ModelRequestParameters(thinking=True)
        settings = {'openai_reasoning_effort': 'high'}

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._translate_thinking(model, settings, params)
        assert result == snapshot({'effort': 'high'})

    def test_provider_specific_minimal_is_not_clamped(self):
        params = ModelRequestParameters(thinking=True)
        settings = {'openai_reasoning_effort': 'minimal'}
        model = FunctionModel(
            _echo,
            profile=OpenAIModelProfile(openai_supports_minimal_reasoning_effort=False),
        )

        result = OpenAIResponsesModel._translate_thinking(model, settings, params)

        assert result == snapshot({'effort': 'minimal'})


@pytest.mark.skipif(not google_imports(), reason='google-genai not installed')
class TestGoogleThinkingTranslation:
    """Test Google model _translate_thinking translation.

    Unit-pinned because cassette matchers aren't sensitive to the request body — asserting the
    translated config directly is what catches resolution drift; wire-level cases live in
    `tests/test_thinking_wire_contract.py`.
    """

    @pytest.fixture
    def default_google_model(self):
        """A model with unspecified thinking_level support (should default to Gemini 3+ behaviour)."""
        return FunctionModel(
            _echo,
            profile=GoogleModelProfile(
                supports_thinking=True,
            ),
        )

    def test_thinking_default_uses_thinking_level(self, default_google_model: FunctionModel):
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(default_google_model, settings, params)
        assert result == {'include_thoughts': True, 'thinking_level': 'HIGH'}

    def test_thinking_false_default_uses_minimal_level(self, default_google_model: FunctionModel):
        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(default_google_model, settings, params)
        assert result == {'thinking_level': 'MINIMAL'}

    @pytest.fixture
    def gemini_3_model(self):
        """A model with thinking_level support (Gemini 3+)."""
        return FunctionModel(
            _echo,
            profile=GoogleModelProfile(
                supports_thinking=True,
                google_supports_thinking_level=True,
            ),
        )

    @pytest.fixture
    def gemini_25_model(self):
        """A model with thinking_budget support (Gemini 2.5)."""
        return FunctionModel(
            _echo,
            profile=GoogleModelProfile(
                supports_thinking=True,
                google_supports_thinking_level=False,
            ),
        )

    @pytest.fixture
    def gemini_3_no_minimal_model(self):
        """A Gemini 3+ model without `minimal` thinking-level support (e.g. `gemini-3.7-flash`)."""
        return FunctionModel(
            _echo,
            profile=GoogleModelProfile(
                supports_thinking=True,
                google_supports_thinking_level=True,
                google_supports_minimal_thinking_level=False,
            ),
        )

    @pytest.fixture
    def flash_lite_image_model(self):
        """`gemini-3.1-flash-lite-image`: documented levels are only `minimal` and `high`."""
        return FunctionModel(
            _echo,
            profile=GoogleModelProfile(
                supports_thinking=True,
                google_supports_thinking_level=True,
                google_thinking_levels=frozenset({'MINIMAL', 'HIGH'}),
            ),
        )

    @pytest.fixture
    def low_high_model(self):
        """A model with a non-contiguous level set that skips `medium` (e.g. `gemini-3-pro-preview`)."""
        return FunctionModel(
            _echo,
            profile=GoogleModelProfile(
                supports_thinking=True,
                google_supports_thinking_level=True,
                google_thinking_levels=frozenset({'LOW', 'HIGH'}),
            ),
        )

    def test_thinking_true_gemini_3(self, gemini_3_model: FunctionModel):
        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True})

    def test_thinking_high_gemini_3(self, gemini_3_model: FunctionModel):
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'HIGH'})

    def test_thinking_low_gemini_3(self, gemini_3_model: FunctionModel):
        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'LOW'})

    def test_thinking_minimal_gemini_3(self, gemini_3_model: FunctionModel):
        params = ModelRequestParameters(thinking='minimal')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'MINIMAL'})

    def test_thinking_minimal_maps_to_low_when_unsupported(self, gemini_3_no_minimal_model: FunctionModel):
        params = ModelRequestParameters(thinking='minimal')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_no_minimal_model, settings, params)
        assert result is not None
        assert result.get('thinking_level') == 'LOW'

    def test_thinking_true_gemini_25(self, gemini_25_model: FunctionModel):
        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_25_model, settings, params)
        assert result == snapshot({'include_thoughts': True})

    def test_thinking_high_gemini_25(self, gemini_25_model: FunctionModel):
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_25_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_budget': 24576})

    def test_thinking_low_gemini_25(self, gemini_25_model: FunctionModel):
        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_25_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_budget': 2048})

    def test_thinking_false(self, gemini_3_model: FunctionModel):
        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'thinking_level': 'MINIMAL'})

    def test_thinking_false_maps_to_low_when_minimal_unsupported(self, gemini_3_no_minimal_model: FunctionModel):
        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_no_minimal_model, settings, params)
        assert result == snapshot({'thinking_level': 'LOW'})

    def test_thinking_low_snaps_to_minimal(self, flash_lite_image_model: FunctionModel):
        """`gemini-3.1-flash-lite-image` documents only `minimal`/`high`, so `low` snaps down."""
        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(flash_lite_image_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'MINIMAL'})

    def test_thinking_medium_snaps_to_high(self, flash_lite_image_model: FunctionModel):
        """`medium` is unsupported on `gemini-3.1-flash-lite-image` and snaps up to `high`."""
        params = ModelRequestParameters(thinking='medium')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(flash_lite_image_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'HIGH'})

    def test_thinking_xhigh_snaps_to_high(self, flash_lite_image_model: FunctionModel):
        params = ModelRequestParameters(thinking='xhigh')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(flash_lite_image_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'HIGH'})

    def test_thinking_false_maps_to_lowest_supported_level(self, flash_lite_image_model: FunctionModel):
        """`thinking=False` resolves through the lowest supported level, here `MINIMAL`."""
        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(flash_lite_image_model, settings, params)
        assert result == snapshot({'thinking_level': 'MINIMAL'})

    def test_thinking_medium_tie_rounds_down(self, low_high_model: FunctionModel):
        """Equidistant supported levels round down to the cheaper one."""
        params = ModelRequestParameters(thinking='medium')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(low_high_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'LOW'})

    def test_thinking_empty_levels_rejected(self):
        """An explicitly empty `google_thinking_levels` is malformed config, not 'no levels'."""
        model = FunctionModel(
            _echo,
            profile=GoogleModelProfile(
                supports_thinking=True,
                google_supports_thinking_level=True,
                google_thinking_levels=frozenset(),
            ),
        )
        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        with pytest.raises(UserError, match='must contain at least one level'):
            GoogleModel._translate_thinking(model, settings, params)

    def test_thinking_snaps_table_derived_levels(self):
        """The snap applies to a level set derived by `google_model_profile`, not just hand-built ones."""
        model = FunctionModel(_echo, profile=google_model_profile('gemini-3.7-flash'))
        params = ModelRequestParameters(thinking='minimal')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'LOW'})

        params_false = ModelRequestParameters(thinking=False)
        assert GoogleModel._translate_thinking(model, settings, params_false) == snapshot({'thinking_level': 'LOW'})

    def test_thinking_unknown_levels_rejected(self):
        """Levels the resolver can't order (e.g. lowercase misspellings) are rejected as config errors."""
        model = FunctionModel(
            _echo,
            profile=GoogleModelProfile(
                supports_thinking=True,
                google_supports_thinking_level=True,
                google_thinking_levels=frozenset({'minimal'}),
            ),
        )
        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        with pytest.raises(UserError, match='unknown levels'):
            GoogleModel._translate_thinking(model, settings, params)

    def test_thinking_false_gemini_25(self, gemini_25_model: FunctionModel):
        """thinking=False on Gemini 2.5 uses thinking_budget=0."""
        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_25_model, settings, params)
        assert result == snapshot({'thinking_budget': 0})

    def test_thinking_false_gemini_25_pro_uses_min_budget(self):
        """Gemini 2.5 Pro rejects `thinking_budget=0`, so `thinking=False` sends its minimum budget, without
        asking for the thoughts back so it stays distinct from `'minimal'`."""
        model = FunctionModel(_echo, profile=google_model_profile('gemini-2.5-pro'))
        params = ModelRequestParameters(thinking=False)
        result = GoogleModel._translate_thinking(model, {}, params)
        assert result == snapshot({'thinking_budget': 128})

    def test_thinking_false_always_on_gemini_3_uses_lowest_level(self):
        """Gemini 3.1 Pro can't turn thinking off and rejects `MINIMAL`, so `thinking=False` sends `LOW`."""
        model = FunctionModel(_echo, profile=google_model_profile('gemini-3.1-pro-preview'))
        params = ModelRequestParameters(thinking=False)
        result = GoogleModel._translate_thinking(model, {}, params)
        assert result == snapshot({'thinking_level': 'LOW'})

    def test_thinking_none(self, gemini_3_model: FunctionModel):
        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result is None

    def test_provider_specific_takes_precedence(self, gemini_3_model: FunctionModel):
        params = ModelRequestParameters(thinking=True)
        settings = {'google_thinking_config': {'include_thoughts': False}}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': False})


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
class TestGroqThinkingTranslation:
    """Test Groq model _translate_thinking translation."""

    def test_thinking_true(self):
        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params, False)
        assert result == 'parsed'

    def test_thinking_high(self):
        """Effort levels also translate to 'parsed' for Groq."""
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params, False)
        assert result == 'parsed'

    def test_thinking_false(self):
        """thinking=False -> 'hidden' (Groq has no true disable; 'hidden' suppresses output)."""
        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params, False)
        assert result == 'hidden'

    def test_thinking_false_always_on_without_effort_sends_nothing(self):
        """An always-on model with no effort knob (e.g. the legacy `deepseek-r1-distill`) has nothing to turn down."""
        params = ModelRequestParameters(thinking=False)
        model = FunctionModel(_echo, profile=groq_model_profile('deepseek-r1-distill-llama-70b'))
        result = GroqModel._translate_thinking(model, {}, params, False)
        assert result is groq_NOT_GIVEN

    def test_thinking_false_gpt_oss_hides_reasoning(self):
        """gpt-oss reasons at its lowest effort for `thinking=False`, with the reasoning kept out of the response."""
        params = ModelRequestParameters(thinking=False)
        model = FunctionModel(_echo, profile=groq_model_profile('openai/gpt-oss-120b'))
        result = GroqModel._translate_thinking(model, {}, params, False)
        assert result == 'hidden'

    def test_thinking_none(self):
        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params, False)
        assert result is groq_NOT_GIVEN

    def test_provider_specific_takes_precedence(self):
        params = ModelRequestParameters(thinking=True)
        settings = {'groq_reasoning_format': 'raw'}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params, False)
        assert result == 'raw'


def _thinking_settings(anthropic_thinking: BetaThinkingConfigParam | None) -> ModelSettings:
    """Provider-specific thinking when a config is given, unified `thinking='high'` otherwise.

    The settings are built here rather than in the `parametrize` decorators because
    `AnthropicModelSettings` is imported behind `try_import`, and a decorator argument is evaluated
    at collection time — before the class's skip mark can spare a shard that lacks the SDK.
    """
    if anthropic_thinking:
        return AnthropicModelSettings(anthropic_thinking=anthropic_thinking)
    return ModelSettings(thinking='high')


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
class TestAnthropicThinkingOutputToolsConflict:
    """Tool Output resolves to a forced `tool_choice`, which Anthropic rejects alongside extended thinking and
    answers without thinking alongside adaptive thinking, so a bare structured `output_type` switches to Native
    Output whenever the request thinks, including on models that think without a thinking setting.

    These are pre-request decisions, so no request is ever made and there is nothing to record. Real
    model names are used so the shipped profile flags — not hand-built ones — decide each case.
    """

    @pytest.fixture
    def output_tool_params(self) -> ModelRequestParameters:
        return ModelRequestParameters(
            output_tools=[ToolDefinition(name='output', description='', parameters_json_schema={}, kind='output')],
            output_object=OutputObjectDefinition(json_schema={'type': 'object', 'properties': {}}),
            output_mode='auto',
        )

    @pytest.mark.parametrize(
        'model_name,anthropic_thinking,expected_output_mode',
        [
            pytest.param(
                'claude-opus-4-6',
                None,
                'native',
                id='unified_thinking_on_adaptive_profile_switches_to_native',
            ),
            pytest.param(
                'claude-sonnet-4-5',
                None,
                'native',
                id='unified_thinking_on_non_adaptive_profile_switches_to_native',
            ),
            pytest.param(
                'claude-fable-5-1',
                None,
                'native',
                id='adaptive_profile_that_cannot_force_switches_to_native',
            ),
            pytest.param(
                'claude-opus-4-6',
                {'type': 'adaptive'},
                'native',
                id='explicit_adaptive_switches_to_native',
            ),
            pytest.param(
                'claude-opus-4-6',
                {'type': 'enabled', 'budget_tokens': 1024},
                'native',
                id='explicit_extended_thinking_switches_to_native',
            ),
        ],
    )
    def test_auto_output_mode(
        self,
        anthropic_api_key: str,
        output_tool_params: ModelRequestParameters,
        model_name: str,
        anthropic_thinking: BetaThinkingConfigParam | None,
        expected_output_mode: str,
    ):
        model = AnthropicModel(model_name, provider=AnthropicProvider(api_key=anthropic_api_key))

        _, resolved_params = model.prepare_request(_thinking_settings(anthropic_thinking), output_tool_params)

        assert resolved_params.output_mode == expected_output_mode

    @pytest.mark.parametrize(
        'model_name,thinking,expected_output_mode',
        [
            pytest.param('claude-opus-4-6', None, 'tool', id='opus_4_6_thinks_only_when_asked'),
            pytest.param('claude-opus-5', None, 'native', id='opus_5_thinks_by_default'),
            pytest.param('claude-sonnet-5', False, 'tool', id='sonnet_5_thinking_turned_off'),
            pytest.param('claude-opus-5-5', False, 'native', id='opus_5_5_cannot_turn_thinking_off'),
        ],
    )
    def test_auto_output_mode_follows_the_model_default(
        self,
        anthropic_api_key: str,
        output_tool_params: ModelRequestParameters,
        model_name: str,
        thinking: bool | None,
        expected_output_mode: str,
    ):
        model = AnthropicModel(model_name, provider=AnthropicProvider(api_key=anthropic_api_key))
        settings = ModelSettings() if thinking is None else ModelSettings(thinking=thinking)

        _, resolved_params = model.prepare_request(settings, output_tool_params)

        assert resolved_params.output_mode == expected_output_mode

    @pytest.mark.parametrize(
        'model_name,anthropic_thinking',
        [
            pytest.param('claude-fable-5-1', None, id='model_that_cannot_force'),
            pytest.param('claude-opus-4-6', {'type': 'enabled', 'budget_tokens': 1024}, id='extended_thinking'),
        ],
    )
    def test_explicit_tool_output_is_kept(
        self,
        anthropic_api_key: str,
        output_tool_params: ModelRequestParameters,
        model_name: str,
        anthropic_thinking: BetaThinkingConfigParam | None,
    ):
        """An explicit `ToolOutput` keeps Tool Output where forcing isn't available; the output tool is then
        offered with `tool_choice='auto'`, and a text response is retried."""
        settings = _thinking_settings(anthropic_thinking)
        model = AnthropicModel(model_name, provider=AnthropicProvider(api_key=anthropic_api_key))
        params = replace(output_tool_params, output_mode='tool', allow_text_output=False)

        _, resolved_params = model.prepare_request(settings, params)

        assert resolved_params.output_mode == 'tool'


def _bedrock_model(profile: ModelProfile) -> BedrockConverseModel:
    client = MagicMock()
    client.meta.endpoint_url = 'https://bedrock-runtime.us-east-1.amazonaws.com'
    return BedrockConverseModel('test-model', provider=BedrockProvider(bedrock_client=client), profile=profile)


@pytest.mark.skipif(not bedrock_imports(), reason='boto3 not installed')
class TestBedrockThinkingTranslation:
    """Test Bedrock thinking translation in `_build_additional_model_request_fields` for each variant."""

    def test_anthropic_variant_thinking_true(self):
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='anthropic',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result == {'thinking': {'type': 'enabled', 'budget_tokens': 10000}}

    def test_anthropic_variant_thinking_false(self):
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='anthropic',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result == {'thinking': {'type': 'disabled'}}

    @pytest.mark.parametrize(
        ('level', 'effort'),
        [(False, 'low'), ('minimal', 'low'), ('low', 'low'), (True, 'medium'), ('high', 'high'), ('xhigh', 'high')],
    )
    def test_openai_variant_effort_map(self, level: ThinkingLevel, effort: str):
        """gpt-oss accepts only `low`, `medium`, and `high`, so `thinking=False` and `'minimal'` send `low`."""
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='openai',
                supports_thinking=True,
                thinking_always_enabled=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=level)
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result == {'reasoning_effort': effort}

    def test_openai_variant_thinking_high(self):
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='openai',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result == {'reasoning_effort': 'high'}

    def test_qwen_variant_thinking_true(self):
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='qwen',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result == {'reasoning_config': 'high'}

    def test_qwen_variant_thinking_false(self):
        """thinking=False on Qwen sends `reasoning_config='none'`."""
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='qwen',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result == {'reasoning_config': 'none'}

    @pytest.mark.parametrize(
        ('thinking', 'expected_effort'),
        [
            (True, 'medium'),
            ('minimal', 'low'),
            ('low', 'low'),
            ('medium', 'medium'),
            ('high', 'high'),
            ('xhigh', 'xhigh'),
        ],
    )
    def test_xai_variant_thinking_levels(self, thinking: ThinkingLevel, expected_effort: str):
        """Grok on Converse takes effort nested as `reasoning.effort`, not the flat gpt-oss `reasoning_effort`.

        Not a VCR test: the recording account has no Grok access on Converse.
        """
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='xai',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=thinking)
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result == {'reasoning': {'effort': expected_effort}}

    def test_xai_variant_thinking_false(self):
        """thinking=False on the xAI Bedrock variant sends Grok's lowest effort, as it always reasons.

        Not a VCR test: the recording account has no Grok access on Converse.
        """
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='xai',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result == {'reasoning': {'effort': 'low'}}

    def test_xai_variant_thinking_true_keeps_model_default(self):
        """thinking=True on Grok 4.7 omits `reasoning` so its documented `high` default applies.

        Not a VCR test: the recording account has no Grok access on Converse.
        """
        profile = BedrockProvider.model_profile('us.xai.grok-4.7')
        assert profile is not None
        model = _bedrock_model(profile)

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result is None

    def test_xai_variant_user_reasoning_wins(self):
        """A user-supplied `reasoning` in `bedrock_additional_model_requests_fields` is left untouched.

        Not a VCR test: the recording account has no Grok access on Converse.
        """
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='xai',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings(bedrock_additional_model_requests_fields={'reasoning': {'effort': 'low'}})
        params = ModelRequestParameters(thinking='high')
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result == {'reasoning': {'effort': 'low'}}

    def test_no_variant_thinking_passthrough(self):
        """When bedrock_thinking_variant is None, unified thinking is a no-op."""
        model = _bedrock_model(BedrockModelProfile(bedrock_thinking_variant=None))

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = model._build_additional_model_request_fields(settings, params, [])
        # No variant set, so no thinking fields are added
        assert result is None

    def test_thinking_none_returns_existing(self):
        model = _bedrock_model(BedrockModelProfile(bedrock_thinking_variant='anthropic'))

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=None)
        result = model._build_additional_model_request_fields(settings, params, [])
        assert result is None

    def _adaptive_model(self, *, supports_effort: bool = True):
        return _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='anthropic',
                bedrock_supports_adaptive_thinking=True,
                bedrock_supports_effort=supports_effort,
                supports_thinking=True,
            )
        )

    def test_anthropic_variant_adaptive_thinking_true(self):
        """Bare thinking=True enables adaptive but doesn't set effort (matches AnthropicModel)."""
        model = self._adaptive_model()
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking=True), []
        )
        assert result == {'thinking': {'type': 'adaptive'}}

    def test_anthropic_variant_adaptive_thinking_high_sets_effort(self):
        """thinking='high' on adaptive model adds output_config.effort sibling per AWS docs."""
        model = self._adaptive_model()
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking='high'), []
        )
        assert result == {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'}}

    @pytest.mark.parametrize(
        'level,effort',
        [('minimal', 'low'), ('low', 'low'), ('medium', 'medium'), ('high', 'high'), ('xhigh', 'max')],
    )
    def test_anthropic_variant_adaptive_effort_map(self, level: ThinkingLevel, effort: str):
        """Effort map: minimal/low → low, medium → medium, high → high, xhigh → max (best-effort)."""
        model = self._adaptive_model()
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking=level), []
        )
        assert result == {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': effort}}

    def test_anthropic_variant_adaptive_xhigh_passes_through_when_profile_supports_it(self):
        """`xhigh` reaches the wire as `xhigh` when the merged profile carries `anthropic_supports_xhigh_effort`.

        `BedrockProvider.model_profile` merges the downstream Anthropic profile into the Bedrock one, so the
        flag is present for the same models the direct Anthropic path passes `xhigh` through for. Bedrock accepts
        `xhigh` on exactly those models and rejects it on the rest, which the `max` fallback above covers.
        """
        model = _bedrock_model(
            merge_profile(
                BedrockModelProfile(
                    bedrock_thinking_variant='anthropic',
                    bedrock_supports_adaptive_thinking=True,
                    bedrock_supports_effort=True,
                    supports_thinking=True,
                ),
                AnthropicModelProfile(anthropic_supports_xhigh_effort=True),
            )
        )
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking='xhigh'), []
        )
        assert result == {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'xhigh'}}

    def test_anthropic_variant_adaptive_no_effort_when_unsupported(self):
        """Effort is omitted when the profile doesn't advertise bedrock_supports_effort."""
        model = self._adaptive_model(supports_effort=False)
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking='high'), []
        )
        assert result == {'thinking': {'type': 'adaptive'}}

    def test_anthropic_variant_adaptive_user_output_config_wins(self):
        """User-provided output_config in bedrock_additional_model_requests_fields is preserved."""
        model = self._adaptive_model()
        settings = BedrockModelSettings(bedrock_additional_model_requests_fields={'output_config': {'effort': 'low'}})
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking='high'), [])
        assert result == {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'low'}}

    def test_anthropic_variant_adaptive_thinking_false_omits(self):
        """thinking=False on adaptive model omits both thinking and output_config."""
        model = self._adaptive_model()
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking=False), []
        )
        assert result is None

    def test_anthropic_variant_adaptive_thinking_false_disables_thinking_on_by_default(self):
        """thinking=False on a model that thinks by default sends `disabled`, since omitting it leaves thinking on."""
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='anthropic',
                bedrock_supports_adaptive_thinking=True,
                supports_thinking=True,
                thinking_enabled_by_default=True,
            )
        )
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking=False), []
        )
        assert result == {'thinking': {'type': 'disabled'}}

    def test_anthropic_variant_adaptive_thinking_false_between_tools(self):
        """thinking=False sends `between_tools` on a model that turns thinking off that way."""
        model = _bedrock_model(
            merge_profile(
                BedrockModelProfile(
                    bedrock_thinking_variant='anthropic',
                    bedrock_supports_adaptive_thinking=True,
                    supports_thinking=True,
                    thinking_enabled_by_default=True,
                ),
                AnthropicModelProfile(anthropic_disables_thinking_with_between_tools=True),
            )
        )
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking=False), []
        )
        assert result == {'thinking': {'type': 'between_tools'}}

    @pytest.mark.parametrize(
        ('supports_effort', 'expected'),
        [(True, {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'low'}}), (False, None)],
    )
    def test_anthropic_variant_adaptive_thinking_false_always_on(
        self, supports_effort: bool, expected: dict[str, Any] | None
    ):
        """thinking=False on a model that can't turn thinking off sends adaptive thinking at `low` effort, or nothing
        when the model has no effort setting."""
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='anthropic',
                bedrock_supports_adaptive_thinking=True,
                bedrock_supports_effort=supports_effort,
                supports_thinking=True,
                thinking_enabled_by_default=True,
                thinking_always_enabled=True,
            )
        )
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking=False), []
        )
        assert result == expected


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestOpenRouterThinkingTranslation:
    """Test OpenRouter unified thinking fallback in _openrouter_settings_to_openai_settings."""

    def test_thinking_true(self):
        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = _openrouter_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'medium', 'enabled': True}

    def test_thinking_high(self):
        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = _openrouter_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'high', 'enabled': True}

    def test_thinking_false_emits_effort_none(self):
        """`thinking=False` is forwarded as `reasoning.effort='none'` — the documented OpenRouter disable signal."""
        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = _openrouter_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'none'}

    def test_thinking_false_emits_lowest_effort_when_always_enabled(self):
        """A route that can't stop reasoning gets the lowest effort, without asking for reasoning to be enabled."""
        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = _openrouter_settings_to_openai_settings(settings, params, thinking_always_enabled=True)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'low'}

    @pytest.mark.parametrize(
        ('model_name', 'expected_always_enabled'),
        [
            # Hybrid routes — upstream can honor `effort='none'`; gate forwards `thinking=False`.
            pytest.param('moonshotai/kimi-k2.6', False, id='kimi-k2-hybrid'),
            pytest.param('qwen/qwen3-235b-a22b', False, id='qwen3-hybrid'),
            pytest.param('z-ai/glm-4.6', False, id='glm-hybrid'),
            pytest.param('anthropic/claude-sonnet-4.5', False, id='claude-hybrid'),
            # Grok 3 Mini takes Grok 4.3's efforts, `'none'` included (verified live on xAI).
            pytest.param('x-ai/grok-3-mini', False, id='grok-3-mini-hybrid'),
            # Always-on routes — upstream cannot disable, so `thinking=False` sends the lowest effort,
            # matching the same model's direct-route behavior.
            pytest.param('openai/o3', True, id='o3-always-on'),
            pytest.param('openai/gpt-5', True, id='gpt-5-always-on'),
            pytest.param('mistralai/magistral-medium-2509', True, id='magistral-always-on'),
            pytest.param('deepseek/deepseek-r1', True, id='deepseek-r1-always-on'),
            # OpenRouter makes reasoning mandatory on gpt-oss, though it takes `effort='none'` elsewhere.
            pytest.param('openai/gpt-oss-120b', True, id='gpt-oss-mandatory-on-openrouter'),
        ],
    )
    def test_provider_profile_propagates_thinking_capability(self, model_name: str, expected_always_enabled: bool):
        """OpenRouter's base profile sets `supports_thinking=True` so the gate forwards
        `thinking` to the transformer for every routed model; `thinking_always_enabled`
        propagates from the sub-profile via `merge_profile` so always-on upstream routes
        send their lowest effort for `thinking=False` — matching their direct-route behavior
        per the `ModelProfile.thinking_always_enabled` docstring."""
        profile = OpenRouterProvider.model_profile(model_name)
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is expected_always_enabled

    def test_openai_reasoning_effort_passthrough(self):
        """Explicit openai_reasoning_effort on OpenRouter is passed through."""
        model = OpenRouterModel(
            'openai/gpt-5',
            provider=OpenRouterProvider(api_key='mock-api-key'),
            profile=ModelProfile(supports_thinking=True),
        )

        settings: dict[str, Any] = {'openai_reasoning_effort': 'low'}
        params = ModelRequestParameters(thinking='high')
        result = model._translate_thinking(settings, params)
        assert result == 'low'


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestCerebrasThinkingTranslation:
    """Test Cerebras unified thinking fallback.

    Disabling is emitted as the standard `reasoning_effort='none'` rather than the upstream-deprecated
    `extra_body['disable_reasoning']`; other unified thinking levels are omitted because Cerebras reasons by default.
    """

    def test_thinking_false_sets_reasoning_effort_none(self):
        settings = CerebrasModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = _cerebras_settings_to_openai_settings(settings, params)
        assert result.get('openai_reasoning_effort') == 'none'
        assert 'extra_body' not in result

    def test_thinking_false_sets_lowest_effort_when_always_enabled(self):
        """gpt-oss can't turn reasoning off, so `thinking=False` sends its lowest documented effort."""
        settings = CerebrasModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = _cerebras_settings_to_openai_settings(settings, params, thinking_always_enabled=True)
        assert result.get('openai_reasoning_effort') == 'low'

    def test_thinking_false_keeps_explicit_effort_when_always_enabled(self):
        """An explicit `openai_reasoning_effort` wins over the lowest effort `thinking=False` would send."""
        settings: dict[str, Any] = {'openai_reasoning_effort': 'high'}
        params = ModelRequestParameters(thinking=False)
        result = _cerebras_settings_to_openai_settings(settings, params, thinking_always_enabled=True)
        assert result.get('openai_reasoning_effort') == 'high'

    def test_thinking_true_omits_reasoning_effort(self):
        settings = CerebrasModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = _cerebras_settings_to_openai_settings(settings, params)
        assert 'openai_reasoning_effort' not in result

    def test_thinking_effort_omits_reasoning_effort(self):
        settings = CerebrasModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = _cerebras_settings_to_openai_settings(settings, params)
        assert 'openai_reasoning_effort' not in result

    def test_explicit_cerebras_disable_takes_precedence(self):
        settings = CerebrasModelSettings(cerebras_disable_reasoning=True)
        params = ModelRequestParameters(thinking=True)
        with pytest.warns(PydanticAIDeprecationWarning, match=r'`cerebras_disable_reasoning` is deprecated'):
            result = _cerebras_settings_to_openai_settings(settings, params)
        assert result.get('openai_reasoning_effort') == 'none'

    def test_disable_overrides_explicit_reasoning_effort(self):
        """Disabling reasoning wins over an explicit `openai_reasoning_effort`, forcing `'none'`."""
        settings: dict[str, Any] = {'cerebras_disable_reasoning': True, 'openai_reasoning_effort': 'low'}
        params = ModelRequestParameters()
        with pytest.warns(PydanticAIDeprecationWarning, match=r'`cerebras_disable_reasoning` is deprecated'):
            result = _cerebras_settings_to_openai_settings(settings, params)
        assert result.get('openai_reasoning_effort') == 'none'

    def test_explicit_openai_reasoning_effort_passthrough(self):
        """Explicit openai_reasoning_effort on Cerebras is passed through."""
        model = CerebrasModel(
            'zai-glm-4.7',
            provider=CerebrasProvider(api_key='mock-api-key'),
            profile=ModelProfile(supports_thinking=True),
        )

        settings: dict[str, Any] = {'openai_reasoning_effort': 'low'}
        params = ModelRequestParameters(thinking='high')
        result = model._translate_thinking(settings, params)
        assert result == 'low'

    def test_clear_thinking_survives_repeated_requests(self):
        """`prepare_request` must not mutate the model's own `settings` dict.

        `merge_model_settings(self.settings, None)` returns `self.settings` by identity when there's no
        run-level override, so popping `cerebras_clear_thinking` in place would drop it on the next request.
        Driven through `prepare_request` rather than VCR because the footgun is a cross-request mutation of
        in-memory settings that no single recorded request can exercise.
        """
        model = CerebrasModel(
            'zai-glm-4.7',
            provider=CerebrasProvider(api_key='mock-api-key'),
            settings=CerebrasModelSettings(cerebras_clear_thinking=False),
        )
        params = ModelRequestParameters()

        first, _ = model.prepare_request(None, params)
        second, _ = model.prepare_request(None, params)

        assert first is not None and second is not None
        assert first.get('extra_body') == {'clear_thinking': False}
        assert second.get('extra_body') == {'clear_thinking': False}
        assert model._settings is not None
        assert 'cerebras_clear_thinking' in model._settings

    def test_disable_reasoning_survives_repeated_requests(self):
        """Same non-mutation guarantee for the deprecated `cerebras_disable_reasoning` pop."""
        model = CerebrasModel(
            'zai-glm-4.7',
            provider=CerebrasProvider(api_key='mock-api-key'),
            settings=CerebrasModelSettings(cerebras_disable_reasoning=True),
        )
        params = ModelRequestParameters()

        with pytest.warns(PydanticAIDeprecationWarning, match=r'`cerebras_disable_reasoning` is deprecated'):
            first, _ = model.prepare_request(None, params)
        with pytest.warns(PydanticAIDeprecationWarning, match=r'`cerebras_disable_reasoning` is deprecated'):
            second, _ = model.prepare_request(None, params)

        assert first is not None and second is not None
        assert first.get('openai_reasoning_effort') == 'none'
        assert second.get('openai_reasoning_effort') == 'none'
        assert model._settings is not None
        assert 'cerebras_disable_reasoning' in model._settings


@pytest.mark.skipif(not xai_imports(), reason='xai_sdk not installed')
class TestXaiThinkingTranslation:
    """Test xAI unified thinking fallback."""

    def test_thinking_high(self):
        model = XaiModel(
            'grok-4', provider=XaiProvider(api_key='mock-api-key'), profile=ModelProfile(supports_thinking=True)
        )

        settings = XaiModelSettings()
        params = ModelRequestParameters(thinking='high')
        # We can't call _create_chat directly, but we can verify prepare_request resolves
        _, resolved_params = model.prepare_request(settings, params)
        assert resolved_params.thinking == 'high'

    def test_thinking_true(self):
        model = XaiModel(
            'grok-4', provider=XaiProvider(api_key='mock-api-key'), profile=ModelProfile(supports_thinking=True)
        )

        settings = XaiModelSettings()
        params = ModelRequestParameters(thinking=True)
        _, resolved_params = model.prepare_request(settings, params)
        assert resolved_params.thinking is True

    @pytest.mark.parametrize('model_name', ['grok-3-mini', 'grok-4.5'])
    def test_thinking_false_reaches_the_adapter(self, model_name: str):
        """`thinking=False` survives the gate on both a model that can turn reasoning off (grok-3-mini) and
        one that can't (grok-4.5), where the adapter sends its lowest effort instead."""
        model = XaiModel(
            model_name, provider=XaiProvider(api_key='mock-api-key'), profile=grok_model_profile(model_name)
        )
        _, resolved = model.prepare_request(XaiModelSettings(thinking=False), ModelRequestParameters())
        assert resolved.thinking is False


# ---------------------------------------------------------------------------
# 3. Thinking capability tests
# ---------------------------------------------------------------------------


class TestThinkingCapability:
    def test_default_effort(self):
        cap = Thinking()
        assert cap.effort is True

    def test_get_model_settings_default(self):
        cap = Thinking()
        assert cap.get_model_settings() == snapshot(ModelSettings(thinking=True))

    def test_get_model_settings_high(self):
        cap = Thinking(effort='high')
        assert cap.get_model_settings() == snapshot(ModelSettings(thinking='high'))

    def test_get_model_settings_false(self):
        cap = Thinking(effort=False)
        assert cap.get_model_settings() == snapshot(ModelSettings(thinking=False))

    def test_get_model_settings_low(self):
        cap = Thinking(effort='low')
        assert cap.get_model_settings() == snapshot(ModelSettings(thinking='low'))

    def test_serialization_name(self):
        assert Thinking.get_serialization_name() == 'Thinking'

    def test_in_capability_types(self):
        assert 'Thinking' in CAPABILITY_TYPES
        assert CAPABILITY_TYPES['Thinking'] is Thinking

    def test_from_spec_default(self):
        cap = Thinking.from_spec()
        assert isinstance(cap, Thinking)
        assert cap.effort is True

    def test_from_spec_with_effort(self):
        cap = Thinking.from_spec(effort='high')
        assert isinstance(cap, Thinking)
        assert cap.effort == 'high'

    def test_agent_from_spec_with_thinking(self):
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [
                    {'Thinking': {'effort': 'high'}},
                ],
            }
        )
        assert agent.model is not None

    def test_agent_from_spec_with_thinking_shorthand(self):
        """Thinking with no args can be specified as a bare string."""
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': ['Thinking'],
            }
        )
        assert agent.model is not None


# ---------------------------------------------------------------------------
# 4. Integration tests
# ---------------------------------------------------------------------------


class TestThinkingIntegration:
    async def test_thinking_setting_produces_output(self):
        """Basic smoke test: agent with thinking=True runs successfully."""
        model = _make_model(supports_thinking=True)
        agent = Agent(model, model_settings=ModelSettings(thinking=True))
        result = await agent.run('test')
        assert result.output == 'ok'

    async def test_capability_flows_through_to_model(self):
        """Thinking capability's model settings flow through to resolved params."""
        captured_params: list[ModelRequestParameters] = []

        def _capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            captured_params.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart(content='done')])

        model = FunctionModel(
            _capture,
            profile=ModelProfile(supports_thinking=True),
        )
        agent = Agent(model, capabilities=[Thinking(effort='high')])
        result = await agent.run('test')
        assert result.output == 'done'
        assert len(captured_params) == 1
        assert captured_params[0].thinking == 'high'

    async def test_capability_default_effort_flows_through(self):
        """Thinking() with default effort=True flows through."""
        captured_params: list[ModelRequestParameters] = []

        def _capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            captured_params.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart(content='done')])

        model = FunctionModel(
            _capture,
            profile=ModelProfile(supports_thinking=True),
        )
        agent = Agent(model, capabilities=[Thinking()])
        result = await agent.run('test')
        assert result.output == 'done'
        assert captured_params[0].thinking is True

    async def test_capability_silently_ignored_on_unsupported_model(self):
        """Thinking capability on unsupported model -> params.thinking stays None."""
        captured_params: list[ModelRequestParameters] = []

        def _capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            captured_params.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart(content='done')])

        model = FunctionModel(
            _capture,
            profile=ModelProfile(supports_thinking=False),
        )
        agent = Agent(model, capabilities=[Thinking(effort='high')])
        result = await agent.run('test')
        assert result.output == 'done'
        assert captured_params[0].thinking is None

    async def test_model_settings_override_with_thinking(self):
        """run-level model_settings with thinking override agent-level capability."""
        captured_params: list[ModelRequestParameters] = []
        captured_settings: list[ModelSettings | None] = []

        def _capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            captured_params.append(info.model_request_parameters)
            captured_settings.append(info.model_settings)
            return ModelResponse(parts=[TextPart(content='done')])

        model = FunctionModel(
            _capture,
            profile=ModelProfile(supports_thinking=True),
        )
        agent = Agent(model, capabilities=[Thinking(effort='low')])
        result = await agent.run('test', model_settings=ModelSettings(thinking='high'))
        assert result.output == 'done'
        # Run-level settings override capability settings via merge_model_settings
        assert captured_params[0].thinking == 'high'

    async def test_thinking_false_capability_on_always_enabled(self):
        """Thinking(effort=False) on always-on model -> passed to the adapter, which maps it to its lowest setting."""
        captured_params: list[ModelRequestParameters] = []

        def _capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            captured_params.append(info.model_request_parameters)
            return ModelResponse(parts=[TextPart(content='done')])

        model = FunctionModel(
            _capture,
            profile=ModelProfile(thinking_always_enabled=True),
        )
        agent = Agent(model, capabilities=[Thinking(effort=False)])
        result = await agent.run('test')
        assert result.output == 'done'
        assert captured_params[0].thinking is False

    async def test_prepare_request_does_not_mutate_model_settings(self):
        """Regression: prepare_request() must not mutate the original model_settings dict."""
        model = _make_model(supports_thinking=True)
        settings = ModelSettings(thinking='high')
        original_keys = set(settings.keys())
        params = ModelRequestParameters()
        model.prepare_request(settings, params)
        assert set(settings.keys()) == original_keys
        assert 'thinking' in settings

    async def test_thinking_stripped_from_model_settings(self):
        """After prepare_request(), returned model_settings should not contain 'thinking'."""
        model = _make_model(supports_thinking=True)
        settings: ModelSettings = {'thinking': 'high', 'max_tokens': 500}
        returned_settings, resolved_params = model.prepare_request(settings, ModelRequestParameters())
        assert returned_settings is not None
        assert 'thinking' not in returned_settings
        assert returned_settings.get('max_tokens') == 500
        assert resolved_params.thinking == 'high'

    async def test_thinking_only_setting_returns_none(self):
        """When thinking is the only model setting, stripping it should return None."""
        model = _make_model(supports_thinking=True)
        settings: ModelSettings = {'thinking': 'high'}
        returned_settings, resolved_params = model.prepare_request(settings, ModelRequestParameters())
        assert returned_settings is None
        assert resolved_params.thinking == 'high'


@pytest.mark.skipif(not google_imports(), reason='google-genai not installed')
class TestGoogleBudgetApiConstraints:
    """Budget values respect the Google API's documented limits."""

    def test_all_budgets_within_flash_range(self):
        """Every effort budget must be within Gemini 2.5 Flash's [0, 24576] range."""
        model = FunctionModel(
            _echo, profile=GoogleModelProfile(supports_thinking=True, google_supports_thinking_level=False)
        )
        for effort in ('minimal', 'low', 'medium', 'high', 'xhigh'):
            params = ModelRequestParameters(thinking=effort)
            result = GoogleModel._translate_thinking(model, {}, params)
            assert result is not None
            budget = result.get('thinking_budget')
            assert budget is not None, f"effort='{effort}' should produce a thinking_budget"
            assert 0 <= budget <= 24576, f"effort='{effort}' budget={budget} exceeds Flash max 24576"

    def test_all_budgets_within_pro_range(self):
        """Every effort budget must be within Gemini 2.5 Pro's [128, 32768] range."""
        model = FunctionModel(
            _echo, profile=GoogleModelProfile(supports_thinking=True, google_supports_thinking_level=False)
        )
        for effort in ('minimal', 'low', 'medium', 'high', 'xhigh'):
            params = ModelRequestParameters(thinking=effort)
            result = GoogleModel._translate_thinking(model, {}, params)
            assert result is not None
            budget = result.get('thinking_budget')
            assert budget is not None, f"effort='{effort}' should produce a thinking_budget"
            assert 128 <= budget <= 32768, f"effort='{effort}' budget={budget} outside Pro range [128, 32768]"

    def test_budgets_are_monotonically_increasing(self):
        """low < medium < high — effort levels should map to increasing budgets."""
        model = FunctionModel(
            _echo, profile=GoogleModelProfile(supports_thinking=True, google_supports_thinking_level=False)
        )
        budgets = {}
        for effort in ('low', 'medium', 'high'):
            params = ModelRequestParameters(thinking=effort)
            result = GoogleModel._translate_thinking(model, {}, params)
            assert result is not None
            budgets[effort] = result.get('thinking_budget')

        assert budgets['low'] is not None
        assert budgets['medium'] is not None
        assert budgets['high'] is not None
        assert budgets['low'] < budgets['medium'] < budgets['high']


class TestProfileThinkingCapabilities:
    """Model profiles correctly detect thinking-capable models."""

    def test_anthropic_profile_thinking_support(self):
        from pydantic_ai.profiles.anthropic import anthropic_model_profile

        # All Anthropic models support thinking in our implementation
        profile = anthropic_model_profile('claude-3-7-sonnet')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True

        profile = anthropic_model_profile('claude-sonnet-4-5')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True

        # Newer models support adaptive thinking
        profile = anthropic_model_profile('claude-sonnet-4-6')
        assert profile is not None
        assert isinstance(profile, dict)
        assert profile.get('anthropic_supports_adaptive_thinking', False) is True

    @pytest.mark.parametrize('model_name', ['claude-opus-4-7', 'claude-opus-4-8'])
    def test_anthropic_profile_thinking_support_opus_47_plus(self, model_name: str):
        from pydantic_ai.profiles.anthropic import anthropic_model_profile

        profile = anthropic_model_profile(model_name)
        assert profile is not None
        assert isinstance(profile, dict)
        assert profile.get('anthropic_supports_adaptive_thinking', False) is True
        assert profile.get('anthropic_supports_xhigh_effort', False) is True
        assert profile.get('anthropic_disallows_budget_thinking', False) is True
        assert profile.get('anthropic_supports_task_budgets', False) is True

    def test_google_profile_thinking_support(self):
        profile = google_model_profile('gemini-2.5-flash')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is False
        assert profile.get('google_supports_thinking_level') is False

        profile = google_model_profile('gemini-2.5-pro')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is True
        assert profile.get('google_supports_thinking_level') is False

        profile = google_model_profile('gemini-2.0-flash')
        assert profile is not None
        assert profile.get('supports_thinking', False) is False
        assert profile.get('google_supports_thinking_level') is False

        profile = google_model_profile('gemini-1.5-flash')
        assert profile is not None
        assert profile.get('supports_thinking', False) is False
        assert profile.get('google_supports_thinking_level') is False

        profile = google_model_profile('gemini-3.0-pro')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is True
        assert profile.get('google_supports_thinking_level') is True

        profile = google_model_profile('gemini-3-flash-preview')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('google_supports_thinking_level') is True

        # Future/unversioned and -latest alias models default to Gemini 3 behaviour
        for name in ('gemini-9-flash', 'gemini-flash-latest', 'gemini-pro-latest'):
            profile = google_model_profile(name)
            assert profile is not None
            assert profile.get('supports_thinking', False) is True
            assert profile.get('google_supports_thinking_level') is True

    def test_openai_profile_thinking_support(self):
        profile = openai_model_profile('o3')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is True

        profile = openai_model_profile('gpt-4o')
        assert profile is not None
        assert profile.get('supports_thinking', False) is False

    def test_groq_profile_thinking_support(self):
        profile = groq_model_profile('deepseek-r1-distill-llama-70b')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True

        profile = groq_model_profile('llama-3.1-8b-instant')
        assert profile is not None
        assert profile.get('supports_thinking', False) is False

    @pytest.mark.parametrize(
        'model_name',
        [
            'grok-4.3',
            'grok-4.3-latest',
            # Floating alias for the newest Grok, currently 4.3.
            'grok-latest',
            'grok-4-1-fast-reasoning',
            'grok-4-fast-non-reasoning',
            'grok-3',
        ],
    )
    def test_grok_43_profile_thinking_support(self, model_name: str):
        profile = grok_model_profile(model_name)
        assert profile is not None
        assert isinstance(profile, dict)
        assert profile.get('supports_thinking', False) is True
        assert profile.get('grok_reasoning_efforts') == frozenset({'none', 'low', 'medium', 'high'})

    def test_grok_3_mini_profile_thinking_support(self):
        profile = grok_model_profile('grok-3-mini')
        assert profile is not None
        assert isinstance(profile, dict)
        assert profile.get('supports_thinking', False) is True
        assert profile.get('grok_reasoning_efforts') == frozenset({'none', 'low', 'high'})
        assert profile.get('thinking_enabled_by_default', False) is True

    def test_grok_3_mini_latest_profile_keeps_basic_efforts(self):
        """Only the `grok-3-mini` ids verified to take Grok 4.3's efforts get them; other ids keep `low`/`high`."""
        profile = grok_model_profile('grok-3-mini-fast-latest')
        assert profile is not None
        assert profile.get('grok_reasoning_efforts') == frozenset({'low', 'high'})
        assert profile.get('thinking_always_enabled', False) is True

    def test_grok_3_fast_profile_thinking_support(self):
        profile = grok_model_profile('grok-3-fast')
        assert profile is not None
        assert isinstance(profile, dict)
        assert profile.get('supports_thinking', False) is False
        assert profile.get('grok_reasoning_efforts') == frozenset()

    def test_cohere_profile_thinking_support(self):
        profile = cohere_model_profile('command-a-reasoning')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True

    def test_mistral_profile_thinking_support(self):
        profile = mistral_model_profile('magistral-medium')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is True

    def test_grok_profile_thinking_always_enabled(self):
        """grok-4.5's `reasoning_effort` has no `'none'` value, so the profile is marked always-on and
        `thinking=False` sends its lowest effort; grok-3-mini accepts `'none'`, so it isn't."""
        profile = grok_model_profile('grok-4.5')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is True

        profile = grok_model_profile('grok-3-mini')
        assert profile is not None
        assert profile.get('thinking_always_enabled', False) is False

        # grok-4 reasoning models reject `reasoning_effort` outright, so the profile
        # leaves thinking unsupported — guarding against accidental promotion to always-on.
        profile = grok_model_profile('grok-4')
        assert profile is not None
        assert profile.get('supports_thinking', False) is False
        assert profile.get('thinking_always_enabled', False) is False

    @pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
    def test_bedrock_openai_variant_thinking_always_enabled(self):
        """Bedrock-Converse rejects `reasoning_effort='none'` for gpt-oss; marked always-on."""
        from pydantic_ai.providers.bedrock import BedrockProvider

        profile = BedrockProvider.model_profile('openai.gpt-oss-120b-1:0')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is True

    @pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
    @pytest.mark.parametrize(
        ('model_name', 'supports_thinking'),
        [
            ('qwen.qwen3-32b-v1:0', True),
            ('qwen.qwen3-next-80b-a3b', True),
            ('qwen.qwen3-vl-235b-a22b', True),
            ('qwen.qwen3-coder-30b-a3b-v1:0', False),
            ('qwen.qwen3-coder-next', False),
        ],
    )
    def test_bedrock_qwen_variant_thinking_support(self, model_name: str, supports_thinking: bool):
        """Qwen3 on Bedrock-Converse reasons only when asked to and accepts `reasoning_config='none'`, so it can turn
        thinking off. Qwen3-Coder returns no reasoning at any level. Verified live."""
        from pydantic_ai.providers.bedrock import BedrockProvider

        profile = BedrockProvider.model_profile(model_name)
        assert profile is not None
        assert profile.get('supports_thinking', False) is supports_thinking
        assert profile.get('thinking_always_enabled', False) is False


class TestCrossProviderPortability:
    """Same unified settings produce sensible results across providers."""

    @pytest.mark.skipif(
        not (anthropic_imports() and openai_imports() and groq_imports()),
        reason='anthropic, openai, and groq must all be installed',
    )
    def test_same_settings_all_main_providers(self):
        """The same thinking=True + effort='high' should produce non-None results
        on supported models across all providers."""
        thinking_profile = ModelProfile(supports_thinking=True)
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        # Anthropic: budget-based
        result = AnthropicModel._translate_thinking(FunctionModel(_echo, profile=thinking_profile), settings, params)
        assert result is not None

        # OpenAI Chat: direct effort mapping
        result = OpenAIChatModel._translate_thinking(FunctionModel(_echo, profile=thinking_profile), settings, params)
        assert result == 'high'

        # Groq: effort silently ignored, just enables
        result = GroqModel._translate_thinking(FunctionModel(_echo, profile=thinking_profile), settings, params, False)
        assert result == 'parsed'

    def test_unsupported_models_silently_dropped_via_prepare_request(self):
        """thinking settings on unsupported models → not resolved by prepare_request."""
        model = _make_model(supports_thinking=False)
        settings: ModelSettings = {'thinking': 'high'}
        _merged, params = model.prepare_request(settings, ModelRequestParameters())
        assert params.thinking is None

    def test_thinking_false_reaches_adapter_on_always_enabled_models(self):
        """`thinking=False` stays on the params of always-on profiles, so the adapter can send the least thinking
        the model accepts rather than ignoring it. Other values flow through unchanged."""
        model = _make_model(supports_thinking=True, thinking_always_enabled=True)
        _merged, params = model.prepare_request({'thinking': False}, ModelRequestParameters())
        assert params.thinking is False

        _merged, params = model.prepare_request({'thinking': 'high'}, ModelRequestParameters())
        assert params.thinking == 'high'

    @pytest.mark.parametrize(
        'model_name',
        [
            'openai/o3',
            'openai/gpt-5',
            'mistralai/magistral-medium-2509',
            'deepseek/deepseek-r1',
        ],
    )
    def test_thinking_false_sends_lowest_effort_on_always_on_openrouter_routes(self, model_name: str):
        """Aggregator-route consistency: OpenRouter routes to always-on upstream models send their lowest
        effort for `thinking=False`, as the direct route to the same model does, since OpenRouter rejects
        `effort='none'` on them ("Reasoning is mandatory"). `thinking_always_enabled` propagates from the
        sub-profile, no aggregator-specific override required."""
        model = OpenRouterModel(model_name, provider=OpenRouterProvider(api_key='mock-api-key'))
        settings, _params = model.prepare_request({'thinking': False}, ModelRequestParameters())
        extra_body: dict[str, Any] = (settings or {}).get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'low'}

        settings, _params = model.prepare_request({'thinking': 'high'}, ModelRequestParameters())
        extra_body = (settings or {}).get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'high', 'enabled': True}


class TestPrepareRequestNoMutationDetailed:
    """prepare_request doesn't leak state across sequential calls."""

    def test_sequential_calls_no_leakage(self):
        """Sequential prepare_request calls don't leak thinking into model._settings."""
        model = _make_model(supports_thinking=True)

        # First call with thinking
        settings1: ModelSettings = {'thinking': True}
        _merged1, params1 = model.prepare_request(settings1, ModelRequestParameters())
        assert params1.thinking is True

        # Second call with False should not see True
        settings2: ModelSettings = {'thinking': False}
        _merged2, params2 = model.prepare_request(settings2, ModelRequestParameters())
        assert params2.thinking is False

    def test_no_settings_after_thinking_call(self):
        """Calling without settings after a thinking call should not carry state."""
        model = _make_model(supports_thinking=True)

        settings1: ModelSettings = {'thinking': 'high'}
        _merged1, params1 = model.prepare_request(settings1, ModelRequestParameters())
        assert params1.thinking == 'high'

        _merged2, params2 = model.prepare_request(None, ModelRequestParameters())
        assert params2.thinking is None


# ---------------------------------------------------------------------------
# Every effort level through every provider mapping
# ---------------------------------------------------------------------------

_SKIP_WITHOUT_OPENAI = pytest.mark.skipif(not openai_imports(), reason='openai not installed')
_SKIP_WITHOUT_ANTHROPIC = pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
_SKIP_WITHOUT_GOOGLE = pytest.mark.skipif(not google_imports(), reason='google-genai not installed')
_SKIP_WITHOUT_BEDROCK = pytest.mark.skipif(not bedrock_imports(), reason='boto3 not installed')


@dataclass(frozen=True)
class EffortMapping:
    """How one provider maps each unified effort level, and the value it should produce for each."""

    resolve: Callable[[ThinkingEffort], object]
    expected: dict[str, object]


EFFORT_MAPPINGS = [
    pytest.param(
        EffortMapping(
            lambda level: OpenAIChatModel._translate_thinking(
                FunctionModel(_echo), {}, ModelRequestParameters(thinking=level)
            ),
            {'minimal': 'minimal', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'xhigh', 'max': 'xhigh'},
        ),
        id='openai-chat',
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    pytest.param(
        EffortMapping(
            lambda level: OpenAIChatModel._translate_thinking(
                FunctionModel(_echo, profile=OpenAIModelProfile(openai_chat_supports_max_reasoning_effort=True)),
                {},
                ModelRequestParameters(thinking=level),
            ),
            {'minimal': 'minimal', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'xhigh', 'max': 'max'},
        ),
        id='openai-chat-max',
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    pytest.param(
        EffortMapping(
            lambda level: OpenAIResponsesModel._translate_thinking(
                FunctionModel(_echo), {}, ModelRequestParameters(thinking=level)
            ),
            {
                'minimal': {'effort': 'minimal'},
                'low': {'effort': 'low'},
                'medium': {'effort': 'medium'},
                'high': {'effort': 'high'},
                'xhigh': {'effort': 'xhigh'},
                'max': {'effort': 'xhigh'},
            },
        ),
        id='openai-responses',
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    pytest.param(
        EffortMapping(
            lambda level: OpenAIResponsesModel._translate_thinking(
                FunctionModel(_echo, profile=OpenAIModelProfile(openai_responses_supports_max_reasoning_effort=True)),
                {},
                ModelRequestParameters(thinking=level),
            ),
            {
                'minimal': {'effort': 'minimal'},
                'low': {'effort': 'low'},
                'medium': {'effort': 'medium'},
                'high': {'effort': 'high'},
                'xhigh': {'effort': 'xhigh'},
                'max': {'effort': 'max'},
            },
        ),
        id='openai-responses-max',
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    pytest.param(
        EffortMapping(
            lambda level: resolve_anthropic_effort(level, supports_xhigh=True, supports_max=True),
            {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'xhigh', 'max': 'max'},
        ),
        id='anthropic-effort',
    ),
    pytest.param(
        EffortMapping(
            lambda level: resolve_anthropic_effort(level, supports_xhigh=False, supports_max=True),
            {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'max', 'max': 'max'},
        ),
        id='anthropic-effort-without-xhigh',
    ),
    pytest.param(
        EffortMapping(
            lambda level: resolve_anthropic_effort(level, supports_xhigh=False, supports_max=False),
            {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'high', 'max': 'high'},
        ),
        id='anthropic-effort-without-xhigh-or-max',
    ),
    pytest.param(
        EffortMapping(
            lambda level: resolve_anthropic_effort(level, supports_xhigh=True, supports_max=False),
            {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'xhigh', 'max': 'xhigh'},
        ),
        id='anthropic-effort-without-max',
    ),
    pytest.param(
        EffortMapping(
            lambda level: AnthropicModel._translate_thinking(
                FunctionModel(_echo, profile=AnthropicModelProfile(supports_thinking=True)),
                {},
                ModelRequestParameters(thinking=level),
            ),
            {
                'minimal': {'type': 'enabled', 'budget_tokens': 1024},
                'low': {'type': 'enabled', 'budget_tokens': 2048},
                'medium': {'type': 'enabled', 'budget_tokens': 10000},
                'high': {'type': 'enabled', 'budget_tokens': 16384},
                'xhigh': {'type': 'enabled', 'budget_tokens': 32768},
                'max': {'type': 'enabled', 'budget_tokens': 32768},
            },
        ),
        id='anthropic-budget',
        marks=_SKIP_WITHOUT_ANTHROPIC,
    ),
    pytest.param(
        EffortMapping(
            lambda level: GoogleModel._translate_thinking(
                FunctionModel(_echo, profile=GoogleModelProfile(supports_thinking=True)),
                {},
                ModelRequestParameters(thinking=level),
            ),
            {
                'minimal': {'include_thoughts': True, 'thinking_level': 'MINIMAL'},
                'low': {'include_thoughts': True, 'thinking_level': 'LOW'},
                'medium': {'include_thoughts': True, 'thinking_level': 'MEDIUM'},
                'high': {'include_thoughts': True, 'thinking_level': 'HIGH'},
                'xhigh': {'include_thoughts': True, 'thinking_level': 'HIGH'},
                'max': {'include_thoughts': True, 'thinking_level': 'HIGH'},
            },
        ),
        id='google',
        marks=_SKIP_WITHOUT_GOOGLE,
    ),
    pytest.param(
        EffortMapping(
            lambda level: GoogleModel._translate_thinking(
                FunctionModel(
                    _echo, profile=GoogleModelProfile(supports_thinking=True, google_supports_thinking_level=False)
                ),
                {},
                ModelRequestParameters(thinking=level),
            ),
            {
                'minimal': {'include_thoughts': True, 'thinking_budget': 128},
                'low': {'include_thoughts': True, 'thinking_budget': 2048},
                'medium': {'include_thoughts': True, 'thinking_budget': 8192},
                'high': {'include_thoughts': True, 'thinking_budget': 24576},
                'xhigh': {'include_thoughts': True, 'thinking_budget': 24576},
                'max': {'include_thoughts': True, 'thinking_budget': 24576},
            },
        ),
        id='google-budget',
        marks=_SKIP_WITHOUT_GOOGLE,
    ),
    pytest.param(
        EffortMapping(
            lambda level: GROQ_GPT_OSS_REASONING_EFFORT_MAP[level],
            {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'high', 'max': 'high'},
        ),
        id='groq-gpt-oss',
    ),
    pytest.param(
        EffortMapping(
            lambda level: _MISTRAL_REASONING_EFFORT_MAP[level],
            {'minimal': 'high', 'low': 'high', 'medium': 'high', 'high': 'high', 'xhigh': 'high', 'max': 'high'},
        ),
        id='mistral',
        marks=pytest.mark.skipif(not mistral_imports(), reason='mistral not installed'),
    ),
    pytest.param(
        EffortMapping(
            lambda level: _THINKING_TOKEN_BUDGETS[level],
            {'minimal': 512, 'low': 2048, 'medium': 8192, 'high': 16384, 'xhigh': 31000, 'max': 31000},
        ),
        id='cohere',
        marks=pytest.mark.skipif(not cohere_imports(), reason='cohere not installed'),
    ),
    pytest.param(
        EffortMapping(
            lambda level: _SNOWFLAKE_REASONING_EFFORT_MAP[level],
            {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'high', 'max': 'high'},
        ),
        id='snowflake',
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    pytest.param(
        EffortMapping(
            lambda level: _OPENROUTER_EFFORT_MAP[level],
            {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'high', 'max': 'max'},
        ),
        id='openrouter',
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    pytest.param(
        EffortMapping(
            lambda level: _zai_settings_to_openai_settings(
                {},
                ModelRequestParameters(thinking=level),
                supports_thinking=False,
                thinking_always_enabled=False,
                supports_reasoning_effort=True,
            ).get('extra_body'),
            {
                level: {'thinking': {'type': 'enabled'}, 'reasoning_effort': level}
                for level in ('minimal', 'low', 'medium', 'high', 'xhigh', 'max')
            },
        ),
        id='zai-glm-5.2',
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    pytest.param(
        EffortMapping(
            lambda level: _zai_settings_to_openai_settings(
                {},
                ModelRequestParameters(thinking=level),
                supports_thinking=False,
                thinking_always_enabled=True,
                supports_reasoning_effort=True,
                reasoning_effort_mapping=_GLM_5_3_REASONING_EFFORT_MAPPING,
            ).get('extra_body'),
            {
                'minimal': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'},
                'low': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'},
                'medium': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'},
                'high': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'},
                'xhigh': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'max'},
                'max': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'max'},
            },
        ),
        id='zai-glm-5.3',
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    pytest.param(
        EffortMapping(
            lambda level: _map_reasoning_effort(
                level, GrokModelProfile(grok_reasoning_efforts=frozenset({'low', 'medium', 'high'}))
            ),
            {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'high', 'max': 'high'},
        ),
        id='xai',
        marks=pytest.mark.skipif(not xai_imports(), reason='xai_sdk not installed'),
    ),
    pytest.param(
        EffortMapping(
            lambda level: _bedrock_model(
                BedrockModelProfile(bedrock_thinking_variant='qwen', supports_thinking=True)
            )._build_additional_model_request_fields(
                BedrockModelSettings(), ModelRequestParameters(thinking=level), []
            ),
            {
                'minimal': {'reasoning_config': 'low'},
                'low': {'reasoning_config': 'low'},
                'medium': {'reasoning_config': 'high'},
                'high': {'reasoning_config': 'high'},
                'xhigh': {'reasoning_config': 'high'},
                'max': {'reasoning_config': 'high'},
            },
        ),
        id='bedrock-qwen',
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    pytest.param(
        EffortMapping(
            lambda level: _bedrock_model(
                BedrockModelProfile(bedrock_thinking_variant='openai', supports_thinking=True)
            )._build_additional_model_request_fields(
                BedrockModelSettings(), ModelRequestParameters(thinking=level), []
            ),
            {
                # gpt-oss on Converse accepts only `low`, `medium`, and `high`.
                'minimal': {'reasoning_effort': 'low'},
                'low': {'reasoning_effort': 'low'},
                'medium': {'reasoning_effort': 'medium'},
                'high': {'reasoning_effort': 'high'},
                'xhigh': {'reasoning_effort': 'high'},
                'max': {'reasoning_effort': 'high'},
            },
        ),
        id='bedrock-openai',
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    pytest.param(
        EffortMapping(
            lambda level: _bedrock_model(
                BedrockModelProfile(bedrock_thinking_variant='openai_responses', supports_thinking=True)
            )._build_additional_model_request_fields(
                BedrockModelSettings(), ModelRequestParameters(thinking=level), []
            ),
            {
                'minimal': {'reasoning': {'effort': 'low'}},
                'low': {'reasoning': {'effort': 'low'}},
                'medium': {'reasoning': {'effort': 'medium'}},
                'high': {'reasoning': {'effort': 'high'}},
                'xhigh': {'reasoning': {'effort': 'xhigh'}},
                'max': {'reasoning': {'effort': 'xhigh'}},
            },
        ),
        id='bedrock-openai-responses',
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    pytest.param(
        EffortMapping(
            lambda level: _bedrock_model(
                BedrockModelProfile(bedrock_thinking_variant='xai', supports_thinking=True)
            )._build_additional_model_request_fields(
                BedrockModelSettings(), ModelRequestParameters(thinking=level), []
            ),
            {
                'minimal': {'reasoning': {'effort': 'low'}},
                'low': {'reasoning': {'effort': 'low'}},
                'medium': {'reasoning': {'effort': 'medium'}},
                'high': {'reasoning': {'effort': 'high'}},
                'xhigh': {'reasoning': {'effort': 'xhigh'}},
                'max': {'reasoning': {'effort': 'xhigh'}},
            },
        ),
        id='bedrock-xai',
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    pytest.param(
        EffortMapping(
            lambda level: _bedrock_model(
                merge_profile(
                    BedrockModelProfile(
                        bedrock_thinking_variant='anthropic',
                        bedrock_supports_adaptive_thinking=True,
                        bedrock_supports_effort=True,
                        supports_thinking=True,
                    ),
                    AnthropicModelProfile(anthropic_supports_max_effort=False),
                )
            )._build_additional_model_request_fields(
                BedrockModelSettings(), ModelRequestParameters(thinking=level), []
            ),
            {
                'minimal': {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'low'}},
                'low': {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'low'}},
                'medium': {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'medium'}},
                'high': {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'}},
                'xhigh': {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'}},
                'max': {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'}},
            },
        ),
        id='bedrock-anthropic-without-max',
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
]


@pytest.mark.parametrize('mapping', EFFORT_MAPPINGS)
def test_every_effort_level_resolves(mapping: EffortMapping):
    """Each provider's mapping of every unified effort level, `'max'` included.

    Not a VCR test: a level-keyed mapping that misses a level still type-checks and only raises `KeyError`
    once a request uses that level, so iterating `ThinkingEffort` here catches a mapping a new level skips.
    """
    assert {level: mapping.resolve(level) for level in get_args(ThinkingEffort)} == mapping.expected


# ---------------------------------------------------------------------------
# 5. Cross-scope precedence warnings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CrossScopeCase:
    """A provider-specific thinking setting probed against unified `thinking` at one scope and across two."""

    id: str
    probe: Callable[[ModelSettings | None], Probe]
    """Builds a probe whose model carries the given `settings=`, sending the settings each probe call passes."""
    setting: Callable[[], ModelSettings]
    """Builds the provider-specific setting: its settings class is imported behind `try_import`, so it can't be built
    at collection time on a shard that lacks the SDK."""
    thinking: ThinkingLevel
    sent: Callable[[dict[str, Any]], object]
    """The part of a probed request that both the setting and `thinking` control."""
    overrides: bool
    marks: pytest.MarkDecorator


_SKIP_WITHOUT_GROQ = pytest.mark.skipif(not groq_imports(), reason='groq not installed')
_SKIP_WITHOUT_XAI = pytest.mark.skipif(not xai_imports(), reason='xai_sdk not installed')


def _bedrock_probe(model_name: str) -> Callable[[ModelSettings | None], Probe]:
    return lambda settings: botocore_probe(
        lambda: BedrockConverseModel(
            model_name,
            provider=BedrockProvider(
                aws_access_key_id=PROBE_KEY, aws_secret_access_key=PROBE_KEY, region_name='us-east-1'
            ),
            settings=settings,
        )
    )


def _bedrock_sent(payload: dict[str, Any]) -> object:
    return json.loads(payload['body']).get('additionalModelRequestFields')


_CROSS_SCOPE_CASES = [
    _CrossScopeCase(
        'openai-chat-reasoning-effort',
        lambda s: http_probe(
            lambda c: OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=c), settings=s)
        ),
        lambda: OpenAIChatModelSettings(openai_reasoning_effort='high'),
        'low',
        lambda p: p['body'].get('reasoning_effort'),
        overrides=True,
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    _CrossScopeCase(
        'openai-responses-reasoning-effort',
        lambda s: http_probe(
            lambda c: OpenAIResponsesModel(
                'gpt-5', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: OpenAIResponsesModelSettings(openai_reasoning_effort='high'),
        'low',
        lambda p: p['body'].get('reasoning'),
        overrides=True,
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    _CrossScopeCase(
        'anthropic-thinking',
        lambda s: http_probe(
            lambda c: AnthropicModel(
                'claude-sonnet-4-5', provider=AnthropicProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 5000}),
        'low',
        lambda p: p['body'].get('thinking'),
        overrides=True,
        marks=_SKIP_WITHOUT_ANTHROPIC,
    ),
    _CrossScopeCase(
        # Only the effort is replaced: the thinking budget still follows `thinking`.
        'anthropic-effort',
        lambda s: http_probe(
            lambda c: AnthropicModel(
                'claude-opus-4-5', provider=AnthropicProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: AnthropicModelSettings(anthropic_effort='high'),
        'low',
        lambda p: p['body']['output_config']['effort'],
        overrides=True,
        marks=_SKIP_WITHOUT_ANTHROPIC,
    ),
    _CrossScopeCase(
        # Without effort support `thinking` maps to a budget alone, which an explicit effort leaves in place.
        'anthropic-effort-unsupported',
        lambda s: http_probe(
            lambda c: AnthropicModel(
                'claude-sonnet-4-5', provider=AnthropicProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: AnthropicModelSettings(anthropic_effort='high'),
        'low',
        lambda p: p['body'].get('thinking'),
        overrides=False,
        marks=_SKIP_WITHOUT_ANTHROPIC,
    ),
    _CrossScopeCase(
        'google-thinking-config',
        lambda s: http_probe(
            lambda c: GoogleModel(
                'gemini-2.5-flash', provider=GoogleProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: GoogleModelSettings(google_thinking_config={'thinking_budget': 1000}),
        'low',
        lambda p: p['body']['generationConfig'].get('thinkingConfig'),
        overrides=True,
        marks=_SKIP_WITHOUT_GOOGLE,
    ),
    _CrossScopeCase(
        'groq-reasoning-format',
        lambda s: legacy_http_probe(
            lambda c: GroqModel('qwen/qwen3-32b', provider=GroqProvider(api_key=PROBE_KEY, http_client=c), settings=s)
        ),
        lambda: GroqModelSettings(groq_reasoning_format='raw'),
        'low',
        lambda p: p['body'].get('reasoning_format'),
        overrides=True,
        marks=_SKIP_WITHOUT_GROQ,
    ),
    _CrossScopeCase(
        # qwen3 disables reasoning through `reasoning_effort='none'`, which Groq lets win over any format.
        'groq-reasoning-format-qwen3-disabled',
        lambda s: legacy_http_probe(
            lambda c: GroqModel('qwen/qwen3-32b', provider=GroqProvider(api_key=PROBE_KEY, http_client=c), settings=s)
        ),
        lambda: GroqModelSettings(groq_reasoning_format='raw'),
        False,
        lambda p: p['body'].get('reasoning_effort'),
        overrides=False,
        marks=_SKIP_WITHOUT_GROQ,
    ),
    _CrossScopeCase(
        # Only the effort is replaced: the reasoning format still follows `thinking`.
        'groq-reasoning-effort',
        lambda s: legacy_http_probe(
            lambda c: GroqModel(
                'openai/gpt-oss-120b', provider=GroqProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: GroqModelSettings(groq_reasoning_effort='high'),
        'low',
        lambda p: p['body'].get('reasoning_effort'),
        overrides=True,
        marks=_SKIP_WITHOUT_GROQ,
    ),
    _CrossScopeCase(
        # qwen3 has no graded effort, but an explicit `'none'` disables the reasoning `thinking` enables.
        'groq-qwen3-reasoning-effort-none',
        lambda s: legacy_http_probe(
            lambda c: GroqModel('qwen/qwen3-32b', provider=GroqProvider(api_key=PROBE_KEY, http_client=c), settings=s)
        ),
        lambda: GroqModelSettings(groq_reasoning_effort='none'),
        'high',
        lambda p: p['body'].get('reasoning_effort'),
        overrides=True,
        marks=_SKIP_WITHOUT_GROQ,
    ),
    _CrossScopeCase(
        # Any other qwen3 effort keeps the reasoning `thinking` enables.
        'groq-qwen3-reasoning-effort-default',
        lambda s: legacy_http_probe(
            lambda c: GroqModel('qwen/qwen3-32b', provider=GroqProvider(api_key=PROBE_KEY, http_client=c), settings=s)
        ),
        lambda: GroqModelSettings(groq_reasoning_effort='default'),
        'high',
        lambda p: p['body'].get('reasoning_effort') == 'none',
        overrides=False,
        marks=_SKIP_WITHOUT_GROQ,
    ),
    _CrossScopeCase(
        'bedrock-anthropic-thinking',
        _bedrock_probe('anthropic.claude-sonnet-4-5-20250929-v1:0'),
        lambda: BedrockModelSettings(
            bedrock_additional_model_requests_fields={'thinking': {'type': 'enabled', 'budget_tokens': 5000}}
        ),
        'low',
        _bedrock_sent,
        overrides=True,
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    _CrossScopeCase(
        # Adaptive Claude takes the effort an effort-level `thinking` maps to from `output_config`.
        'bedrock-anthropic-output-config',
        _bedrock_probe('us.anthropic.claude-sonnet-4-6'),
        lambda: BedrockModelSettings(bedrock_additional_model_requests_fields={'output_config': {'effort': 'low'}}),
        'high',
        lambda p: json.loads(p['body'])['additionalModelRequestFields']['output_config'],
        overrides=True,
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    _CrossScopeCase(
        # Budget-based Claude maps `thinking` to `thinking` alone, which an explicit `output_config` leaves in place.
        'bedrock-anthropic-output-config-without-effort',
        _bedrock_probe('anthropic.claude-sonnet-4-5-20250929-v1:0'),
        lambda: BedrockModelSettings(bedrock_additional_model_requests_fields={'output_config': {'effort': 'low'}}),
        'high',
        lambda p: json.loads(p['body'])['additionalModelRequestFields']['thinking'],
        overrides=False,
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    _CrossScopeCase(
        'bedrock-openai-reasoning-effort',
        _bedrock_probe('openai.gpt-oss-120b-1:0'),
        lambda: BedrockModelSettings(bedrock_additional_model_requests_fields={'reasoning_effort': 'high'}),
        'low',
        _bedrock_sent,
        overrides=True,
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    _CrossScopeCase(
        'bedrock-openai-responses-reasoning',
        _bedrock_probe('us.openai.gpt-5.6-luna'),
        lambda: BedrockModelSettings(bedrock_additional_model_requests_fields={'reasoning': {'effort': 'high'}}),
        'low',
        _bedrock_sent,
        overrides=True,
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    _CrossScopeCase(
        'bedrock-qwen-reasoning-config',
        _bedrock_probe('qwen.qwen3-32b-v1:0'),
        lambda: BedrockModelSettings(bedrock_additional_model_requests_fields={'reasoning_config': 'high'}),
        'low',
        _bedrock_sent,
        overrides=True,
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    _CrossScopeCase(
        'bedrock-xai-reasoning',
        _bedrock_probe('us.xai.grok-4.7'),
        lambda: BedrockModelSettings(bedrock_additional_model_requests_fields={'reasoning': {'effort': 'high'}}),
        'low',
        _bedrock_sent,
        overrides=True,
        marks=_SKIP_WITHOUT_BEDROCK,
    ),
    _CrossScopeCase(
        'openrouter-reasoning',
        lambda s: http_probe(
            lambda c: OpenRouterModel(
                'openai/gpt-5', provider=OpenRouterProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: OpenRouterModelSettings(openrouter_reasoning={'effort': 'high'}),
        'low',
        lambda p: p['body'].get('reasoning'),
        overrides=True,
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    _CrossScopeCase(
        # Sent beside the unified mapping, which still goes out unchanged.
        'openrouter-openai-reasoning-effort',
        lambda s: http_probe(
            lambda c: OpenRouterModel(
                'openai/gpt-5', provider=OpenRouterProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: OpenAIChatModelSettings(openai_reasoning_effort='high'),
        'low',
        lambda p: p['body'].get('reasoning'),
        overrides=False,
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    _CrossScopeCase(
        'xai-reasoning-effort',
        lambda s: xai_sdk_probe(
            lambda client: XaiModel('grok-3-mini', provider=XaiProvider(xai_client=client), settings=s)
        ),
        lambda: XaiModelSettings(xai_reasoning_effort='high'),
        'low',
        lambda p: p.get('reasoning_effort'),
        overrides=True,
        marks=_SKIP_WITHOUT_XAI,
    ),
    _CrossScopeCase(
        # Cerebras reasons by default and sends no effort for an enabled `thinking`, so an explicit one wins.
        'cerebras-reasoning-effort',
        lambda s: http_probe(
            lambda c: CerebrasModel(
                'zai-glm-4.7', provider=CerebrasProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: OpenAIChatModelSettings(openai_reasoning_effort='low'),
        'high',
        lambda p: p['body'].get('reasoning_effort'),
        overrides=True,
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    _CrossScopeCase(
        # `thinking=False` is sent as `reasoning_effort='none'`, replacing the explicit effort.
        'cerebras-thinking-disabled',
        lambda s: http_probe(
            lambda c: CerebrasModel(
                'zai-glm-4.7', provider=CerebrasProvider(api_key=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: OpenAIChatModelSettings(openai_reasoning_effort='high'),
        False,
        lambda p: p['body'].get('reasoning_effort'),
        overrides=False,
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    _CrossScopeCase(
        'snowflake-claude-reasoning',
        lambda s: http_probe(
            lambda c: SnowflakeModel(
                'claude-sonnet-4-5',
                provider=SnowflakeProvider(account='probe', token=PROBE_KEY, http_client=c),
                settings=s,
            )
        ),
        lambda: SnowflakeModelSettings(snowflake_reasoning={'effort': 'high'}),
        'low',
        lambda p: p['body'].get('reasoning'),
        overrides=True,
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    _CrossScopeCase(
        'snowflake-openai-reasoning-effort',
        lambda s: http_probe(
            lambda c: SnowflakeModel(
                'openai-gpt-5', provider=SnowflakeProvider(account='probe', token=PROBE_KEY, http_client=c), settings=s
            )
        ),
        lambda: OpenAIChatModelSettings(openai_reasoning_effort='high'),
        'low',
        lambda p: p['body'].get('reasoning_effort'),
        overrides=True,
        marks=_SKIP_WITHOUT_OPENAI,
    ),
    _CrossScopeCase(
        # Z.AI maps `thinking` to `extra_body.thinking` and never sends `openai_reasoning_effort`.
        'zai-openai-reasoning-effort',
        lambda s: http_probe(
            lambda c: ZaiModel('glm-4.6', provider=ZaiProvider(api_key=PROBE_KEY, http_client=c), settings=s)
        ),
        lambda: OpenAIChatModelSettings(openai_reasoning_effort='high'),
        'low',
        lambda p: (p['body'].get('thinking'), p['body'].get('reasoning_effort')),
        overrides=False,
        marks=_SKIP_WITHOUT_OPENAI,
    ),
]


async def _send(probe: Probe, settings: ModelSettings) -> tuple[dict[str, Any], list[str]]:
    """The probed request, and any cross-scope precedence warnings it raised."""
    # The probe records the warnings its request raises; `always` keeps the suite's `error` filter from aborting it.
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        result = await probe(settings)
    assert result.payload is not None
    return json.loads(result.payload), [
        m for m in result.observed.warning_messages if 'takes precedence over `thinking' in m
    ]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in _CROSS_SCOPE_CASES])
async def test_cross_scope_thinking_override_warning_matches_the_wire(
    case: _CrossScopeCase, allow_model_requests: None
):
    """A model's setting warns over a request's `thinking` exactly when, at one scope, it changes what `thinking` sends.

    Not a VCR test, for the reason `tests/models/test_model_settings_support.py` gives: a frozen recording keeps
    matching after an adapter changes what it sends, and this pins each adapter's precedence against its own wire.
    """
    setting = case.setting()
    key = next(iter(setting))
    thinking = ModelSettings(thinking=case.thinking)
    same_scope = case.probe(None)
    sent_thinking, _ = await _send(same_scope, thinking)
    sent_setting, _ = await _send(same_scope, setting)
    sent_both, same_scope_warnings = await _send(same_scope, {**setting, **thinking})
    _, cross_scope_warnings = await _send(case.probe(setting), thinking)

    assert same_scope_warnings == []
    if case.overrides:
        assert case.sent(sent_both) == case.sent(sent_setting) != case.sent(sent_thinking)
        assert cross_scope_warnings == [
            f"`{key}` from the model's `settings` takes precedence over `thinking={case.thinking!r}` from the "
            f"request's `model_settings`. Set `{key}` in the request's `model_settings` instead of `thinking`, or "
            "remove it from the model's `settings`."
        ]
    else:
        assert case.sent(sent_both) == case.sent(sent_thinking)
        assert cross_scope_warnings == []


def _mock_openai_chat(model_name: str, settings: ModelSettings) -> tuple[OpenAIChatModel, AsyncOpenAI]:
    client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='4', role='assistant')))
    return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client), settings=settings), client


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestCrossScopeThinkingOverrideWarning:
    """Which agent layers the warning names. The per-adapter precedence is pinned against the wire above.

    A mocked client stands in for recordings: the warning is decided before any request goes out, and the recorded
    runs in `tests/test_thinking_wire_contract.py` already pin the model, agent and direct-request scopes. Assertions
    keep only `UserWarning`s, since `simplefilter('always')` also records a `ResourceWarning` for any socket another
    test left for the garbage collector.
    """

    async def test_agent_thinking_overridden_by_model_setting(self, allow_model_requests: None):
        model, client = _mock_openai_chat('gpt-5', OpenAIChatModelSettings(openai_reasoning_effort='low'))
        agent = Agent(model, model_settings=ModelSettings(thinking='high'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await agent.run('What is 2+2?')
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == [
            "`openai_reasoning_effort` from the model's `settings` takes precedence over `thinking='high'` "
            "from the agent's `model_settings`. Set `openai_reasoning_effort` in the agent's `model_settings` "
            "instead of `thinking`, or remove it from the model's `settings`."
        ]
        assert get_mock_chat_completion_kwargs(client)[0]['reasoning_effort'] == 'low'

    async def test_wrapped_model_setting_overriding_run_thinking(self, allow_model_requests: None):
        model, _ = _mock_openai_chat('gpt-5', OpenAIChatModelSettings(openai_reasoning_effort='low'))
        agent = Agent(WrapperModel(model))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await agent.run('What is 2+2?', model_settings=ModelSettings(thinking='high'))
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == [
            "`openai_reasoning_effort` from the model's `settings` takes precedence over `thinking='high'` "
            "from the run's `model_settings`. Set `openai_reasoning_effort` in the run's `model_settings` "
            "instead of `thinking`, or remove it from the model's `settings`."
        ]

    async def test_fallback_child_setting_overriding_run_thinking(self, allow_model_requests: None):
        """A `FallbackModel` has no settings of its own, so its child checks its own as for a direct request."""
        model, client = _mock_openai_chat('gpt-5', OpenAIChatModelSettings(openai_reasoning_effort='low'))
        agent = Agent(FallbackModel(model))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await agent.run('What is 2+2?', model_settings=ModelSettings(thinking='high'))
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == [
            "`openai_reasoning_effort` from the model's `settings` takes precedence over `thinking='high'` "
            "from the request's `model_settings`. Set `openai_reasoning_effort` in the request's `model_settings` "
            "instead of `thinking`, or remove it from the model's `settings`."
        ]
        assert get_mock_chat_completion_kwargs(client)[0]['reasoning_effort'] == 'low'

    async def test_fallback_agent_setting_overriding_run_thinking(self, allow_model_requests: None):
        """The agent's layers reach each model a `FallbackModel` may use."""
        model, client = _mock_openai_chat('gpt-5', ModelSettings())
        agent = Agent(FallbackModel(model), model_settings=OpenAIChatModelSettings(openai_reasoning_effort='low'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await agent.run('What is 2+2?', model_settings=ModelSettings(thinking='high'))
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == [
            "`openai_reasoning_effort` from the agent's `model_settings` takes precedence over `thinking='high'` "
            "from the run's `model_settings`. Set `openai_reasoning_effort` in the run's `model_settings` "
            "instead of `thinking`, or remove it from the agent's `model_settings`."
        ]
        assert get_mock_chat_completion_kwargs(client)[0]['reasoning_effort'] == 'low'

    async def test_override_thinking_overridden_by_model_setting(self, allow_model_requests: None):
        model, _ = _mock_openai_chat('gpt-5', OpenAIChatModelSettings(openai_reasoning_effort='low'))
        agent = Agent(model)
        with (
            agent.override(model_settings=ModelSettings(thinking='high')),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter('always')
            await agent.run('What is 2+2?')
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == [
            "`openai_reasoning_effort` from the model's `settings` takes precedence over `thinking='high'` "
            'from `override(model_settings=...)`. Set `openai_reasoning_effort` in `override(model_settings=...)` '
            "instead of `thinking`, or remove it from the model's `settings`."
        ]

    async def test_thinking_capability_overridden_by_model_setting(self, allow_model_requests: None):
        model, client = _mock_openai_chat('gpt-5', OpenAIChatModelSettings(openai_reasoning_effort='low'))
        agent = Agent(model, capabilities=[Thinking(effort='high')])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await agent.run('What is 2+2?')
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == [
            "`openai_reasoning_effort` from the model's `settings` takes precedence over `thinking='high'` "
            "from a capability. Remove `openai_reasoning_effort` from the model's `settings` to apply `thinking`."
        ]
        assert get_mock_chat_completion_kwargs(client)[0]['reasoning_effort'] == 'low'

    async def test_each_model_a_step_selects_is_checked_once(self, allow_model_requests: None):
        """`SelectModel` can switch models between steps, so each model is checked when a step first uses it."""
        plain, plain_client = _mock_openai_chat('gpt-5', ModelSettings())
        tuned_client = MockOpenAI.create_mock(
            [completion_message(ChatCompletionMessage(content='4', role='assistant'))] * 2
        )
        tuned = OpenAIChatModel(
            'gpt-5',
            provider=OpenAIProvider(openai_client=tuned_client),
            settings=OpenAIChatModelSettings(openai_reasoning_effort='low'),
        )
        agent = Agent(capabilities=[SelectModel(lambda ctx: plain if ctx.run_step == 1 else tuned)], retries=2)
        attempts = 0

        @agent.output_validator
        def retry_twice(output: str) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ModelRetry('Try again.')
            return output

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await agent.run('What is 2+2?', model_settings=ModelSettings(thinking='high'))
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == [
            "`openai_reasoning_effort` from the model's `settings` takes precedence over `thinking='high'` "
            "from the run's `model_settings`. Set `openai_reasoning_effort` in the run's `model_settings` "
            "instead of `thinking`, or remove it from the model's `settings`."
        ]
        assert [kwargs['reasoning_effort'] for kwargs in get_mock_chat_completion_kwargs(plain_client)] == ['high']
        assert [kwargs['reasoning_effort'] for kwargs in get_mock_chat_completion_kwargs(tuned_client)] == [
            'low',
            'low',
        ]

    async def test_model_replaced_in_for_run_is_silent(self, allow_model_requests: None):
        """A model that a capability's `for_run` replaces sends no request, so its settings override nothing."""
        replaced, _ = _mock_openai_chat('gpt-5', OpenAIChatModelSettings(openai_reasoning_effort='low'))
        replacement, client = _mock_openai_chat('gpt-5', ModelSettings())

        @dataclass
        class ReplaceModel(AbstractCapability[None]):
            replacement_model: Model | None = None

            def get_model(self) -> Model | None:
                return self.replacement_model

            async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
                return ReplaceModel(replacement)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await Agent(replaced, capabilities=[ReplaceModel()], deps_type=type(None)).run(
                'What is 2+2?', model_settings=ModelSettings(thinking='high')
            )
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == []
        assert get_mock_chat_completion_kwargs(client)[0]['reasoning_effort'] == 'high'

    async def test_narrower_setting_over_broader_thinking_is_silent(self, allow_model_requests: None):
        """A provider-specific setting narrower than `thinking` is the caller's explicit choice, so nothing warns."""
        model, client = _mock_openai_chat('gpt-5', ModelSettings(thinking='high'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await Agent(model).run(
                'What is 2+2?', model_settings=OpenAIChatModelSettings(openai_reasoning_effort='low')
            )
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == []
        assert get_mock_chat_completion_kwargs(client)[0]['reasoning_effort'] == 'low'

    async def test_thinking_the_model_ignores_is_silent(self, allow_model_requests: None):
        """`gpt-4o` drops `thinking` whatever else is set, so no setting took precedence over it."""
        model, _ = _mock_openai_chat('gpt-4o', OpenAIChatModelSettings(openai_reasoning_effort='low'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await Agent(model).run('What is 2+2?', model_settings=ModelSettings(thinking='high'))
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == []

    async def test_disabling_thinking_on_an_always_on_model_is_overridden_too(self, allow_model_requests: None):
        """`gpt-5` always reasons, but `thinking=False` still asks for its lowest effort, which a broader setting replaces."""
        model, client = _mock_openai_chat('gpt-5', OpenAIChatModelSettings(openai_reasoning_effort='low'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await Agent(model).run('What is 2+2?', model_settings=ModelSettings(thinking=False))
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == [
            "`openai_reasoning_effort` from the model's `settings` takes precedence over `thinking=False` "
            "from the run's `model_settings`. Set `openai_reasoning_effort` in the run's `model_settings` "
            "instead of `thinking`, or remove it from the model's `settings`."
        ]
        assert get_mock_chat_completion_kwargs(client)[0]['reasoning_effort'] == 'low'
