"""Tests for the unified thinking/reasoning feature.

Tests the base Model.prepare_request() thinking resolution, per-provider translation,
the Thinking capability, and end-to-end integration via FunctionModel.
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false
from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import CAPABILITY_TYPES, Thinking
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.google import GoogleModelProfile, google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.groq import groq_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
from pydantic_ai.settings import ModelSettings, ThinkingLevel
from pydantic_ai.tools import ToolDefinition

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as anthropic_imports:
    from anthropic import omit as anthropic_omit
    from anthropic.types.beta import BetaThinkingConfigParam

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_imports:
    from openai import omit as openai_omit
    from openai.types.chat import ChatCompletionMessage

    from pydantic_ai.models.cerebras import (
        CerebrasModel,
        CerebrasModelSettings,
        _cerebras_settings_to_openai_settings,
    )
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import (
        OpenRouterModel,
        OpenRouterModelSettings,
        _openrouter_settings_to_openai_settings,
    )
    from pydantic_ai.providers.cerebras import CerebrasProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.vllm import VLLMProvider

    from .models.mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel

with try_import() as groq_imports:
    from groq import NOT_GIVEN as groq_NOT_GIVEN

    from pydantic_ai.models.groq import GroqModel

with try_import() as bedrock_imports:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider

with try_import() as xai_imports:
    from pydantic_ai.models.xai import XaiModel, XaiModelSettings
    from pydantic_ai.providers.xai import XaiProvider

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
        """Cannot disable thinking on always-on models; silently ignored."""
        model = _make_model(thinking_always_enabled=True)
        assert _resolve_thinking(model, thinking=False) is None

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
        assert result == snapshot({'type': 'adaptive', 'display': 'summarized'})

    def test_thinking_true_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking=True with non-adaptive model -> {'type': 'enabled', 'budget_tokens': 10000}."""
        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 10000, 'display': 'summarized'})

    def test_thinking_high_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking='high' with non-adaptive -> budget_tokens=16384."""
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 16384, 'display': 'summarized'})

    def test_thinking_low_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking='low' with non-adaptive -> budget_tokens=2048."""
        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 2048, 'display': 'summarized'})

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
        assert thinking_param == snapshot({'type': 'adaptive', 'display': 'summarized'})

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
        assert result == {'type': 'adaptive', 'display': 'summarized'}

    def test_low_uses_adaptive_on_adaptive(self, adaptive_model: FunctionModel):
        """thinking='low' on adaptive model -> adaptive (effort controlled via output_config)."""
        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == {'type': 'adaptive', 'display': 'summarized'}

    def test_high_uses_adaptive_on_adaptive(self, adaptive_model: FunctionModel):
        """thinking='high' on adaptive model -> adaptive (effort controlled via output_config)."""
        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == {'type': 'adaptive', 'display': 'summarized'}

    def test_display_summarized_on_adaptive(self, adaptive_model: FunctionModel):
        params = ModelRequestParameters(thinking=True)
        settings: AnthropicModelSettings = {'anthropic_thinking_display': 'summarized'}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == snapshot({'type': 'adaptive', 'display': 'summarized'})

    def test_display_omitted_on_adaptive(self, adaptive_model: FunctionModel):
        params = ModelRequestParameters(thinking=True)
        settings: AnthropicModelSettings = {'anthropic_thinking_display': 'omitted'}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == snapshot({'type': 'adaptive', 'display': 'omitted'})

    def test_display_on_enabled_shape(self, non_adaptive_model: FunctionModel):
        """display is also emitted on the budget-based 'enabled' shape (pre-adaptive models)."""
        params = ModelRequestParameters(thinking='high')
        settings: AnthropicModelSettings = {'anthropic_thinking_display': 'summarized'}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 16384, 'display': 'summarized'})

    def test_display_ignored_when_thinking_false(self, adaptive_model: FunctionModel):
        """display setting is dropped on the OMIT path — no thinking block to attach to."""
        params = ModelRequestParameters(thinking=False)
        settings: AnthropicModelSettings = {'anthropic_thinking_display': 'summarized'}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result is anthropic_omit

    def test_display_ignored_when_thinking_none(self, adaptive_model: FunctionModel):
        """display setting is dropped on the OMIT path — no thinking block to attach to."""
        params = ModelRequestParameters(thinking=None)
        settings: AnthropicModelSettings = {'anthropic_thinking_display': 'summarized'}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result is anthropic_omit

    def test_raw_anthropic_thinking_wins_over_display(self, adaptive_model: FunctionModel):
        """Raw `anthropic_thinking` dict wins; `anthropic_thinking_display` is not injected on top."""
        params = ModelRequestParameters(thinking=True)
        settings: AnthropicModelSettings = {
            'anthropic_thinking': {'type': 'adaptive'},
            'anthropic_thinking_display': 'summarized',
        }
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == snapshot({'type': 'adaptive'})


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
        assert result == {'thinking': {'type': 'enabled', 'budget_tokens': 10000, 'display': 'summarized'}}

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

    def test_openai_variant_thinking_false(self):
        """thinking=False on OpenAI Bedrock variant is a no-op (Bedrock rejects 'none')."""
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='openai',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = model._build_additional_model_request_fields(settings, params, [])
        # thinking=False: no reasoning_effort set, returns None
        assert result is None

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
        """thinking=False on Qwen variant is a no-op (Qwen has no disable mechanism)."""
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='qwen',
                supports_thinking=True,
            )
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = model._build_additional_model_request_fields(settings, params, [])
        # thinking=False on Qwen: no reasoning_config set, returns None (empty dict is falsy)
        assert result is None

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
        """thinking=False on the xAI Bedrock variant is a no-op (Grok always reasons).

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
        assert result is None

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
        assert result == {'thinking': {'type': 'adaptive', 'display': 'summarized'}}

    def test_anthropic_variant_adaptive_thinking_high_sets_effort(self):
        """thinking='high' on adaptive model adds output_config.effort sibling per AWS docs."""
        model = self._adaptive_model()
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking='high'), []
        )
        assert result == {
            'thinking': {'type': 'adaptive', 'display': 'summarized'},
            'output_config': {'effort': 'high'},
        }

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
        assert result == {
            'thinking': {'type': 'adaptive', 'display': 'summarized'},
            'output_config': {'effort': effort},
        }

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
        assert result == {
            'thinking': {'type': 'adaptive', 'display': 'summarized'},
            'output_config': {'effort': 'xhigh'},
        }

    def test_anthropic_variant_adaptive_no_effort_when_unsupported(self):
        """Effort is omitted when the profile doesn't advertise bedrock_supports_effort."""
        model = self._adaptive_model(supports_effort=False)
        result = model._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(thinking='high'), []
        )
        assert result == {'thinking': {'type': 'adaptive', 'display': 'summarized'}}

    def test_anthropic_variant_adaptive_user_output_config_wins(self):
        """User-provided output_config in bedrock_additional_model_requests_fields is preserved."""
        model = self._adaptive_model()
        settings = BedrockModelSettings(bedrock_additional_model_requests_fields={'output_config': {'effort': 'low'}})
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking='high'), [])
        assert result == {'thinking': {'type': 'adaptive', 'display': 'summarized'}, 'output_config': {'effort': 'low'}}

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

    def test_anthropic_variant_adaptive_display_summarized(self):
        model = self._adaptive_model()
        settings = BedrockModelSettings(bedrock_thinking_display='summarized')
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking=True), [])
        assert result == snapshot({'thinking': {'type': 'adaptive', 'display': 'summarized'}})

    def test_anthropic_variant_adaptive_display_omitted(self):
        model = self._adaptive_model()
        settings = BedrockModelSettings(bedrock_thinking_display='omitted')
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking=True), [])
        assert result == snapshot({'thinking': {'type': 'adaptive', 'display': 'omitted'}})

    def test_anthropic_variant_adaptive_display_with_effort(self):
        """display coexists with output_config.effort on adaptive models."""
        model = self._adaptive_model()
        settings = BedrockModelSettings(bedrock_thinking_display='summarized')
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking='high'), [])
        assert result == snapshot(
            {'thinking': {'type': 'adaptive', 'display': 'summarized'}, 'output_config': {'effort': 'high'}}
        )

    def test_anthropic_variant_enabled_display(self):
        """display is also attached to the budget-based 'enabled' shape (pre-adaptive Anthropic models)."""
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='anthropic',
                supports_thinking=True,
            )
        )
        settings = BedrockModelSettings(bedrock_thinking_display='summarized')
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking='high'), [])
        assert result == snapshot({'thinking': {'type': 'enabled', 'budget_tokens': 16384, 'display': 'summarized'}})

    def test_anthropic_variant_display_dropped_when_thinking_false(self):
        """display is not injected on the 'disabled' shape — Anthropic's API doesn't accept it there."""
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='anthropic',
                supports_thinking=True,
            )
        )
        settings = BedrockModelSettings(bedrock_thinking_display='summarized')
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking=False), [])
        assert result == snapshot({'thinking': {'type': 'disabled'}})

    def test_anthropic_variant_adaptive_display_dropped_when_thinking_false(self):
        model = self._adaptive_model()
        settings = BedrockModelSettings(bedrock_thinking_display='summarized')
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking=False), [])
        assert result is None

    def test_openai_variant_ignores_display(self):
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='openai',
                supports_thinking=True,
            )
        )
        settings = BedrockModelSettings(bedrock_thinking_display='summarized')
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking='high'), [])
        assert result == snapshot({'reasoning_effort': 'high'})

    def test_qwen_variant_ignores_display(self):
        model = _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='qwen',
                supports_thinking=True,
            )
        )
        settings = BedrockModelSettings(bedrock_thinking_display='summarized')
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking=True), [])
        assert result == snapshot({'reasoning_config': 'high'})

    def test_anthropic_variant_raw_thinking_dict_wins_over_display(self):
        """Raw `thinking` dict in additionalModelRequestFields wins; display is not injected on top."""
        model = self._adaptive_model()
        settings = BedrockModelSettings(
            bedrock_thinking_display='summarized',
            bedrock_additional_model_requests_fields={'thinking': {'type': 'adaptive'}},
        )
        result = model._build_additional_model_request_fields(settings, ModelRequestParameters(thinking=True), [])
        assert result == snapshot({'thinking': {'type': 'adaptive'}})


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

    @pytest.mark.parametrize(
        ('model_name', 'expected_always_enabled'),
        [
            # Hybrid routes — upstream can honor `effort='none'`; gate forwards `thinking=False`.
            pytest.param('moonshotai/kimi-k2.6', False, id='kimi-k2-hybrid'),
            pytest.param('qwen/qwen3-235b-a22b', False, id='qwen3-hybrid'),
            pytest.param('z-ai/glm-4.6', False, id='glm-hybrid'),
            pytest.param('openai/gpt-oss-120b', False, id='gpt-oss-hybrid'),
            pytest.param('anthropic/claude-sonnet-4.5', False, id='claude-hybrid'),
            # Always-on routes — upstream cannot disable; gate silently drops `thinking=False`,
            # matching the same model's direct-route behavior.
            pytest.param('openai/o3', True, id='o3-always-on'),
            pytest.param('openai/gpt-5', True, id='gpt-5-always-on'),
            pytest.param('mistralai/magistral-medium-2509', True, id='magistral-always-on'),
            pytest.param('deepseek/deepseek-r1', True, id='deepseek-r1-always-on'),
            pytest.param('x-ai/grok-3-mini', True, id='grok-3-mini-always-on'),
        ],
    )
    def test_provider_profile_propagates_thinking_capability(self, model_name: str, expected_always_enabled: bool):
        """OpenRouter's base profile sets `supports_thinking=True` so the gate forwards
        `thinking` to the transformer for every routed model; `thinking_always_enabled`
        propagates from the sub-profile via `merge_profile` so always-on upstream routes
        silently drop `thinking=False` at the gate — matching their direct-route behavior
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

    def test_grok_3_mini_profile_drops_thinking_false_via_gate(self):
        """The `thinking_always_enabled` flag is what enforces the gate drop for grok-3-mini.

        Two instances because `Model.profile` is a `cached_property`."""
        always_on = XaiModel(
            'grok-3-mini', provider=XaiProvider(api_key='mock-api-key'), profile=grok_model_profile('grok-3-mini')
        )
        _, resolved = always_on.prepare_request(XaiModelSettings(thinking=False), ModelRequestParameters())
        assert resolved.thinking is None

        non_always_on = XaiModel(
            'grok-4',
            provider=XaiProvider(api_key='mock-api-key'),
            profile=ModelProfile(supports_thinking=True, thinking_always_enabled=False),
        )
        _, resolved = non_always_on.prepare_request(XaiModelSettings(thinking=False), ModelRequestParameters())
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

    @pytest.mark.parametrize('summary', [True, False])
    def test_get_model_settings_summary(self, summary: bool):
        cap = Thinking(effort='high', summary=summary)
        assert cap.get_model_settings() == ModelSettings(thinking='high', thinking_summary=summary)

    def test_summary_unset_leaves_thinking_summary_out(self):
        assert 'thinking_summary' not in (Thinking().get_model_settings() or {})

    def test_from_spec_with_summary(self):
        cap = Thinking.from_spec(summary=True)
        assert isinstance(cap, Thinking)
        assert cap.summary is True

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
        """Thinking(effort=False) on always-on model -> silently ignored."""
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
        assert captured_params[0].thinking is None

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
        assert profile.get('grok_reasoning_efforts') == frozenset({'low', 'high'})

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
        """grok-3-mini's `reasoning_effort` has no `'none'` value, so the profile
        is marked always-on and the gate drops `thinking=False`."""
        profile = grok_model_profile('grok-3-mini')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is True

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
    def test_bedrock_qwen_variant_thinking_always_enabled(self):
        """Bedrock-Converse exposes only `reasoning_config ∈ {low, high}` for Qwen3;
        marked always-on so the gate drops `thinking=False`."""
        from pydantic_ai.providers.bedrock import BedrockProvider

        profile = BedrockProvider.model_profile('qwen.qwen3-32b-v1:0')
        assert profile is not None
        assert profile.get('supports_thinking', False) is True
        assert profile.get('thinking_always_enabled', False) is True


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

    def test_thinking_false_silently_dropped_on_always_enabled_models(self):
        """`thinking=False` is stripped from params on always-on profiles, codifying
        the contract advertised at docs/thinking.md ("silently ignored on always-on
        models"). Non-False values flow through unchanged."""
        model = _make_model(supports_thinking=True, thinking_always_enabled=True)
        _merged, params = model.prepare_request({'thinking': False}, ModelRequestParameters())
        assert params.thinking is None

        _merged, params = model.prepare_request({'thinking': 'high'}, ModelRequestParameters())
        assert params.thinking == 'high'

    @pytest.mark.parametrize(
        'model_name',
        [
            'openai/o3',
            'openai/gpt-5',
            'mistralai/magistral-medium-2509',
            'deepseek/deepseek-r1',
            'x-ai/grok-3-mini',
        ],
    )
    def test_thinking_false_silently_dropped_on_always_on_openrouter_routes(self, model_name: str):
        """Aggregator-route consistency: OpenRouter routes to always-on upstream models
        silently drop `thinking=False` at the gate — same behavior the direct route to
        the same model would exhibit. `ModelProfile.update()` propagates
        `thinking_always_enabled=True` from the sub-profile, no aggregator-specific
        override required."""
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        profile = OpenRouterProvider.model_profile(model_name)
        assert profile is not None
        model = FunctionModel(_echo, profile=profile)
        _merged, params = model.prepare_request({'thinking': False}, ModelRequestParameters())
        assert params.thinking is None
        _merged, params = model.prepare_request({'thinking': 'high'}, ModelRequestParameters())
        assert params.thinking == 'high'


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
# Unified `thinking_summary`
# ---------------------------------------------------------------------------
# Unit-pinned like the translation tests above: the cassette matchers don't read the body, so asserting the
# translated field directly is what catches drift. The headline paths are recorded in
# `tests/test_thinking_summary.py`.


@pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')
class TestAnthropicThinkingSummary:
    """`thinking_summary` as Anthropic's `thinking.display`."""

    @pytest.fixture
    def adaptive_model(self):
        return FunctionModel(
            _echo, profile=AnthropicModelProfile(supports_thinking=True, anthropic_supports_adaptive_thinking=True)
        )

    @pytest.fixture
    def thinks_by_default_model(self):
        """Claude Opus 5 and later: they think without a `thinking` parameter and can turn it off."""
        return FunctionModel(
            _echo,
            profile=AnthropicModelProfile(
                supports_thinking=True, anthropic_supports_adaptive_thinking=True, thinking_enabled_by_default=True
            ),
        )

    @pytest.mark.parametrize(
        ('thinking_summary', 'display'), [(True, 'summarized'), (False, 'omitted')], ids=['true', 'false']
    )
    def test_with_unified_thinking(self, adaptive_model: FunctionModel, thinking_summary: bool, display: str):
        settings: AnthropicModelSettings = {'thinking_summary': thinking_summary}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, ModelRequestParameters(thinking='high'))
        assert result == {'type': 'adaptive', 'display': display}

    def test_unset_without_thinking_sends_nothing(self, thinks_by_default_model: FunctionModel):
        """Neither setting: byte-for-byte what was sent before `thinking_summary` existed."""
        result = AnthropicModel._translate_thinking(thinks_by_default_model, {}, ModelRequestParameters())
        assert result is anthropic_omit

    @pytest.mark.parametrize(
        ('thinking_summary', 'display'), [(True, 'summarized'), (False, 'omitted')], ids=['true', 'false']
    )
    def test_model_that_thinks_by_default_gets_an_explicit_adaptive_block(
        self, thinks_by_default_model: FunctionModel, thinking_summary: bool, display: str
    ):
        settings: AnthropicModelSettings = {'thinking_summary': thinking_summary}
        result = AnthropicModel._translate_thinking(thinks_by_default_model, settings, ModelRequestParameters())
        assert result == {'type': 'adaptive', 'display': display}

    def test_model_that_only_thinks_when_asked_sends_nothing(self, adaptive_model: FunctionModel):
        settings: AnthropicModelSettings = {'thinking_summary': True}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, ModelRequestParameters())
        assert result is anthropic_omit

    def test_disabled_block_takes_no_display(self, thinks_by_default_model: FunctionModel):
        """`display` on `{'type': 'disabled'}` is a 400 ("Extra inputs are not permitted")."""
        settings: AnthropicModelSettings = {'thinking_summary': True}
        result = AnthropicModel._translate_thinking(
            thinks_by_default_model, settings, ModelRequestParameters(thinking=False)
        )
        assert result == {'type': 'disabled'}

    def test_thinking_false_sends_nothing(self, adaptive_model: FunctionModel):
        settings: AnthropicModelSettings = {'thinking_summary': True}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, ModelRequestParameters(thinking=False))
        assert result is anthropic_omit

    def test_provider_display_wins(self, adaptive_model: FunctionModel):
        settings: AnthropicModelSettings = {'thinking_summary': True, 'anthropic_thinking_display': 'omitted'}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, ModelRequestParameters(thinking=True))
        assert result == {'type': 'adaptive', 'display': 'omitted'}

    def test_provider_display_alone_on_a_model_that_thinks_by_default(self, thinks_by_default_model: FunctionModel):
        settings: AnthropicModelSettings = {'anthropic_thinking_display': 'summarized'}
        result = AnthropicModel._translate_thinking(thinks_by_default_model, settings, ModelRequestParameters())
        assert result == {'type': 'adaptive', 'display': 'summarized'}

    def test_raw_anthropic_thinking_wins(self, adaptive_model: FunctionModel):
        """`between_tools` (like `disabled`) rejects `display`, so a raw dict is never touched."""
        settings: AnthropicModelSettings = {'thinking_summary': True, 'anthropic_thinking': {'type': 'between_tools'}}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, ModelRequestParameters(thinking=True))
        assert result == {'type': 'between_tools'}


@pytest.mark.skipif(not bedrock_imports(), reason='boto3 not installed')
class TestBedrockThinkingSummary:
    """`thinking_summary` on Bedrock: Claude's `thinking.display` and gpt-oss's `include_reasoning`."""

    @staticmethod
    def _claude(*, thinks_by_default: bool = False) -> BedrockConverseModel:
        return _bedrock_model(
            BedrockModelProfile(
                bedrock_thinking_variant='anthropic',
                bedrock_supports_adaptive_thinking=True,
                supports_thinking=True,
                thinking_enabled_by_default=thinks_by_default,
            )
        )

    @staticmethod
    def _gpt_oss() -> BedrockConverseModel:
        return _bedrock_model(
            BedrockModelProfile(bedrock_thinking_variant='openai', supports_thinking=True, thinking_always_enabled=True)
        )

    def test_claude_false_with_thinking(self):
        settings = BedrockModelSettings(thinking_summary=False)
        result = self._claude()._build_additional_model_request_fields(
            settings, ModelRequestParameters(thinking=True), []
        )
        assert result == {'thinking': {'type': 'adaptive', 'display': 'omitted'}}

    @pytest.mark.parametrize(
        ('thinking_summary', 'display'), [(True, 'summarized'), (False, 'omitted')], ids=['true', 'false']
    )
    def test_claude_that_thinks_by_default(self, thinking_summary: bool, display: str):
        settings = BedrockModelSettings(thinking_summary=thinking_summary)
        result = self._claude(thinks_by_default=True)._build_additional_model_request_fields(
            settings, ModelRequestParameters(), []
        )
        assert result == {'thinking': {'type': 'adaptive', 'display': display}}

    def test_claude_unset_without_thinking_sends_nothing(self):
        result = self._claude(thinks_by_default=True)._build_additional_model_request_fields(
            BedrockModelSettings(), ModelRequestParameters(), []
        )
        assert result is None

    def test_claude_that_only_thinks_when_asked_sends_nothing(self):
        settings = BedrockModelSettings(thinking_summary=True)
        result = self._claude()._build_additional_model_request_fields(settings, ModelRequestParameters(), [])
        assert result is None

    def test_claude_disabled_block_takes_no_display(self):
        settings = BedrockModelSettings(thinking_summary=True)
        result = self._claude(thinks_by_default=True)._build_additional_model_request_fields(
            settings, ModelRequestParameters(thinking=False), []
        )
        assert result == {'thinking': {'type': 'disabled'}}

    def test_claude_provider_display_wins(self):
        settings = BedrockModelSettings(thinking_summary=False, bedrock_thinking_display='summarized')
        result = self._claude()._build_additional_model_request_fields(
            settings, ModelRequestParameters(thinking=True), []
        )
        assert result == {'thinking': {'type': 'adaptive', 'display': 'summarized'}}

    def test_claude_raw_thinking_wins(self):
        settings = BedrockModelSettings(
            thinking_summary=True, bedrock_additional_model_requests_fields={'thinking': {'type': 'adaptive'}}
        )
        result = self._claude(thinks_by_default=True)._build_additional_model_request_fields(
            settings, ModelRequestParameters(), []
        )
        assert result == {'thinking': {'type': 'adaptive'}}

    def test_gpt_oss_false_drops_the_reasoning(self):
        settings = BedrockModelSettings(thinking_summary=False)
        result = self._gpt_oss()._build_additional_model_request_fields(
            settings, ModelRequestParameters(thinking='high'), []
        )
        assert result == {'reasoning_effort': 'high', 'include_reasoning': False}

    def test_gpt_oss_true_sends_nothing(self):
        """Any `include_reasoning` value, `True` included, drops gpt-oss's reasoning block on Bedrock."""
        settings = BedrockModelSettings(thinking_summary=True)
        result = self._gpt_oss()._build_additional_model_request_fields(settings, ModelRequestParameters(), [])
        assert result is None

    def test_gpt_oss_explicit_include_reasoning_wins(self):
        settings = BedrockModelSettings(
            thinking_summary=False, bedrock_additional_model_requests_fields={'include_reasoning': True}
        )
        result = self._gpt_oss()._build_additional_model_request_fields(settings, ModelRequestParameters(), [])
        assert result == {'include_reasoning': True}

    def test_qwen_ignores_it(self):
        model = _bedrock_model(BedrockModelProfile(bedrock_thinking_variant='qwen', supports_thinking=True))
        settings = BedrockModelSettings(thinking_summary=False)
        assert model._build_additional_model_request_fields(settings, ModelRequestParameters(), []) is None


@pytest.mark.skipif(not google_imports(), reason='google-genai not installed')
class TestGoogleThinkingSummary:
    """`thinking_summary` as Gemini's `include_thoughts`."""

    @staticmethod
    def _model(**profile: Any) -> FunctionModel:
        return FunctionModel(_echo, profile=GoogleModelProfile(supports_thinking=True, **profile))

    def test_false_keeps_the_level(self):
        result = GoogleModel._translate_thinking(
            self._model(), {'thinking_summary': False}, ModelRequestParameters(thinking='high')
        )
        assert result == {'include_thoughts': False, 'thinking_level': 'HIGH'}

    def test_false_keeps_the_budget(self):
        result = GoogleModel._translate_thinking(
            self._model(google_supports_thinking_level=False),
            {'thinking_summary': False},
            ModelRequestParameters(thinking='high'),
        )
        assert result == {'include_thoughts': False, 'thinking_budget': 24576}

    def test_false_with_bare_thinking(self):
        result = GoogleModel._translate_thinking(
            self._model(), {'thinking_summary': False}, ModelRequestParameters(thinking=True)
        )
        assert result == {'include_thoughts': False}

    def test_true_on_a_model_that_thinks_by_default(self):
        result = GoogleModel._translate_thinking(
            self._model(thinking_enabled_by_default=True), {'thinking_summary': True}, ModelRequestParameters()
        )
        assert result == {'include_thoughts': True}

    def test_true_on_a_model_that_does_not_think_by_default(self):
        """`gemini-2.5-flash-image` 400s on any thinking config, so nothing is sent where the model doesn't think."""
        result = GoogleModel._translate_thinking(self._model(), {'thinking_summary': True}, ModelRequestParameters())
        assert result is None

    def test_false_without_thinking_sends_nothing(self):
        """Thoughts are left out by default, so there is nothing to ask for."""
        result = GoogleModel._translate_thinking(
            self._model(thinking_enabled_by_default=True), {'thinking_summary': False}, ModelRequestParameters()
        )
        assert result is None

    def test_google_thinking_config_wins(self):
        settings: dict[str, Any] = {'thinking_summary': False, 'google_thinking_config': {'include_thoughts': True}}
        result = GoogleModel._translate_thinking(self._model(), settings, ModelRequestParameters(thinking=True))
        assert result == {'include_thoughts': True}

    @pytest.mark.parametrize(
        ('model_name', 'thinks_by_default'),
        [
            ('gemini-2.5-flash', True),
            ('gemini-2.5-pro', True),
            ('gemini-3.8-flash', True),
            ('gemini-3-pro-image-preview', True),
            ('gemini-2.5-flash-lite', False),
            ('gemini-3.1-flash-lite', False),
            ('gemini-2.5-flash-image', False),
            ('gemini-3.1-flash-image', False),
            ('gemini-2.0-flash', False),
        ],
    )
    def test_profile_thinks_by_default(self, model_name: str, thinks_by_default: bool):
        """Verified live by the thinking tokens a request without a thinking config reports."""
        profile = google_model_profile(model_name)
        assert profile is not None
        assert profile.get('thinking_enabled_by_default', False) is thinks_by_default


@pytest.mark.skipif(not groq_imports(), reason='groq not installed')
class TestGroqThinkingSummary:
    """`thinking_summary` as Groq's `reasoning_format`."""

    @pytest.fixture
    def reasoning_model(self):
        return FunctionModel(_echo, profile=ModelProfile(supports_thinking=True, thinking_always_enabled=True))

    @pytest.mark.parametrize(('thinking_summary', 'fmt'), [(True, 'parsed'), (False, 'hidden')], ids=['true', 'false'])
    def test_without_thinking(self, reasoning_model: FunctionModel, thinking_summary: bool, fmt: str):
        """Groq's reasoning models all reason by default."""
        result = GroqModel._translate_thinking(
            reasoning_model, {'thinking_summary': thinking_summary}, ModelRequestParameters(), False
        )
        assert result == fmt

    def test_false_with_thinking(self, reasoning_model: FunctionModel):
        result = GroqModel._translate_thinking(
            reasoning_model, {'thinking_summary': False}, ModelRequestParameters(thinking='high'), False
        )
        assert result == 'hidden'

    def test_non_reasoning_model_sends_nothing(self):
        """Groq rejects `reasoning_format` on models that don't reason."""
        result = GroqModel._translate_thinking(
            FunctionModel(_echo), {'thinking_summary': False}, ModelRequestParameters(), False
        )
        assert result is groq_NOT_GIVEN

    def test_disabled_reasoning_sends_nothing(self, reasoning_model: FunctionModel):
        result = GroqModel._translate_thinking(
            reasoning_model, {'thinking_summary': True}, ModelRequestParameters(thinking=False), True
        )
        assert result is groq_NOT_GIVEN

    def test_groq_reasoning_format_wins(self, reasoning_model: FunctionModel):
        settings: dict[str, Any] = {'thinking_summary': False, 'groq_reasoning_format': 'raw'}
        assert GroqModel._translate_thinking(reasoning_model, settings, ModelRequestParameters(), False) == 'raw'


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestOpenAIThinkingSummary:
    """`thinking_summary` as the Responses API's `reasoning.summary`, and as provider fields on Chat Completions."""

    @staticmethod
    def _responses_model(model_name: str) -> OpenAIResponsesModel:
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key='mock-api-key'))

    def test_responses_true_when_reasoning(self):
        model = self._responses_model('gpt-6.1-sol')
        result = model._translate_thinking({'thinking_summary': True}, ModelRequestParameters(thinking='high'))
        assert result == snapshot({'effort': 'high', 'context': 'all_turns', 'summary': 'auto'})

    def test_responses_true_on_a_model_that_reasons_by_default(self):
        model = self._responses_model('gpt-6.1-sol')
        result = model._translate_thinking({'thinking_summary': True}, ModelRequestParameters())
        assert result == snapshot({'context': 'all_turns', 'summary': 'auto'})

    def test_responses_true_without_reasoning_sends_nothing(self):
        model = self._responses_model('gpt-6.1-sol')
        result = model._translate_thinking({'thinking_summary': True}, ModelRequestParameters(thinking=False))
        assert result == snapshot({'effort': 'none', 'context': 'all_turns'})

    @pytest.mark.parametrize('thinking_summary', [None, False], ids=['unset', 'false'])
    def test_responses_unset_or_false_sends_nothing(self, thinking_summary: bool | None):
        """Summaries may require organization verification, so they're only requested with `True`."""
        model = self._responses_model('gpt-6.1-sol')
        settings: dict[str, Any] = {} if thinking_summary is None else {'thinking_summary': thinking_summary}
        result = model._translate_thinking(settings, ModelRequestParameters(thinking=True))
        assert result == snapshot({'effort': 'medium', 'context': 'all_turns'})

    def test_responses_openai_reasoning_summary_wins(self):
        model = self._responses_model('gpt-6.1-sol')
        settings: dict[str, Any] = {'thinking_summary': True, 'openai_reasoning_summary': 'detailed'}
        result = model._translate_thinking(settings, ModelRequestParameters(thinking=True))
        assert result == snapshot({'effort': 'medium', 'context': 'all_turns', 'summary': 'detailed'})

    @staticmethod
    async def _chat_extra_body(model: OpenAIChatModel, settings: ModelSettings) -> object:
        await Agent(model, model_settings=settings).run('hello')
        return get_mock_chat_completion_kwargs(model.client)[0].get('extra_body')

    async def test_vllm_false_excludes_reasoning(self, allow_model_requests: None):
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))
        model = OpenAIChatModel('Qwen/Qwen3-32B', provider=VLLMProvider(openai_client=mock_client))
        extra_body = await self._chat_extra_body(model, ModelSettings(thinking_summary=False))
        assert extra_body == {'include_reasoning': False}

    async def test_vllm_explicit_extra_body_wins(self, allow_model_requests: None):
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))
        model = OpenAIChatModel('Qwen/Qwen3-32B', provider=VLLMProvider(openai_client=mock_client))
        settings = ModelSettings(thinking_summary=False, extra_body={'include_reasoning': True, 'top_k': 5})
        assert await self._chat_extra_body(model, settings) == {'include_reasoning': True, 'top_k': 5}

    async def test_vllm_leaves_an_extra_body_it_cannot_merge_alone(self, allow_model_requests: None):
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))
        model = OpenAIChatModel('Qwen/Qwen3-32B', provider=VLLMProvider(openai_client=mock_client))
        settings = ModelSettings(thinking_summary=False, extra_body=['opaque'])
        assert await self._chat_extra_body(model, settings) == ['opaque']

    @pytest.mark.parametrize('thinking_summary', [None, True], ids=['unset', 'true'])
    async def test_vllm_unset_or_true_sends_nothing(self, allow_model_requests: None, thinking_summary: bool | None):
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))
        model = OpenAIChatModel('Qwen/Qwen3-32B', provider=VLLMProvider(openai_client=mock_client))
        settings = ModelSettings() if thinking_summary is None else ModelSettings(thinking_summary=thinking_summary)
        assert await self._chat_extra_body(model, settings) is None

    async def test_openai_chat_false_sends_nothing(self, allow_model_requests: None):
        """OpenAI's Chat Completions API has no switch, so only providers that declare one send anything."""
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))
        model = OpenAIChatModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=mock_client))
        assert await self._chat_extra_body(model, ModelSettings(thinking_summary=False)) is None

    @pytest.mark.parametrize(
        ('model_name', 'expected'),
        [('gpt-oss-120b', {'reasoning_format': 'hidden'}), ('zai-glm-4.7', {'clear_thinking': False})],
    )
    async def test_cerebras_false_hides_reasoning_only_on_gpt_oss(
        self, allow_model_requests: None, model_name: str, expected: dict[str, object]
    ):
        """Cerebras' other models reject `reasoning_format`."""
        mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))
        model = CerebrasModel(model_name, provider=CerebrasProvider(openai_client=mock_client))
        assert await self._chat_extra_body(model, ModelSettings(thinking_summary=False)) == expected


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
class TestOpenRouterThinkingSummary:
    """`thinking_summary=False` as OpenRouter's `reasoning.exclude`."""

    def test_false_without_thinking(self):
        result = _openrouter_settings_to_openai_settings(
            OpenRouterModelSettings(thinking_summary=False), ModelRequestParameters()
        )
        assert result.get('extra_body') == {'reasoning': {'exclude': True}}
        # Kept off `openrouter_reasoning`, where it would read as a request to reason.
        assert 'openrouter_reasoning' not in result

    def test_false_with_thinking(self):
        result = _openrouter_settings_to_openai_settings(
            OpenRouterModelSettings(thinking_summary=False), ModelRequestParameters(thinking='high')
        )
        assert result.get('extra_body') == {'reasoning': {'effort': 'high', 'enabled': True, 'exclude': True}}

    def test_true_sends_nothing(self):
        """OpenRouter returns reasoning unless excluded."""
        result = _openrouter_settings_to_openai_settings(
            OpenRouterModelSettings(thinking_summary=True), ModelRequestParameters()
        )
        assert result.get('extra_body') == {}

    def test_openrouter_reasoning_wins(self):
        settings = OpenRouterModelSettings(thinking_summary=False, openrouter_reasoning={'effort': 'low'})
        result = _openrouter_settings_to_openai_settings(settings, ModelRequestParameters())
        assert result.get('extra_body') == {'reasoning': {'effort': 'low'}}
