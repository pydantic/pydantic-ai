"""Tests for the unified thinking/reasoning feature.

Tests the base Model.prepare_request() thinking resolution, per-provider translation,
the Thinking capability, and end-to-end integration via FunctionModel.
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false
from __future__ import annotations

from typing import Any, Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import CAPABILITY_TYPES, Thinking
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from ._inline_snapshot import snapshot

pytestmark = [
    pytest.mark.anyio,
]


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


class TestAnthropicThinkingTranslation:
    """Test Anthropic _translate_thinking and _build_output_config translation."""

    @pytest.fixture(autouse=True)
    def _require_anthropic(self):
        pytest.importorskip('anthropic', reason='anthropic not installed')

    @pytest.fixture
    def adaptive_model(self):
        from pydantic_ai.profiles.anthropic import AnthropicModelProfile

        return FunctionModel(
            _echo,
            profile=AnthropicModelProfile(
                supports_thinking=True,
                anthropic_supports_adaptive_thinking=True,
            ),
        )

    @pytest.fixture
    def non_adaptive_model(self):
        from pydantic_ai.profiles.anthropic import AnthropicModelProfile

        return FunctionModel(
            _echo,
            profile=AnthropicModelProfile(
                supports_thinking=True,
                anthropic_supports_adaptive_thinking=False,
            ),
        )

    def test_thinking_true_adaptive(self, adaptive_model: FunctionModel):
        """thinking=True with adaptive model -> {'type': 'adaptive'}."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == snapshot({'type': 'adaptive'})

    def test_thinking_true_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking=True with non-adaptive model -> {'type': 'enabled', 'budget_tokens': 10000}."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 10000})

    def test_thinking_high_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking='high' with non-adaptive -> budget_tokens=16384."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 16384})

    def test_thinking_low_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking='low' with non-adaptive -> budget_tokens=2048."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 2048})

    def test_thinking_false_returns_omit(self, adaptive_model: FunctionModel):
        """thinking=False -> OMIT (not sent to API)."""
        from anthropic import omit

        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result is omit

    def test_thinking_none_returns_omit(self, adaptive_model: FunctionModel):
        """thinking=None -> OMIT (not sent to API)."""
        from anthropic import omit

        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result is omit

    def test_provider_specific_takes_precedence(self, adaptive_model: FunctionModel):
        """anthropic_thinking set -> unified thinking ignored."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=True)
        settings = {'anthropic_thinking': {'type': 'disabled'}}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == snapshot({'type': 'disabled'})

    def test_effort_level_on_output_config(self):
        """thinking='high' sets effort on output_config when model supports it."""
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.profiles.anthropic import AnthropicModelProfile

        model = AnthropicModel.__new__(AnthropicModel)
        model._profile = AnthropicModelProfile(
            supports_thinking=True,
            anthropic_supports_effort=True,
        )

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = model._build_output_config(params, settings)
        assert result == snapshot({'effort': 'high'})

    def test_output_config_no_effort_for_bool(self):
        """thinking=True does NOT set effort on output_config (only str values do)."""
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.profiles.anthropic import AnthropicModelProfile

        model = AnthropicModel.__new__(AnthropicModel)
        model._profile = AnthropicModelProfile(
            supports_thinking=True,
            anthropic_supports_effort=True,
        )

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = model._build_output_config(params, settings)
        assert result is None

    def test_adaptive_model_with_effort_level(self):
        """thinking='high' on adaptive+effort model uses adaptive thinking and output_config effort."""
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.profiles.anthropic import AnthropicModelProfile

        model = AnthropicModel.__new__(AnthropicModel)
        model._profile = AnthropicModelProfile(
            supports_thinking=True,
            anthropic_supports_adaptive_thinking=True,
            anthropic_supports_effort=True,
        )

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        # All truthy thinking values use adaptive on adaptive models
        thinking_param = AnthropicModel._translate_thinking(model, settings, params)
        assert thinking_param == snapshot({'type': 'adaptive'})

        # output_config: effort controls depth separately
        output_config = model._build_output_config(params, settings)
        assert output_config == snapshot({'effort': 'high'})

    def test_medium_uses_adaptive(self, adaptive_model: FunctionModel):
        """thinking='medium' on adaptive model -> adaptive (not budget)."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking='medium')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == {'type': 'adaptive'}

    def test_low_uses_adaptive_on_adaptive(self, adaptive_model: FunctionModel):
        """thinking='low' on adaptive model -> adaptive (effort controlled via output_config)."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == {'type': 'adaptive'}

    def test_high_uses_adaptive_on_adaptive(self, adaptive_model: FunctionModel):
        """thinking='high' on adaptive model -> adaptive (effort controlled via output_config)."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = AnthropicModel._translate_thinking(adaptive_model, settings, params)
        assert result == {'type': 'adaptive'}


class TestOpenAIChatThinkingTranslation:
    """Test OpenAI Chat model _translate_thinking translation."""

    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip('openai', reason='openai not installed')

    def test_thinking_true(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}

        # We need a model-like object to call the method; use a FunctionModel with the right profile
        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result == 'medium'

    def test_thinking_high(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result == 'high'

    def test_thinking_false(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result == 'none'

    def test_thinking_none_returns_omit(self):
        from openai import omit

        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result is omit

    def test_provider_specific_takes_precedence(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking=True)
        settings = {'openai_reasoning_effort': 'low'}

        model = FunctionModel(_echo)
        result = OpenAIChatModel._translate_thinking(model, settings, params)
        assert result == 'low'


class TestOpenAIResponsesThinkingTranslation:
    """Test OpenAI Responses model _translate_thinking translation."""

    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip('openai', reason='openai not installed')

    def test_thinking_true(self):
        from pydantic_ai.models.openai import OpenAIResponsesModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._translate_thinking(model, settings, params)
        assert result == snapshot({'effort': 'medium'})

    def test_thinking_high(self):
        from pydantic_ai.models.openai import OpenAIResponsesModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._translate_thinking(model, settings, params)
        assert result == snapshot({'effort': 'high'})

    def test_thinking_false(self):
        """thinking=False -> reasoning_effort='none'."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._translate_thinking(model, settings, params)
        # 'none' is falsy for dict truthiness check, but the effort_map maps False -> 'none'
        # which gets set as reasoning_effort. Then `if reasoning_effort:` is truthy for 'none'.
        assert result == snapshot({'effort': 'none'})

    def test_provider_specific_takes_precedence(self):
        from pydantic_ai.models.openai import OpenAIResponsesModel

        params = ModelRequestParameters(thinking=True)
        settings = {'openai_reasoning_effort': 'high'}

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._translate_thinking(model, settings, params)
        assert result == snapshot({'effort': 'high'})


class TestGoogleThinkingTranslation:
    """Test Google model _translate_thinking translation."""

    @pytest.fixture(autouse=True)
    def _require_google(self):
        pytest.importorskip('google.genai', reason='google-genai not installed')

    @pytest.fixture
    def gemini_3_model(self):
        """A model with thinking_level support (Gemini 3+)."""
        from pydantic_ai.profiles.google import GoogleModelProfile

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
        from pydantic_ai.profiles.google import GoogleModelProfile

        return FunctionModel(
            _echo,
            profile=GoogleModelProfile(
                supports_thinking=True,
                google_supports_thinking_level=False,
            ),
        )

    def test_thinking_true_gemini_3(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True})

    def test_thinking_high_gemini_3(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'HIGH'})

    def test_thinking_low_gemini_3(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'LOW'})

    def test_thinking_true_gemini_25(self, gemini_25_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_25_model, settings, params)
        assert result == snapshot({'include_thoughts': True})

    def test_thinking_high_gemini_25(self, gemini_25_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_25_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_budget': 24576})

    def test_thinking_low_gemini_25(self, gemini_25_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_25_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_budget': 2048})

    def test_thinking_false(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'thinking_level': 'MINIMAL'})

    def test_thinking_false_gemini_25(self, gemini_25_model: FunctionModel):
        """thinking=False on Gemini 2.5 uses thinking_budget=0."""
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_25_model, settings, params)
        assert result == snapshot({'thinking_budget': 0})

    def test_thinking_none(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result is None

    def test_provider_specific_takes_precedence(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=True)
        settings = {'google_thinking_config': {'include_thoughts': False}}
        result = GoogleModel._translate_thinking(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': False})


class TestGroqThinkingTranslation:
    """Test Groq model _translate_thinking translation."""

    @pytest.fixture(autouse=True)
    def _require_groq(self):
        pytest.importorskip('groq', reason='groq not installed')

    def test_thinking_true(self):
        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params)
        assert result == 'parsed'

    def test_thinking_high(self):
        """Effort levels also translate to 'parsed' for Groq."""
        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params)
        assert result == 'parsed'

    def test_thinking_false(self):
        """thinking=False -> 'hidden' (Groq has no true disable; 'hidden' suppresses output)."""
        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params)
        assert result == 'hidden'

    def test_thinking_none(self):
        from groq import NOT_GIVEN

        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params)
        assert result is NOT_GIVEN

    def test_provider_specific_takes_precedence(self):
        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking=True)
        settings = {'groq_reasoning_format': 'raw'}

        model = FunctionModel(_echo)
        result = GroqModel._translate_thinking(model, settings, params)
        assert result == 'raw'


class TestAnthropicUnifiedThinkingConflict:
    """Test that unified thinking triggers the output tools conflict path in prepare_request."""

    def test_unified_thinking_with_output_tools_auto_mode(self):
        """thinking='high' (unified) + output tools + auto mode -> switches to native."""
        pytest.importorskip('anthropic', reason='anthropic not installed')
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.output import OutputObjectDefinition
        from pydantic_ai.profiles.anthropic import AnthropicModelProfile
        from pydantic_ai.tools import ToolDefinition

        model = AnthropicModel.__new__(AnthropicModel)
        model._profile = AnthropicModelProfile(
            supports_thinking=True,
            supports_json_schema_output=True,
            anthropic_supports_adaptive_thinking=True,
        )
        model._settings = None

        output_tool = ToolDefinition(name='output', description='', parameters_json_schema={}, kind='output')
        output_object = OutputObjectDefinition(json_schema={'type': 'object', 'properties': {}})
        params = ModelRequestParameters(
            output_tools=[output_tool],
            output_object=output_object,
            output_mode='auto',
        )
        settings = ModelSettings(thinking='high')

        _, resolved_params = model.prepare_request(settings, params)
        # Should have switched from auto to native (since supports_json_schema_output=True)
        assert resolved_params.output_mode == 'native'
        assert resolved_params.thinking == 'high'


class TestBedrockThinkingTranslation:
    """Test Bedrock _translate_thinking translation for each variant."""

    @pytest.fixture(autouse=True)
    def _require_boto3(self):
        pytest.importorskip('boto3', reason='boto3 not installed')

    def test_anthropic_variant_thinking_true(self):
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='anthropic',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = model._translate_thinking(settings, params)
        assert result == {'thinking': {'type': 'enabled', 'budget_tokens': 10000}}

    def test_anthropic_variant_thinking_false(self):
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='anthropic',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = model._translate_thinking(settings, params)
        assert result == {'thinking': {'type': 'disabled'}}

    def test_openai_variant_thinking_false(self):
        """thinking=False on OpenAI Bedrock variant is a no-op (Bedrock rejects 'none')."""
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='openai',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = model._translate_thinking(settings, params)
        # thinking=False: no reasoning_effort set, returns None
        assert result is None

    def test_openai_variant_thinking_high(self):
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='openai',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = model._translate_thinking(settings, params)
        assert result == {'reasoning_effort': 'high'}

    def test_qwen_variant_thinking_true(self):
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='qwen',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = model._translate_thinking(settings, params)
        assert result == {'reasoning_config': 'high'}

    def test_qwen_variant_thinking_false(self):
        """thinking=False on Qwen variant is a no-op (Qwen has no disable mechanism)."""
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='qwen',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = model._translate_thinking(settings, params)
        # thinking=False on Qwen: no reasoning_config set, returns None (empty dict is falsy)
        assert result is None

    def test_no_variant_thinking_passthrough(self):
        """When bedrock_thinking_variant is None, unified thinking is a no-op."""
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(bedrock_thinking_variant=None)

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = model._translate_thinking(settings, params)
        # No variant set, so no thinking fields are added
        assert result is None

    def test_thinking_none_returns_existing(self):
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(bedrock_thinking_variant='anthropic')

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=None)
        result = model._translate_thinking(settings, params)
        assert result is None


class TestOpenRouterThinkingTranslation:
    """Test OpenRouter unified thinking fallback in _openrouter_settings_to_openai_settings."""

    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip('openai', reason='openai not installed')

    def test_thinking_true(self):
        from pydantic_ai.models.openrouter import OpenRouterModelSettings, _openrouter_settings_to_openai_settings

        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = _openrouter_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'medium'}

    def test_thinking_high(self):
        from pydantic_ai.models.openrouter import OpenRouterModelSettings, _openrouter_settings_to_openai_settings

        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = _openrouter_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'high'}

    def test_thinking_false_no_reasoning(self):
        from pydantic_ai.models.openrouter import OpenRouterModelSettings, _openrouter_settings_to_openai_settings

        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = _openrouter_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert 'reasoning' not in extra_body

    def test_openai_reasoning_effort_passthrough(self):
        """Explicit openai_reasoning_effort on OpenRouter is passed through."""
        from pydantic_ai.models.openrouter import OpenRouterModel

        model = OpenRouterModel.__new__(OpenRouterModel)
        model._profile = ModelProfile(supports_thinking=True)
        model._settings = None

        settings: dict[str, Any] = {'openai_reasoning_effort': 'low'}
        params = ModelRequestParameters(thinking='high')
        result = model._translate_thinking(settings, params)
        assert result == 'low'


class TestCerebrasThinkingTranslation:
    """Test Cerebras unified thinking fallback."""

    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip('openai', reason='openai not installed')

    def test_thinking_false_sets_disable_reasoning(self):
        from pydantic_ai.models.cerebras import CerebrasModelSettings, _cerebras_settings_to_openai_settings

        settings = CerebrasModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = _cerebras_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('disable_reasoning') is True

    def test_thinking_true_sets_disable_reasoning_false(self):
        from pydantic_ai.models.cerebras import CerebrasModelSettings, _cerebras_settings_to_openai_settings

        settings = CerebrasModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = _cerebras_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('disable_reasoning') is False

    def test_thinking_effort_sets_disable_reasoning_false(self):
        from pydantic_ai.models.cerebras import CerebrasModelSettings, _cerebras_settings_to_openai_settings

        settings = CerebrasModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = _cerebras_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('disable_reasoning') is False

    def test_explicit_cerebras_disable_takes_precedence(self):
        from pydantic_ai.models.cerebras import CerebrasModelSettings, _cerebras_settings_to_openai_settings

        settings = CerebrasModelSettings(cerebras_disable_reasoning=True)
        params = ModelRequestParameters(thinking=True)
        result = _cerebras_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('disable_reasoning') is True

    def test_explicit_openai_reasoning_effort_passthrough(self):
        """Explicit openai_reasoning_effort on Cerebras is passed through."""
        from pydantic_ai.models.cerebras import CerebrasModel

        model = CerebrasModel.__new__(CerebrasModel)
        model._profile = ModelProfile(supports_thinking=True)
        model._settings = None

        settings: dict[str, Any] = {'openai_reasoning_effort': 'low'}
        params = ModelRequestParameters(thinking='high')
        result = model._translate_thinking(settings, params)
        assert result == 'low'


class TestMistralThinkingTranslation:
    """Test Mistral model _translate_thinking translation."""

    @pytest.fixture(autouse=True)
    def _require_mistral(self):
        pytest.importorskip('mistralai', reason='mistralai not installed')

    def test_thinking_true(self):
        from pydantic_ai.models.mistral import MistralModel, MistralModelSettings

        params = ModelRequestParameters(thinking=True)
        settings = MistralModelSettings()
        result = MistralModel._translate_thinking(settings, params)
        assert result == 'high'

    def test_thinking_high(self):
        from pydantic_ai.models.mistral import MistralModel, MistralModelSettings

        params = ModelRequestParameters(thinking='high')
        settings = MistralModelSettings()
        result = MistralModel._translate_thinking(settings, params)
        assert result == 'high'

    def test_thinking_low(self):
        """thinking='low' maps to 'high' (Mistral has no in-between; any active thinking uses 'high')."""
        from pydantic_ai.models.mistral import MistralModel, MistralModelSettings

        params = ModelRequestParameters(thinking='low')
        settings = MistralModelSettings()
        result = MistralModel._translate_thinking(settings, params)
        assert result == 'high'

    def test_thinking_minimal(self):
        """thinking='minimal' maps to 'none' — disables the thinking chunk per Mistral docs."""
        from pydantic_ai.models.mistral import MistralModel, MistralModelSettings

        params = ModelRequestParameters(thinking='minimal')
        settings = MistralModelSettings()
        result = MistralModel._translate_thinking(settings, params)
        assert result == 'none'

    def test_thinking_false(self):
        from pydantic_ai.models.mistral import MistralModel, MistralModelSettings

        params = ModelRequestParameters(thinking=False)
        settings = MistralModelSettings()
        result = MistralModel._translate_thinking(settings, params)
        assert result == 'none'

    def test_thinking_none_returns_unset(self):
        from mistralai.client.types.basemodel import Unset as MistralUnset

        from pydantic_ai.models.mistral import MistralModel, MistralModelSettings

        params = ModelRequestParameters(thinking=None)
        settings = MistralModelSettings()
        result = MistralModel._translate_thinking(settings, params)
        assert isinstance(result, MistralUnset)

    def test_provider_specific_takes_precedence(self):
        """mistral_reasoning_effort set -> unified thinking is ignored."""
        from pydantic_ai.models.mistral import MistralModel, MistralModelSettings

        params = ModelRequestParameters(thinking=True)
        settings = MistralModelSettings(mistral_reasoning_effort='none')
        result = MistralModel._translate_thinking(settings, params)
        assert result == 'none'

    def test_provider_specific_high(self):
        from pydantic_ai.models.mistral import MistralModel, MistralModelSettings

        params = ModelRequestParameters(thinking=False)
        settings = MistralModelSettings(mistral_reasoning_effort='high')
        result = MistralModel._translate_thinking(settings, params)
        assert result == 'high'


class TestXaiThinkingTranslation:
    """Test xAI unified thinking fallback."""

    @pytest.fixture(autouse=True)
    def _require_xai_sdk(self):
        pytest.importorskip('xai_sdk', reason='xai_sdk not installed')

    def test_thinking_high(self):
        from pydantic_ai.models.xai import XaiModel, XaiModelSettings

        model = XaiModel.__new__(XaiModel)
        model._profile = ModelProfile(supports_thinking=True)
        model._settings = None

        settings = XaiModelSettings()
        params = ModelRequestParameters(thinking='high')
        # We can't call _create_chat directly, but we can verify prepare_request resolves
        _, resolved_params = model.prepare_request(settings, params)
        assert resolved_params.thinking == 'high'

    def test_thinking_true(self):
        from pydantic_ai.models.xai import XaiModel, XaiModelSettings

        model = XaiModel.__new__(XaiModel)
        model._profile = ModelProfile(supports_thinking=True)
        model._settings = None

        settings = XaiModelSettings()
        params = ModelRequestParameters(thinking=True)
        _, resolved_params = model.prepare_request(settings, params)
        assert resolved_params.thinking is True


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


class TestGoogleBudgetApiConstraints:
    """Budget values respect the Google API's documented limits."""

    @pytest.fixture(autouse=True)
    def _require_google(self):
        pytest.importorskip('google.genai', reason='google-genai not installed')

    def test_all_budgets_within_flash_range(self):
        """Every effort budget must be within Gemini 2.5 Flash's [0, 24576] range."""
        from pydantic_ai.models.google import GoogleModel

        model = FunctionModel(_echo, profile=ModelProfile(supports_thinking=True))
        for effort in ('minimal', 'low', 'medium', 'high', 'xhigh'):
            params = ModelRequestParameters(thinking=effort)
            result = GoogleModel._translate_thinking(model, {}, params)
            assert result is not None
            budget = result.get('thinking_budget')
            assert budget is not None, f"effort='{effort}' should produce a thinking_budget"
            assert 0 <= budget <= 24576, f"effort='{effort}' budget={budget} exceeds Flash max 24576"

    def test_all_budgets_within_pro_range(self):
        """Every effort budget must be within Gemini 2.5 Pro's [128, 32768] range."""
        from pydantic_ai.models.google import GoogleModel

        model = FunctionModel(_echo, profile=ModelProfile(supports_thinking=True))
        for effort in ('minimal', 'low', 'medium', 'high', 'xhigh'):
            params = ModelRequestParameters(thinking=effort)
            result = GoogleModel._translate_thinking(model, {}, params)
            assert result is not None
            budget = result.get('thinking_budget')
            assert budget is not None, f"effort='{effort}' should produce a thinking_budget"
            assert 128 <= budget <= 32768, f"effort='{effort}' budget={budget} outside Pro range [128, 32768]"

    def test_budgets_are_monotonically_increasing(self):
        """low < medium < high — effort levels should map to increasing budgets."""
        from pydantic_ai.models.google import GoogleModel

        model = FunctionModel(_echo, profile=ModelProfile(supports_thinking=True))
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
        from pydantic_ai.profiles.anthropic import AnthropicModelProfile, anthropic_model_profile

        # All Anthropic models support thinking in our implementation
        profile = anthropic_model_profile('claude-3-7-sonnet')
        assert profile is not None
        assert profile.supports_thinking is True

        profile = anthropic_model_profile('claude-sonnet-4-5')
        assert profile is not None
        assert profile.supports_thinking is True

        # Newer models support adaptive thinking
        profile = anthropic_model_profile('claude-sonnet-4-6')
        assert profile is not None
        assert isinstance(profile, AnthropicModelProfile)
        assert profile.anthropic_supports_adaptive_thinking is True

        profile = anthropic_model_profile('claude-opus-4-7')
        assert profile is not None
        assert isinstance(profile, AnthropicModelProfile)
        assert profile.anthropic_supports_adaptive_thinking is True
        assert profile.anthropic_supports_xhigh_effort is True
        assert profile.anthropic_disallows_budget_thinking is True

    def test_google_profile_thinking_support(self):
        from pydantic_ai.profiles.google import google_model_profile

        profile = google_model_profile('gemini-2.5-flash')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is False

        profile = google_model_profile('gemini-2.5-pro')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True

        profile = google_model_profile('gemini-2.0-flash')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_openai_profile_thinking_support(self):
        from pydantic_ai.profiles.openai import openai_model_profile

        profile = openai_model_profile('o3')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True

        profile = openai_model_profile('gpt-4o')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_groq_profile_thinking_support(self):
        from pydantic_ai.profiles.groq import groq_model_profile

        profile = groq_model_profile('deepseek-r1-distill-llama-70b')
        assert profile is not None
        assert profile.supports_thinking is True

        profile = groq_model_profile('llama-3.1-8b-instant')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_cohere_profile_thinking_support(self):
        from pydantic_ai.profiles.cohere import cohere_model_profile

        profile = cohere_model_profile('command-a-reasoning')
        assert profile is not None
        assert profile.supports_thinking is True

    def test_mistral_profile_thinking_support(self):
        from pydantic_ai.profiles.mistral import mistral_model_profile

        profile = mistral_model_profile('magistral-medium')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True


class TestCrossProviderPortability:
    """Same unified settings produce sensible results across providers."""

    def test_same_settings_all_main_providers(self):
        """The same thinking=True + effort='high' should produce non-None results
        on supported models across all providers."""
        pytest.importorskip('anthropic', reason='anthropic not installed')
        pytest.importorskip('openai', reason='openai not installed')
        pytest.importorskip('groq', reason='groq not installed')
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.models.openai import OpenAIChatModel

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
        result = GroqModel._translate_thinking(FunctionModel(_echo, profile=thinking_profile), settings, params)
        assert result == 'parsed'

    def test_unsupported_models_silently_dropped_via_prepare_request(self):
        """thinking settings on unsupported models → not resolved by prepare_request."""
        model = _make_model(supports_thinking=False)
        settings: ModelSettings = {'thinking': 'high'}
        _merged, params = model.prepare_request(settings, ModelRequestParameters())
        assert params.thinking is None


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
