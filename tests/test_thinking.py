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


def _make_model(
    *,
    supports_thinking: bool = False,
    thinking_always_enabled: bool = False,
) -> FunctionModel:
    """Create a FunctionModel with a specific thinking profile."""

    def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='ok')])

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
    """Test Anthropic _get_thinking_param and _build_output_config translation."""

    @pytest.fixture(autouse=True)
    def _require_anthropic(self):
        pytest.importorskip('anthropic', reason='anthropic not installed')

    @pytest.fixture
    def adaptive_model(self):
        from pydantic_ai.profiles.anthropic import AnthropicModelProfile

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

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

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

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
        result = AnthropicModel._get_thinking_param(adaptive_model, settings, params)
        assert result == snapshot({'type': 'adaptive'})

    def test_thinking_true_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking=True with non-adaptive model -> {'type': 'enabled', 'budget_tokens': 10000}."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = AnthropicModel._get_thinking_param(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 10000})

    def test_thinking_high_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking='high' with non-adaptive -> budget_tokens=16384."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = AnthropicModel._get_thinking_param(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 16384})

    def test_thinking_low_non_adaptive(self, non_adaptive_model: FunctionModel):
        """thinking='low' with non-adaptive -> budget_tokens=2048."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = AnthropicModel._get_thinking_param(non_adaptive_model, settings, params)
        assert result == snapshot({'type': 'enabled', 'budget_tokens': 2048})

    def test_thinking_false_returns_omit(self, adaptive_model: FunctionModel):
        """thinking=False -> OMIT (not sent to API)."""
        from anthropic import omit

        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = AnthropicModel._get_thinking_param(adaptive_model, settings, params)
        assert result is omit

    def test_thinking_none_returns_omit(self, adaptive_model: FunctionModel):
        """thinking=None -> OMIT (not sent to API)."""
        from anthropic import omit

        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}
        result = AnthropicModel._get_thinking_param(adaptive_model, settings, params)
        assert result is omit

    def test_provider_specific_takes_precedence(self, adaptive_model: FunctionModel):
        """anthropic_thinking set -> unified thinking ignored."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=True)
        settings = {'anthropic_thinking': {'type': 'disabled'}}
        result = AnthropicModel._get_thinking_param(adaptive_model, settings, params)
        assert result == snapshot({'type': 'disabled'})

    def test_effort_level_on_output_config(self, non_adaptive_model: FunctionModel):
        """thinking='high' sets effort on output_config."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = AnthropicModel._build_output_config(params, settings)
        assert result == snapshot({'effort': 'high'})

    def test_output_config_no_effort_for_bool(self, non_adaptive_model: FunctionModel):
        """thinking=True does NOT set effort on output_config (only str values do)."""
        from pydantic_ai.models.anthropic import AnthropicModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = AnthropicModel._build_output_config(params, settings)
        assert result is None


class TestOpenAIChatThinkingTranslation:
    """Test OpenAI Chat model _get_reasoning_effort translation."""

    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip('openai', reason='openai not installed')

    def test_thinking_true(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}

        # We need a model-like object to call the method; use a FunctionModel with the right profile
        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = OpenAIChatModel._get_reasoning_effort(model, settings, params)
        assert result == 'medium'

    def test_thinking_high(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = OpenAIChatModel._get_reasoning_effort(model, settings, params)
        assert result == 'high'

    def test_thinking_false(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = OpenAIChatModel._get_reasoning_effort(model, settings, params)
        assert result == 'none'

    def test_thinking_none_returns_omit(self):
        from openai import omit

        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = OpenAIChatModel._get_reasoning_effort(model, settings, params)
        assert result is omit

    def test_provider_specific_takes_precedence(self):
        from pydantic_ai.models.openai import OpenAIChatModel

        params = ModelRequestParameters(thinking=True)
        settings = {'openai_reasoning_effort': 'low'}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = OpenAIChatModel._get_reasoning_effort(model, settings, params)
        assert result == 'low'


class TestOpenAIResponsesThinkingTranslation:
    """Test OpenAI Responses model _get_reasoning translation."""

    @pytest.fixture(autouse=True)
    def _require_openai(self):
        pytest.importorskip('openai', reason='openai not installed')

    def test_thinking_true(self):
        from pydantic_ai.models.openai import OpenAIResponsesModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._get_reasoning(model, settings, params)
        assert result == snapshot({'effort': 'medium'})

    def test_thinking_high(self):
        from pydantic_ai.models.openai import OpenAIResponsesModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._get_reasoning(model, settings, params)
        assert result == snapshot({'effort': 'high'})

    def test_thinking_false(self):
        """thinking=False -> reasoning_effort='none'."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._get_reasoning(model, settings, params)
        # 'none' is falsy for dict truthiness check, but the effort_map maps False -> 'none'
        # which gets set as reasoning_effort. Then `if reasoning_effort:` is truthy for 'none'.
        assert result == snapshot({'effort': 'none'})

    def test_provider_specific_takes_precedence(self):
        from pydantic_ai.models.openai import OpenAIResponsesModel

        params = ModelRequestParameters(thinking=True)
        settings = {'openai_reasoning_effort': 'high'}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = OpenAIResponsesModel._get_reasoning(model, settings, params)
        assert result == snapshot({'effort': 'high'})


class TestGoogleThinkingTranslation:
    """Test Google model _get_thinking_config translation."""

    @pytest.fixture(autouse=True)
    def _require_google(self):
        pytest.importorskip('google.genai', reason='google-genai not installed')

    @pytest.fixture
    def gemini_3_model(self):
        """A model with thinking_level support (Gemini 3+)."""
        from pydantic_ai.profiles.google import GoogleModelProfile

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

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

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

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
        result = GoogleModel._get_thinking_config(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True})

    def test_thinking_high_gemini_3(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = GoogleModel._get_thinking_config(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'HIGH'})

    def test_thinking_low_gemini_3(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = GoogleModel._get_thinking_config(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_level': 'LOW'})

    def test_thinking_true_gemini_25(self, gemini_25_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}
        result = GoogleModel._get_thinking_config(gemini_25_model, settings, params)
        assert result == snapshot({'include_thoughts': True})

    def test_thinking_high_gemini_25(self, gemini_25_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}
        result = GoogleModel._get_thinking_config(gemini_25_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_budget': 24576})

    def test_thinking_low_gemini_25(self, gemini_25_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking='low')
        settings: ModelSettings = {}
        result = GoogleModel._get_thinking_config(gemini_25_model, settings, params)
        assert result == snapshot({'include_thoughts': True, 'thinking_budget': 2048})

    def test_thinking_false(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}
        result = GoogleModel._get_thinking_config(gemini_3_model, settings, params)
        assert result == snapshot({'thinking_budget': 0})

    def test_thinking_none(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}
        result = GoogleModel._get_thinking_config(gemini_3_model, settings, params)
        assert result is None

    def test_provider_specific_takes_precedence(self, gemini_3_model: FunctionModel):
        from pydantic_ai.models.google import GoogleModel

        params = ModelRequestParameters(thinking=True)
        settings = {'google_thinking_config': {'include_thoughts': False}}
        result = GoogleModel._get_thinking_config(gemini_3_model, settings, params)
        assert result == snapshot({'include_thoughts': False})


class TestGroqThinkingTranslation:
    """Test Groq model _get_reasoning_format translation."""

    @pytest.fixture(autouse=True)
    def _require_groq(self):
        pytest.importorskip('groq', reason='groq not installed')

    def test_thinking_true(self):
        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking=True)
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = GroqModel._get_reasoning_format(model, settings, params)
        assert result == 'parsed'

    def test_thinking_high(self):
        """Effort levels also translate to 'parsed' for Groq."""
        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking='high')
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = GroqModel._get_reasoning_format(model, settings, params)
        assert result == 'parsed'

    def test_thinking_false(self):
        """thinking=False -> NOT_GIVEN (Groq treats False as 'do not send')."""
        from groq import NOT_GIVEN

        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking=False)
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = GroqModel._get_reasoning_format(model, settings, params)
        assert result is NOT_GIVEN

    def test_thinking_none(self):
        from groq import NOT_GIVEN

        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking=None)
        settings: ModelSettings = {}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = GroqModel._get_reasoning_format(model, settings, params)
        assert result is NOT_GIVEN

    def test_provider_specific_takes_precedence(self):
        from pydantic_ai.models.groq import GroqModel

        params = ModelRequestParameters(thinking=True)
        settings = {'groq_reasoning_format': 'raw'}

        def _echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ok')])

        model = FunctionModel(_echo)
        result = GroqModel._get_reasoning_format(model, settings, params)
        assert result == 'raw'


class TestAnthropicUnifiedThinkingConflict:
    """Test that unified thinking triggers the output tools conflict path in prepare_request."""

    def test_unified_thinking_with_output_tools_auto_mode(self):
        """thinking='high' (unified) + output tools + auto mode -> switches to native."""
        pytest.importorskip('anthropic')
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
    """Test Bedrock _get_thinking_fields translation for each variant."""

    def test_anthropic_variant_thinking_true(self):
        pytest.importorskip('boto3')
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='anthropic',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = model._get_thinking_fields(settings, params)
        assert result == {'thinking': {'type': 'enabled', 'budget_tokens': 10000}}

    def test_anthropic_variant_thinking_false(self):
        pytest.importorskip('boto3')
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='anthropic',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = model._get_thinking_fields(settings, params)
        assert result == {'thinking': {'type': 'disabled'}}

    def test_openai_variant_thinking_high(self):
        pytest.importorskip('boto3')
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='openai',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = model._get_thinking_fields(settings, params)
        assert result == {'reasoning_effort': 'high'}

    def test_qwen_variant_thinking_true(self):
        pytest.importorskip('boto3')
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(
            bedrock_thinking_variant='qwen',
            supports_thinking=True,
        )

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = model._get_thinking_fields(settings, params)
        assert result == {'reasoning_config': 'high'}

    def test_thinking_none_returns_existing(self):
        pytest.importorskip('boto3')
        from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
        from pydantic_ai.providers.bedrock import BedrockModelProfile

        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._profile = BedrockModelProfile(bedrock_thinking_variant='anthropic')

        settings = BedrockModelSettings()
        params = ModelRequestParameters(thinking=None)
        result = model._get_thinking_fields(settings, params)
        assert result is None


class TestOpenRouterThinkingTranslation:
    """Test OpenRouter unified thinking fallback in _openrouter_settings_to_openai_settings."""

    def test_thinking_true(self):
        pytest.importorskip('openai')
        from pydantic_ai.models.openrouter import OpenRouterModelSettings, _openrouter_settings_to_openai_settings

        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking=True)
        result = _openrouter_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'medium'}

    def test_thinking_high(self):
        pytest.importorskip('openai')
        from pydantic_ai.models.openrouter import OpenRouterModelSettings, _openrouter_settings_to_openai_settings

        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking='high')
        result = _openrouter_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('reasoning') == {'effort': 'high'}

    def test_thinking_false_no_reasoning(self):
        pytest.importorskip('openai')
        from pydantic_ai.models.openrouter import OpenRouterModelSettings, _openrouter_settings_to_openai_settings

        settings = OpenRouterModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = _openrouter_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert 'reasoning' not in extra_body


class TestCerebrasThinkingTranslation:
    """Test Cerebras unified thinking fallback."""

    def test_thinking_false_sets_disable_reasoning(self):
        pytest.importorskip('openai')
        from pydantic_ai.models.cerebras import CerebrasModelSettings, _cerebras_settings_to_openai_settings

        settings = CerebrasModelSettings()
        params = ModelRequestParameters(thinking=False)
        result = _cerebras_settings_to_openai_settings(settings, params)
        extra_body: dict[str, Any] = result.get('extra_body') or {}  # type: ignore[assignment]
        assert extra_body.get('disable_reasoning') is True


class TestXaiThinkingTranslation:
    """Test xAI unified thinking fallback."""

    def test_thinking_high(self):
        pytest.importorskip('xai_sdk')
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
        pytest.importorskip('xai_sdk')
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
