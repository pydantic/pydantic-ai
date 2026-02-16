"""Tests for unified thinking settings across all model providers.

Tests the three-layer architecture:
1. ModelSettings (user input): `thinking: bool` + `thinking_effort: Literal['low', 'medium', 'high']`
2. resolve_thinking_config() (pure normalization): no validation, no errors
3. Model._resolve_*() (per-provider translation): silent-drop for unsupported settings
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import pytest

from pydantic_ai.profiles import ModelProfile

from .conftest import try_import

with try_import() as imports_successful:
    from google.genai.types import ThinkingLevel

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
    from pydantic_ai.models.cohere import CohereModel, CohereModelSettings
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.models.groq import GroqModel, GroqModelSettings
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings, OpenAIResponsesModelSettings
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.models.xai import XaiModel, XaiModelSettings


pytestmark = pytest.mark.skipif(not imports_successful(), reason='model extras not installed')

# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def thinking_profile() -> ModelProfile:
    """A model profile that supports thinking."""
    return ModelProfile(supports_thinking=True)


@pytest.fixture
def non_thinking_profile() -> ModelProfile:
    """A model profile that does NOT support thinking."""
    return ModelProfile(supports_thinking=False)


@pytest.fixture
def always_on_thinking_profile() -> ModelProfile:
    """A model profile where thinking is always enabled (e.g. o-series, DeepSeek R1)."""
    return ModelProfile(supports_thinking=True, thinking_always_enabled=True)


# ============================================================================
# Core resolve_thinking_config() tests
# ============================================================================


class TestResolveThinkingConfig:
    """Direct unit tests for pydantic_ai.thinking.resolve_thinking_config."""

    def test_no_settings_returns_none(self):
        """No thinking fields set → None (provider uses defaults)."""
        from pydantic_ai.thinking import resolve_thinking_config

        result = resolve_thinking_config({})
        assert result is None

    def test_thinking_true_enabled(self):
        """thinking=True → enabled=True, effort=None."""
        from pydantic_ai.thinking import resolve_thinking_config

        result = resolve_thinking_config({'thinking': True})
        assert result is not None
        assert result.enabled is True
        assert result.effort is None

    def test_thinking_false_disabled(self):
        """thinking=False → enabled=False (effort ignored per precedence rule 2)."""
        from pydantic_ai.thinking import resolve_thinking_config

        result = resolve_thinking_config({'thinking': False})
        assert result is not None
        assert result.enabled is False

    def test_effort_alone_implicit_enable(self):
        """thinking_effort without thinking → implicit enable (precedence rule 3)."""
        from pydantic_ai.thinking import resolve_thinking_config

        result = resolve_thinking_config({'thinking_effort': 'high'})
        assert result is not None
        assert result.enabled is True
        assert result.effort == 'high'

    def test_thinking_true_with_effort(self):
        """thinking=True + thinking_effort → enabled with effort."""
        from pydantic_ai.thinking import resolve_thinking_config

        result = resolve_thinking_config({'thinking': True, 'thinking_effort': 'low'})
        assert result is not None
        assert result.enabled is True
        assert result.effort == 'low'

    def test_thinking_false_ignores_effort(self):
        """thinking=False + thinking_effort → disabled (False overrides effort)."""
        from pydantic_ai.thinking import resolve_thinking_config

        result = resolve_thinking_config({'thinking': False, 'thinking_effort': 'high'})
        assert result is not None
        assert result.enabled is False
        # Effort is not checked when disabled


# ============================================================================
# Anthropic unified thinking tests
# ============================================================================


class TestAnthropicUnifiedThinking:
    """Tests for unified thinking settings on Anthropic models."""

    def test_thinking_true_budget_based_model(self, thinking_profile: ModelProfile):
        """thinking=True on budget-based model → enabled with default budget."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 4096}

    def test_thinking_true_adaptive_model(self, thinking_profile: ModelProfile):
        """thinking=True on adaptive model → type: adaptive."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'adaptive'}

    def test_thinking_false_disables(self, thinking_profile: ModelProfile):
        """thinking=False → type: disabled."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': False}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'disabled'}

    @pytest.mark.parametrize(
        'effort,expected_budget',
        [('low', 1024), ('medium', 4096), ('high', 16384)],
    )
    def test_effort_maps_to_budget(self, thinking_profile: ModelProfile, effort: str, expected_budget: int):
        """thinking_effort maps to budget_tokens on budget-based models."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking_effort': effort}  # type: ignore[typeddict-item]
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': expected_budget}

    def test_effort_on_adaptive_model_returns_adaptive(self, thinking_profile: ModelProfile):
        """thinking_effort on adaptive model → type: adaptive (effort flows through output_config)."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-opus-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking_effort': 'high'}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'adaptive'}

    def test_provider_specific_takes_precedence(self, thinking_profile: ModelProfile):
        """anthropic_thinking takes precedence over unified fields."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {
            'thinking': True,
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 5000},
        }
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 5000}

    def test_empty_settings_returns_none(self, thinking_profile: ModelProfile):
        """No thinking fields → None."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4'
        model._profile = thinking_profile

        result = model._resolve_thinking_config({})
        assert result is None

    def test_silent_drop_unsupported_model(self, non_thinking_profile: ModelProfile):
        """thinking=True on unsupported model → None (silent drop)."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-3-opus-20240229'
        model._profile = non_thinking_profile

        settings: AnthropicModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result is None


# ============================================================================
# Google unified thinking tests
# ============================================================================


class TestGoogleUnifiedThinking:
    """Tests for unified thinking settings on Google models."""

    @pytest.fixture
    def google_thinking_profile(self) -> ModelProfile:
        """A Google model profile that supports thinking."""
        return ModelProfile(supports_thinking=True)

    def test_thinking_true_gemini25(self, google_thinking_profile: ModelProfile):
        """thinking=True on Gemini 2.5 → None (enables thinking with provider defaults)."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        # No explicit config means provider defaults (thinking already on for 2.5)
        assert result is None

    def test_thinking_false_gemini25(self, google_thinking_profile: ModelProfile):
        """thinking=False on Gemini 2.5 → thinking_budget: 0."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking': False}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_budget': 0}

    @pytest.mark.parametrize(
        'effort,expected_budget',
        [('low', 1024), ('medium', 8192), ('high', 32768)],
    )
    def test_effort_maps_to_budget_gemini25(
        self, google_thinking_profile: ModelProfile, effort: str, expected_budget: int
    ):
        """thinking_effort maps to thinking_budget on Gemini 2.5."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking_effort': effort}  # type: ignore[typeddict-item]
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_budget': expected_budget}

    def test_thinking_true_gemini3(self, google_thinking_profile: ModelProfile):
        """thinking=True on Gemini 3 → thinking_level: HIGH."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_level': ThinkingLevel.HIGH}

    def test_thinking_false_gemini3(self, google_thinking_profile: ModelProfile):
        """thinking=False on Gemini 3 → thinking_level: LOW + include_thoughts: False."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking': False}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_level': ThinkingLevel.LOW, 'include_thoughts': False}

    @pytest.mark.parametrize(
        'effort,expected_level',
        [('low', ThinkingLevel.LOW), ('medium', ThinkingLevel.MEDIUM), ('high', ThinkingLevel.HIGH)],
    )
    def test_effort_maps_to_level_gemini3(
        self, google_thinking_profile: ModelProfile, effort: str, expected_level: ThinkingLevel
    ):
        """thinking_effort maps to thinking_level on Gemini 3."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking_effort': effort}  # type: ignore[typeddict-item]
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_level': expected_level}

    def test_provider_specific_takes_precedence(self, google_thinking_profile: ModelProfile):
        """google_thinking_config takes precedence over unified fields."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {
            'thinking': True,
            'google_thinking_config': {'thinking_budget': 5000},
        }
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_budget': 5000}

    def test_silent_drop_unsupported_model(self, non_thinking_profile: ModelProfile):
        """thinking=True on unsupported model → None (silent drop)."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.0-flash'
        model._profile = non_thinking_profile

        settings: GoogleModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result is None


# ============================================================================
# OpenAI Chat unified thinking tests
# ============================================================================


class TestOpenAIChatUnifiedThinking:
    """Tests for unified thinking settings on OpenAI Chat Completions models."""

    @pytest.fixture
    def openai_reasoning_profile(self) -> ModelProfile:
        """An OpenAI model profile that supports reasoning (like o3)."""
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)

    def test_thinking_true_uses_medium(self, openai_reasoning_profile: ModelProfile):
        """thinking=True → reasoning_effort: 'medium'."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': True}
        result = model._resolve_reasoning_effort(settings)

        assert result == 'medium'

    @pytest.mark.parametrize('effort', ['low', 'medium', 'high'])
    def test_effort_direct_mapping(self, openai_reasoning_profile: ModelProfile, effort: str):
        """thinking_effort maps 1:1 to reasoning_effort."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking_effort': effort}  # type: ignore[typeddict-item]
        result = model._resolve_reasoning_effort(settings)

        assert result == effort

    def test_thinking_false_returns_none(self, openai_reasoning_profile: ModelProfile):
        """thinking=False → None (silent drop on always-on model)."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': False}
        result = model._resolve_reasoning_effort(settings)

        assert result is None

    def test_provider_specific_takes_precedence(self, openai_reasoning_profile: ModelProfile):
        """openai_reasoning_effort takes precedence over unified fields."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {
            'thinking_effort': 'low',
            'openai_reasoning_effort': 'high',
        }
        result = model._resolve_reasoning_effort(settings)

        assert result == 'high'

    def test_silent_drop_unsupported_model(self, non_thinking_profile: ModelProfile):
        """thinking=True on unsupported model → None (silent drop)."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'gpt-4o'
        model._profile = non_thinking_profile

        settings: OpenAIChatModelSettings = {'thinking': True}
        result = model._resolve_reasoning_effort(settings)

        assert result is None

    def test_empty_settings_returns_none(self, openai_reasoning_profile: ModelProfile):
        """No thinking fields → None."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        result = model._resolve_reasoning_effort({})
        assert result is None


# ============================================================================
# OpenAI Responses unified thinking tests
# ============================================================================


class TestOpenAIResponsesUnifiedThinking:
    """Tests for unified thinking settings on OpenAI Responses API models."""

    @pytest.fixture
    def openai_responses_reasoning_profile(self) -> ModelProfile:
        """An OpenAI Responses model profile that supports reasoning."""
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)

    def test_thinking_true_uses_medium(self, openai_responses_reasoning_profile: ModelProfile):
        """thinking=True → effort='medium'."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        settings: OpenAIResponsesModelSettings = {'thinking': True}
        effort, summary = model._apply_unified_thinking(settings, None, None)

        assert effort == 'medium'
        assert summary is None

    def test_effort_direct_mapping(self, openai_responses_reasoning_profile: ModelProfile):
        """thinking_effort maps 1:1 to reasoning effort."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        settings: OpenAIResponsesModelSettings = {'thinking_effort': 'high'}
        effort, _ = model._apply_unified_thinking(settings, None, None)

        assert effort == 'high'

    def test_thinking_false_silent_drop(self, openai_responses_reasoning_profile: ModelProfile):
        """thinking=False on always-on model → no change (silent drop)."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        settings: OpenAIResponsesModelSettings = {'thinking': False}
        effort, summary = model._apply_unified_thinking(settings, None, None)

        assert effort is None
        assert summary is None

    def test_preserves_existing_reasoning_effort(self, openai_responses_reasoning_profile: ModelProfile):
        """Provider-specific reasoning_effort is preserved when unified also set."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        settings: OpenAIResponsesModelSettings = {'thinking_effort': 'low'}
        effort, _ = model._apply_unified_thinking(settings, 'high', None)

        # Existing 'high' should be preserved
        assert effort == 'high'

    def test_silent_drop_unsupported_model(self, non_thinking_profile: ModelProfile):
        """thinking=True on unsupported model → no change (silent drop)."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'gpt-4o'
        model._profile = non_thinking_profile

        settings: OpenAIResponsesModelSettings = {'thinking': True}
        effort, summary = model._apply_unified_thinking(settings, None, None)

        assert effort is None
        assert summary is None


# ============================================================================
# Bedrock unified thinking tests
# ============================================================================


class TestBedrockUnifiedThinking:
    """Tests for unified thinking settings on Bedrock models."""

    def test_thinking_true_uses_default_budget(self, thinking_profile: ModelProfile):
        """thinking=True → enabled with default budget (4096)."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 4096}

    def test_thinking_false_disables(self, thinking_profile: ModelProfile):
        """thinking=False → type: disabled."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking': False}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'disabled'}

    @pytest.mark.parametrize(
        'effort,expected_budget',
        [('low', 1024), ('medium', 4096), ('high', 16384)],
    )
    def test_effort_maps_to_budget(self, thinking_profile: ModelProfile, effort: str, expected_budget: int):
        """thinking_effort maps to budget_tokens on Bedrock Claude."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking_effort': effort}  # type: ignore[typeddict-item]
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': expected_budget}

    def test_silent_drop_unsupported_model(self, non_thinking_profile: ModelProfile):
        """thinking=True on unsupported model → None (silent drop)."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-3-opus-20240229-v1:0'
        model._profile = non_thinking_profile

        settings: BedrockModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result is None

    def test_empty_settings_returns_none(self, thinking_profile: ModelProfile):
        """No thinking fields → None."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        result = model._resolve_thinking_config({})
        assert result is None


# ============================================================================
# OpenRouter unified thinking tests
# ============================================================================


class TestOpenRouterUnifiedThinking:
    """Tests for unified thinking settings on OpenRouter models."""

    def test_thinking_true_enables_reasoning(self):
        """thinking=True → {enabled: True}."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'anthropic/claude-sonnet-4-5'
        model._profile = ModelProfile(supports_thinking=True)

        settings: OpenRouterModelSettings = {'thinking': True}
        result = model._resolve_reasoning_config(settings)

        assert result == {'enabled': True}

    def test_thinking_false_disables_reasoning(self):
        """thinking=False → {enabled: False}."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'anthropic/claude-sonnet-4-5'
        model._profile = ModelProfile(supports_thinking=True)

        settings: OpenRouterModelSettings = {'thinking': False}
        result = model._resolve_reasoning_config(settings)

        assert result == {'enabled': False}

    @pytest.mark.parametrize('effort', ['low', 'medium', 'high'])
    def test_effort_passthrough(self, effort: str):
        """thinking_effort passes through directly to OpenRouter."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'openai/o3'
        model._profile = ModelProfile(supports_thinking=True)

        settings: OpenRouterModelSettings = {'thinking_effort': effort}  # type: ignore[typeddict-item]
        result = model._resolve_reasoning_config(settings)

        assert result == {'effort': effort}

    def test_empty_settings_returns_none(self):
        """No thinking fields → None."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'openai/o3'
        model._profile = ModelProfile(supports_thinking=True)

        result = model._resolve_reasoning_config({})
        assert result is None


# ============================================================================
# Groq unified thinking tests
# ============================================================================


class TestGroqUnifiedThinking:
    """Tests for unified thinking settings on Groq models."""

    def test_thinking_true_uses_parsed(self, thinking_profile: ModelProfile):
        """thinking=True → 'parsed'."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        settings: GroqModelSettings = {'thinking': True}
        result = model._resolve_reasoning_format(settings)

        assert result == 'parsed'

    def test_thinking_false_uses_hidden(self, thinking_profile: ModelProfile):
        """thinking=False → 'hidden'."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        settings: GroqModelSettings = {'thinking': False}
        result = model._resolve_reasoning_format(settings)

        assert result == 'hidden'

    def test_effort_silently_ignored(self, thinking_profile: ModelProfile):
        """thinking_effort is silently ignored (Groq has no effort control)."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        settings: GroqModelSettings = {'thinking_effort': 'high'}
        result = model._resolve_reasoning_format(settings)

        # Effort triggers enable, but effort value itself is dropped
        assert result == 'parsed'

    def test_silent_drop_unsupported_model(self, non_thinking_profile: ModelProfile):
        """thinking=True on unsupported model → None (silent drop)."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'llama-3.1-8b-instant'
        model._profile = non_thinking_profile

        settings: GroqModelSettings = {'thinking': True}
        result = model._resolve_reasoning_format(settings)

        assert result is None

    def test_empty_settings_returns_none(self, thinking_profile: ModelProfile):
        """No thinking fields → None."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        result = model._resolve_reasoning_format({})
        assert result is None


# ============================================================================
# Cerebras unified thinking tests
# ============================================================================


class TestCerebrasUnifiedThinking:
    """Tests for unified thinking settings on Cerebras models."""

    def test_thinking_true_returns_none(self, thinking_profile: ModelProfile):
        """thinking=True → None (use default enabled behavior)."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = thinking_profile

        settings: CerebrasModelSettings = {'thinking': True}
        result = model._resolve_reasoning_config(settings)

        assert result is None

    def test_thinking_false_disables(self, thinking_profile: ModelProfile):
        """thinking=False → True (disable_reasoning=True)."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = thinking_profile

        settings: CerebrasModelSettings = {'thinking': False}
        result = model._resolve_reasoning_config(settings)

        assert result is True

    def test_effort_silently_ignored(self, thinking_profile: ModelProfile):
        """thinking_effort is silently ignored (Cerebras has no effort control)."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = thinking_profile

        settings: CerebrasModelSettings = {'thinking_effort': 'high'}
        result = model._resolve_reasoning_config(settings)

        # Enabled (None → don't set disable_reasoning), effort is dropped
        assert result is None

    def test_silent_drop_unsupported_model(self, non_thinking_profile: ModelProfile):
        """thinking=True on unsupported model → None (silent drop)."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'llama-3.3-70b'
        model._profile = non_thinking_profile

        settings: CerebrasModelSettings = {'thinking': True}
        result = model._resolve_reasoning_config(settings)

        assert result is None

    def test_empty_settings_returns_none(self, thinking_profile: ModelProfile):
        """No thinking fields → None."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = thinking_profile

        result = model._resolve_reasoning_config({})
        assert result is None


# ============================================================================
# Cohere unified thinking tests
# ============================================================================


class TestCohereUnifiedThinking:
    """Tests for unified thinking settings on Cohere models."""

    def test_thinking_true_enables(self, thinking_profile: ModelProfile):
        """thinking=True → Thinking(type='enabled')."""
        from cohere import Thinking

        model = CohereModel.__new__(CohereModel)
        model._model_name = 'command-a-reasoning'
        model._profile = thinking_profile

        settings: CohereModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result == Thinking(type='enabled')

    def test_thinking_false_disables(self, thinking_profile: ModelProfile):
        """thinking=False → Thinking(type='disabled')."""
        from cohere import Thinking

        model = CohereModel.__new__(CohereModel)
        model._model_name = 'command-a-reasoning'
        model._profile = thinking_profile

        settings: CohereModelSettings = {'thinking': False}
        result = model._resolve_thinking_config(settings)

        assert result == Thinking(type='disabled')

    def test_effort_silently_ignored(self, thinking_profile: ModelProfile):
        """thinking_effort is silently ignored (Cohere has no effort control)."""
        from cohere import Thinking

        model = CohereModel.__new__(CohereModel)
        model._model_name = 'command-a-reasoning'
        model._profile = thinking_profile

        settings: CohereModelSettings = {'thinking_effort': 'high'}
        result = model._resolve_thinking_config(settings)

        # Enabled, but effort is dropped
        assert result == Thinking(type='enabled')

    def test_silent_drop_unsupported_model(self, non_thinking_profile: ModelProfile):
        """thinking=True on unsupported model → None (silent drop)."""
        model = CohereModel.__new__(CohereModel)
        model._model_name = 'command-r-plus'
        model._profile = non_thinking_profile

        settings: CohereModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result is None

    def test_empty_settings_returns_none(self, thinking_profile: ModelProfile):
        """No thinking fields → None."""
        model = CohereModel.__new__(CohereModel)
        model._model_name = 'command-a-reasoning'
        model._profile = thinking_profile

        result = model._resolve_thinking_config({})
        assert result is None


# ============================================================================
# xAI unified thinking tests
# ============================================================================


class TestXaiUnifiedThinking:
    """Tests for unified thinking settings on xAI models."""

    @pytest.fixture
    def grok3_mini_profile(self) -> ModelProfile:
        """Profile for grok-3-mini (supports thinking, effort control)."""
        return ModelProfile(supports_thinking=True)

    @pytest.fixture
    def grok4_profile(self) -> ModelProfile:
        """Profile for grok-4 (supports thinking, always-on, no effort control)."""
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)

    def test_effort_low_maps_to_low(self, grok3_mini_profile: ModelProfile):
        """thinking_effort='low' on grok-3-mini → 'low'."""
        model = XaiModel.__new__(XaiModel)
        model._model_name = 'grok-3-mini'
        model._profile = grok3_mini_profile

        settings: XaiModelSettings = {'thinking_effort': 'low'}
        result = model._resolve_reasoning_effort(settings)

        assert result == 'low'

    def test_effort_medium_downmaps_to_low(self, grok3_mini_profile: ModelProfile):
        """thinking_effort='medium' on grok-3-mini → 'low' (conservative downmap)."""
        model = XaiModel.__new__(XaiModel)
        model._model_name = 'grok-3-mini'
        model._profile = grok3_mini_profile

        settings: XaiModelSettings = {'thinking_effort': 'medium'}
        result = model._resolve_reasoning_effort(settings)

        assert result == 'low'

    def test_effort_high_maps_to_high(self, grok3_mini_profile: ModelProfile):
        """thinking_effort='high' on grok-3-mini → 'high'."""
        model = XaiModel.__new__(XaiModel)
        model._model_name = 'grok-3-mini'
        model._profile = grok3_mini_profile

        settings: XaiModelSettings = {'thinking_effort': 'high'}
        result = model._resolve_reasoning_effort(settings)

        assert result == 'high'

    def test_thinking_true_grok3_mini_no_effort(self, grok3_mini_profile: ModelProfile):
        """thinking=True without effort on grok-3-mini → None (no explicit effort)."""
        model = XaiModel.__new__(XaiModel)
        model._model_name = 'grok-3-mini'
        model._profile = grok3_mini_profile

        settings: XaiModelSettings = {'thinking': True}
        result = model._resolve_reasoning_effort(settings)

        assert result is None

    def test_thinking_false_grok3_mini(self, grok3_mini_profile: ModelProfile):
        """thinking=False on grok-3-mini → None (can't disable via API)."""
        model = XaiModel.__new__(XaiModel)
        model._model_name = 'grok-3-mini'
        model._profile = grok3_mini_profile

        settings: XaiModelSettings = {'thinking': False}
        result = model._resolve_reasoning_effort(settings)

        assert result is None

    def test_effort_on_grok4_silently_ignored(self, grok4_profile: ModelProfile):
        """thinking_effort on grok-4 → None (only grok-3-mini has effort control)."""
        model = XaiModel.__new__(XaiModel)
        model._model_name = 'grok-4-fast-reasoning'
        model._profile = grok4_profile

        settings: XaiModelSettings = {'thinking_effort': 'high'}
        result = model._resolve_reasoning_effort(settings)

        assert result is None

    def test_silent_drop_unsupported_model(self, non_thinking_profile: ModelProfile):
        """thinking=True on unsupported model → None (silent drop)."""
        model = XaiModel.__new__(XaiModel)
        model._model_name = 'grok-4-fast-non-reasoning'
        model._profile = non_thinking_profile

        settings: XaiModelSettings = {'thinking': True}
        result = model._resolve_reasoning_effort(settings)

        assert result is None

    def test_empty_settings_returns_none(self, grok3_mini_profile: ModelProfile):
        """No thinking fields → None."""
        model = XaiModel.__new__(XaiModel)
        model._model_name = 'grok-3-mini'
        model._profile = grok3_mini_profile

        result = model._resolve_reasoning_effort({})
        assert result is None


# ============================================================================
# Profile capability tests
# ============================================================================


class TestProfileThinkingCapabilities:
    """Tests for thinking capabilities in model profiles."""

    def test_anthropic_profile_thinking_support(self):
        """Anthropic profiles correctly detect thinking-capable models."""
        from pydantic_ai.profiles.anthropic import anthropic_model_profile

        # Claude 3.7+ supports thinking
        profile = anthropic_model_profile('claude-3-7-sonnet')
        assert profile is not None
        assert profile.supports_thinking is True

        # Claude 4 supports thinking
        profile = anthropic_model_profile('claude-sonnet-4-5')
        assert profile is not None
        assert profile.supports_thinking is True

        # Older models don't support thinking
        profile = anthropic_model_profile('claude-3-opus-20240229')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_google_profile_thinking_support(self):
        """Google profiles correctly detect thinking-capable models."""
        from pydantic_ai.profiles.google import google_model_profile

        # Gemini 2.5 supports thinking
        profile = google_model_profile('gemini-2.5-flash')
        assert profile is not None
        assert profile.supports_thinking is True

        # Gemini 3 supports thinking
        profile = google_model_profile('gemini-3-flash')
        assert profile is not None
        assert profile.supports_thinking is True

        # Older models don't support thinking
        profile = google_model_profile('gemini-2.0-flash')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_openai_profile_thinking_support(self):
        """OpenAI profiles correctly detect reasoning models."""
        from pydantic_ai.profiles.openai import openai_model_profile

        # o-series supports thinking (always on)
        profile = openai_model_profile('o3')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True

        # GPT-5 supports thinking
        profile = openai_model_profile('gpt-5')
        assert profile is not None
        assert profile.supports_thinking is True

        # Non-reasoning models don't support thinking
        profile = openai_model_profile('gpt-4o')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_deepseek_profile_thinking_support(self):
        """DeepSeek profiles correctly detect R1 reasoning models."""
        from pydantic_ai.profiles.deepseek import deepseek_model_profile

        profile = deepseek_model_profile('deepseek-r1')
        assert profile is not None
        assert profile.supports_thinking is True

        profile = deepseek_model_profile('deepseek-chat')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_groq_profile_thinking_support(self):
        """Groq profiles correctly detect reasoning models."""
        from pydantic_ai.profiles.groq import groq_model_profile

        profile = groq_model_profile('deepseek-r1-distill-llama-70b')
        assert profile is not None
        assert profile.supports_thinking is True

        profile = groq_model_profile('llama-3.1-8b-instant')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_mistral_profile_thinking_support(self):
        """Mistral profiles correctly detect Magistral reasoning models."""
        from pydantic_ai.profiles.mistral import mistral_model_profile

        # Magistral models: thinking always on
        profile = mistral_model_profile('magistral-medium')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True

        # Regular Mistral models: no special profile
        profile = mistral_model_profile('mistral-large')
        assert profile is None

    def test_cohere_profile_thinking_support(self):
        """Cohere profiles correctly detect reasoning models."""
        from pydantic_ai.profiles.cohere import cohere_model_profile

        profile = cohere_model_profile('command-a-reasoning')
        assert profile is not None
        assert profile.supports_thinking is True

        # Non-reasoning Cohere models: no special profile
        profile = cohere_model_profile('command-r-plus')
        assert profile is None

    def test_grok_profile_thinking_support(self):
        """Grok profiles correctly detect reasoning models and model variants."""
        from pydantic_ai.profiles.grok import grok_model_profile

        # grok-3-mini: supports thinking, NOT always-on (has effort control)
        profile = grok_model_profile('grok-3-mini')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is False

        # grok-4: supports thinking, always-on
        profile = grok_model_profile('grok-4-fast-reasoning')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True

        # non-reasoning variant: no thinking
        profile = grok_model_profile('grok-4-fast-non-reasoning')
        assert profile is not None
        assert profile.supports_thinking is False


# ============================================================================
# Cross-provider portability tests
# ============================================================================


class TestCrossProviderPortability:
    """Tests verifying that the same unified settings work across providers."""

    def test_same_settings_all_providers(self):
        """The same settings dict should work across all providers (silent drop)."""
        # Anthropic (budget-based model)
        anthropic_model = AnthropicModel.__new__(AnthropicModel)
        anthropic_model._model_name = 'claude-sonnet-4'
        anthropic_model._profile = ModelProfile(supports_thinking=True)
        result = anthropic_model._resolve_thinking_config(AnthropicModelSettings(thinking=True, thinking_effort='high'))
        assert result == {'type': 'enabled', 'budget_tokens': 16384}

        # OpenAI Chat
        openai_model = OpenAIChatModel.__new__(OpenAIChatModel)
        openai_model._model_name = 'o3'
        openai_model._profile = ModelProfile(supports_thinking=True, thinking_always_enabled=True)
        result = openai_model._resolve_reasoning_effort(OpenAIChatModelSettings(thinking=True, thinking_effort='high'))
        assert result == 'high'

        # Groq (effort silently ignored)
        groq_model = GroqModel.__new__(GroqModel)
        groq_model._model_name = 'deepseek-r1-distill-llama-70b'
        groq_model._profile = ModelProfile(supports_thinking=True)
        result = groq_model._resolve_reasoning_format(GroqModelSettings(thinking=True, thinking_effort='high'))
        assert result == 'parsed'

        # Cerebras (effort silently ignored)
        cerebras_model = CerebrasModel.__new__(CerebrasModel)
        cerebras_model._model_name = 'zai-glm-4.6'
        cerebras_model._profile = ModelProfile(supports_thinking=True)
        result = cerebras_model._resolve_reasoning_config(CerebrasModelSettings(thinking=True, thinking_effort='high'))
        assert result is None  # enabled is default, effort dropped

    def test_settings_on_unsupported_models_silently_dropped(self):
        """Thinking settings on models without thinking support → silently dropped."""
        non_thinking = ModelProfile(supports_thinking=False)

        # All providers should return None (or equivalent no-op)
        anthropic_model = AnthropicModel.__new__(AnthropicModel)
        anthropic_model._model_name = 'claude-3-opus'
        anthropic_model._profile = non_thinking
        assert (
            anthropic_model._resolve_thinking_config(AnthropicModelSettings(thinking=True, thinking_effort='high'))
            is None
        )

        openai_model = OpenAIChatModel.__new__(OpenAIChatModel)
        openai_model._model_name = 'gpt-4o'
        openai_model._profile = non_thinking
        assert (
            openai_model._resolve_reasoning_effort(OpenAIChatModelSettings(thinking=True, thinking_effort='high'))
            is None
        )

        groq_model = GroqModel.__new__(GroqModel)
        groq_model._model_name = 'llama-3.1-8b'
        groq_model._profile = non_thinking
        assert groq_model._resolve_reasoning_format(GroqModelSettings(thinking=True, thinking_effort='high')) is None


# ============================================================================
# Model settings merge tests for thinking configuration
# ============================================================================


class TestMergeModelSettingsThinking:
    """Tests for merge_model_settings with unified thinking fields."""

    def test_merge_thinking_bool_override(self):
        """Override thinking bool replaces base."""
        from pydantic_ai.settings import merge_model_settings

        base: AnthropicModelSettings = {'thinking': True}
        overrides: AnthropicModelSettings = {'thinking': False}

        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('thinking') is False

    def test_merge_effort_override(self):
        """Override thinking_effort replaces base."""
        from pydantic_ai.settings import merge_model_settings

        base: AnthropicModelSettings = {'thinking_effort': 'low'}
        overrides: AnthropicModelSettings = {'thinking_effort': 'high'}

        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('thinking_effort') == 'high'

    def test_merge_preserves_non_thinking_settings(self):
        """Non-thinking settings preserved during merge."""
        from pydantic_ai.settings import merge_model_settings

        base: AnthropicModelSettings = {'max_tokens': 1000, 'temperature': 0.5}
        overrides: AnthropicModelSettings = {'thinking': True}

        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('max_tokens') == 1000
        assert result.get('temperature') == 0.5
        assert result.get('thinking') is True

    def test_merge_with_none_base(self):
        """Merging with None base returns overrides."""
        from pydantic_ai.settings import merge_model_settings

        overrides: AnthropicModelSettings = {'thinking': True, 'thinking_effort': 'high'}
        result = merge_model_settings(None, overrides)
        assert result == overrides

    def test_merge_with_none_overrides(self):
        """Merging with None overrides returns base."""
        from pydantic_ai.settings import merge_model_settings

        base: AnthropicModelSettings = {'thinking': True}
        result = merge_model_settings(base, None)
        assert result == base

    def test_merge_with_both_none(self):
        """Merging both None returns None."""
        from pydantic_ai.settings import merge_model_settings

        result = merge_model_settings(None, None)
        assert result is None
