"""Tests for unified thinking settings across all model providers."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.anthropic import ANTHROPIC_EFFORT_TO_BUDGET
from pydantic_ai.profiles.google import GOOGLE_EFFORT_TO_BUDGET

from .conftest import try_import

with try_import() as imports_successful:
    # Import ThinkingLevel for Gemini 3 tests
    from google.genai.types import ThinkingLevel

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.models.cerebras import CerebrasModel, CerebrasModelSettings
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.models.groq import GroqModel, GroqModelSettings
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings, OpenAIResponsesModelSettings
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings


pytestmark = pytest.mark.skipif(not imports_successful(), reason='model extras not installed')

# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def thinking_profile() -> ModelProfile:
    """A model profile that supports thinking."""
    return ModelProfile(
        supports_thinking=True,
        default_thinking_budget=4096,
        effort_to_budget_map=ANTHROPIC_EFFORT_TO_BUDGET,
    )


@pytest.fixture
def non_thinking_profile() -> ModelProfile:
    """A model profile that does NOT support thinking."""
    return ModelProfile(
        supports_thinking=False,
    )


@pytest.fixture
def always_on_thinking_profile() -> ModelProfile:
    """A model profile where thinking is always enabled (like OpenAI o-series)."""
    return ModelProfile(
        supports_thinking=True,
        thinking_always_enabled=True,
    )


# ============================================================================
# Anthropic unified thinking tests
# ============================================================================


class TestAnthropicUnifiedThinking:
    """Tests for unified thinking settings on Anthropic models."""

    def test_thinking_true_uses_default_budget(self, thinking_profile: ModelProfile):
        """thinking=True should enable thinking with the default budget."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 4096}

    def test_thinking_false_disables_thinking(self, thinking_profile: ModelProfile):
        """thinking=False should disable thinking."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': False}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'disabled'}

    def test_thinking_config_with_budget(self, thinking_profile: ModelProfile):
        """ThinkingConfig with explicit budget_tokens should use that budget."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': {'budget_tokens': 2048}}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 2048}

    def test_thinking_config_with_effort_low(self, thinking_profile: ModelProfile):
        """ThinkingConfig with effort='low' should map to 1024 tokens."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': {'effort': 'low'}}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 1024}

    def test_thinking_config_with_effort_medium(self, thinking_profile: ModelProfile):
        """ThinkingConfig with effort='medium' should map to 4096 tokens."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': {'effort': 'medium'}}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 4096}

    def test_thinking_config_with_effort_high(self, thinking_profile: ModelProfile):
        """ThinkingConfig with effort='high' should map to 16384 tokens."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': {'effort': 'high'}}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 16384}

    def test_thinking_config_enabled_false(self, thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=False should disable thinking."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': {'enabled': False}}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'disabled'}

    def test_provider_specific_takes_precedence(self, thinking_profile: ModelProfile):
        """anthropic_thinking should take precedence over thinking."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {
            'thinking': {'budget_tokens': 1000},  # Unified setting
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 5000},  # Provider-specific
        }
        result = model._resolve_thinking_config(settings)

        # Provider-specific should win
        assert result == {'type': 'enabled', 'budget_tokens': 5000}

    def test_thinking_none_returns_none(self, thinking_profile: ModelProfile):
        """No thinking setting should return None."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {}
        result = model._resolve_thinking_config(settings)

        assert result is None

    def test_unsupported_model_raises_error(self, non_thinking_profile: ModelProfile):
        """Using thinking on a model that doesn't support it should raise UserError."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-3-opus-20240229'
        model._profile = non_thinking_profile

        settings: AnthropicModelSettings = {'thinking': True}

        with pytest.raises(UserError, match='does not support thinking/reasoning'):
            model._resolve_thinking_config(settings)

    def test_thinking_config_enabled_true_uses_default_budget(self, thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=True but no effort/budget should use default budget."""
        model = AnthropicModel.__new__(AnthropicModel)
        model._model_name = 'claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: AnthropicModelSettings = {'thinking': {'enabled': True}}
        result = model._resolve_thinking_config(settings)

        # Should use the default budget from the profile (4096)
        assert result == {'type': 'enabled', 'budget_tokens': 4096}


# ============================================================================
# Google unified thinking tests
# ============================================================================


class TestGoogleUnifiedThinking:
    """Tests for unified thinking settings on Google models."""

    @pytest.fixture
    def google_thinking_profile(self) -> ModelProfile:
        """A Google model profile that supports thinking (Gemini 2.5 - uses thinking_budget)."""
        return ModelProfile(
            supports_thinking=True,
            effort_to_budget_map=GOOGLE_EFFORT_TO_BUDGET,
        )

    @pytest.fixture
    def google_gemini3_profile(self) -> ModelProfile:
        """A Google model profile for Gemini 3 (uses thinking_level instead of budget)."""
        return ModelProfile(
            supports_thinking=True,
            supports_thinking_level=True,
        )

    def test_thinking_true_enables_with_include_thoughts(self, google_thinking_profile: ModelProfile):
        """thinking=True should enable thinking with include_thoughts=True."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result == {'include_thoughts': True}

    def test_thinking_false_disables_with_zero_budget(self, google_thinking_profile: ModelProfile):
        """thinking=False should disable thinking with budget=0."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking': False}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_budget': 0}

    def test_thinking_config_with_budget(self, google_thinking_profile: ModelProfile):
        """ThinkingConfig with budget_tokens should set thinking_budget."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking': {'budget_tokens': 8192}}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_budget': 8192, 'include_thoughts': True}

    def test_thinking_config_with_effort_high(self, google_thinking_profile: ModelProfile):
        """ThinkingConfig with effort='high' should map to 32768 tokens for Google."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking': {'effort': 'high'}}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_budget': 32768, 'include_thoughts': True}

    def test_thinking_config_include_in_response_false(self, google_thinking_profile: ModelProfile):
        """ThinkingConfig with include_in_response=False should set include_thoughts=False."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {'thinking': {'include_in_response': False}}
        result = model._resolve_thinking_config(settings)

        assert result == {'include_thoughts': False}

    def test_provider_specific_takes_precedence(self, google_thinking_profile: ModelProfile):
        """google_thinking_config should take precedence over thinking."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.5-flash'
        model._profile = google_thinking_profile

        settings: GoogleModelSettings = {
            'thinking': {'budget_tokens': 1000},  # Unified setting
            'google_thinking_config': {'thinking_budget': 5000},  # Provider-specific
        }
        result = model._resolve_thinking_config(settings)

        # Provider-specific should win
        assert result == {'thinking_budget': 5000}

    # --- Gemini 3 thinking_level tests ---

    def test_gemini3_thinking_true_uses_level(self, google_gemini3_profile: ModelProfile):
        """Gemini 3: thinking=True should enable thinking with thinking_level=HIGH."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_gemini3_profile

        settings: GoogleModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_level': ThinkingLevel.HIGH, 'include_thoughts': True}

    def test_gemini3_thinking_config_effort_low(self, google_gemini3_profile: ModelProfile):
        """Gemini 3: ThinkingConfig with effort='low' should set thinking_level='low'."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_gemini3_profile

        settings: GoogleModelSettings = {'thinking': {'effort': 'low'}}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_level': ThinkingLevel.LOW, 'include_thoughts': True}

    def test_gemini3_thinking_config_effort_medium(self, google_gemini3_profile: ModelProfile):
        """Gemini 3 Flash: ThinkingConfig with effort='medium' should set thinking_level='medium'."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_gemini3_profile

        settings: GoogleModelSettings = {'thinking': {'effort': 'medium'}}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_level': ThinkingLevel.MEDIUM, 'include_thoughts': True}

    def test_gemini3_thinking_config_effort_high(self, google_gemini3_profile: ModelProfile):
        """Gemini 3: ThinkingConfig with effort='high' should set thinking_level='high'."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_gemini3_profile

        settings: GoogleModelSettings = {'thinking': {'effort': 'high'}}
        result = model._resolve_thinking_config(settings)

        assert result == {'thinking_level': ThinkingLevel.HIGH, 'include_thoughts': True}

    def test_gemini3_pro_medium_effort_warns_and_maps_to_high(self, google_gemini3_profile: ModelProfile):
        """Gemini 3 Pro: effort='medium' should warn and map to 'high'."""
        import warnings
        from unittest.mock import Mock

        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-pro'
        model._profile = google_gemini3_profile
        mock_provider = Mock()
        mock_provider.name = 'google-gla'  # Set name as attribute, not Mock(name=...)
        model._provider = mock_provider

        settings: GoogleModelSettings = {'thinking': {'effort': 'medium'}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = model._resolve_thinking_config(settings)

            # Should emit a warning about medium being mapped to high
            assert len(w) == 1
            assert "mapped effort='medium' to 'high'" in str(w[0].message)
            assert 'gemini-3-pro' in str(w[0].message)

        # Should map to 'high' instead
        assert result == {'thinking_level': ThinkingLevel.HIGH, 'include_thoughts': True}

    def test_gemini3_thinking_false_disables(self, google_gemini3_profile: ModelProfile):
        """Gemini 3: thinking=False should disable thinking with thinking_level=LOW."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_gemini3_profile

        settings: GoogleModelSettings = {'thinking': False}
        result = model._resolve_thinking_config(settings)

        # Gemini 3 uses thinking_level=LOW with include_thoughts=False to disable
        assert result == {'thinking_level': ThinkingLevel.LOW, 'include_thoughts': False}

    def test_gemini3_thinking_config_enabled_false(self, google_gemini3_profile: ModelProfile):
        """Gemini 3: ThinkingConfig with enabled=False should disable thinking."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_gemini3_profile

        settings: GoogleModelSettings = {'thinking': {'enabled': False}}
        result = model._resolve_thinking_config(settings)

        # Gemini 3 uses thinking_level=LOW with include_thoughts=False to disable
        assert result == {'thinking_level': ThinkingLevel.LOW, 'include_thoughts': False}

    def test_gemini3_thinking_config_enabled_true_defaults_to_high(self, google_gemini3_profile: ModelProfile):
        """Gemini 3: ThinkingConfig with enabled=True but no effort should default to HIGH."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-3-flash'
        model._profile = google_gemini3_profile

        settings: GoogleModelSettings = {'thinking': {'enabled': True}}
        result = model._resolve_thinking_config(settings)

        # Should default to HIGH when no effort is specified
        assert result == {'thinking_level': ThinkingLevel.HIGH, 'include_thoughts': True}

    def test_unsupported_model_raises_error(self, non_thinking_profile: ModelProfile):
        """Using thinking on a Google model that doesn't support it should raise UserError."""
        model = GoogleModel.__new__(GoogleModel)
        model._model_name = 'gemini-2.0-flash'
        model._profile = non_thinking_profile

        settings: GoogleModelSettings = {'thinking': True}

        with pytest.raises(UserError, match='does not support thinking/reasoning'):
            model._resolve_thinking_config(settings)


# ============================================================================
# OpenAI unified thinking tests
# ============================================================================


class TestOpenAIUnifiedThinking:
    """Tests for unified thinking settings on OpenAI models."""

    @pytest.fixture
    def openai_reasoning_profile(self) -> ModelProfile:
        """An OpenAI model profile that supports reasoning (like o3)."""
        return ModelProfile(
            supports_thinking=True,
            thinking_always_enabled=True,
        )

    @pytest.fixture
    def openai_optional_reasoning_profile(self) -> ModelProfile:
        """An OpenAI model profile with optional reasoning (like o1-preview)."""
        return ModelProfile(
            supports_thinking=True,
            thinking_always_enabled=False,
        )

    def test_thinking_true_uses_medium_effort(self, openai_reasoning_profile: ModelProfile):
        """thinking=True should use medium effort by default."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': True}
        result = model._resolve_reasoning_effort(settings)

        assert result == 'medium'

    def test_thinking_config_with_effort_low(self, openai_reasoning_profile: ModelProfile):
        """ThinkingConfig with effort='low' should return 'low'."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': {'effort': 'low'}}
        result = model._resolve_reasoning_effort(settings)

        assert result == 'low'

    def test_thinking_config_with_effort_high(self, openai_reasoning_profile: ModelProfile):
        """ThinkingConfig with effort='high' should return 'high'."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': {'effort': 'high'}}
        result = model._resolve_reasoning_effort(settings)

        assert result == 'high'

    def test_thinking_false_raises_error_for_always_on(self, openai_reasoning_profile: ModelProfile):
        """thinking=False should raise UserError for always-on thinking models."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': False}

        with pytest.raises(UserError, match='has reasoning always enabled and cannot be disabled'):
            model._resolve_reasoning_effort(settings)

    def test_thinking_enabled_false_raises_error_for_always_on(self, openai_reasoning_profile: ModelProfile):
        """ThinkingConfig with enabled=False should raise UserError for always-on models."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': {'enabled': False}}

        with pytest.raises(UserError, match='has reasoning always enabled and cannot be disabled'):
            model._resolve_reasoning_effort(settings)

    def test_provider_specific_takes_precedence(self, openai_reasoning_profile: ModelProfile):
        """openai_reasoning_effort should take precedence over thinking."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o3'
        model._profile = openai_reasoning_profile

        settings: OpenAIChatModelSettings = {
            'thinking': {'effort': 'low'},  # Unified setting
            'openai_reasoning_effort': 'high',  # Provider-specific
        }
        result = model._resolve_reasoning_effort(settings)

        # Provider-specific should win
        assert result == 'high'

    def test_unsupported_model_raises_error(self, non_thinking_profile: ModelProfile):
        """Using thinking on a model that doesn't support it should raise UserError."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'gpt-4o'
        model._profile = non_thinking_profile

        settings: OpenAIChatModelSettings = {'thinking': True}

        with pytest.raises(UserError, match='does not support reasoning'):
            model._resolve_reasoning_effort(settings)

    def test_thinking_false_disables_on_optional_model(self, openai_optional_reasoning_profile: ModelProfile):
        """thinking=False should return None on models where reasoning is optional."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o1-preview'
        model._profile = openai_optional_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': False}
        result = model._resolve_reasoning_effort(settings)

        assert result is None

    def test_thinking_config_enabled_false_disables_on_optional_model(
        self, openai_optional_reasoning_profile: ModelProfile
    ):
        """ThinkingConfig with enabled=False should return None on optional reasoning models."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o1-preview'
        model._profile = openai_optional_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': {'enabled': False}}
        result = model._resolve_reasoning_effort(settings)

        assert result is None

    def test_thinking_config_no_effort_defaults_to_medium(self, openai_optional_reasoning_profile: ModelProfile):
        """ThinkingConfig without effort should default to 'medium'."""
        model = OpenAIChatModel.__new__(OpenAIChatModel)
        model._model_name = 'o1-preview'
        model._profile = openai_optional_reasoning_profile

        settings: OpenAIChatModelSettings = {'thinking': {'enabled': True}}
        result = model._resolve_reasoning_effort(settings)

        assert result == 'medium'


class TestOpenAIResponsesUnifiedThinking:
    """Tests for unified thinking settings on OpenAI Responses API models."""

    @pytest.fixture
    def openai_responses_reasoning_profile(self) -> ModelProfile:
        """An OpenAI Responses model profile that supports reasoning."""
        return ModelProfile(
            supports_thinking=True,
            thinking_always_enabled=True,
        )

    @pytest.fixture
    def openai_responses_optional_reasoning_profile(self) -> ModelProfile:
        """An OpenAI Responses model profile with optional reasoning."""
        return ModelProfile(
            supports_thinking=True,
            thinking_always_enabled=False,
        )

    def test_thinking_config_with_budget_tokens_warns(self, openai_responses_reasoning_profile: ModelProfile):
        """ThinkingConfig with budget_tokens should warn about ignored setting."""
        import warnings
        from unittest.mock import Mock

        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile
        mock_provider = Mock()
        mock_provider.name = 'openai'
        model._provider = mock_provider

        settings: OpenAIResponsesModelSettings = {'thinking': {'budget_tokens': 4096, 'effort': 'high'}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = model._apply_unified_thinking(settings, None, None)

            assert len(w) == 1
            assert 'budget_tokens' in str(w[0].message)
            assert "effort='high'" in str(w[0].message)

        effort, _ = result
        assert effort == 'high'

    def test_thinking_config_with_summary(self, openai_responses_reasoning_profile: ModelProfile):
        """ThinkingConfig with summary should map to OpenAI format."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        # Test summary='concise'
        settings: OpenAIResponsesModelSettings = {'thinking': {'summary': 'concise'}}
        _, summary = model._apply_unified_thinking(settings, None, None)
        assert summary == 'concise'

        # Test summary='detailed'
        settings = {'thinking': {'summary': 'detailed'}}
        _, summary = model._apply_unified_thinking(settings, None, None)
        assert summary == 'detailed'

        # Test summary=True maps to 'auto'
        settings = {'thinking': {'summary': True}}
        _, summary = model._apply_unified_thinking(settings, None, None)
        assert summary == 'auto'

        # Test summary='none' maps to None
        settings = {'thinking': {'summary': 'none'}}
        _, summary = model._apply_unified_thinking(settings, None, None)
        assert summary is None

    def test_unsupported_model_raises_error(self, non_thinking_profile: ModelProfile):
        """Using thinking on a non-reasoning model should raise UserError."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'gpt-4o'
        model._profile = non_thinking_profile

        settings: OpenAIResponsesModelSettings = {'thinking': True}

        with pytest.raises(UserError, match='does not support reasoning'):
            model._apply_unified_thinking(settings, None, None)

    def test_thinking_true_uses_medium_effort(self, openai_responses_reasoning_profile: ModelProfile):
        """thinking=True should use 'medium' as the default effort."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        settings: OpenAIResponsesModelSettings = {'thinking': True}
        effort, summary = model._apply_unified_thinking(settings, None, None)

        assert effort == 'medium'
        assert summary is None

    def test_thinking_false_raises_for_always_on(self, openai_responses_reasoning_profile: ModelProfile):
        """thinking=False on always-enabled reasoning model should raise UserError."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        settings: OpenAIResponsesModelSettings = {'thinking': False}

        with pytest.raises(UserError, match='has reasoning always enabled'):
            model._apply_unified_thinking(settings, None, None)

    def test_thinking_config_enabled_false_raises_for_always_on(self, openai_responses_reasoning_profile: ModelProfile):
        """ThinkingConfig with enabled=False on always-enabled reasoning model should raise."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        settings: OpenAIResponsesModelSettings = {'thinking': {'enabled': False}}

        with pytest.raises(UserError, match='has reasoning always enabled'):
            model._apply_unified_thinking(settings, None, None)

    def test_thinking_false_disables_on_optional_model(self, openai_responses_optional_reasoning_profile: ModelProfile):
        """thinking=False should return defaults on models where reasoning is optional."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o1-preview'
        model._profile = openai_responses_optional_reasoning_profile

        settings: OpenAIResponsesModelSettings = {'thinking': False}
        effort, summary = model._apply_unified_thinking(settings, None, None)

        # Returns the defaults (None values) indicating no thinking config applied
        assert effort is None
        assert summary is None

    def test_thinking_config_enabled_false_disables_on_optional_model(
        self, openai_responses_optional_reasoning_profile: ModelProfile
    ):
        """ThinkingConfig with enabled=False should return defaults on optional models."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o1-preview'
        model._profile = openai_responses_optional_reasoning_profile

        settings: OpenAIResponsesModelSettings = {'thinking': {'enabled': False}}
        effort, summary = model._apply_unified_thinking(settings, None, None)

        assert effort is None
        assert summary is None

    def test_preserves_existing_reasoning_effort(self, openai_responses_reasoning_profile: ModelProfile):
        """Existing reasoning_effort value should be preserved."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        # Test with explicit effort that should NOT override existing value
        settings: OpenAIResponsesModelSettings = {'thinking': {'effort': 'low'}}
        effort, _ = model._apply_unified_thinking(settings, 'high', None)

        # Existing 'high' should be preserved
        assert effort == 'high'

    def test_preserves_existing_reasoning_summary(self, openai_responses_reasoning_profile: ModelProfile):
        """Existing reasoning_summary value should be preserved."""
        from pydantic_ai.models.openai import OpenAIResponsesModel

        model = OpenAIResponsesModel.__new__(OpenAIResponsesModel)
        model._model_name = 'o3'
        model._profile = openai_responses_reasoning_profile

        # Test with explicit summary that should NOT override existing value
        settings: OpenAIResponsesModelSettings = {'thinking': {'summary': 'detailed'}}
        _, summary = model._apply_unified_thinking(settings, None, 'concise')

        # Existing 'concise' should be preserved
        assert summary == 'concise'


# ============================================================================
# Profile capability tests
# ============================================================================


class TestProfileThinkingCapabilities:
    """Tests for thinking capabilities in model profiles."""

    def test_anthropic_profile_thinking_support(self):
        """Anthropic profile should set thinking capabilities for supported models."""
        from pydantic_ai.profiles.anthropic import anthropic_model_profile

        # Claude 3.7 models support thinking
        profile = anthropic_model_profile('claude-3-7-sonnet')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.default_thinking_budget == 4096

        # Claude 4 models support thinking
        profile = anthropic_model_profile('claude-sonnet-4-5')
        assert profile is not None
        assert profile.supports_thinking is True

        # Older models don't support thinking
        profile = anthropic_model_profile('claude-3-opus-20240229')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_google_profile_thinking_support(self):
        """Google profile should set thinking capabilities for Gemini 2.5+ models."""
        from pydantic_ai.profiles.google import google_model_profile

        # Gemini 2.5 models support thinking (via thinking_budget)
        profile = google_model_profile('gemini-2.5-flash')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.supports_thinking_level is False  # Uses budget, not level
        assert profile.effort_to_budget_map is not None

        # Gemini 3 models support thinking (via thinking_level)
        profile = google_model_profile('gemini-3-flash')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.supports_thinking_level is True  # Uses level, not budget
        assert profile.effort_to_budget_map is None  # No budget map for Gemini 3

        # Older models don't support thinking
        profile = google_model_profile('gemini-2.0-flash')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_openai_profile_thinking_support(self):
        """OpenAI profile should set thinking capabilities for reasoning models."""
        from pydantic_ai.profiles.openai import openai_model_profile

        # o-series models support thinking (always on)
        profile = openai_model_profile('o3')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True

        # GPT-5 reasoning models support thinking
        profile = openai_model_profile('gpt-5')
        assert profile is not None
        assert profile.supports_thinking is True

        # Non-reasoning models don't support thinking
        profile = openai_model_profile('gpt-4o')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_deepseek_profile_thinking_support(self):
        """DeepSeek profile should set thinking always enabled for R1 models."""
        from pydantic_ai.profiles.deepseek import deepseek_model_profile

        # R1 models have thinking always enabled
        profile = deepseek_model_profile('deepseek-r1')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True

        # Non-R1 models don't support thinking
        profile = deepseek_model_profile('deepseek-chat')
        assert profile is not None
        assert profile.supports_thinking is False


# ============================================================================
# Effort-to-budget mapping tests
# ============================================================================


class TestEffortToBudgetMapping:
    """Tests for effort level to token budget mapping."""

    def test_anthropic_effort_mapping(self):
        """Anthropic effort mapping should match design spec."""
        assert ANTHROPIC_EFFORT_TO_BUDGET == {
            'low': 1024,
            'medium': 4096,
            'high': 16384,
        }

    def test_google_effort_mapping(self):
        """Google effort mapping should match design spec (higher values)."""
        assert GOOGLE_EFFORT_TO_BUDGET == {
            'low': 1024,
            'medium': 8192,
            'high': 32768,
        }


# ============================================================================
# OpenRouter unified thinking tests
# ============================================================================


class TestOpenRouterUnifiedThinking:
    """Tests for unified thinking settings on OpenRouter models."""

    def test_thinking_true_enables_reasoning(self, thinking_profile: ModelProfile):
        """thinking=True should enable reasoning."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'anthropic/claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: OpenRouterModelSettings = {'thinking': True}
        result = model._resolve_reasoning_config(settings)

        assert result == {'enabled': True}

    def test_thinking_false_disables_reasoning(self, thinking_profile: ModelProfile):
        """thinking=False should disable reasoning."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'anthropic/claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: OpenRouterModelSettings = {'thinking': False}
        result = model._resolve_reasoning_config(settings)

        assert result == {'enabled': False}

    def test_thinking_config_with_effort(self, thinking_profile: ModelProfile):
        """ThinkingConfig with effort should set effort level."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'openai/o3'
        model._profile = thinking_profile

        settings: OpenRouterModelSettings = {'thinking': {'effort': 'high'}}
        result = model._resolve_reasoning_config(settings)

        assert result == {'effort': 'high'}

    def test_thinking_config_with_budget(self, thinking_profile: ModelProfile):
        """ThinkingConfig with budget_tokens should set max_tokens."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'anthropic/claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: OpenRouterModelSettings = {'thinking': {'budget_tokens': 4096}}
        result = model._resolve_reasoning_config(settings)

        assert result == {'max_tokens': 4096}

    def test_thinking_config_include_in_response_false(self, thinking_profile: ModelProfile):
        """ThinkingConfig with include_in_response=False should set exclude=True."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'anthropic/claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: OpenRouterModelSettings = {'thinking': {'include_in_response': False}}
        result = model._resolve_reasoning_config(settings)

        assert result == {'exclude': True}

    def test_thinking_none_returns_none(self, thinking_profile: ModelProfile):
        """No thinking setting should return None."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'anthropic/claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: OpenRouterModelSettings = {}
        result = model._resolve_reasoning_config(settings)

        assert result is None

    def test_unsupported_model_raises_error(self, non_thinking_profile: ModelProfile):
        """Using thinking on a model that doesn't support it should raise UserError."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'openai/gpt-4o'
        model._profile = non_thinking_profile

        settings: OpenRouterModelSettings = {'thinking': True}

        with pytest.raises(UserError, match='does not support reasoning'):
            model._resolve_reasoning_config(settings)

    def test_thinking_false_raises_for_always_on(self, always_on_thinking_profile: ModelProfile):
        """thinking=False should raise UserError for always-on thinking models."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'deepseek/deepseek-r1'
        model._profile = always_on_thinking_profile

        settings: OpenRouterModelSettings = {'thinking': False}

        with pytest.raises(UserError, match='has reasoning always enabled and cannot be disabled'):
            model._resolve_reasoning_config(settings)

    def test_thinking_config_enabled_false_for_optional(self, thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=False should return enabled=False for optional models."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'anthropic/claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: OpenRouterModelSettings = {'thinking': {'enabled': False}}
        result = model._resolve_reasoning_config(settings)

        assert result == {'enabled': False}

    def test_thinking_config_enabled_false_raises_for_always_on(self, always_on_thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=False should raise UserError for always-on models."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'deepseek/deepseek-r1'
        model._profile = always_on_thinking_profile

        settings: OpenRouterModelSettings = {'thinking': {'enabled': False}}

        with pytest.raises(UserError, match='has reasoning always enabled and cannot be disabled'):
            model._resolve_reasoning_config(settings)

    def test_thinking_config_enabled_true_defaults_to_enabled(self, thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=True but no other settings should return enabled=True."""
        model = OpenRouterModel.__new__(OpenRouterModel)
        model._model_name = 'anthropic/claude-sonnet-4-5'
        model._profile = thinking_profile

        settings: OpenRouterModelSettings = {'thinking': {'enabled': True}}
        result = model._resolve_reasoning_config(settings)

        assert result == {'enabled': True}


# ============================================================================
# Groq unified thinking tests
# ============================================================================


class TestGroqUnifiedThinking:
    """Tests for unified thinking settings on Groq models."""

    def test_thinking_true_uses_parsed_format(self, thinking_profile: ModelProfile):
        """thinking=True should use 'parsed' format for structured output."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        settings: GroqModelSettings = {'thinking': True}
        result = model._resolve_reasoning_format(settings)

        assert result == 'parsed'

    def test_thinking_false_returns_none(self, thinking_profile: ModelProfile):
        """thinking=False should return None (no reasoning format)."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        settings: GroqModelSettings = {'thinking': False}
        result = model._resolve_reasoning_format(settings)

        assert result is None

    def test_thinking_config_include_in_response_false(self, thinking_profile: ModelProfile):
        """ThinkingConfig with include_in_response=False should use 'hidden' format."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        settings: GroqModelSettings = {'thinking': {'include_in_response': False}}
        result = model._resolve_reasoning_format(settings)

        assert result == 'hidden'

    def test_thinking_config_include_in_response_true(self, thinking_profile: ModelProfile):
        """ThinkingConfig with include_in_response=True should use 'parsed' format."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        settings: GroqModelSettings = {'thinking': {'include_in_response': True}}
        result = model._resolve_reasoning_format(settings)

        assert result == 'parsed'

    def test_thinking_none_returns_none(self, thinking_profile: ModelProfile):
        """No thinking setting should return None."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        settings: GroqModelSettings = {}
        result = model._resolve_reasoning_format(settings)

        assert result is None

    def test_unsupported_model_raises_error(self, non_thinking_profile: ModelProfile):
        """Using thinking on a model that doesn't support it should raise UserError."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'llama-3.1-8b-instant'
        model._profile = non_thinking_profile

        settings: GroqModelSettings = {'thinking': True}

        with pytest.raises(UserError, match='does not support reasoning'):
            model._resolve_reasoning_format(settings)

    def test_thinking_false_raises_for_always_on(self, always_on_thinking_profile: ModelProfile):
        """thinking=False should raise UserError for always-on thinking models."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1'
        model._profile = always_on_thinking_profile

        settings: GroqModelSettings = {'thinking': False}

        with pytest.raises(UserError, match='has reasoning always enabled and cannot be disabled'):
            model._resolve_reasoning_format(settings)

    def test_thinking_config_enabled_false_raises_for_always_on(self, always_on_thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=False should raise UserError for always-on models."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1'
        model._profile = always_on_thinking_profile

        settings: GroqModelSettings = {'thinking': {'enabled': False}}

        with pytest.raises(UserError, match='has reasoning always enabled and cannot be disabled'):
            model._resolve_reasoning_format(settings)

    def test_thinking_config_enabled_false_returns_none_for_optional(self, thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=False should return None for non-always-on models."""
        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile

        settings: GroqModelSettings = {'thinking': {'enabled': False}}
        result = model._resolve_reasoning_format(settings)

        # Should return None (don't set reasoning format, effectively disabling)
        assert result is None

    def test_thinking_config_with_budget_tokens_warns(self, thinking_profile: ModelProfile):
        """ThinkingConfig with budget_tokens should warn about ignored setting."""
        import warnings
        from unittest.mock import Mock

        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile
        mock_provider = Mock()
        mock_provider.name = 'groq'
        model._provider = mock_provider

        settings: GroqModelSettings = {'thinking': {'budget_tokens': 4096}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = model._resolve_reasoning_format(settings)

            assert len(w) == 1
            assert 'budget_tokens' in str(w[0].message)

        assert result == 'parsed'

    def test_thinking_config_with_effort_warns(self, thinking_profile: ModelProfile):
        """ThinkingConfig with effort should warn about ignored setting."""
        import warnings
        from unittest.mock import Mock

        model = GroqModel.__new__(GroqModel)
        model._model_name = 'deepseek-r1-distill-llama-70b'
        model._profile = thinking_profile
        mock_provider = Mock()
        mock_provider.name = 'groq'
        model._provider = mock_provider

        settings: GroqModelSettings = {'thinking': {'effort': 'high'}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = model._resolve_reasoning_format(settings)

            assert len(w) == 1
            assert 'effort' in str(w[0].message)

        assert result == 'parsed'


# ============================================================================
# Cerebras unified thinking tests
# ============================================================================


class TestCerebrasUnifiedThinking:
    """Tests for unified thinking settings on Cerebras models."""

    def test_thinking_true_returns_none(self, thinking_profile: ModelProfile):
        """thinking=True should return None (enable default reasoning behavior)."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = thinking_profile

        settings: CerebrasModelSettings = {'thinking': True}
        result = model._resolve_reasoning_config(settings)

        assert result is None

    def test_thinking_false_disables_reasoning(self, thinking_profile: ModelProfile):
        """thinking=False should return True (disable reasoning)."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = thinking_profile

        settings: CerebrasModelSettings = {'thinking': False}
        result = model._resolve_reasoning_config(settings)

        assert result is True

    def test_thinking_config_enabled_false(self, thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=False should return True (disable reasoning)."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = thinking_profile

        settings: CerebrasModelSettings = {'thinking': {'enabled': False}}
        result = model._resolve_reasoning_config(settings)

        assert result is True

    def test_thinking_config_with_budget_returns_none_and_warns(self, thinking_profile: ModelProfile):
        """ThinkingConfig with budget_tokens should return None and emit warning."""
        import warnings
        from unittest.mock import Mock

        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = thinking_profile
        mock_provider = Mock()
        mock_provider.name = 'cerebras'  # Set name as attribute, not Mock(name=...)
        model._provider = mock_provider

        settings: CerebrasModelSettings = {'thinking': {'budget_tokens': 4096}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = model._resolve_reasoning_config(settings)

            # Should emit a warning about ignored settings
            assert len(w) == 1
            assert 'budget_tokens' in str(w[0].message)

        # Cerebras doesn't support budget_tokens, but we enable reasoning
        assert result is None

    def test_thinking_none_returns_none(self, thinking_profile: ModelProfile):
        """No thinking setting should return None."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = thinking_profile

        settings: CerebrasModelSettings = {}
        result = model._resolve_reasoning_config(settings)

        assert result is None

    def test_unsupported_model_raises_error(self, non_thinking_profile: ModelProfile):
        """Using thinking on a model that doesn't support it should raise UserError."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'llama-3.3-70b'
        model._profile = non_thinking_profile

        settings: CerebrasModelSettings = {'thinking': True}

        with pytest.raises(UserError, match='does not support reasoning'):
            model._resolve_reasoning_config(settings)

    def test_thinking_false_raises_for_always_on(self, always_on_thinking_profile: ModelProfile):
        """thinking=False should raise UserError for always-on reasoning models."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = always_on_thinking_profile

        settings: CerebrasModelSettings = {'thinking': False}

        with pytest.raises(UserError, match='has reasoning always enabled'):
            model._resolve_reasoning_config(settings)

    def test_thinking_config_enabled_false_raises_for_always_on(self, always_on_thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=False should raise UserError for always-on models."""
        model = CerebrasModel.__new__(CerebrasModel)
        model._model_name = 'zai-glm-4.6'
        model._profile = always_on_thinking_profile

        settings: CerebrasModelSettings = {'thinking': {'enabled': False}}

        with pytest.raises(UserError, match='has reasoning always enabled'):
            model._resolve_reasoning_config(settings)


# ============================================================================
# Bedrock unified thinking tests
# ============================================================================


class TestBedrockUnifiedThinking:
    """Tests for unified thinking settings on Bedrock models."""

    def test_thinking_true_uses_default_budget(self, thinking_profile: ModelProfile):
        """thinking=True should enable thinking with the default budget."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking': True}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 4096}

    def test_thinking_false_disables_thinking(self, thinking_profile: ModelProfile):
        """thinking=False should disable thinking."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking': False}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'disabled'}

    def test_thinking_config_with_budget(self, thinking_profile: ModelProfile):
        """ThinkingConfig with explicit budget_tokens should use that budget."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking': {'budget_tokens': 2048}}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 2048}

    def test_thinking_config_with_effort_low(self, thinking_profile: ModelProfile):
        """ThinkingConfig with effort='low' should map to 1024 tokens."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking': {'effort': 'low'}}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 1024}

    def test_thinking_config_with_effort_high(self, thinking_profile: ModelProfile):
        """ThinkingConfig with effort='high' should map to 16384 tokens."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking': {'effort': 'high'}}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'enabled', 'budget_tokens': 16384}

    def test_thinking_config_enabled_false(self, thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=False should disable thinking."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking': {'enabled': False}}
        result = model._resolve_thinking_config(settings)

        assert result == {'type': 'disabled'}

    def test_thinking_config_enabled_true_uses_default_budget(self, thinking_profile: ModelProfile):
        """ThinkingConfig with enabled=True but no effort/budget should use default budget."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {'thinking': {'enabled': True}}
        result = model._resolve_thinking_config(settings)

        # Should use the default budget from the profile (4096)
        assert result == {'type': 'enabled', 'budget_tokens': 4096}

    def test_thinking_none_returns_none(self, thinking_profile: ModelProfile):
        """No thinking setting should return None."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-sonnet-4-5-20250514-v1:0'
        model._profile = thinking_profile

        settings: BedrockModelSettings = {}
        result = model._resolve_thinking_config(settings)

        assert result is None

    def test_unsupported_model_raises_error(self, non_thinking_profile: ModelProfile):
        """Using thinking on a model that doesn't support it should raise UserError."""
        model = BedrockConverseModel.__new__(BedrockConverseModel)
        model._model_name = 'us.anthropic.claude-3-opus-20240229-v1:0'
        model._profile = non_thinking_profile

        settings: BedrockModelSettings = {'thinking': True}

        with pytest.raises(UserError, match='does not support thinking/reasoning'):
            model._resolve_thinking_config(settings)


# ============================================================================
# Additional profile capability tests
# ============================================================================


class TestAdditionalProfileCapabilities:
    """Tests for thinking capabilities in additional model profiles."""

    def test_groq_profile_thinking_support(self):
        """Groq profile should set thinking capabilities for reasoning models."""
        from pydantic_ai.profiles.groq import groq_model_profile

        # DeepSeek R1 models support thinking
        profile = groq_model_profile('deepseek-r1-distill-llama-70b')
        assert profile is not None
        assert profile.supports_thinking is True
        assert profile.thinking_always_enabled is True

        # QwQ models support thinking
        profile = groq_model_profile('qwen-qwq-32b')
        assert profile is not None
        assert profile.supports_thinking is True

        # Regular models don't support thinking
        profile = groq_model_profile('llama-3.1-8b-instant')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_zai_profile_thinking_support(self):
        """ZAI profile should set thinking capabilities for GLM models."""
        from pydantic_ai.profiles.zai import zai_model_profile

        # ZAI GLM models support reasoning
        profile = zai_model_profile('zai-glm-4.6')
        assert profile is not None
        assert profile.supports_thinking is True

        # Non-GLM models don't support thinking
        profile = zai_model_profile('zai-other')
        assert profile is not None
        assert profile.supports_thinking is False

    def test_harmony_profile_thinking_support(self):
        """Harmony profile should set thinking capabilities for GPT-OSS models."""
        from pydantic_ai.profiles.harmony import harmony_model_profile

        # GPT-OSS models support reasoning
        profile = harmony_model_profile('gpt-oss-120b')
        assert profile is not None
        assert profile.supports_thinking is True

        # Non-GPT-OSS models don't support thinking
        # Note: Use a model name that doesn't start with 'o' to avoid matching OpenAI o-series pattern
        profile = harmony_model_profile('llama-3-70b')
        assert profile is not None
        assert profile.supports_thinking is False
