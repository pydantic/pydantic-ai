"""Tests for model warning utilities."""

from __future__ import annotations

import warnings

import pytest

from pydantic_ai.models._warnings import (
    warn_setting_ignored,
    warn_setting_mapped,
    warn_settings_ignored_batch,
)


class TestWarnSettingIgnored:
    """Tests for warn_setting_ignored helper."""

    def test_basic_warning(self):
        """Basic warning without model name or extra details."""
        with pytest.warns(UserWarning, match="OpenAI ignores 'budget_tokens' setting"):
            warn_setting_ignored(
                setting_name='budget_tokens',
                provider_name='OpenAI',
                stacklevel=2,
            )

    def test_with_model_name(self):
        """Warning includes model name when provided."""
        with pytest.warns(UserWarning, match="OpenAI model 'o3' ignores 'budget_tokens' setting"):
            warn_setting_ignored(
                setting_name='budget_tokens',
                provider_name='OpenAI',
                model_name='o3',
                stacklevel=2,
            )

    def test_with_reason(self):
        """Warning includes reason when provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_setting_ignored(
                setting_name='budget_tokens',
                provider_name='OpenAI',
                reason='OpenAI uses effort levels instead',
                stacklevel=2,
            )

        assert len(w) == 1
        message = str(w[0].message)
        assert 'OpenAI uses effort levels instead' in message

    def test_with_alternative(self):
        """Warning includes alternative suggestion when provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_setting_ignored(
                setting_name='budget_tokens',
                provider_name='OpenAI',
                alternative="Use effort='high' instead",
                stacklevel=2,
            )

        assert len(w) == 1
        message = str(w[0].message)
        assert "Use effort='high' instead" in message

    def test_full_message_format(self):
        """Complete message with all optional parts."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_setting_ignored(
                setting_name='budget_tokens',
                provider_name='OpenAI',
                model_name='o3',
                reason='No token budget support',
                alternative='Use effort instead',
                stacklevel=2,
            )

        assert len(w) == 1
        message = str(w[0].message)
        assert "OpenAI model 'o3' ignores 'budget_tokens' setting" in message
        assert 'No token budget support' in message
        assert 'Use effort instead' in message
        assert message.endswith('.')


class TestWarnSettingMapped:
    """Tests for warn_setting_mapped helper."""

    def test_basic_mapping_warning(self):
        """Basic mapping warning without model name."""
        with pytest.warns(UserWarning, match="Google mapped effort='medium' to 'high'"):
            warn_setting_mapped(
                setting_name='effort',
                setting_value='medium',
                provider_name='Google',
                mapped_to='high',
                stacklevel=2,
            )

    def test_with_model_name(self):
        """Mapping warning includes model name when provided."""
        with pytest.warns(
            UserWarning, match="Google model 'gemini-3-pro' mapped effort='medium' to 'high'"
        ):
            warn_setting_mapped(
                setting_name='effort',
                setting_value='medium',
                provider_name='Google',
                model_name='gemini-3-pro',
                mapped_to='high',
                stacklevel=2,
            )

    def test_with_reason(self):
        """Mapping warning includes reason when provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_setting_mapped(
                setting_name='effort',
                setting_value='medium',
                provider_name='Google',
                mapped_to='high',
                reason="Gemini 3 Pro only supports 'low' and 'high'",
                stacklevel=2,
            )

        assert len(w) == 1
        message = str(w[0].message)
        assert "Gemini 3 Pro only supports 'low' and 'high'" in message

    def test_effort_avoidance_hint(self):
        """Effort mappings include avoidance hint automatically."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_setting_mapped(
                setting_name='effort',
                setting_value='medium',
                provider_name='Google',
                mapped_to='high',
                stacklevel=2,
            )

        assert len(w) == 1
        message = str(w[0].message)
        assert 'to avoid this warning' in message
        assert "effort='low'" in message or "effort='high'" in message

    def test_non_effort_no_avoidance_hint(self):
        """Non-effort mappings don't include avoidance hint."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_setting_mapped(
                setting_name='temperature',
                setting_value='0.5',
                provider_name='Test',
                mapped_to='0.7',
                stacklevel=2,
            )

        assert len(w) == 1
        message = str(w[0].message)
        assert 'to avoid this warning' not in message


class TestWarnSettingsIgnoredBatch:
    """Tests for warn_settings_ignored_batch helper."""

    def test_single_setting(self):
        """Batch warning with a single setting."""
        with pytest.warns(UserWarning, match="Groq ignores these unified thinking settings: 'budget_tokens'"):
            warn_settings_ignored_batch(
                setting_names=['budget_tokens'],
                provider_name='Groq',
                stacklevel=2,
            )

    def test_multiple_settings(self):
        """Batch warning with multiple settings."""
        with pytest.warns(UserWarning, match="'budget_tokens', 'effort'"):
            warn_settings_ignored_batch(
                setting_names=['budget_tokens', 'effort'],
                provider_name='Groq',
                stacklevel=2,
            )

    def test_with_model_name(self):
        """Batch warning includes model name when provided."""
        with pytest.warns(UserWarning, match="Cerebras model 'zai-glm-4.6' ignores these settings"):
            warn_settings_ignored_batch(
                setting_names=['budget_tokens', 'effort'],
                provider_name='Cerebras',
                model_name='zai-glm-4.6',
                stacklevel=2,
            )

    def test_with_reason(self):
        """Batch warning includes reason when provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_settings_ignored_batch(
                setting_names=['budget_tokens', 'effort'],
                provider_name='Cerebras',
                reason='Cerebras only supports enable/disable',
                stacklevel=2,
            )

        assert len(w) == 1
        message = str(w[0].message)
        assert 'Cerebras only supports enable/disable' in message

    def test_empty_list_no_warning(self):
        """Empty settings list does not emit a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_settings_ignored_batch(
                setting_names=[],
                provider_name='Cerebras',
                stacklevel=2,
            )

        assert len(w) == 0

    def test_many_settings(self):
        """Batch warning with many settings formats correctly."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_settings_ignored_batch(
                setting_names=['budget_tokens', 'effort', 'include_in_response', 'summary'],
                provider_name='Cerebras',
                stacklevel=2,
            )

        assert len(w) == 1
        message = str(w[0].message)
        assert "'budget_tokens'" in message
        assert "'effort'" in message
        assert "'include_in_response'" in message
        assert "'summary'" in message
