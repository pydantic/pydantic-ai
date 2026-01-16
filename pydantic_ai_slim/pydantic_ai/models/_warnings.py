"""Warning utilities for model providers.

This module provides centralized warning helpers for emitting consistent
warnings when unified settings don't translate directly to provider-specific
configurations.
"""

from __future__ import annotations

import warnings
from typing import Literal

# Common setting names for type hints
ThinkingSettingName = Literal['budget_tokens', 'effort', 'include_in_response', 'summary', 'enabled']


def warn_setting_ignored(
    *,
    setting_name: ThinkingSettingName | str,
    provider_name: str,
    model_name: str | None = None,
    reason: str | None = None,
    alternative: str | None = None,
    stacklevel: int = 4,
) -> None:
    """Emit a warning when a unified setting is silently ignored by a provider.

    Use this when a provider does not support a particular unified setting
    at all, and the setting will have no effect.

    Args:
        setting_name: Name of the unified setting being ignored (e.g., 'budget_tokens').
        provider_name: Name of the provider ignoring the setting (e.g., 'OpenAI', 'Groq').
        model_name: Optional specific model name for more targeted messaging.
        reason: Optional explanation of why the setting is ignored.
        alternative: Optional suggestion for what to use instead.
        stacklevel: Stack level for warnings.warn(). Default 4 accounts for:
            1. warn_setting_ignored
            2. _resolve_thinking_config / similar
            3. prepare_request / similar
            4. User code (agent.run(), etc.)

    Example:
        >>> warn_setting_ignored(
        ...     setting_name='budget_tokens',
        ...     provider_name='OpenAI',
        ...     reason='OpenAI reasoning models do not support token budgets',
        ...     alternative="Use effort='low', 'medium', or 'high' instead",
        ... )
        # UserWarning: OpenAI ignores 'budget_tokens' setting.
        # OpenAI reasoning models do not support token budgets.
        # Use effort='low', 'medium', or 'high' instead.
    """
    if model_name:
        parts = [f"{provider_name} model '{model_name}' ignores '{setting_name}' setting"]
    else:
        parts = [f"{provider_name} ignores '{setting_name}' setting"]

    if reason:
        parts.append(reason)

    if alternative:
        parts.append(alternative)

    message = '. '.join(parts) + '.'
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def warn_setting_mapped(
    *,
    setting_name: ThinkingSettingName | str,
    setting_value: str | int | bool,
    provider_name: str,
    mapped_to: str,
    model_name: str | None = None,
    reason: str | None = None,
    stacklevel: int = 4,
) -> None:
    """Emit a warning when a unified setting value is mapped to a different provider value.

    Use this when a provider supports the setting concept but maps the specific
    value to something different (e.g., 'medium' effort mapped to 'high').

    Args:
        setting_name: Name of the unified setting (e.g., 'effort').
        setting_value: The original value provided by the user.
        provider_name: Name of the provider doing the mapping.
        mapped_to: What the value was mapped to.
        model_name: Optional specific model name.
        reason: Optional explanation of why the mapping occurs.
        stacklevel: Stack level for warnings.warn().

    Example:
        >>> warn_setting_mapped(
        ...     setting_name='effort',
        ...     setting_value='medium',
        ...     provider_name='Google',
        ...     model_name='gemini-3-pro',
        ...     mapped_to='high',
        ...     reason="Gemini 3 Pro only supports 'low' and 'high' effort levels",
        ... )
        # UserWarning: Google model 'gemini-3-pro' mapped effort='medium' to 'high'.
        # Gemini 3 Pro only supports 'low' and 'high' effort levels.
        # Use effort='low' or effort='high' to avoid this warning.
    """
    if model_name:
        base = f"{provider_name} model '{model_name}' mapped {setting_name}={setting_value!r} to {mapped_to!r}"
    else:
        base = f'{provider_name} mapped {setting_name}={setting_value!r} to {mapped_to!r}'

    parts = [base]

    if reason:
        parts.append(reason)

    # Auto-generate avoidance hint for effort level mappings
    if setting_name == 'effort' and isinstance(setting_value, str):
        parts.append("Use effort='low' or effort='high' to avoid this warning")

    message = '. '.join(parts) + '.'
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def warn_settings_ignored_batch(
    *,
    setting_names: list[ThinkingSettingName | str],
    provider_name: str,
    model_name: str | None = None,
    reason: str | None = None,
    stacklevel: int = 4,
) -> None:
    """Emit a single warning when multiple unified settings are ignored.

    Use this to avoid warning spam when a provider ignores multiple settings.

    Args:
        setting_names: List of setting names being ignored.
        provider_name: Name of the provider.
        model_name: Optional specific model name.
        reason: Optional shared reason for all ignored settings.
        stacklevel: Stack level for warnings.warn().

    Example:
        >>> warn_settings_ignored_batch(
        ...     setting_names=['budget_tokens', 'effort', 'include_in_response'],
        ...     provider_name='Cerebras',
        ...     reason='Cerebras reasoning models have fixed behavior',
        ... )
        # UserWarning: Cerebras ignores these unified thinking settings:
        # 'budget_tokens', 'effort', 'include_in_response'.
        # Cerebras reasoning models have fixed behavior.
    """
    if not setting_names:
        return

    formatted_names = ', '.join(f"'{name}'" for name in setting_names)

    if model_name:
        base = f"{provider_name} model '{model_name}' ignores these settings: {formatted_names}"
    else:
        base = f'{provider_name} ignores these unified thinking settings: {formatted_names}'

    parts = [base]

    if reason:
        parts.append(reason)

    message = '. '.join(parts) + '.'
    warnings.warn(message, UserWarning, stacklevel=stacklevel)
