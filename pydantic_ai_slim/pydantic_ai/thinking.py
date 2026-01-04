"""Centralized thinking configuration resolution.

This module provides the single source of truth for resolving unified thinking
settings into provider-specific configurations. It separates:

1. Resolution logic (common to all providers): validation, bool/dict handling, effort mapping
2. Formatting logic (provider-specific): converting resolved config to API format
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .exceptions import UserError
from .settings import ThinkingConfig

if TYPE_CHECKING:
    from .profiles import ModelProfile


@dataclass
class ResolvedThinkingConfig:
    """Intermediate representation of resolved thinking configuration.

    This is the canonical form after validation and normalization,
    before provider-specific formatting.
    """

    enabled: bool
    """Whether thinking is enabled."""

    budget_tokens: int | None = None
    """Token budget for thinking (if applicable to provider)."""

    effort: Literal['low', 'medium', 'high'] | None = None
    """Effort level (if applicable to provider)."""

    include_in_response: bool = True
    """Whether to include thinking content in response."""

    summary: Literal['none', 'concise', 'detailed', 'auto'] | bool | None = None
    """Summary mode (OpenAI-specific)."""


def resolve_thinking_config(
    thinking: bool | ThinkingConfig | None,
    profile: ModelProfile,
    model_name: str,
) -> ResolvedThinkingConfig | None:
    """Resolve unified thinking settings using profile capabilities.

    This is the single source of truth for thinking configuration resolution.
    All validation, bool/dict handling, and effort-to-budget mapping happens here.
    Provider-specific formatting is done by separate formatter functions.

    Args:
        thinking: The unified thinking setting from ModelSettings.
        profile: The model profile with capability declarations.
        model_name: For error messages.

    Returns:
        ResolvedThinkingConfig if thinking should be configured, None otherwise.

    Raises:
        UserError: If thinking is requested but model doesn't support it,
                   or if thinking cannot be disabled for always-enabled models.
    """
    if thinking is None:
        return None

    # Validate model supports thinking
    if not profile.supports_thinking:
        raise UserError(
            f'Model {model_name!r} does not support thinking/reasoning. '
            f'Remove the `thinking` setting or use a model that supports thinking.'
        )

    # Handle thinking=False
    if thinking is False:
        if profile.thinking_always_enabled:
            raise UserError(
                f'Model {model_name!r} has reasoning always enabled and cannot be disabled. '
                f'Remove the `thinking=False` setting.'
            )
        return ResolvedThinkingConfig(enabled=False)

    # Handle thinking=True
    if thinking is True:
        # Don't set budget_tokens here - let formatters apply provider-specific defaults
        return ResolvedThinkingConfig(enabled=True)

    # Handle ThinkingConfig dict
    config: ThinkingConfig = thinking

    # Check if explicitly disabled
    if config.get('enabled') is False:
        if profile.thinking_always_enabled:
            raise UserError(
                f'Model {model_name!r} has reasoning always enabled and cannot be disabled. '
                f"Remove the `thinking={{'enabled': False}}` setting."
            )
        return ResolvedThinkingConfig(enabled=False)

    # Only set budget_tokens if user explicitly provided it
    # Formatters will derive from effort or apply defaults as needed per provider
    budget_tokens = config.get('budget_tokens')
    effort = config.get('effort')

    return ResolvedThinkingConfig(
        enabled=True,
        budget_tokens=budget_tokens,
        effort=effort,
        include_in_response=config.get('include_in_response', True),
        summary=config.get('summary'),
    )


# =============================================================================
# Provider-specific formatters
# =============================================================================


def format_anthropic_thinking(
    config: ResolvedThinkingConfig | None,
    profile: ModelProfile,
) -> dict[str, Any] | None:
    """Format for Anthropic API (BetaThinkingConfigParam).

    Returns:
        {'type': 'disabled'} or {'type': 'enabled', 'budget_tokens': int}
    """
    if config is None:
        return None
    if not config.enabled:
        return {'type': 'disabled'}

    # Derive budget from effort if not explicitly set
    budget_tokens = config.budget_tokens
    if budget_tokens is None and config.effort and profile.effort_to_budget_map:
        budget_tokens = profile.effort_to_budget_map.get(config.effort)
    if budget_tokens is None:
        budget_tokens = profile.default_thinking_budget or 4096

    return {'type': 'enabled', 'budget_tokens': budget_tokens}


def format_bedrock_thinking(
    config: ResolvedThinkingConfig | None,
    profile: ModelProfile,
) -> dict[str, Any] | None:
    """Format for Bedrock API (same as Anthropic, via additionalModelRequestFields)."""
    return format_anthropic_thinking(config, profile)


def format_google_thinking(
    config: ResolvedThinkingConfig | None,
    profile: ModelProfile,
    model_name: str,
) -> dict[str, Any] | None:
    """Format for Google Gemini API.

    Handles both Gemini 2.5 (thinking_budget) and Gemini 3+ (thinking_level).

    Returns:
        For Gemini 3+: {'thinking_level': ThinkingLevel, 'include_thoughts': bool}
        For Gemini 2.5: {'thinking_budget': int, 'include_thoughts': bool}
    """
    if config is None:
        return None

    # Import here to avoid requiring google-genai when not used
    try:
        from google.genai.types import ThinkingLevel
    except ImportError:  # pragma: no cover
        # Fall back to string values if google-genai not installed
        ThinkingLevel = None  # type: ignore

    uses_thinking_level = profile.supports_thinking_level

    if not config.enabled:
        if uses_thinking_level and ThinkingLevel is not None:
            return {'thinking_level': ThinkingLevel.LOW, 'include_thoughts': False}
        else:
            return {'thinking_budget': 0}

    result: dict[str, Any] = {}

    if uses_thinking_level:
        # Gemini 3+: Map effort to thinking_level
        if ThinkingLevel is not None:
            effort_to_level = {
                'low': ThinkingLevel.LOW,
                'medium': ThinkingLevel.MEDIUM,
                'high': ThinkingLevel.HIGH,
            }

            if config.effort:
                # Check for Pro model medium→high mapping
                is_pro = 'pro' in model_name.lower()
                if config.effort == 'medium' and is_pro:
                    from .models._warnings import warn_setting_mapped

                    warn_setting_mapped(
                        setting_name='effort',
                        setting_value='medium',
                        provider_name='Google',
                        model_name=model_name,
                        mapped_to='high',
                        reason="Gemini 3 Pro only supports 'low' and 'high' effort levels. Thinking will use 'high' effort",
                    )
                    result['thinking_level'] = ThinkingLevel.HIGH
                else:
                    result['thinking_level'] = effort_to_level.get(config.effort, ThinkingLevel.HIGH)
            else:
                result['thinking_level'] = ThinkingLevel.HIGH
        else:  # pragma: no cover
            # Fallback if google-genai not installed
            result['thinking_level'] = config.effort or 'high'
    else:
        # Gemini 2.5: Use thinking_budget
        # Derive budget from effort if not explicitly set
        budget_tokens = config.budget_tokens
        if budget_tokens is None and config.effort and profile.effort_to_budget_map:
            budget_tokens = profile.effort_to_budget_map.get(config.effort)
        if budget_tokens is not None:
            result['thinking_budget'] = budget_tokens

    # Handle include_in_response → include_thoughts
    result['include_thoughts'] = config.include_in_response

    return result if result else None


def format_openai_reasoning(
    config: ResolvedThinkingConfig | None,
    profile: ModelProfile,
) -> Literal['low', 'medium', 'high'] | None:
    """Format for OpenAI reasoning models (ReasoningEffort).

    Returns:
        'low', 'medium', or 'high' effort level, or None if not enabled.
    """
    if config is None:
        return None
    if not config.enabled:
        return None
    return config.effort or 'medium'


def format_groq_reasoning(
    config: ResolvedThinkingConfig | None,
    profile: ModelProfile,
    model_name: str,
    provider_name: str = 'Groq',
) -> Literal['hidden', 'raw', 'parsed'] | None:
    """Format for Groq reasoning models.

    Args:
        config: Resolved thinking config.
        profile: Model profile (unused but kept for consistency).
        model_name: For warning messages.
        provider_name: Provider name for warnings (default 'Groq').

    Returns:
        'parsed' (visible), 'hidden' (not visible), or None.
    """
    if config is None:
        return None
    if not config.enabled:
        return None

    # Warn about ignored settings (Groq doesn't support fine-grained control)
    ignored: list[str] = []
    if config.budget_tokens is not None:
        ignored.append('budget_tokens')
    if config.effort is not None:
        ignored.append('effort')

    if ignored:
        from .models._warnings import warn_settings_ignored_batch

        warn_settings_ignored_batch(
            setting_names=ignored,
            provider_name=provider_name,
            reason='Groq reasoning models do not support fine-grained control. Reasoning will be enabled with default behavior',
        )

    # Map include_in_response to reasoning format
    if config.include_in_response is False:
        return 'hidden'
    return 'parsed'


def format_cerebras_reasoning(
    config: ResolvedThinkingConfig | None,
    profile: ModelProfile,
    model_name: str,
    provider_name: str = 'Cerebras',
) -> bool | None:
    """Format for Cerebras reasoning models.

    Args:
        config: Resolved thinking config.
        profile: Model profile (unused but kept for consistency).
        model_name: For warning messages.
        provider_name: Provider name for warnings.

    Returns:
        True to disable reasoning, None for default (enabled).
    """
    if config is None:
        return None
    if not config.enabled:
        return True  # disable_reasoning=True

    # Warn about ignored settings (Cerebras only supports enable/disable)
    ignored: list[str] = []
    for setting in ('budget_tokens', 'effort', 'include_in_response', 'summary'):
        value = getattr(config, setting, None)
        if value is not None and (setting != 'include_in_response' or value is not True):
            ignored.append(setting)

    if ignored:
        from .models._warnings import warn_settings_ignored_batch

        warn_settings_ignored_batch(
            setting_names=ignored,
            provider_name=provider_name,
            model_name=model_name,
            reason='Cerebras reasoning can only be enabled or disabled. Reasoning will be enabled with default behavior',
        )

    return None  # Don't set disable_reasoning (use default enabled behavior)


def format_openrouter_reasoning(
    config: ResolvedThinkingConfig | None,
    profile: ModelProfile,
) -> dict[str, Any] | None:
    """Format for OpenRouter reasoning models.

    Returns:
        OpenRouter reasoning config dict with 'enabled', 'effort', 'max_tokens', 'exclude'.
    """
    if config is None:
        return None
    if not config.enabled:
        return {'enabled': False}

    result: dict[str, Any] = {}

    # Map effort directly (OpenRouter uses same values)
    if config.effort:
        result['effort'] = config.effort

    # Map budget_tokens to max_tokens
    if config.budget_tokens:
        result['max_tokens'] = config.budget_tokens

    # Map include_in_response=False to exclude=True
    if config.include_in_response is False:
        result['exclude'] = True

    # If no specific settings, just enable
    if not result:
        result['enabled'] = True

    return result
