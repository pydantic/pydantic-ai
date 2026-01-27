"""Centralized thinking configuration resolution.

This module provides the single source of truth for resolving unified thinking
settings into a normalized `ResolvedThinkingConfig`. Provider-specific formatting
is done in each model file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
    All validation, bool/dict handling, and normalization happens here.
    Provider-specific formatting is done in each model file.

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
        # Don't set budget_tokens here - let model files apply provider-specific defaults
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
    # Model files will derive from effort or apply defaults as needed per provider
    budget_tokens = config.get('budget_tokens')
    effort = config.get('effort')

    return ResolvedThinkingConfig(
        enabled=True,
        budget_tokens=budget_tokens,
        effort=effort,
        include_in_response=config.get('include_in_response', True),
        summary=config.get('summary'),
    )
