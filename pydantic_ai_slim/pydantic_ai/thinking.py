"""Centralized thinking configuration resolution.

This module provides the single source of truth for normalizing unified thinking
settings into a canonical `ResolvedThinkingConfig`. Provider-specific formatting
is done in each model class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .profiles import ModelProfile
    from .settings import ModelSettings


@dataclass
class ResolvedThinkingConfig:
    """Normalized thinking configuration after input parsing."""

    enabled: bool
    """Whether thinking is enabled."""

    effort: Literal['low', 'medium', 'high'] | None = None
    """Effort level for thinking depth."""


def resolve_thinking_config(
    model_settings: ModelSettings,
    profile: ModelProfile | None = None,
) -> ResolvedThinkingConfig | None:
    """Normalize unified thinking settings, optionally guarding against profile capabilities.

    Returns None if no thinking settings are specified or if the model doesn't support thinking.

    When ``profile`` is provided, applies these guards:
    1. Silent-drop if the model doesn't support thinking
    2. Silent-ignore ``thinking=False`` on always-on models

    Provider-specific translation of the returned config is each model's responsibility.
    """
    thinking = model_settings.get('thinking')
    effort = model_settings.get('thinking_effort')

    # Nothing set -> no unified thinking config
    if thinking is None and effort is None:
        return None

    # Profile guards (skip if no profile provided)
    if profile is not None:
        if not profile.supports_thinking:
            return None
        # thinking=False on always-on models → no-op (silent ignore)
        if thinking is False and profile.thinking_always_enabled:
            return None

    # thinking=False -> disabled (effort ignored per precedence rule 2)
    if thinking is False:
        return ResolvedThinkingConfig(enabled=False)

    # thinking=True or effort set (implicit enable, precedence rule 3)
    return ResolvedThinkingConfig(enabled=True, effort=effort)
