"""Centralized thinking configuration resolution.

This module provides the single source of truth for normalizing unified thinking
settings into a canonical `ResolvedThinkingConfig`. Provider-specific formatting
is done in each model class.

No validation against model capabilities happens here — that's the model class's
job, using silent-drop semantics for unsupported settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .profiles import ModelProfile
    from .settings import ModelSettings


@dataclass
class ResolvedThinkingConfig:
    """Normalized thinking configuration after input parsing.

    No validation against profile — that's the model class's job.
    """

    enabled: bool
    """Whether thinking is enabled."""

    effort: Literal['low', 'medium', 'high'] | None = None
    """Effort level for thinking depth."""


def resolve_thinking_config(
    model_settings: ModelSettings,
) -> ResolvedThinkingConfig | None:
    """Normalize unified thinking settings into a canonical form.

    Returns None if no thinking settings are specified.
    Does NOT validate against model capabilities — that happens in each model class.
    """
    thinking = model_settings.get('thinking')
    effort = model_settings.get('thinking_effort')

    # Nothing set -> no unified thinking config
    if thinking is None and effort is None:
        return None

    # thinking=False -> disabled (effort ignored per precedence rule 2)
    if thinking is False:
        return ResolvedThinkingConfig(enabled=False)

    # thinking=True or effort set (implicit enable, precedence rule 3)
    return ResolvedThinkingConfig(
        enabled=True,
        effort=effort,
    )


def resolve_with_profile(
    model_settings: ModelSettings,
    profile: ModelProfile,
) -> ResolvedThinkingConfig | None:
    """Resolve and guard unified thinking settings against model profile capabilities.

    This centralizes the common pattern used across all providers:
    1. Resolve unified settings into canonical form
    2. Silent-drop if the model doesn't support thinking
    3. Silent-ignore `thinking=False` on always-on models

    Returns None if thinking settings are not applicable.
    Provider-specific translation of the returned config is each model's responsibility.
    """
    resolved = resolve_thinking_config(model_settings)
    if resolved is None:
        return None

    if not profile.supports_thinking:
        return None

    # thinking=False on always-on models → no-op (silent ignore)
    if not resolved.enabled and profile.thinking_always_enabled:
        return None

    return resolved
