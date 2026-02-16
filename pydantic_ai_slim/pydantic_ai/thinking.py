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
