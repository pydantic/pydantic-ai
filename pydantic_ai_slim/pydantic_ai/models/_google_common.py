"""Wire-format utilities shared by the Google API adapters."""

from ..exceptions import UserError
from ..profiles.google import (
    GOOGLE_THINKING_LEVEL_SCALE,
    GOOGLE_THINKING_LEVELS,
    GoogleModelProfile,
    GoogleThinkingLevel,
)
from ..settings import ThinkingEffort


def character_index(text: str, byte_index: int) -> int | None:
    """Convert a Gemini UTF-8 byte offset to a Python string index."""
    encoded = text.encode()
    if not 0 <= byte_index <= len(encoded):
        return None
    try:
        return len(encoded[:byte_index].decode())
    except UnicodeDecodeError:
        return None


def thinking_effort_to_level(thinking: ThinkingEffort) -> GoogleThinkingLevel:
    """Normalize unified thinking effort to a Gemini thinking level."""
    level_by_effort: dict[ThinkingEffort, GoogleThinkingLevel] = {
        'minimal': 'MINIMAL',
        'low': 'LOW',
        'medium': 'MEDIUM',
        'high': 'HIGH',
        # Gemini has nothing above `HIGH`; map the higher levels to it.
        'xhigh': 'HIGH',
        'max': 'HIGH',
    }
    return level_by_effort[thinking]


_GOOGLE_THINKING_LEVEL_ORDER: dict[GoogleThinkingLevel, int] = {
    level: order for order, level in enumerate(GOOGLE_THINKING_LEVEL_SCALE)
}


def snap_thinking_level(
    level: GoogleThinkingLevel, levels: frozenset[GoogleThinkingLevel] | None
) -> GoogleThinkingLevel:
    """Snap a thinking level to the nearest one the model accepts, on the `MINIMAL < LOW < MEDIUM < HIGH` scale.

    Equidistant levels round down to the cheaper one. `None` means the model takes the whole scale, so
    the level passes through. Shared with the Live path, which differs only in where its level set
    comes from.
    """
    if levels is None:
        return level
    if not levels:
        raise UserError('`google_thinking_levels` must contain at least one level when `thinking` is set')
    if unknown := levels - GOOGLE_THINKING_LEVELS:
        raise UserError(
            f'`google_thinking_levels` contains unknown levels: {sorted(unknown)!r}; '
            f'expected a subset of {sorted(GOOGLE_THINKING_LEVELS)!r}'
        )
    requested = _GOOGLE_THINKING_LEVEL_ORDER[level]
    return min(
        levels,
        key=lambda candidate: (
            abs(_GOOGLE_THINKING_LEVEL_ORDER[candidate] - requested),
            _GOOGLE_THINKING_LEVEL_ORDER[candidate],
        ),
    )


def resolve_google_thinking_level(thinking: ThinkingEffort, profile: GoogleModelProfile) -> GoogleThinkingLevel:
    """Map unified thinking to the closest thinking level the model supports."""
    levels = profile.get('google_thinking_levels')
    if levels is None:
        # Sparse profile without a level set: fall back to the boolean floor flag.
        levels = (
            GOOGLE_THINKING_LEVELS
            if profile.get('google_supports_minimal_thinking_level', True)
            else GOOGLE_THINKING_LEVELS - {'MINIMAL'}
        )
    return snap_thinking_level(thinking_effort_to_level(thinking), levels)
