from __future__ import annotations as _annotations

from . import ModelProfile

# Models with adjustable reasoning via `reasoning_effort` (opt-in, unlike always-on `magistral`).
# See https://docs.mistral.ai/capabilities/reasoning/.
#
# `-latest` aliases resolve to a reasoning model on the public API; older `mistral-small-*` /
# `mistral-medium-*` versions (e.g. `mistral-small-2501`, `mistral-medium-2505`) do not reason,
# so we match the reasoning families by exact name or as a `<base>-<version>` dated variant
# (e.g. `mistral-medium-3-5-26-04`), never by bare `startswith`.
_ADJUSTABLE_REASONING_ALIASES = frozenset({'mistral-small-latest', 'mistral-medium-latest'})
_ADJUSTABLE_REASONING_BASES = ('mistral-small-4', 'mistral-medium-3-5')


def _supports_adjustable_reasoning(model_name: str) -> bool:
    if model_name in _ADJUSTABLE_REASONING_ALIASES:
        return True
    return any(model_name == base or model_name.startswith(f'{base}-') for base in _ADJUSTABLE_REASONING_BASES)


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    if model_name.startswith('magistral'):
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)
    if _supports_adjustable_reasoning(model_name):
        return ModelProfile(supports_thinking=True, thinking_always_enabled=False)
    return None
