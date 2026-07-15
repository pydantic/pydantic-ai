from __future__ import annotations as _annotations

from . import ModelProfile

# Models with adjustable reasoning via `reasoning_effort` (opt-in, unlike always-on `magistral`):
# the Mistral Small 4 and Medium 3.5 families. Older `mistral-small-*` / `mistral-medium-*`
# snapshots (e.g. `mistral-small-2506`, `mistral-medium-2505`) don't support reasoning and are
# deliberately excluded; keep this set in sync with the ids reporting `capabilities.reasoning`
# on the Mistral `/v1/models` API. The `-latest` aliases resolve to a reasoning model on the
# public API; on private deployments they may point to an older non-reasoning snapshot.
# See https://docs.mistral.ai/capabilities/reasoning/.
_ADJUSTABLE_REASONING_MODELS = frozenset(
    {
        'mistral-small-latest',
        'mistral-small-2603',
        'mistral-medium-latest',
        'mistral-medium-3-5',
        'mistral-medium-2604',
    }
)


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    if model_name.startswith('magistral'):
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)
    if model_name in _ADJUSTABLE_REASONING_MODELS:
        return ModelProfile(supports_thinking=True, thinking_always_enabled=False)
    return None
