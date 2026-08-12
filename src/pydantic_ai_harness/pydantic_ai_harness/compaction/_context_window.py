"""Context-window resolution -- how many tokens the model will actually accept.

Every strategy in this package triggers on an absolute budget, which an application
can only supply as a constant.  That constant is wrong for every model it was not
measured against.  This module turns a model into its real window, so a *fraction*
can stand in for the constant.

Pydantic AI does not carry the number yet (`ModelProfile` has no `context_window`
field as of 2.18; see pydantic/pydantic-ai#4538), but `genai-prices` -- already a
transitive dependency via `pydantic-ai-slim` -- does.  When core grows the field,
`resolve_context_window` is the single place that switches over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.models import AbstractModel

DEFAULT_CONTEXT_WINDOW = 200_000
"""Window assumed when the model's real one cannot be resolved.

Deliberately conservative.  Compacting earlier than necessary costs one summary;
overestimating the window costs the whole request.

Every capability that resolves a fraction takes a `fallback_context_window` defaulting to
this, so a deployment the registry cannot resolve -- a local endpoint, a Bedrock-prefixed
reference, a `FallbackModel` -- can supply the number it knows instead.
"""


def split_model_id(model_id: str) -> tuple[str | None, str]:
    """Split a `provider:model` id into its parts.

    A bare model name yields `(None, name)`, leaving the provider for `genai-prices`
    to infer from the name alone.
    """
    provider, separator, model = model_id.partition(':')
    if not separator:
        return None, model_id
    return provider, model


def resolve_context_window(model: AbstractModel | str) -> int | None:
    """Return the model's context window in tokens, or `None` when it is not known.

    `None` is returned both for models `genai-prices` has no entry for and for models
    it knows without a recorded window, so callers cannot mistake "unknown" for a
    number.  Pair it with `DEFAULT_CONTEXT_WINDOW` to get a usable budget.

    A wrapping model resolves to whatever `model_id` it reports: `WrapperModel` (and so
    `InstrumentedModel`) forwards the wrapped model's, while `FallbackModel` reports a
    composite `fallback:...` id that no registry entry matches, so it resolves to `None`.
    """
    from genai_prices.data_snapshot import get_snapshot

    provider_id, model_ref = split_model_id(model if isinstance(model, str) else model.model_id)
    try:
        _, model_info = get_snapshot().find_provider_model(
            model_ref=model_ref,
            provider=None,
            provider_id=provider_id,
            provider_api_url=None,
        )
    except LookupError:
        return None

    window = model_info.context_window
    # A registry entry can carry a zero or negative window; treat it as absent rather
    # than propagating a budget no request could ever fit under.
    return window if window is not None and window > 0 else None
