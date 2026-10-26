"""Context-window resolution -- how many tokens the model will actually accept.

Every strategy in this package triggers on an absolute budget, which an application
can only supply as a constant.  That constant is wrong for every model it was not
measured against.  This module turns a model into its real window, so a *fraction*
can stand in for the constant.

A model reports its own window as `Model.context_window`: the value its profile sets,
filled from `genai-prices` when no profile layer does, and for a `FallbackModel` the
smallest among its candidates.  A model id given as a string is looked up in
`genai-prices` directly.
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
this, so a deployment whose window is unknown -- a local endpoint, a model neither its
profile nor the registry has a window for -- can supply the number it knows instead.
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

    A model instance reports its own `context_window` first: the value its profile sets
    (including a user's `profile=` override), filled from `genai-prices` when no profile
    layer does.  `WrapperModel` (and so `InstrumentedModel`) forwards the wrapped model's,
    and `FallbackModel` reports the smallest among its candidates.  When the model has no
    window of its own, and for a model id given as a string, `genai-prices` is consulted
    directly.

    `None` is returned both for models `genai-prices` has no entry for and for models
    it knows without a recorded window, so callers cannot mistake "unknown" for a
    number.  Pair it with `DEFAULT_CONTEXT_WINDOW` to get a usable budget.
    """
    from genai_prices.data_snapshot import get_snapshot

    if not isinstance(model, str) and (window := model.context_window) is not None and window > 0:
        return window

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
