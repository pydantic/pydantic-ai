from __future__ import annotations as _annotations

from . import ModelProfile
from .decision import decision_model_profile


def typesafe_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a TypeSafe model.

    Jev is a [decision model][pydantic_ai.models.decision.DecisionModel], so this is the
    [decision model profile][pydantic_ai.profiles.decision.decision_model_profile] unchanged.
    """
    # No `context_window`: it comes from genai-prices, whose Jev entry records the 32k tokens `jev-1.13` takes
    # for the state plus the longest question. That is the limit a growing conversation hits, since the state
    # is counted once per request; the 64k for the state and all the questions together only binds when the
    # questions themselves are very large. https://docs.typesafe.ai/model-jaggedness/jev-1.13
    return decision_model_profile(model_name)
