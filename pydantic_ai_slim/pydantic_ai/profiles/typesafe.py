from __future__ import annotations as _annotations

from . import ModelProfile, merge_profile
from .decision import DecisionModelProfile, decision_model_profile

_JEV_MAX_CHOICE_OPTIONS = 255
"""Jev picks from at most this many options in one question; a 256th is a 400 from the API.

https://docs.typesafe.ai/model-jaggedness/jev-1.13
"""

_JEV_MAX_SCORE_LEVELS = 10
"""Jev scores against at most this many rubric levels; an 11th is a 400 from the API.

https://docs.typesafe.ai/primitives/score
"""


def typesafe_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a TypeSafe model.

    Jev is a [decision model][pydantic_ai.models.decision.DecisionModel], so this is the
    [decision model profile][pydantic_ai.profiles.decision.decision_model_profile] with Jev's caps on options and
    rubric levels.
    """
    # No `context_window`: it comes from genai-prices, whose Jev entry records the 32k tokens `jev-1.13` takes
    # for the state plus the longest question. That is the limit a growing conversation hits, since the state
    # is counted once per request; the 64k for the state and all the questions together only binds when the
    # questions themselves are very large. https://docs.typesafe.ai/model-jaggedness/jev-1.13
    return merge_profile(
        decision_model_profile(model_name),
        DecisionModelProfile(
            decision_max_choice_options=_JEV_MAX_CHOICE_OPTIONS,
            decision_max_score_levels=_JEV_MAX_SCORE_LEVELS,
        ),
    )
