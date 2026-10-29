from __future__ import annotations as _annotations

from . import ModelProfile, merge_profile
from .decision import DecisionModelProfile, decision_model_profile

DREX_MAX_QUESTIONS = 16
"""Drex Max answers at most this many questions in one request.

A 17th is rejected before the request is sent, and is not split into a second request.
"""

NACE_MAX_QUESTIONS = 512
"""The Decisions endpoint accepts at most this many questions in one request."""

_DREX_MAX_MODEL = 'drex-max'


class NaceModelProfile(DecisionModelProfile, total=False):
    """Profile for a Nace Drex model.

    Question and image caps that differ by model live here. The class's `max_questions` and `max_images` stay the
    endpoint caps, which a profile cannot raise.
    """

    nace_accepts_images: bool
    """Whether this model accepts the images Drex Max can see. Other Drex models reject any image."""

    nace_max_questions: int
    """The most questions this model answers in one request.

    Drex Max is 16. A larger request fails before it is sent. It is not split.
    """


def nace_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a Nace Drex model.

    Drex is a [decision model][pydantic_ai.models.decision.DecisionModel]. Only `drex-max` accepts images, and only
    `drex-max` is limited to 16 questions. Every other Drex id keeps the endpoint's 512-question cap and rejects images.
    """
    drex_max = model_name == _DREX_MAX_MODEL
    return merge_profile(
        decision_model_profile(model_name),
        NaceModelProfile(
            nace_accepts_images=drex_max,
            nace_max_questions=DREX_MAX_QUESTIONS if drex_max else NACE_MAX_QUESTIONS,
        ),
    )
