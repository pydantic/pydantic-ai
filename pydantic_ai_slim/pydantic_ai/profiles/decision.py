from __future__ import annotations as _annotations

from . import ModelProfile


class DecisionModelProfile(ModelProfile, total=False):
    """Profile for a [decision model][pydantic_ai.models.decision.DecisionModel]: what the model can be asked.

    These are facts about the model behind the URL, not the class that talks to it, so they are set by the provider
    for the model name, or by `profile=`. A key left out falls back to the class's
    [`max_choice_options`][pydantic_ai.models.decision.DecisionModel.max_choice_options] and
    [`max_score_levels`][pydantic_ai.models.decision.DecisionModel.max_score_levels].

    ALL FIELDS MUST BE `decision_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    decision_max_choice_options: int | None
    """The most options the model accepts in one pick-one question, or `None` for no limit.

    A pick-one field with more options, or more routes than this on the route question, is a
    [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent.
    """

    decision_max_score_levels: int | None
    """The most levels the model accepts in one rubric, or `None` for no limit.

    Whole numbers from 0 with more levels than this are not a rubric, so a field of them is asked as a pick-one
    instead, and counts against `decision_max_choice_options`.
    """


def decision_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a [decision model][pydantic_ai.models.decision.DecisionModel].

    A decision model answers typed questions about a state; it does not generate text, call tools, or read anything
    but text. Tool-mode structured output is how a decision model fills an `output_type`, and it rides on
    `supports_tools`, so that stays on. A system prompt anywhere in the history is part of what the model judges, so
    it needs no wrapping. Every other capability flag is off, and what no flag covers, such as a file in a prompt,
    the model refuses itself.
    """
    return ModelProfile(
        supports_tools=True,
        supports_text_output=False,
        supports_inline_system_prompts=True,
        supports_tool_return_schema=False,
        supports_json_schema_output=False,
        supports_json_object_output=False,
        supports_image_output=False,
        supports_audio_input=False,
        default_structured_output_mode='tool',
    )
