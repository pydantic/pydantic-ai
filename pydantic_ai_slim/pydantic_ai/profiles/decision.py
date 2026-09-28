from __future__ import annotations as _annotations

from . import ModelProfile


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
