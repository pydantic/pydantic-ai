from __future__ import annotations as _annotations

from . import ModelProfile


def typesafe_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a TypeSafe model.

    Jev answers typed questions about a state; it does not generate text, call tools, or read
    anything but text. Tool-mode structured output is how
    [`TypeSafeModel`][pydantic_ai.models.typesafe.TypeSafeModel] fills an `output_type`, and it rides on
    `supports_tools`, so that stays on. System prompts anywhere in the history become instructions, so they
    need no wrapping. Every other capability flag is off, and what no flag covers, such as a file in a prompt,
    the model refuses itself.
    """
    return ModelProfile(
        supports_tools=True,
        supports_inline_system_prompts=True,
        supports_tool_return_schema=False,
        supports_json_schema_output=False,
        supports_json_object_output=False,
        supports_image_output=False,
        supports_audio_input=False,
        default_structured_output_mode='tool',
    )
