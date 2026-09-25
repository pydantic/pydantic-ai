from __future__ import annotations as _annotations

from . import ModelProfile


def typesafe_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a TypeSafe model.

    Jev answers typed questions about a state; it does not generate text, call tools, or read
    anything but text. Tool-mode structured output is how
    [`TypeSafeModel`][pydantic_ai.models.typesafe.TypeSafeModel] fills an `output_type`, and it rides on
    `supports_tools`, so that stays on. A system prompt anywhere in the history is part of what Jev judges, so it
    needs no wrapping. Every other capability flag is off, and what no flag covers, such as a file in a prompt,
    the model refuses itself.
    """
    return ModelProfile(
        # `jev-1.13` takes 32k tokens for the state plus the longest question, and 64k for the state and all
        # the questions together. The state is counted once per request, so the 32k limit is the one a growing
        # conversation hits: measured live, 32,878 tokens were accepted and the next size up was refused with
        # `max_tokens_exceeded`. The 64k budget only binds when the questions themselves are very large.
        # https://docs.typesafe.ai/model-jaggedness/jev-1.13
        # This mirrors the genai-prices entry for Jev, and can be removed once core requires a genai-prices
        # version that records it.
        context_window=32_000,
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
