from __future__ import annotations as _annotations

from . import ModelProfile, merge_profile
from .openai import OpenAIModelProfile, openai_model_profile


def harmony_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for the OpenAI Harmony Response format.

    See <https://cookbook.openai.com/articles/openai-harmony> for more details.
    """
    # Reasoning is intrinsic to Harmony: the raw chain of thought goes to the `analysis` channel, the
    # reasoning level is set in the system message as low/medium/high with medium as the default, and
    # there is no off switch. `openai_model_profile` can't tell us this — its prefix table describes
    # the models OpenAI serves on its own Responses API, which doesn't include gpt-oss, so a bare
    # `gpt-oss-120b` matches no prefix and resolves to "doesn't reason".
    #
    # The flags are gated on the model name because two providers route more than gpt-oss here:
    # OVHcloud maps the `gpt` prefix and Nebius the whole `openai/` namespace, so a non-Harmony model
    # such as `gpt-4o` also arrives. Setting the flags unconditionally would advertise mandatory
    # reasoning for it and override the correct `supports_thinking=False` from `openai_model_profile`.
    # The remaining Harmony-format overrides apply to every caller, since a provider only routes a
    # model here if it serves it in that format.
    reasoning = (
        OpenAIModelProfile(supports_thinking=True, thinking_always_enabled=True)
        if model_name.startswith('gpt-oss')
        else None
    )
    return merge_profile(
        openai_model_profile(model_name),
        OpenAIModelProfile(
            openai_supports_tool_choice_required=False,
            ignore_streamed_leading_whitespace=True,
        ),
        reasoning,
    )
