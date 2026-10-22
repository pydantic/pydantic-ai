from __future__ import annotations

from . import ModelProfile, merge_profile
from .decision import DecisionModelProfile, decision_model_profile
from .openai import OpenAIModelProfile


def cloudflare_model_profile(model_name: str) -> ModelProfile | None:
    """Get conservative capabilities for native Workers AI models.

    Unlisted chat models accept text. Override the profile explicitly to enable additional capabilities.
    """
    if model_name == '@cf/cloudflare/clef':
        context_window = 65_536
    elif model_name == '@cf/cloudflare/clef-flash':
        context_window = 24_576
    else:
        return OpenAIModelProfile(
            supports_tools=model_name
            in (
                '@cf/meta/llama-4-scout-17b-16e-instruct',
                '@cf/qwen/qwen3-30b-a3b-fp8',
                '@cf/openai/gpt-oss-20b',
                '@cf/openai/gpt-oss-120b',
            ),
            supports_image_input=model_name == '@cf/meta/llama-4-scout-17b-16e-instruct',
            supports_document_input=False,
            supports_audio_file_input=False,
            supports_video_input=False,
            supports_json_schema_output=model_name == '@cf/meta/llama-4-scout-17b-16e-instruct',
            supports_json_object_output=False,
            openai_supports_strict_tool_definition=False,
        )
    return merge_profile(
        decision_model_profile(model_name),
        DecisionModelProfile(
            decision_max_choice_options=255,
            decision_max_score_levels=10,
            context_window=context_window,
        ),
    )
