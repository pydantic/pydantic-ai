from __future__ import annotations as _annotations

from . import ModelProfile, merge_profile
from .openai import OpenAIModelProfile, openai_model_profile


def openai_codex_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for OpenAI Codex subscription-auth models.

    The Codex backend speaks the Responses API with a narrower dialect than the standard OpenAI
    endpoint: it serves streaming responses only, requires `store=false`, rejects sampling/tuning
    request fields (verified live on PR [#6433](https://github.com/pydantic/pydantic-ai/pull/6433)),
    and does not expose server-side input-token counting.
    """
    return merge_profile(
        openai_model_profile(model_name),
        OpenAIModelProfile(
            # Drop unsupported generic settings for portability. Forward explicit `openai_*`
            # settings so the API reports incompatibilities instead of silently ignoring them.
            openai_unsupported_model_settings=(
                'max_tokens',
                'temperature',
                'top_p',
            ),
            openai_responses_requires_streaming=True,
            openai_responses_requires_store_false=True,
            openai_supports_input_token_counting=False,
        ),
    )
