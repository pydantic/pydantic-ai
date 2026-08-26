from __future__ import annotations as _annotations

from pydantic_ai import ModelProfile
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile


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
            # Rejected server-side per the live probes on #6433: `max_output_tokens` (the wire name
            # for the generic `max_tokens` setting), `temperature`, `top_p`, `top_logprobs`, `user`,
            # and `truncation`. Dropped silently so client code stays portable across providers.
            openai_unsupported_model_settings=(
                'max_tokens',
                'temperature',
                'top_p',
                'openai_top_logprobs',
                'openai_truncation',
                'openai_user',
            ),
            openai_responses_requires_streaming=True,
            openai_responses_requires_store_false=True,
            openai_supports_input_token_counting=False,
            # The official Codex client keys prompt-cache affinity off a stable session id and
            # emits `session-id`/`thread-id` headers on every request; mirror that per conversation.
            openai_responses_session_affinity=True,
        ),
    )
