from __future__ import annotations as _annotations

from ..native_tools import WebSearchTool
from . import ModelProfile, merge_profile
from .openai import OpenAIModelProfile, openai_model_profile


def openai_chatgpt_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for models used through ChatGPT plan usage (Sign in with ChatGPT).

    ChatGPT plan usage serves the public Responses API with the narrower dialect OpenAI documents in its
    [preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations):
    requests must stream and set `store=false`, explicit `system` messages are rejected in favor of
    `developer` ones, sampling, output-limit, and service-tier fields are rejected, only `POST /v1/responses` is served
    (so there is no server-side input-token counting or Batch API), and of the hosted tools only web search is
    accepted. `tool_search` is not accepted, so deferred tools fall back to local search.
    """
    return merge_profile(
        openai_model_profile(model_name),
        OpenAIModelProfile(
            # Drop unsupported generic settings for portability. Forward explicit `openai_*`
            # settings so the API reports incompatibilities instead of silently ignoring them.
            # Service-tier overrides are among the capabilities the route reports as unsupported.
            openai_unsupported_model_settings=(
                'max_tokens',
                'service_tier',
                'temperature',
                'top_p',
            ),
            openai_system_prompt_role='developer',
            openai_responses_requires_streaming=True,
            openai_responses_requires_store_false=True,
            openai_supports_input_token_counting=False,
            openai_supports_batch_api=False,
            # `prompt_cache_retention` is rejected and the explicit caching fields are not documented as
            # supported, so the unified `cache` setting has nothing to turn on.
            openai_supports_prompt_cache_breakpoints=False,
            openai_responses_supports_prompt_cache_diagnostics=False,
            # `reasoning.effort='max'` is unverified on ChatGPT plan usage, so unified `'max'` falls back to `'xhigh'`.
            openai_responses_supports_max_reasoning_effort=False,
            supports_cache=False,
            supported_cache_retentions=(),
            default_cache_retention=None,
            supported_native_tools=frozenset({WebSearchTool}),
            tool_deferral_mode=None,
        ),
    )
