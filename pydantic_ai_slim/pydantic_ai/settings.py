from __future__ import annotations

from typing import Literal

from httpx import Timeout
from typing_extensions import TypedDict


class ModelSettings(TypedDict, total=False):
    """Settings to configure an LLM.

    Includes only settings which apply to multiple models / model providers,
    though not all of these settings are supported by all models.

    All types must be JSON-serializable.
    """

    max_tokens: int
    """The maximum number of tokens to generate before stopping.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * MCP Sampling
    * Outlines (all providers)
    """

    temperature: float
    """Amount of randomness injected into the response.

    Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to a model's
    maximum `temperature` for creative and generative tasks.

    Note that even with `temperature` of `0.0`, the results will not be fully deterministic.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * Outlines (Transformers, LlamaCpp, SgLang, VLLMOffline)
    """

    top_p: float
    """An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.

    So 0.1 means only the tokens comprising the top 10% probability mass are considered.

    You should either alter `temperature` or `top_p`, but not both.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * Outlines (Transformers, LlamaCpp, SgLang, VLLMOffline)
    """

    timeout: float | Timeout
    """Override the client-level default timeout for a request, in seconds.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Mistral
    """

    parallel_tool_calls: bool
    """Whether to allow parallel tool calls.

    Supported by:

    * OpenAI (some models, not o1)
    * Groq
    * Anthropic
    """

    tool_choice: Literal['none', 'required', 'auto'] | list[str] | None
    """Control which function tools the model can use.

    This setting controls the API's tool_choice parameter. When possible, we send all tools and use
    API-native features (e.g., OpenAI's `allowed_tools`, Gemini's `allowed_function_names`) to restrict
    which tools the model can call. This preserves API caching benefits. Filtering tools before sending
    only happens when the provider doesn't support API-native restrictions.

    Output tools (used for structured output) are managed separately and remain available when needed.

    * `None` (default): All tools sent, tool_choice determined by output configuration
    * `'auto'`: All tools sent, model decides whether to use them
    * `'required'`: Only function tools sent (no output tools), model must use one
    * `'none'`: Only output tools available (function tools disabled)
    * `list[str]`: Only specified tools available (can include function or output tool names)
    * `[]` (empty list): Treated as `'none'`

    Supported by:

    * OpenAI (uses `allowed_tools` for multi-tool restrictions)
    * Anthropic (note: `'required'` and specific tools not supported with thinking/extended thinking)
    * Gemini (uses `allowed_function_names` for restrictions)
    * Groq (only supports forcing a single tool; falls back for multi-tool lists)
    * Mistral (no specific tool support; uses `'required'` for lists)
    * HuggingFace (only supports forcing a single tool; falls back for multi-tool lists)
    * Bedrock (only supports forcing a single tool; falls back for multi-tool lists)
    """

    seed: int
    """The random seed to use for the model, theoretically allowing for deterministic results.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Gemini
    * Outlines (LlamaCpp, VLLMOffline)
    """

    presence_penalty: float
    """Penalize new tokens based on whether they have appeared in the text so far.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    * Outlines (LlamaCpp, SgLang, VLLMOffline)
    """

    frequency_penalty: float
    """Penalize new tokens based on their existing frequency in the text so far.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    * Outlines (LlamaCpp, SgLang, VLLMOffline)
    """

    logit_bias: dict[str, int]
    """Modify the likelihood of specified tokens appearing in the completion.

    Supported by:

    * OpenAI
    * Groq
    * Outlines (Transformers, LlamaCpp, VLLMOffline)
    """

    stop_sequences: list[str]
    """Sequences that will cause the model to stop generating.

    Supported by:

    * OpenAI
    * Anthropic
    * Bedrock
    * Mistral
    * Groq
    * Cohere
    * Google
    """

    extra_headers: dict[str, str]
    """Extra headers to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    """

    extra_body: object
    """Extra body to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    * Outlines (all providers)
    """


def merge_model_settings(base: ModelSettings | None, overrides: ModelSettings | None) -> ModelSettings | None:
    """Merge two sets of model settings, preferring the overrides.

    A common use case is: merge_model_settings(<agent settings>, <run settings>)
    """
    # Note: we may want merge recursively if/when we add non-primitive values
    if base and overrides:
        return base | overrides
    else:
        return base or overrides
