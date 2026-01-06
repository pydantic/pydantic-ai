from __future__ import annotations

from typing import Literal

from httpx import Timeout
from typing_extensions import TypedDict


class ThinkingConfig(TypedDict, total=False):
    """Unified configuration for model thinking/reasoning.

    This provides a provider-agnostic way to configure thinking features across
    different LLM providers. The settings are mapped to provider-specific
    configurations automatically.
    """

    enabled: bool
    """Whether thinking is enabled. Defaults to True if ThinkingConfig is provided."""

    budget_tokens: int
    """Maximum tokens allocated for thinking.

    Mapped to:
    - Anthropic: budget_tokens
    - Google: thinking_budget
    - OpenAI: (not directly supported, use effort instead)
    - Bedrock (Claude): budget_tokens
    """

    effort: Literal['low', 'medium', 'high']
    """Thinking effort level. Alternative to budget_tokens.

    Mapped to:
    - OpenAI: reasoning_effort ('low', 'medium', 'high')
    - Anthropic: budget_tokens (low=1024, medium=4096, high=16384)
    - Google: thinking_budget (low=1024, medium=8192, high=32768)
    """

    include_in_response: bool
    """Whether to include thinking content in the response. Defaults to True.

    Mapped to:
    - Google: include_thoughts
    - Groq: reasoning_format ('parsed' vs 'hidden')
    - Others: Always included when thinking is enabled
    """

    summary: Literal['none', 'concise', 'detailed', 'auto'] | bool
    """Request a summary of the thinking process.

    Mapped to:
    - OpenAI: reasoning_summary
    - Others: Not directly supported (thinking content is returned)
    """


class ModelSettings(TypedDict, total=False):
    """Settings to configure an LLM.

    Here we include only settings which apply to multiple models / model providers,
    though not all of these settings are supported by all models.
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

    thinking: bool | ThinkingConfig
    """Enable or configure thinking/reasoning for the model.

    Basic usage:
    - `thinking=True`: Enable thinking with provider defaults
    - `thinking=False`: Disable thinking (if provider enables by default)
    - `thinking={'budget_tokens': 2048}`: Enable with specific budget
    - `thinking={'effort': 'high'}`: Enable with effort level

    Provider-specific settings (e.g., `anthropic_thinking`) take precedence
    when specified alongside this unified setting.

    Supported by:

    * Anthropic
    * Google (Gemini)
    * OpenAI (reasoning models)
    * Bedrock (Claude models)
    """


def merge_model_settings(base: ModelSettings | None, overrides: ModelSettings | None) -> ModelSettings | None:
    """Merge two sets of model settings, preferring the overrides.

    A common use case is: merge_model_settings(<agent settings>, <run settings>)

    For nested dict values (like `thinking` or `extra_headers`), performs a shallow merge
    so that override fields are applied on top of base fields rather than replacing entirely.
    """
    if base and overrides:
        result = dict(base)
        for key, override_value in overrides.items():
            base_value = result.get(key)
            # Shallow merge for nested dicts (e.g., thinking, extra_headers)
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                result[key] = base_value | override_value
            else:
                result[key] = override_value
        return result
    else:
        return base or overrides
