from __future__ import annotations

from typing import Literal, cast

from httpx import Timeout
from typing_extensions import TypedDict


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
    * xAI
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
    * xAI
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
    * xAI
    """

    timeout: float | Timeout
    """Override the client-level default timeout for a request, in seconds.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Mistral
    * xAI
    """

    parallel_tool_calls: bool
    """Whether to allow parallel tool calls.

    Supported by:

    * OpenAI (some models, not o1)
    * Groq
    * Anthropic
    * xAI
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
    * xAI
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
    * xAI
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
    * xAI
    """

    extra_headers: dict[str, str]
    """Extra headers to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Gemini
    * Groq
    * xAI
    """

    extra_body: object
    """Extra body to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    * Outlines (all providers)
    """

    thinking: bool
    """Enable or disable thinking/reasoning.

    - `True`: Enable thinking. Provider picks the best mode automatically
      (adaptive for Anthropic Opus 4.6, enabled for older Anthropic,
      default-on for OpenAI o-series, etc.)
    - `False`: Disable thinking. Silently ignored on models where
      thinking cannot be disabled (o-series, DeepSeek R1, etc.)
    - Omitted: Use provider default behavior.

    When `thinking` is `False`, `thinking_effort` is ignored.

    Provider-specific settings (e.g. `anthropic_thinking`, `openai_reasoning_effort`)
    always take precedence over this unified field.

    Supported by:

    * Anthropic (Claude 3.7+)
    * Gemini (2.5+)
    * OpenAI (o-series, GPT-5+)
    * Bedrock (Claude, DeepSeek R1, and Amazon Nova 2)
    * OpenRouter
    * Groq (reasoning models)
    * Cerebras (GLM, GPT-OSS)
    * Mistral (Magistral models — always-on)
    * Cohere (Command A Reasoning)
    * xAI (Grok 3 Mini, Grok 4)
    * DeepSeek (R1 models)
    * Harmony (GPT-OSS models — always-on)
    * ZAI (GLM models)
    """

    thinking_effort: Literal['low', 'medium', 'high']
    """Control the depth of thinking/reasoning.

    - `'low'`: Minimal thinking, faster responses, lower cost
    - `'medium'`: Balanced thinking depth (typical default)
    - `'high'`: Deep thinking, most thorough analysis

    Setting `thinking_effort` without `thinking` implicitly enables thinking.
    Silently ignored on models that don't support effort control.

    Not all providers support all effort levels. xAI only supports `'low'` and
    `'high'` — `'medium'` is mapped to `'low'`. Cohere and Groq ignore effort
    entirely (thinking is either on or off). Provider-specific effort levels
    (OpenAI's `xhigh`/`minimal`, Anthropic's `max`) are available through
    provider-specific settings.

    Supported by:

    * Anthropic (Opus 4.5+ via `output_config.effort`)
    * Gemini 3 (via `thinking_level`)
    * OpenAI (o-series, GPT-5+ via `reasoning_effort`)
    * OpenRouter (via `reasoning.effort`)
    * Bedrock (Claude and DeepSeek R1 via `budget_tokens`, Nova 2 via `maxReasoningEffort`)
    * xAI (Grok 3 Mini only — `'low'` and `'high'` only)
    """


def merge_model_settings(base: ModelSettings | None, overrides: ModelSettings | None) -> ModelSettings | None:
    """Merge two sets of model settings, preferring the overrides.

    A common use case is: merge_model_settings(<agent settings>, <run settings>)

    For nested dict values (like `extra_headers`), performs a shallow merge
    so that override fields are applied on top of base fields rather than replacing entirely.
    """
    if base and overrides:
        result = dict(base)
        for key, override_value in overrides.items():
            base_value = result.get(key)
            # Shallow merge for nested dicts (e.g., extra_headers)
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                result[key] = base_value | override_value
            else:
                result[key] = override_value
        return cast(ModelSettings, result)
    else:
        return base or overrides
