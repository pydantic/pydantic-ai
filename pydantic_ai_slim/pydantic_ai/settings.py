from __future__ import annotations

from typing import Literal, TypeAlias

from httpx import Timeout
from typing_extensions import TypedDict

ThinkingEffort: TypeAlias = Literal['minimal', 'low', 'medium', 'high', 'xhigh']
"""The string effort levels for thinking/reasoning configuration."""

ThinkingLevel: TypeAlias = bool | ThinkingEffort
"""Type alias for thinking/reasoning configuration values.

- `True`: Enable thinking with the provider's default effort.
- `False`: Disable thinking (silently ignored on always-on models).
- `'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'`: Enable thinking at a specific effort level.

Not all providers support all levels. When a level is not natively supported,
it maps to the closest available value (e.g. `'xhigh'` -> `'high'` on providers
that don't support it, `'minimal'` -> `'low'` on providers without a minimal level).
"""

ServiceTier: TypeAlias = Literal['auto', 'default', 'flex', 'priority']
"""Cross-provider service-tier value for the [`service_tier`][pydantic_ai.settings.ModelSettings.service_tier] model setting.

- `'auto'`: let the provider decide (the setting is omitted from the request).
- `'default'`: ask for the provider's standard service tier explicitly.
- `'flex'`: ask for a lower-cost, latency-tolerant tier where available.
- `'priority'`: ask for a higher-priority tier where available.

See [`ModelSettings.service_tier`][pydantic_ai.settings.ModelSettings.service_tier]
for per-provider mapping details and precedence rules.
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

    thinking: ThinkingLevel
    """Enable or configure thinking/reasoning for the model.

    - `True`: Enable thinking with the provider's default effort level.
    - `False`: Disable thinking (silently ignored if the model always thinks).
    - `'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'`: Enable thinking at a specific effort level.

    When omitted, the model uses its default behavior (which may include thinking
    for reasoning models).

    Provider-specific thinking settings (e.g., `anthropic_thinking`,
    `openai_reasoning_effort`) take precedence over this unified field.

    Supported by:

    * Anthropic
    * OpenAI
    * Gemini
    * Groq
    * Bedrock
    * OpenRouter
    * Cerebras
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

    service_tier: ServiceTier
    """Unified cross-provider request for a pricing/latency service tier.

    Values are `'auto'`, `'default'`, `'flex'`, and `'priority'`. `'auto'` is
    functionally identical to not setting the field (both omit it from the request)
    — it exists so you can explicitly override an inherited `service_tier` from a
    higher-level settings object. Provider-specific fields (e.g. `openai_service_tier`,
    `bedrock_service_tier`, `google_gla_service_tier`, `google_vertex_service_tier`)
    always take precedence.

    Mapping summary:

    * OpenAI: pass-through (identical values).
    * Bedrock: pass-through for `default`/`flex`/`priority`; `auto` omits the field.
      `bedrock_service_tier` is the only way to request `reserved`.
    * Google (Gemini API / GLA only): `default` -> `'standard'`, `flex` -> `'flex'`,
      `priority` -> `'priority'`, `auto` omits the field.
    * Google (Vertex AI): **ignored.** Vertex's routing model (Provisioned Throughput /
      Flex PayGo) doesn't map cleanly onto `service_tier`'s pricing/latency semantics,
      so there is no silent cross-map. Use
      [`google_vertex_service_tier`][pydantic_ai.models.google.GoogleModelSettings.google_vertex_service_tier]
      to request Vertex routing explicitly.

    Supported by:

    * OpenAI
    * Bedrock
    * Google (Gemini API only)
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
