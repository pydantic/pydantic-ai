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
"""Cross-provider value set for [`ModelSettings.service_tier`][pydantic_ai.settings.ModelSettings.service_tier].

Values:

- `'auto'`: Let the provider decide — typically means "use a higher tier (scale credits, priority capacity)
  when available, otherwise standard." On providers without a server-side auto concept the field is
  omitted so the provider's natural default applies.
- `'default'`: Explicitly request the provider's standard tier — opts out of any server-side
  auto-promotion to premium tiers.
- `'flex'`: Lower-cost, latency-tolerant tier where the provider offers one. Silently ignored on
  providers that don't (e.g. Anthropic).
- `'priority'`: Higher-priority / lower-latency tier where the provider offers one. Silently ignored
  on providers that don't.

Per-provider mapping:

| value | OpenAI | Anthropic | Bedrock | Google (Gemini API) | Google (Vertex AI) |
|---|---|---|---|---|---|
| `'auto'` | `'auto'` | `'auto'` | _(omitted)_ | _(omitted)_ | _no headers (PT then on-demand)_ |
| `'default'` | `'default'` | `'standard_only'` | `{'type': 'default'}` | `'standard'` | _no headers (PT then on-demand)_ |
| `'flex'` | `'flex'` | _(omitted)_ | `{'type': 'flex'}` | `'flex'` | header `Shared-Request-Type: flex` (PT then Flex PayGo) |
| `'priority'` | `'priority'` | _(omitted)_ | `{'type': 'priority'}` | `'priority'` | header `Shared-Request-Type: priority` (PT then Priority PayGo) |

On Vertex AI the unified field maps only to safe PT-with-spillover variants so customers with
Provisioned Throughput keep using their reserved capacity first; to bypass PT entirely use
[`google_vertex_service_tier`][pydantic_ai.models.google.GoogleModelSettings.google_vertex_service_tier]
with `'flex_only'` or `'priority_only'`. Likewise, provider-specific values not in the unified set
(Bedrock's `'reserved'`, Anthropic's `'standard_only'`, Vertex's PT routing tiers) are reachable
only through the per-provider field.

Per-provider settings (`openai_service_tier`, `anthropic_service_tier`, `bedrock_service_tier`,
`google_vertex_service_tier`) always take precedence over this unified field when set.
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

    service_tier: ServiceTier
    """The cross-provider service tier to use for the model request.

    See [`ServiceTier`][pydantic_ai.settings.ServiceTier] for the value semantics and
    the per-provider mapping table. Provider-specific settings (`openai_service_tier`,
    `anthropic_service_tier`, `bedrock_service_tier`, `google_vertex_service_tier`)
    take precedence over this unified field when set.

    Supported by:

    * OpenAI
    * Anthropic
    * Bedrock
    * Google (Gemini API and Vertex AI)
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
