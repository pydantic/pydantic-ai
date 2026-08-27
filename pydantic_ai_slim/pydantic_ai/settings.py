from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from typing_extensions import TypedDict

from ._http import legacy_httpx

if TYPE_CHECKING:
    from httpx import Timeout
else:
    # Legacy HTTPX is optional: without it no `Timeout` instance can reach `ModelSettings`, so the
    # union member collapses onto the numeric one it already allows.
    Timeout = legacy_httpx.Timeout if legacy_httpx is not None else float

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

ToolChoiceScalar = Literal['none', 'required', 'auto']


@dataclass
class ToolOrOutput:
    """Restricts function tools while keeping output tools and direct text/image output available.

    Use this when you want to control which function tools the model can use
    in an agent run while still allowing the agent to complete with structured output,
    text, or images.

    See the [Tool Choice guide](../tools-advanced.md#tool-choice) for examples.
    """

    function_tools: list[str]
    """The names of function tools available to the model."""


ToolChoice = ToolChoiceScalar | list[str] | ToolOrOutput | None
"""Type alias for all valid tool_choice values."""

ServiceTier: TypeAlias = Literal['auto', 'default', 'flex', 'priority']
"""Cross-provider value set for [`ModelSettings.service_tier`][pydantic_ai.settings.ModelSettings.service_tier].

Values:

- `'auto'`: Let the provider decide — typically means "use a higher tier (scale credits, priority capacity)
  when available, otherwise standard." On providers without a server-side auto concept the field is
  omitted so the provider's natural default applies.
- `'default'`: Explicitly request the provider's standard tier — opts out of any server-side
  auto-promotion to premium tiers.
- `'flex'`: Lower-cost, latency-tolerant tier where the provider offers one. Silently ignored on
  providers that don't (e.g. Anthropic) — though a few reject the field outright rather than ignore it,
  as noted on the [`service_tier`][pydantic_ai.settings.ModelSettings.service_tier] entries.
- `'priority'`: Higher-priority / lower-latency tier where the provider offers one. Silently ignored
  on providers that don't.

Per-provider mapping:

| value | OpenAI | Anthropic | Bedrock | Google (Gemini API) | Google Cloud |
|---|---|---|---|---|---|
| `'auto'` | `'auto'` | `'auto'` | _(omitted)_ | _(omitted)_ | _no headers (PT then on-demand)_ |
| `'default'` | `'default'` | `'standard_only'` | `{'type': 'default'}` | `'standard'` | _no headers (PT then on-demand)_ |
| `'flex'` | `'flex'` | _(omitted)_ | `{'type': 'flex'}` | `'flex'` | header `Shared-Request-Type: flex` (PT then Flex PayGo) |
| `'priority'` | `'priority'` | _(omitted)_ | `{'type': 'priority'}` | `'priority'` | header `Shared-Request-Type: priority` (PT then Priority PayGo) |

On Google Cloud the unified field maps only to safe PT-with-spillover variants so customers with
Provisioned Throughput keep using their reserved capacity first; to bypass PT entirely use
[`google_cloud_service_tier`][pydantic_ai.models.google.GoogleModelSettings.google_cloud_service_tier]
with `'flex_only'` or `'priority_only'`. Likewise, provider-specific values not in the unified set
(Bedrock's `'reserved'`, Anthropic's `'standard_only'`, Google Cloud's PT routing tiers) are reachable
only through the per-provider field.

Per-provider settings (`openai_service_tier`, `anthropic_service_tier`, `bedrock_service_tier`,
`google_cloud_service_tier`) always take precedence over this unified field when set.
"""


class ModelSettings(TypedDict, total=False):
    """Settings to configure an LLM.

    Includes only settings which apply to multiple models / model providers,
    though not all of these settings are supported by all models.

    Each field's `Supported by:` list names the model classes that put the setting on the wire. A bare
    name covers every interface that model serves, so `OpenAI` means both
    [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] and
    [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]; a name qualified with an
    interface, like `OpenAI Chat Completions`, covers only that one, because the Responses API does
    not accept the setting at all.

    These lists are parsed and checked against the wire by
    `tests/models/test_model_settings_support.py`, so keep the `* Name` bullet shape and put any nuance in
    parentheses after the name.

    Being listed means Pydantic AI sends the setting, not that the service honors it: the
    OpenAI-compatible model classes forward whatever the OpenAI schema accepts, and an individual
    provider behind one of them may ignore a field its own API doesn't define, or reject it. Where we
    know of such a case it is noted on the entry, but the provider's own API reference is the
    authority.

    All types must be serializable using Pydantic.
    """

    max_tokens: int
    """The maximum number of tokens to generate before stopping.

    Supported by:

    * OpenAI
    * Anthropic
    * Google
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * MCP Sampling
    * xAI
    * HuggingFace
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle
    """

    temperature: float
    """Amount of randomness injected into the response.

    Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to a model's
    maximum `temperature` for creative and generative tasks.

    Note that even with `temperature` of `0.0`, the results will not be fully deterministic.

    Supported by:

    * OpenAI
    * Anthropic
    * Google
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * MCP Sampling
    * xAI
    * HuggingFace
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle
    """

    top_p: float
    """An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.

    So 0.1 means only the tokens comprising the top 10% probability mass are considered.

    You should either alter `temperature` or `top_p`, but not both.

    Supported by:

    * OpenAI
    * Anthropic
    * Google
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * xAI
    * HuggingFace
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle
    """

    top_k: int
    """Only sample from the top K options for each subsequent token.

    Used to remove "long tail" low probability responses.

    Supported by:

    * Anthropic
    * Google
    * Cohere
    * Bedrock (Anthropic and Amazon Nova models only)
    """

    timeout: int | float | Timeout
    """Override the client-level default timeout for a request, in seconds.

    Numeric seconds work everywhere. A legacy `httpx.Timeout` is also accepted and is converted to an
    `httpx2.Timeout` on the paths whose SDK expects one. `httpx2.Timeout` is deliberately not part of
    this contract, because some SDKs behind these settings still reject it.

    Supported by:

    * OpenAI
    * Anthropic
    * Google (numeric seconds only, not `httpx.Timeout`)
    * Groq
    * Mistral (numeric seconds only, not `httpx.Timeout`)
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle
    """

    parallel_tool_calls: bool
    """Whether to allow parallel tool calls.

    Supported by:

    * OpenAI (some models, not o1)
    * Anthropic
    * Groq
    * Mistral
    * xAI
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle
    """

    tool_choice: ToolChoice
    """Control which function tools the model can use.

    See the [Tool Choice guide](../tools-advanced.md#tool-choice) for detailed documentation
    and examples.

    * `None` (default): Defaults to `'auto'` behavior
    * `'auto'`: All tools available, model decides whether to use them
    * `'none'`: Disables function tools; model responds with text only (output tools remain for structured output)
    * `'required'`: Forces tool use; excludes output tools so the agent cannot produce a final response when set statically
    * `list[str]`: Only specified tools; excludes output tools so the agent cannot produce a final response when set statically
    * [`ToolOrOutput`][pydantic_ai.settings.ToolOrOutput]: Specified function tools plus output tools/text/image

    Note: setting `'required'` or `list[str]` *statically* (via the `model_settings` argument
    of [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] or the agent's own `model_settings`) raises a
    `UserError`, because it would force a tool call on every step and prevent the agent from
    producing a final response. To vary `tool_choice` per step (e.g. force a tool on the
    first step only), return a callable from a capability's
    [`get_model_settings`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] —
    those values are trusted to adapt across steps. For single API calls without an agent
    loop, use [`pydantic_ai.direct.model_request`][pydantic_ai.direct.model_request].

    Supported by:

    * OpenAI
    * Anthropic (`'required'` and specific tools not supported with thinking enabled)
    * Google
    * Groq
    * Cohere (a named subset is honored by filtering the tool list, not sent as a parameter)
    * Mistral (a named subset is honored by filtering the tool list, not sent as a parameter)
    * Bedrock
    * xAI
    * HuggingFace
    * Cerebras
    * Crusoe
    * Ollama (sent, but Ollama documents `tool_choice` as unsupported)
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle
    """

    seed: int
    """The random seed to use for the model, theoretically allowing for deterministic results.

    Supported by:

    * OpenAI Chat Completions
    * Google
    * Groq
    * Cohere
    * Mistral
    * xAI
    * HuggingFace
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle Chat Completions
    """

    presence_penalty: float
    """Penalize new tokens based on whether they have appeared in the text so far.

    Supported by:

    * OpenAI Chat Completions
    * Google
    * Groq
    * Cohere
    * Mistral
    * xAI
    * HuggingFace
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle Chat Completions
    """

    frequency_penalty: float
    """Penalize new tokens based on their existing frequency in the text so far.

    Supported by:

    * OpenAI Chat Completions
    * Google
    * Groq
    * Cohere
    * Mistral
    * xAI
    * HuggingFace
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle Chat Completions
    """

    logit_bias: dict[str, int]
    """Modify the likelihood of specified tokens appearing in the completion.

    Supported by:

    * OpenAI Chat Completions
    * Groq
    * HuggingFace
    * Crusoe
    * Ollama (sent, but Ollama documents `logit_bias` as unsupported)
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle Chat Completions
    """

    stop_sequences: list[str]
    """Sequences that will cause the model to stop generating.

    Supported by:

    * OpenAI Chat Completions
    * Anthropic
    * Google
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * MCP Sampling
    * xAI
    * HuggingFace
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle Chat Completions
    """

    extra_headers: dict[str, str]
    """Extra headers to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Google
    * Groq
    * Bedrock
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle
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

    Listed below are the model classes that translate this field onto the request. A class whose models
    always reason and take no thinking parameter is not listed at all (Cohere); where only some of a
    class's models are always-on it stays listed, and the per-model behavior is on the
    [Thinking page](../capabilities/thinking.md) (Mistral's `magistral`).

    Supported by:

    * OpenAI
    * Anthropic
    * Google
    * Groq
    * Mistral
    * Bedrock
    * xAI
    * Cerebras (only `False` is forwarded, as `reasoning_effort='none'`; the enable levels are not
      sent because Cerebras models reason by default, and `gpt-oss` ignores the disable too)
    * Crusoe
    * Ollama
    * OpenRouter (as `extra_body['reasoning']`)
    * Snowflake (as `extra_body['reasoning']` on Claude models, otherwise as `reasoning_effort`)
    * Z.AI (as `extra_body['thinking']`)
    * Bedrock Mantle (the Responses interface only; the Chat Completions interface serves only the
      `gpt-oss-safeguard` models, which take no thinking parameter)
    """

    service_tier: ServiceTier
    """The cross-provider service tier to use for the model request.

    See [`ServiceTier`][pydantic_ai.settings.ServiceTier] for the value semantics and
    the per-provider mapping table. Provider-specific settings (`openai_service_tier`,
    `anthropic_service_tier`, `bedrock_service_tier`, `google_cloud_service_tier`)
    take precedence over this unified field when set.

    Supported by:

    * OpenAI
    * Anthropic
    * Google (Gemini API and Google Cloud)
    * Bedrock
    * Cerebras (sent and enum-validated — HTTP 200 on an ordinary key; the tiers are in private
      preview, so no effect is observable without access)
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake (sent, but Snowflake Cortex rejects `service_tier` with an error)
    * Z.AI
    * Bedrock Mantle

    The OpenAI-derived model classes send the OpenAI value unchanged, so the OpenAI column of the
    mapping table applies to them.
    """

    extra_body: object
    """Extra body to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    * HuggingFace
    * Cerebras
    * Crusoe
    * Ollama
    * OpenRouter
    * Snowflake
    * Z.AI
    * Bedrock Mantle

    On the OpenAI-derived models that build their own `extra_body` (Cerebras, OpenRouter, Snowflake,
    Z.AI), the model's own derived keys overwrite yours when the keys collide.
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
