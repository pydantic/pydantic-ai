from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from httpx import Timeout
from typing_extensions import TypedDict

ToolChoiceScalar = Literal['none', 'required', 'auto']


@dataclass
class ToolOrOutput:
    """Restricts function tools while keeping output tools and direct text/image output available.

    Use this when you want to control which function tools the model can use
    in an agent run while still allowing the agent to complete with structured output,
    text, or images.

    See the [Tool Choice guide](TODO fill this out in https://github.com/dsfaccini/pydantic-ai/pull/2 i.e. prepare_model_settings hook PR)
    for examples.
    """

    function_tools: list[str]
    """The names of function tools available to the model."""


ToolChoice = ToolChoiceScalar | list[str] | ToolOrOutput | None
"""Type alias for all valid tool_choice values."""


class ModelSettings(TypedDict, total=False):
    """Settings to configure an LLM.

    Includes only settings which apply to multiple models / model providers,
    though not all of these settings are supported by all models.

    All types must be serializable using Pydantic.
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

    tool_choice: ToolChoice
    """Control which function tools the model can use.

    See the [Tool Choice guide](tool-choice.md) for detailed documentation
    and examples.

    * `None` (default): Defaults to `'auto'` behavior
    * `'auto'`: All tools available, model decides whether to use them
    * `'none'`: Disables function tools; model responds with text only (output tools remain for structured output)
    * `'required'`: Forces tool use; excludes output tools so agent cannot complete (use with `model.request()` only)
    * `list[str]`: Only specified tools; excludes output tools so agent cannot complete (use with `model.request()` only)
    * [`ToolOrOutput`][pydantic_ai.settings.ToolOrOutput]: Specified function tools plus output tools/text/image

    Note: `'required'` and `list[str]` raise an error in `agent.run()` because they prevent the agent from
    producing a final response. Use [`ToolOrOutput`][pydantic_ai.settings.ToolOrOutput] to combine specific
    function tools with output capability, or use [direct model requests](direct.md) for single API calls.

    TODO(prepare_model_settings): Update this to note that the hook CAN return 'required' or list[str]
    for per-step control since the hook applies to individual model requests, not the entire run.

    Supported by:

    * OpenAI
    * Anthropic (`'required'` and specific tools not supported with thinking enabled)
    * Google
    * Groq
    * Mistral
    * HuggingFace
    * Bedrock
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


def merge_model_settings(base: ModelSettings | None, overrides: ModelSettings | None) -> ModelSettings | None:
    """Merge two sets of model settings, preferring the overrides.

    A common use case is: merge_model_settings(<agent settings>, <run settings>)
    """
    # Note: we may want merge recursively if/when we add non-primitive values
    if base and overrides:
        return base | overrides
    else:
        return base or overrides
