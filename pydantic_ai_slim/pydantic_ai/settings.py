from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from httpx import Timeout
from typing_extensions import TypedDict

ToolChoiceScalar = Literal['none', 'required', 'auto']


@dataclass
class ToolsPlusOutput:
    """Allows the user to specify a list of tool names, while also allowing the model to generate output."""

    function_tools: list[str]
    """The names of function tools available to the model."""


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

    tool_choice: ToolChoiceScalar | list[str] | ToolsPlusOutput | None
    """Control which function tools the model can use.

    Warning: when using 'required' or passing a list of tool names, the model will be unable to
    respond with text directly. Use these settings only if you know what you're doing
    (for instance when making [direct](TODO: link) model requests).

    This setting controls the API's tool_choice parameter. When possible, we send all tools and use
    API-native features (e.g., OpenAI's `allowed_tools`, Gemini's `allowed_function_names`) to restrict
    which tools the model can call. This preserves API caching benefits.

    About built-in tools: some providers like anthropic

    * `None` (default): Defaults to 'auto' behavior
    * `'auto'`: All tools sent including output tools, model decides whether to use them
    * `'none'`: Tool definitions will be sent to maintain caching, but the model
        will be prevented from calling them. This also means that the model will only be able to generate text output.
    * `'required'`: Forces the model to use one or more function tools. No output tools will be sent.
        Use 'required' only when using [direct](TODO: link) model requests, since the model will be unable to
        respond with text directly.
    * `list[str]`: Sends a list of specific function tool names which the model is allowed to use.
        As with `required`, no output tools will be sent, so use only with [direct](TODO: link) model requests.
    * `ToolChoiceToolList`: Both the specified function tools and all output tools will be sent.
        TODO clear up when `str` is one of the output types: do we default to 'auto' or 'required' depending on whether text output is allowed?

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
