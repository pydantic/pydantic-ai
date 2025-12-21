from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from httpx import Timeout
from typing_extensions import TypedDict

ToolChoiceScalar = Literal['none', 'required', 'auto']


@dataclass
class ToolsPlusOutput:
    """Restricts function tools while keeping output tools available.

    Use this when you want to control which function tools the model can use
    in an agent run while still allowing the agent to complete with structured output.

    See the [Tool Choice guide](https://ai.pydantic.dev/tool-choice/#toolsplusoutput---specific-tools-with-output)
    for examples.
    """

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

    See the [Tool Choice guide](https://ai.pydantic.dev/tool-choice/) for detailed documentation
    and examples.

    Warning: when using `'required'` or passing a list of tool names, the model will be unable to
    respond with text directly. Use these settings only with [direct model requests](https://ai.pydantic.dev/direct/).

    * `None` (default): Defaults to `'auto'` behavior
    * `'auto'`: All tools available, model decides whether to use them
    * `'none'`: Disables function tools; model responds with text only (output tools remain for structured output)
    * `'required'`: Forces tool use; no output tools sent (for direct model requests only)
    * `list[str]`: Only specified tools available; no output tools sent (for direct model requests only)
    * [`ToolsPlusOutput`][pydantic_ai.settings.ToolsPlusOutput]: Specified function tools plus all output tools

    Supported by:

    * OpenAI (full support)
    * Anthropic (`'required'` and specific tools not supported with thinking enabled)
    * Google (full support)
    * Groq (single tool forcing only)
    * Mistral (limited specific tool support)
    * HuggingFace (single tool forcing only)
    * Bedrock (single tool forcing only; no native `'none'`)
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
