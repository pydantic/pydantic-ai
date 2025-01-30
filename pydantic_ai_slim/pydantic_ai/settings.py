from __future__ import annotations

from typing import TYPE_CHECKING

from httpx import Timeout
from typing_extensions import TypedDict

if TYPE_CHECKING:
    pass


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
    """

    presence_penalty: float
    """Penalize new tokens based on whether they have appeared in the text so far.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    """

    frequency_penalty: float
    """Penalize new tokens based on their existing frequency in the text so far.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    """

    logit_bias: dict[str, int]
    """Modify the likelihood of specified tokens appearing in the completion.

    Supported by:

    * OpenAI
    * Groq
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
