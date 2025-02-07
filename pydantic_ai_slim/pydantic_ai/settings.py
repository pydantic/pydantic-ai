from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from httpx import Timeout
from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError

if TYPE_CHECKING:
    pass


class GeminiSafetySettings(TypedDict):
    """Settings for Gemini safety features."""

    category: Literal[
        'HARM_CATEGORY_HARASSMENT',
        'HARM_CATEGORY_HATE_SPEECH',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'HARM_CATEGORY_DANGEROUS_CONTENT',
        'HARM_CATEGORY_CIVIC_INTEGRITY',
    ]
    threshold: Literal['BLOCK_LOW_AND_ABOVE', 'BLOCK_MEDIUM_AND_ABOVE', 'BLOCK_ONLY_HIGH', 'BLOCK_NONE']


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

    gemini_safety_settings: list[GeminiSafetySettings]
    """Settings for Gemini safety features."""

    anthropic_metadata: dict[str, str]
    """An object describing metadata about the request.

    Contains `user_id`, an external identifier for the user who is associated with the request."""

    openai_metadata: dict[str, str]
    """Metadata for OpenAI requests."""

    cohere_metadata: dict[str, str]
    """Metadata for Cohere requests."""

    mistral_metadata: dict[str, str]
    """Metadata for Mistral requests."""

    vertexai_metadata: dict[str, str]
    """Metadata for VertexAI requests."""


def validate_model_settings(settings: ModelSettings) -> None:
    """Validate model settings to ensure only valid parameters are used.

    Args:
        settings: Dictionary of model settings to validate

    Raises:
        UserError: If any invalid parameters are found
    """
    valid_params = {
        'max_tokens',
        'temperature',
        'top_p',
        'timeout',
        'parallel_tool_calls',
        'seed',
        'presence_penalty',
        'frequency_penalty',
        'logit_bias',
        'gemini_safety_settings',
        'anthropic_metadata',
        'openai_metadata',
        'cohere_metadata',
        'mistral_metadata',
        'vertexai_metadata',
    }

    invalid_params = set(settings.keys()) - valid_params
    if invalid_params:
        raise UserError(
            f'Invalid model setting parameter(s): {", ".join(sorted(invalid_params))}. '
            f'Valid parameters are: {", ".join(sorted(valid_params))}'
        )


def merge_model_settings(base: ModelSettings | None, overrides: ModelSettings | None) -> ModelSettings | None:
    """Merge two sets of model settings, preferring the overrides.

    A common use case is: merge_model_settings(<agent settings>, <run settings>)
    """
    # Validate settings before merging
    if base:
        validate_model_settings(base)
    if overrides:
        validate_model_settings(overrides)

    # Note: we may want merge recursively if/when we add non-primitive values
    if base and overrides:
        return base | overrides
    return base or overrides
