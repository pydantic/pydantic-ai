from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from httpx import Timeout
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ._run_context import RunContext

ToolChoiceValue = Literal['none', 'required', 'auto'] | list[str] | None
"""Type for static tool_choice values."""

ToolChoiceFunc = Callable[['RunContext[Any]'], ToolChoiceValue]
"""A callable that returns a tool_choice value based on the current run context.

This allows dynamic control of tool_choice per model request.

Example:
```python {test="skip" lint="skip"}
def my_tool_choice(ctx: RunContext) -> str | list[str] | None:
    if ctx.run_step == 1:
        return ['search']  # Force search tool on first request
    return None  # Default behavior for subsequent requests
```
"""


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

    tool_choice: ToolChoiceValue | Callable[..., ToolChoiceValue]
    """Control which function tools the model can use.

    This setting only affects function tools registered on the agent, not output tools
    used for structured output.

    **Static values:**

    * `None` (default): Automatically determined based on output configuration
    * `'auto'`: Model decides whether to use function tools
    * `'required'`: Model must use one of the available function tools
    * `'none'`: Model cannot use function tools (output tools remain available if needed)
    * `list[str]`: Model must use one of the specified function tools (validated against registered tools)

    **Dynamic callable:**

    You can also pass a callable that receives a [`RunContext`][pydantic_ai.tools.RunContext]
    and returns a static value. This allows dynamic control per model request:

    ```python {test="skip" lint="skip"}
    def my_tool_choice(ctx: RunContext) -> str | list[str] | None:
        if ctx.run_step == 1:
            return ['search']  # Force search tool on first request
        return None  # Default behavior for subsequent requests

    agent = Agent(..., model_settings={'tool_choice': my_tool_choice})
    ```

    When the callable returns `None`, the default behavior (based on output configuration) is used.

    If the agent has a structured output type that requires an output tool and `tool_choice='none'`
    is set, the output tool will still be available and a warning will be logged. Consider using
    native or prompted output modes if you need `tool_choice='none'` with structured output.

    Supported by:

    * OpenAI
    * Anthropic (note: `'required'` and specific tools not supported with thinking/extended thinking)
    * Gemini
    * Groq
    * Mistral
    * HuggingFace
    * Bedrock (note: `'none'` not supported, will fall back to `'auto'` with a warning)
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
