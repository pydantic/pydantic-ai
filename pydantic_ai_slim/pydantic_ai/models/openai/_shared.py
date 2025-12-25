from __future__ import annotations as _annotations

from typing import Any, Literal

from typing_extensions import deprecated

from ... import _utils, usage
from ...builtin_tools import (
    ImageAspectRatio,
    ImageGenerationTool,
)
from ...exceptions import UserError
from ...settings import ModelSettings

try:
    from openai.types import AllModels, chat, responses
    from openai.types.chat import (
        ChatCompletionChunk,
        chat_completion,
        chat_completion_chunk,
        chat_completion_token_logprob,
    )
    from openai.types.chat.chat_completion_prediction_content_param import ChatCompletionPredictionContentParam
    from openai.types.responses import ComputerToolParam, FileSearchToolParam, WebSearchToolParam
    from openai.types.shared import ReasoningEffort
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'OpenAIModelName',
    'OpenAIChatModelSettings',
    'OpenAIModelSettings',
    'OpenAIResponsesModelSettings',
    'MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME',
    '_OPENAI_ASPECT_RATIO_TO_SIZE',
    '_OPENAI_IMAGE_SIZE',
    '_OPENAI_IMAGE_SIZES',
    '_resolve_openai_image_generation_size',
    '_make_raw_content_updater',
    '_map_logprobs',
    '_map_usage',
    '_map_provider_details',
)

OpenAIModelName = str | AllModels
"""
Possible OpenAI model names.

Since OpenAI supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the OpenAI docs](https://platform.openai.com/docs/models) for a full list.

Using this more broad type for the model name instead of the ChatModel definition
allows this model to be used more easily with other model types (ie, Ollama, Deepseek).
"""

MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME: Literal['x-openai-connector'] = 'x-openai-connector'
"""
Prefix for OpenAI connector IDs. OpenAI supports either a URL or a connector ID when passing MCP configuration to a model,
by using that prefix like `x-openai-connector:<connector-id>` in a URL, you can pass a connector ID to a model.
"""

_OPENAI_ASPECT_RATIO_TO_SIZE: dict[ImageAspectRatio, Literal['1024x1024', '1024x1536', '1536x1024']] = {
    '1:1': '1024x1024',
    '2:3': '1024x1536',
    '3:2': '1536x1024',
}

_OPENAI_IMAGE_SIZE = Literal['auto', '1024x1024', '1024x1536', '1536x1024']
_OPENAI_IMAGE_SIZES: tuple[_OPENAI_IMAGE_SIZE, ...] = _utils.get_args(_OPENAI_IMAGE_SIZE)


def _resolve_openai_image_generation_size(
    tool: ImageGenerationTool,
) -> _OPENAI_IMAGE_SIZE:
    """Map `ImageGenerationTool.aspect_ratio` to an OpenAI size string when provided."""
    aspect_ratio = tool.aspect_ratio
    if aspect_ratio is None:
        if tool.size is None:
            return 'auto'  # default
        if tool.size not in _OPENAI_IMAGE_SIZES:
            raise UserError(
                f'OpenAI image generation only supports `size` values: {_OPENAI_IMAGE_SIZES}. '
                f'Got: {tool.size}. Omit `size` to use the default (auto).'
            )
        return tool.size

    mapped_size = _OPENAI_ASPECT_RATIO_TO_SIZE.get(aspect_ratio)
    if mapped_size is None:
        supported = ', '.join(_OPENAI_ASPECT_RATIO_TO_SIZE)
        raise UserError(
            f'OpenAI image generation only supports `aspect_ratio` values: {supported}. Specify one of those values or omit `aspect_ratio`.'
        )
    # When aspect_ratio is set, size must be None, 'auto', or match the mapped size
    if tool.size not in (None, 'auto', mapped_size):
        raise UserError(
            '`ImageGenerationTool` cannot combine `aspect_ratio` with a conflicting `size` when using OpenAI.'
        )

    return mapped_size


class OpenAIChatModelSettings(ModelSettings, total=False):
    """Settings used for an OpenAI model request."""

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openai_reasoning_effort: ReasoningEffort
    """Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning).

    Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
    result in faster responses and fewer tokens used on reasoning in a response.
    """

    openai_logprobs: bool
    """Include log probabilities in the response.

    For Chat models, these will be included in `ModelResponse.provider_details['logprobs']`.
    For Responses models, these will be included in the response output parts `TextPart.provider_details['logprobs']`.
    """

    openai_top_logprobs: int
    """Include log probabilities of the top n tokens in the response."""

    openai_user: str
    """A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.

    See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.
    """

    openai_service_tier: Literal['auto', 'default', 'flex', 'priority']
    """The service tier to use for the model request.

    Currently supported values are `auto`, `default`, `flex`, and `priority`.
    For more information, see [OpenAI's service tiers documentation](https://platform.openai.com/docs/api-reference/chat/object#chat/object-service_tier).
    """

    openai_prediction: ChatCompletionPredictionContentParam
    """Enables [predictive outputs](https://platform.openai.com/docs/guides/predicted-outputs).

    This feature is currently only supported for some OpenAI models.
    """

    openai_prompt_cache_key: str
    """Used by OpenAI to cache responses for similar requests to optimize your cache hit rates.

    See the [OpenAI Prompt Caching documentation](https://platform.openai.com/docs/guides/prompt-caching#how-it-works) for more information.
    """

    openai_prompt_cache_retention: Literal['in-memory', '24h']
    """The retention policy for the prompt cache. Set to 24h to enable extended prompt caching, which keeps cached prefixes active for longer, up to a maximum of 24 hours.

    See the [OpenAI Prompt Caching documentation](https://platform.openai.com/docs/guides/prompt-caching#how-it-works) for more information.
    """


@deprecated('Use `OpenAIChatModelSettings` instead.')
class OpenAIModelSettings(OpenAIChatModelSettings, total=False):
    """Deprecated alias for `OpenAIChatModelSettings`."""


class OpenAIResponsesModelSettings(OpenAIChatModelSettings, total=False):
    """Settings used for an OpenAI Responses model request.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_builtin_tools: list[FileSearchToolParam | WebSearchToolParam | ComputerToolParam]
    """The provided OpenAI built-in tools to use.

    See [OpenAI's built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses) for more details.
    """

    openai_reasoning_generate_summary: Literal['detailed', 'concise']
    """Deprecated alias for `openai_reasoning_summary`."""

    openai_reasoning_summary: Literal['detailed', 'concise', 'auto']
    """A summary of the reasoning performed by the model.

    This can be useful for debugging and understanding the model's reasoning process.
    One of `concise`, `detailed`, or `auto`.

    Check the [OpenAI Reasoning documentation](https://platform.openai.com/docs/guides/reasoning?api-mode=responses#reasoning-summaries)
    for more details.
    """

    openai_send_reasoning_ids: bool
    """Whether to send the unique IDs of reasoning, text, and function call parts from the message history to the model. Enabled by default for reasoning models.

    This can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."`
    if the message history you're sending does not match exactly what was received from the Responses API in a previous response,
    for example if you're using a [history processor](../../message-history.md#processing-message-history).
    In that case, you'll want to disable this.
    """

    openai_truncation: Literal['disabled', 'auto']
    """The truncation strategy to use for the model response.

    It can be either:
    - `disabled` (default): If a model response will exceed the context window size for a model, the
        request will fail with a 400 error.
    - `auto`: If the context of this response and previous ones exceeds the model's context window size,
        the model will truncate the response to fit the context window by dropping input items in the
        middle of the conversation.
    """

    openai_text_verbosity: Literal['low', 'medium', 'high']
    """Constrains the verbosity of the model's text response.

    Lower values will result in more concise responses, while higher values will
    result in more verbose responses. Currently supported values are `low`,
    `medium`, and `high`.
    """

    openai_previous_response_id: Literal['auto'] | str
    """The ID of a previous response from the model to use as the starting point for a continued conversation.

    When set to `'auto'`, the request automatically uses the most recent
    `provider_response_id` from the message history and omits earlier messages.

    This enables the model to use server-side conversation state and faithfully reference previous reasoning.
    See the [OpenAI Responses API documentation](https://platform.openai.com/docs/guides/reasoning#keeping-reasoning-items-in-context)
    for more information.
    """

    openai_include_code_execution_outputs: bool
    """Whether to include the code execution results in the response.

    Corresponds to the `code_interpreter_call.outputs` value of the `include` parameter in the Responses API.
    """

    openai_include_web_search_sources: bool
    """Whether to include the web search results in the response.

    Corresponds to the `web_search_call.action.sources` value of the `include` parameter in the Responses API.
    """

    openai_include_file_search_results: bool
    """Whether to include the file search results in the response.

    Corresponds to the `file_search_call.results` value of the `include` parameter in the Responses API.
    """


# Helper functions used by both completions and responses modules


def _make_raw_content_updater(
    delta: str, index: int
) -> Any:  # Returns Callable[[dict[str, Any] | None], dict[str, Any]]
    """Create a callback that updates `provider_details['raw_content']`.

    This is used for streaming raw CoT from gpt-oss models. The callback pattern keeps
    `raw_content` logic in OpenAI code while the parts manager stays provider-agnostic.
    """

    def update_provider_details(existing: dict[str, Any] | None) -> dict[str, Any]:
        details = {**(existing or {})}
        raw_list: list[str] = list(details.get('raw_content', []))
        while len(raw_list) <= index:
            raw_list.append('')
        raw_list[index] += delta
        details['raw_content'] = raw_list
        return details

    return update_provider_details


def _map_logprobs(
    logprobs: list[chat_completion_token_logprob.ChatCompletionTokenLogprob]
    | list[responses.response_output_text.Logprob],
) -> list[dict[str, Any]]:
    """Convert logprobs to a serializable format."""
    return [
        {
            'token': lp.token,
            'bytes': lp.bytes,
            'logprob': lp.logprob,
            'top_logprobs': [
                {'token': tlp.token, 'bytes': tlp.bytes, 'logprob': tlp.logprob} for tlp in lp.top_logprobs
            ],
        }
        for lp in logprobs
    ]


def _map_usage(
    response: chat.ChatCompletion | ChatCompletionChunk | responses.Response,
    provider: str,
    provider_url: str,
    model: str,
) -> usage.RequestUsage:
    response_usage = response.usage
    if response_usage is None:
        return usage.RequestUsage()

    usage_data = response_usage.model_dump(exclude_none=True)
    details = {
        k: v
        for k, v in usage_data.items()
        if k not in {'prompt_tokens', 'completion_tokens', 'input_tokens', 'output_tokens', 'total_tokens'}
        if isinstance(v, int)
    }
    response_data = dict(model=model, usage=usage_data)
    if isinstance(response_usage, responses.ResponseUsage):
        api_flavor = 'responses'

        if getattr(response_usage, 'output_tokens_details', None) is not None:
            details['reasoning_tokens'] = response_usage.output_tokens_details.reasoning_tokens
        else:
            details['reasoning_tokens'] = 0
    else:
        api_flavor = 'chat'

        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))

    return usage.RequestUsage.extract(
        response_data,
        provider=provider,
        provider_url=provider_url,
        provider_fallback='openai',
        api_flavor=api_flavor,
        details=details,
    )


def _map_provider_details(
    choice: chat_completion_chunk.Choice | chat_completion.Choice,
) -> dict[str, Any] | None:
    """Map provider details from a chat completion choice."""
    provider_details: dict[str, Any] = {}

    # Add logprobs to vendor_details if available
    if choice.logprobs is not None and choice.logprobs.content:
        provider_details['logprobs'] = _map_logprobs(choice.logprobs.content)
    if raw_finish_reason := choice.finish_reason:
        provider_details['finish_reason'] = raw_finish_reason

    return provider_details or None
