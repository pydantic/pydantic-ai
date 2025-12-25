from __future__ import annotations as _annotations

import base64
import json
import warnings
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, cast, overload

from pydantic_core import to_json
from typing_extensions import assert_never

from ... import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from ..._output import DEFAULT_OUTPUT_TOOL_NAME, OutputObjectDefinition
from ..._run_context import RunContext
from ..._utils import guard_tool_call_id as _guard_tool_call_id, now_utc as _now_utc, number_to_datetime
from ...builtin_tools import (
    AbstractBuiltinTool,
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    MCPServerTool,
    WebSearchTool,
)
from ...exceptions import UserError
from ...messages import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FilePart,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ...profiles import ModelProfileSpec
from ...profiles.openai import OpenAIModelProfile
from ...providers import Provider, infer_provider
from ...settings import ModelSettings
from ...tools import ToolDefinition
from .. import (
    Model,
    ModelRequestParameters,
    OpenAIResponsesCompatibleProvider,
    StreamedResponse,
    check_allow_model_requests,
    download_item,
    get_user_agent,
)
from ._shared import (
    MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME,
    OpenAIModelName,
    OpenAIResponsesModelSettings,
    _make_raw_content_updater,  # pyright: ignore[reportPrivateUsage]
    _map_logprobs,  # pyright: ignore[reportPrivateUsage]
    _map_usage,  # pyright: ignore[reportPrivateUsage]
    _resolve_openai_image_generation_size,  # pyright: ignore[reportPrivateUsage]
)

try:
    from openai import NOT_GIVEN, APIConnectionError, APIStatusError, AsyncOpenAI, AsyncStream
    from openai._types import Omit
    from openai.types import responses
    from openai.types.responses.response_input_param import FunctionCallOutput, Message
    from openai.types.responses.response_reasoning_item_param import (
        Content as ReasoningContent,
        Summary as ReasoningSummary,
    )
    from openai.types.shared_params import Reasoning

    def OMIT() -> Omit:
        """Get the omit sentinel value from openai."""
        from openai import omit

        return omit

except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

_RESPONSES_FINISH_REASON_MAP: dict[
    Literal['max_output_tokens', 'content_filter'] | responses.ResponseStatus, FinishReason
] = {
    'max_output_tokens': 'length',
    'content_filter': 'content_filter',
    'completed': 'stop',
    'cancelled': 'error',
    'failed': 'error',
}


@dataclass(init=False)
class OpenAIResponsesModel(Model):
    """A model that uses the OpenAI Responses API.

    The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) is the
    new API for OpenAI models.

    If you are interested in the differences between the Responses API and the Chat Completions API,
    see the [OpenAI API docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions).
    """

    client: AsyncOpenAI = field(repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: OpenAIResponsesCompatibleProvider
        | Literal[
            'openai',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Responses model.

        Args:
            model_name: The name of the OpenAI model to use.
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type[AbstractBuiltinTool]]:
        """Return the set of builtin tool types this model can handle."""
        return frozenset({WebSearchTool, CodeExecutionTool, FileSearchTool, MCPServerTool, ImageGenerationTool})

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._responses_create(
            messages, False, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        return self._process_response(response, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._responses_create(
            messages, True, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    def _process_response(  # noqa: C901
        self, response: responses.Response, model_request_parameters: ModelRequestParameters
    ) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        items: list[ModelResponsePart] = []
        for item in response.output:
            if isinstance(item, responses.ResponseReasoningItem):
                signature = item.encrypted_content
                # Handle raw CoT content from gpt-oss models
                provider_details: dict[str, Any] = {}
                raw_content: list[str] | None = [c.text for c in item.content] if item.content else None
                if raw_content:
                    provider_details['raw_content'] = raw_content

                if item.summary:
                    for summary in item.summary:
                        # We use the same id for all summaries so that we can merge them on the round trip.
                        items.append(
                            ThinkingPart(
                                content=summary.text,
                                id=item.id,
                                signature=signature,
                                provider_name=self.system if (signature or provider_details) else None,
                                provider_details=provider_details or None,
                            )
                        )
                        # We only need to store the signature and raw_content once.
                        signature = None
                        provider_details = None
                elif signature or provider_details:
                    items.append(
                        ThinkingPart(
                            content='',
                            id=item.id,
                            signature=signature,
                            provider_name=self.system if (signature or provider_details) else None,
                            provider_details=provider_details or None,
                        )
                    )
            elif isinstance(item, responses.ResponseOutputMessage):
                for content in item.content:
                    if isinstance(content, responses.ResponseOutputText):  # pragma: no branch
                        part_provider_details: dict[str, Any] | None = None
                        if content.logprobs:
                            part_provider_details = {'logprobs': _map_logprobs(content.logprobs)}
                        items.append(TextPart(content.text, id=item.id, provider_details=part_provider_details))
            elif isinstance(item, responses.ResponseFunctionToolCall):
                items.append(
                    ToolCallPart(
                        item.name,
                        item.arguments,
                        tool_call_id=item.call_id,
                        id=item.id,
                    )
                )
            elif isinstance(item, responses.ResponseCodeInterpreterToolCall):
                call_part, return_part, file_parts = _map_code_interpreter_tool_call(item, self.system)
                items.append(call_part)
                if file_parts:
                    items.extend(file_parts)
                items.append(return_part)
            elif isinstance(item, responses.ResponseFunctionWebSearch):
                call_part, return_part = _map_web_search_tool_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.ImageGenerationCall):
                call_part, return_part, file_part = _map_image_generation_tool_call(item, self.system)
                items.append(call_part)
                if file_part:  # pragma: no branch
                    items.append(file_part)
                items.append(return_part)
            elif isinstance(item, responses.ResponseComputerToolCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the ComputerUse built-in tool
                pass
            elif isinstance(item, responses.ResponseCustomToolCall):  # pragma: no cover
                # Support is being implemented in https://github.com/pydantic/pydantic-ai/pull/2572
                pass
            elif isinstance(item, responses.response_output_item.LocalShellCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the `codex-mini-latest` LocalShell built-in tool
                pass
            elif isinstance(item, responses.ResponseFileSearchToolCall):
                call_part, return_part = _map_file_search_tool_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpCall):
                call_part, return_part = _map_mcp_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpListTools):
                call_part, return_part = _map_mcp_list_tools(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpApprovalRequest):  # pragma: no cover
                # Pydantic AI doesn't yet support McpApprovalRequest (explicit tool usage approval)
                pass

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] = {}
        raw_finish_reason = details.reason if (details := response.incomplete_details) else response.status
        if raw_finish_reason:
            provider_details['finish_reason'] = raw_finish_reason
            finish_reason = _RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)
        if response.created_at:  # pragma: no branch
            provider_details['timestamp'] = number_to_datetime(response.created_at)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response, self._provider.name, self._provider.base_url, self.model_name),
            model_name=response.model,
            provider_response_id=response.id,
            timestamp=_now_utc(),
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            finish_reason=finish_reason,
            provider_details=provider_details or None,
        )

    async def _process_streamed_response(
        self,
        response: AsyncStream[responses.ResponseStreamEvent],
        model_request_parameters: ModelRequestParameters,
    ) -> OpenAIResponsesStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        assert isinstance(first_chunk, responses.ResponseCreatedEvent)
        return OpenAIResponsesStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.response.model,
            _response=peekable_response,
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
            _provider_timestamp=number_to_datetime(first_chunk.response.created_at)
            if first_chunk.response.created_at
            else None,
        )

    @overload
    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[False],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response: ...

    @overload
    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[True],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[responses.ResponseStreamEvent]: ...

    async def _responses_create(  # noqa: C901
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent]:
        tools = (
            self._get_builtin_tools(model_request_parameters)
            + list(model_settings.get('openai_builtin_tools', []))
            + self._get_tools(model_request_parameters)
        )
        profile = OpenAIModelProfile.from_profile(self.profile)
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output and profile.openai_supports_tool_choice_required:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        previous_response_id = model_settings.get('openai_previous_response_id')
        if previous_response_id == 'auto':
            previous_response_id, messages = self._get_previous_response_id_and_new_messages(messages)

        instructions, openai_messages = await self._map_messages(messages, model_settings, model_request_parameters)
        reasoning = self._get_reasoning(model_settings)

        omit = OMIT()
        text: responses.ResponseTextConfigParam | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            text = {'format': self._map_json_schema(output_object)}
        elif (
            model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output
        ):  # pragma: no branch
            text = {'format': {'type': 'json_object'}}

            # Without this trick, we'd hit this error:
            # > Response input messages must contain the word 'json' in some form to use 'text.format' of type 'json_object'.
            # Apparently they're only checking input messages for "JSON", not instructions.
            assert isinstance(instructions, str)
            system_prompt_count = sum(1 for m in openai_messages if m.get('role') == 'system')
            openai_messages.insert(
                system_prompt_count, responses.EasyInputMessageParam(role='system', content=instructions)
            )
            instructions = omit

        if verbosity := model_settings.get('openai_text_verbosity'):
            text = text or {}
            text['verbosity'] = verbosity

        unsupported_model_settings = profile.openai_unsupported_model_settings
        for setting in unsupported_model_settings:
            model_settings.pop(setting, None)

        include: list[responses.ResponseIncludable] = []
        if profile.openai_supports_encrypted_reasoning_content:
            include.append('reasoning.encrypted_content')
        if model_settings.get('openai_include_code_execution_outputs'):
            include.append('code_interpreter_call.outputs')
        if model_settings.get('openai_include_web_search_sources'):
            include.append('web_search_call.action.sources')
        if model_settings.get('openai_include_file_search_results'):
            include.append('file_search_call.results')
        if model_settings.get('openai_logprobs'):
            include.append('message.output_text.logprobs')

        # When there are no input messages and we're not reusing a previous response,
        # the OpenAI API will reject a request without any input,
        # even if there are instructions.
        # To avoid this provide an explicit empty user message.
        if not openai_messages and not previous_response_id:
            openai_messages.append(
                responses.EasyInputMessageParam(
                    role='user',
                    content='',
                )
            )

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.responses.create(
                input=openai_messages,
                model=self.model_name,
                instructions=instructions,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', omit),
                tools=tools or omit,
                tool_choice=tool_choice or omit,
                max_output_tokens=model_settings.get('max_tokens', omit),
                stream=stream,
                temperature=model_settings.get('temperature', omit),
                top_p=model_settings.get('top_p', omit),
                truncation=model_settings.get('openai_truncation', omit),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                service_tier=model_settings.get('openai_service_tier', omit),
                previous_response_id=previous_response_id or omit,
                top_logprobs=model_settings.get('openai_top_logprobs', omit),
                reasoning=reasoning,
                user=model_settings.get('openai_user', omit),
                text=text or omit,
                include=include or omit,
                prompt_cache_key=model_settings.get('openai_prompt_cache_key', omit),
                prompt_cache_retention=model_settings.get('openai_prompt_cache_retention', omit),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

    def _get_reasoning(self, model_settings: OpenAIResponsesModelSettings) -> Reasoning | Omit:
        reasoning_effort = model_settings.get('openai_reasoning_effort', None)
        reasoning_summary = model_settings.get('openai_reasoning_summary', None)
        reasoning_generate_summary = model_settings.get('openai_reasoning_generate_summary', None)

        if reasoning_summary and reasoning_generate_summary:  # pragma: no cover
            raise ValueError('`openai_reasoning_summary` and `openai_reasoning_generate_summary` cannot both be set.')

        if reasoning_generate_summary is not None:  # pragma: no cover
            warnings.warn(
                '`openai_reasoning_generate_summary` is deprecated, use `openai_reasoning_summary` instead',
                DeprecationWarning,
            )
            reasoning_summary = reasoning_generate_summary

        omit = OMIT()
        reasoning: Reasoning = {}
        if reasoning_effort:
            reasoning['effort'] = reasoning_effort
        if reasoning_summary:
            reasoning['summary'] = reasoning_summary
        return reasoning or omit

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.FunctionToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_builtin_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.ToolParam]:
        tools: list[responses.ToolParam] = []
        has_image_generating_tool = False
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                web_search_tool = responses.WebSearchToolParam(
                    type='web_search', search_context_size=tool.search_context_size
                )
                if tool.user_location:
                    web_search_tool['user_location'] = responses.web_search_tool_param.UserLocation(
                        type='approximate', **tool.user_location
                    )
                tools.append(web_search_tool)
            elif isinstance(tool, FileSearchTool):
                file_search_tool = cast(
                    responses.FileSearchToolParam,
                    {'type': 'file_search', 'vector_store_ids': list(tool.file_store_ids)},
                )
                tools.append(file_search_tool)
            elif isinstance(tool, CodeExecutionTool):
                has_image_generating_tool = True
                tools.append({'type': 'code_interpreter', 'container': {'type': 'auto'}})
            elif isinstance(tool, MCPServerTool):
                mcp_tool = responses.tool_param.Mcp(
                    type='mcp',
                    server_label=tool.id,
                    require_approval='never',
                )

                if tool.authorization_token:  # pragma: no branch
                    mcp_tool['authorization'] = tool.authorization_token

                if tool.allowed_tools is not None:  # pragma: no branch
                    mcp_tool['allowed_tools'] = tool.allowed_tools

                if tool.description:  # pragma: no branch
                    mcp_tool['server_description'] = tool.description

                if tool.headers:  # pragma: no branch
                    mcp_tool['headers'] = tool.headers

                if tool.url.startswith(MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME + ':'):
                    _, connector_id = tool.url.split(':', maxsplit=1)
                    mcp_tool['connector_id'] = connector_id  # pyright: ignore[reportGeneralTypeIssues]
                else:
                    mcp_tool['server_url'] = tool.url

                tools.append(mcp_tool)
            elif isinstance(tool, ImageGenerationTool):  # pragma: no branch
                has_image_generating_tool = True
                size = _resolve_openai_image_generation_size(tool)
                output_compression = tool.output_compression if tool.output_compression is not None else 100
                tools.append(
                    responses.tool_param.ImageGeneration(
                        type='image_generation',
                        background=tool.background,
                        input_fidelity=tool.input_fidelity,
                        moderation=tool.moderation,
                        output_compression=output_compression,
                        output_format=tool.output_format or 'png',
                        partial_images=tool.partial_images,
                        quality=tool.quality,
                        size=size,
                    )
                )
            else:
                raise UserError(  # pragma: no cover
                    f'`{tool.__class__.__name__}` is not supported by `OpenAIResponsesModel`. If it should be, please file an issue.'
                )

        if model_request_parameters.allow_image_output and not has_image_generating_tool:
            tools.append({'type': 'image_generation'})
        return tools

    def _map_tool_definition(self, f: ToolDefinition) -> responses.FunctionToolParam:
        return {
            'name': f.name,
            'parameters': f.parameters_json_schema,
            'type': 'function',
            'description': f.description,
            'strict': bool(
                f.strict and OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition
            ),
        }

    def _get_previous_response_id_and_new_messages(
        self, messages: list[ModelMessage]
    ) -> tuple[str | None, list[ModelMessage]]:
        # When `openai_previous_response_id` is set to 'auto', the most recent
        # `provider_response_id` from the message history is selected and all
        # earlier messages are omitted. This allows the OpenAI SDK to reuse
        # server-side history for efficiency. The returned tuple contains the
        # `previous_response_id` (if found) and the trimmed list of messages.
        previous_response_id = None
        trimmed_messages: list[ModelMessage] = []
        for m in reversed(messages):
            if isinstance(m, ModelResponse) and m.provider_name == self.system:
                previous_response_id = m.provider_response_id
                break
            else:
                trimmed_messages.append(m)

        if previous_response_id and trimmed_messages:
            return previous_response_id, list(reversed(trimmed_messages))
        else:
            return None, messages

    async def _map_messages(  # noqa: C901
        self,
        messages: list[ModelMessage],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[str | Omit, list[responses.ResponseInputItemParam]]:
        """Maps a `pydantic_ai.Message` to a `openai.types.responses.ResponseInputParam` i.e. the OpenAI Responses API input format.

        For `ThinkingParts`, this method:
        - Sends `signature` back as `encrypted_content` (for official OpenAI reasoning)
        - Sends `content` back as `summary` text
        - Sends `provider_details['raw_content']` back as `content` items (for gpt-oss raw CoT)

        Raw CoT is sent back to improve model performance in multi-turn conversations.
        """
        omit = OMIT()
        profile = OpenAIModelProfile.from_profile(self.profile)
        send_item_ids = model_settings.get(
            'openai_send_reasoning_ids', profile.openai_supports_encrypted_reasoning_content
        )

        openai_messages: list[responses.ResponseInputItemParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        openai_messages.append(responses.EasyInputMessageParam(role='system', content=part.content))
                    elif isinstance(part, UserPromptPart):
                        openai_messages.append(await self._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        call_id = _guard_tool_call_id(t=part)
                        call_id, _ = _split_combined_tool_call_id(call_id)
                        item = FunctionCallOutput(
                            type='function_call_output',
                            call_id=call_id,
                            output=part.model_response_str(),
                        )
                        openai_messages.append(item)
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            openai_messages.append(
                                Message(role='user', content=[{'type': 'input_text', 'text': part.model_response()}])
                            )
                        else:
                            call_id = _guard_tool_call_id(t=part)
                            call_id, _ = _split_combined_tool_call_id(call_id)
                            item = FunctionCallOutput(
                                type='function_call_output',
                                call_id=call_id,
                                output=part.model_response(),
                            )
                            openai_messages.append(item)
                    else:
                        assert_never(part)
            elif isinstance(message, ModelResponse):
                send_item_ids = send_item_ids and message.provider_name == self.system

                message_item: responses.ResponseOutputMessageParam | None = None
                reasoning_item: responses.ResponseReasoningItemParam | None = None
                web_search_item: responses.ResponseFunctionWebSearchParam | None = None
                file_search_item: responses.ResponseFileSearchToolCallParam | None = None
                code_interpreter_item: responses.ResponseCodeInterpreterToolCallParam | None = None
                for item in message.parts:
                    if isinstance(item, TextPart):
                        if item.id and send_item_ids:
                            if message_item is None or message_item['id'] != item.id:  # pragma: no branch
                                message_item = responses.ResponseOutputMessageParam(
                                    role='assistant',
                                    id=item.id,
                                    content=[],
                                    type='message',
                                    status='completed',
                                )
                                openai_messages.append(message_item)

                            message_item['content'] = [
                                *message_item['content'],
                                responses.ResponseOutputTextParam(
                                    text=item.content, type='output_text', annotations=[]
                                ),
                            ]
                        else:
                            openai_messages.append(
                                responses.EasyInputMessageParam(role='assistant', content=item.content)
                            )
                    elif isinstance(item, ToolCallPart):
                        call_id = _guard_tool_call_id(t=item)
                        call_id, id = _split_combined_tool_call_id(call_id)
                        id = id or item.id

                        param = responses.ResponseFunctionToolCallParam(
                            name=item.tool_name,
                            arguments=item.args_as_json_str(),
                            call_id=call_id,
                            type='function_call',
                        )
                        if profile.openai_responses_requires_function_call_status_none:
                            param['status'] = None  # type: ignore[reportGeneralTypeIssues]
                        if id and send_item_ids:  # pragma: no branch
                            param['id'] = id
                        openai_messages.append(param)
                    elif isinstance(item, BuiltinToolCallPart):
                        if item.provider_name == self.system and send_item_ids:  # pragma: no branch
                            if (
                                item.tool_name == CodeExecutionTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                                and (container_id := args.get('container_id'))
                            ):
                                code_interpreter_item = responses.ResponseCodeInterpreterToolCallParam(
                                    id=item.tool_call_id,
                                    code=args.get('code'),
                                    container_id=container_id,
                                    outputs=None,  # These can be read server-side
                                    status='completed',
                                    type='code_interpreter_call',
                                )
                                openai_messages.append(code_interpreter_item)
                            elif (
                                item.tool_name == WebSearchTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                            ):
                                # We need to exclude None values because of https://github.com/pydantic/pydantic-ai/issues/3653
                                args = {k: v for k, v in args.items() if v is not None}
                                web_search_item = responses.ResponseFunctionWebSearchParam(
                                    id=item.tool_call_id,
                                    action=cast(responses.response_function_web_search_param.Action, args),
                                    status='completed',
                                    type='web_search_call',
                                )
                                openai_messages.append(web_search_item)
                            elif (  # pragma: no cover
                                item.tool_name == FileSearchTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                            ):
                                file_search_item = cast(
                                    responses.ResponseFileSearchToolCallParam,
                                    {
                                        'id': item.tool_call_id,
                                        'queries': args.get('queries', []),
                                        'status': 'completed',
                                        'type': 'file_search_call',
                                    },
                                )
                                openai_messages.append(file_search_item)
                            elif item.tool_name == ImageGenerationTool.kind and item.tool_call_id:
                                # The cast is necessary because of https://github.com/openai/openai-python/issues/2648
                                image_generation_item = cast(
                                    responses.response_input_item_param.ImageGenerationCall,
                                    {
                                        'id': item.tool_call_id,
                                        'type': 'image_generation_call',
                                    },
                                )
                                openai_messages.append(image_generation_item)
                            elif (  # pragma: no branch
                                item.tool_name.startswith(MCPServerTool.kind)
                                and item.tool_call_id
                                and (server_id := item.tool_name.split(':', 1)[1])
                                and (args := item.args_as_dict())
                                and (action := args.get('action'))
                            ):
                                if action == 'list_tools':
                                    mcp_list_tools_item = responses.response_input_item_param.McpListTools(
                                        id=item.tool_call_id,
                                        type='mcp_list_tools',
                                        server_label=server_id,
                                        tools=[],  # These can be read server-side
                                    )
                                    openai_messages.append(mcp_list_tools_item)
                                elif (  # pragma: no branch
                                    action == 'call_tool'
                                    and (tool_name := args.get('tool_name'))
                                    and (tool_args := args.get('tool_args'))
                                ):
                                    mcp_call_item = responses.response_input_item_param.McpCall(
                                        id=item.tool_call_id,
                                        server_label=server_id,
                                        name=tool_name,
                                        arguments=to_json(tool_args).decode(),
                                        error=None,  # These can be read server-side
                                        output=None,  # These can be read server-side
                                        type='mcp_call',
                                    )
                                    openai_messages.append(mcp_call_item)

                    elif isinstance(item, BuiltinToolReturnPart):
                        if item.provider_name == self.system and send_item_ids:  # pragma: no branch
                            if (
                                item.tool_name == CodeExecutionTool.kind
                                and code_interpreter_item is not None
                                and isinstance(item.content, dict)
                                and (content := cast(dict[str, Any], item.content))  # pyright: ignore[reportUnknownMemberType]
                                and (status := content.get('status'))
                            ):
                                code_interpreter_item['status'] = status
                            elif (
                                item.tool_name == WebSearchTool.kind
                                and web_search_item is not None
                                and isinstance(item.content, dict)  # pyright: ignore[reportUnknownMemberType]
                                and (content := cast(dict[str, Any], item.content))  # pyright: ignore[reportUnknownMemberType]
                                and (status := content.get('status'))
                            ):
                                web_search_item['status'] = status
                            elif (  # pragma: no cover
                                item.tool_name == FileSearchTool.kind
                                and file_search_item is not None
                                and isinstance(item.content, dict)  # pyright: ignore[reportUnknownMemberType]
                                and (content := cast(dict[str, Any], item.content))  # pyright: ignore[reportUnknownMemberType]
                                and (status := content.get('status'))
                            ):
                                file_search_item['status'] = status
                            elif item.tool_name == ImageGenerationTool.kind:
                                # Image generation result does not need to be sent back, just the `id` off of `BuiltinToolCallPart`.
                                pass
                            elif item.tool_name.startswith(MCPServerTool.kind):  # pragma: no branch
                                # MCP call result does not need to be sent back, just the fields off of `BuiltinToolCallPart`.
                                pass
                    elif isinstance(item, FilePart):
                        # This was generated by the `ImageGenerationTool` or `CodeExecutionTool`,
                        # and does not need to be sent back separately from the corresponding `BuiltinToolReturnPart`.
                        # If `send_item_ids` is false, we won't send the `BuiltinToolReturnPart`, but OpenAI does not have a type for files from the assistant.
                        pass
                    elif isinstance(item, ThinkingPart):
                        # Get raw CoT content from provider_details if present and from this provider
                        raw_content: list[str] | None = None
                        if item.provider_name == self.system:
                            raw_content = (item.provider_details or {}).get('raw_content')

                        if item.id and (send_item_ids or raw_content):
                            signature: str | None = None
                            if (
                                item.signature
                                and item.provider_name == self.system
                                and profile.openai_supports_encrypted_reasoning_content
                            ):
                                signature = item.signature

                            if (reasoning_item is None or reasoning_item['id'] != item.id) and (
                                signature or item.content or raw_content
                            ):  # pragma: no branch
                                reasoning_item = responses.ResponseReasoningItemParam(
                                    id=item.id,
                                    summary=[],
                                    encrypted_content=signature,
                                    type='reasoning',
                                )
                                openai_messages.append(reasoning_item)

                            if item.content:
                                # The check above guarantees that `reasoning_item` is not None
                                assert reasoning_item is not None
                                reasoning_item['summary'] = [
                                    *reasoning_item['summary'],
                                    ReasoningSummary(text=item.content, type='summary_text'),
                                ]

                            if raw_content:
                                # Send raw CoT back
                                assert reasoning_item is not None
                                reasoning_item['content'] = [
                                    ReasoningContent(text=text, type='reasoning_text') for text in raw_content
                                ]
                        else:
                            start_tag, end_tag = profile.thinking_tags
                            openai_messages.append(
                                responses.EasyInputMessageParam(
                                    role='assistant', content='\n'.join([start_tag, item.content, end_tag])
                                )
                            )
                    else:
                        assert_never(item)
            else:
                assert_never(message)
        instructions = self._get_instructions(messages, model_request_parameters) or omit
        return instructions, openai_messages

    def _map_json_schema(self, o: OutputObjectDefinition) -> responses.ResponseFormatTextJSONSchemaConfigParam:
        response_format_param: responses.ResponseFormatTextJSONSchemaConfigParam = {
            'type': 'json_schema',
            'name': o.name or DEFAULT_OUTPUT_TOOL_NAME,
            'schema': o.json_schema,
        }
        if o.description:
            response_format_param['description'] = o.description
        if OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:  # pragma: no branch
            response_format_param['strict'] = o.strict
        return response_format_param

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> responses.EasyInputMessageParam:  # noqa: C901
        content: str | list[responses.ResponseInputContentParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(responses.ResponseInputTextParam(text=item, type='input_text'))
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        detail: Literal['auto', 'low', 'high'] = 'auto'
                        if metadata := item.vendor_metadata:
                            detail = cast(
                                Literal['auto', 'low', 'high'],
                                metadata.get('detail', 'auto'),
                            )
                        content.append(
                            responses.ResponseInputImageParam(
                                image_url=item.data_uri,
                                type='input_image',
                                detail=detail,
                            )
                        )
                    elif item.is_document:
                        content.append(
                            responses.ResponseInputFileParam(
                                type='input_file',
                                file_data=item.data_uri,
                                # NOTE: Type wise it's not necessary to include the filename, but it's required by the
                                # API itself. If we add empty string, the server sends a 500 error - which OpenAI needs
                                # to fix. In any case, we add a placeholder name.
                                filename=f'filename.{item.format}',
                            )
                        )
                    elif item.is_audio:
                        raise NotImplementedError('Audio as binary content is not supported for OpenAI Responses API.')
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, ImageUrl):
                    detail: Literal['auto', 'low', 'high'] = 'auto'
                    image_url = item.url
                    if metadata := item.vendor_metadata:
                        detail = cast(Literal['auto', 'low', 'high'], metadata.get('detail', 'auto'))
                    if item.force_download:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        image_url = downloaded_item['data']

                    content.append(
                        responses.ResponseInputImageParam(
                            image_url=image_url,
                            type='input_image',
                            detail=detail,
                        )
                    )
                elif isinstance(item, AudioUrl | DocumentUrl):
                    if item.force_download:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        content.append(
                            responses.ResponseInputFileParam(
                                type='input_file',
                                file_data=downloaded_item['data'],
                                filename=f'filename.{downloaded_item["data_type"]}',
                            )
                        )
                    else:
                        content.append(
                            responses.ResponseInputFileParam(
                                type='input_file',
                                file_url=item.url,
                            )
                        )
                elif isinstance(item, VideoUrl):  # pragma: no cover
                    raise NotImplementedError('VideoUrl is not supported for OpenAI.')
                elif isinstance(item, CachePoint):
                    # OpenAI doesn't support prompt caching via CachePoint, so we filter it out
                    pass
                else:
                    assert_never(item)
        return responses.EasyInputMessageParam(role='user', content=content)


@dataclass
class OpenAIResponsesStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI Responses API."""

    _model_name: OpenAIModelName
    _response: AsyncIterable[responses.ResponseStreamEvent]
    _provider_name: str
    _provider_url: str
    _provider_timestamp: datetime | None = None
    _timestamp: datetime = field(default_factory=_now_utc)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        if self._provider_timestamp is not None:  # pragma: no branch
            self.provider_details = {'timestamp': self._provider_timestamp}

        async for chunk in self._response:
            # NOTE: You can inspect the builtin tools used checking the `ResponseCompletedEvent`.
            if isinstance(chunk, responses.ResponseCompletedEvent):
                self._usage += self._map_usage(chunk.response)

                raw_finish_reason = (
                    details.reason if (details := chunk.response.incomplete_details) else chunk.response.status
                )
                if raw_finish_reason:  # pragma: no branch
                    self.provider_details = {**(self.provider_details or {}), 'finish_reason': raw_finish_reason}
                    self.finish_reason = _RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)

            elif isinstance(chunk, responses.ResponseContentPartAddedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseContentPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCreatedEvent):
                if chunk.response.id:  # pragma: no branch
                    self.provider_response_id = chunk.response.id

            elif isinstance(chunk, responses.ResponseFailedEvent):  # pragma: no cover
                self._usage += self._map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDeltaEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=chunk.item_id,
                    args=chunk.delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseIncompleteEvent):  # pragma: no cover
                self._usage += self._map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseInProgressEvent):
                self._usage += self._map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseOutputItemAddedEvent):
                if isinstance(chunk.item, responses.ResponseFunctionToolCall):
                    yield self._parts_manager.handle_tool_call_part(
                        vendor_part_id=chunk.item.id,
                        tool_name=chunk.item.name,
                        args=chunk.item.arguments,
                        tool_call_id=chunk.item.call_id,
                        id=chunk.item.id,
                    )
                elif isinstance(chunk.item, responses.ResponseReasoningItem):
                    pass
                elif isinstance(chunk.item, responses.ResponseOutputMessage):
                    pass
                elif isinstance(chunk.item, responses.ResponseFunctionWebSearch):
                    call_part, _ = _map_web_search_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                elif isinstance(chunk.item, responses.ResponseFileSearchToolCall):
                    call_part, _ = _map_file_search_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                elif isinstance(chunk.item, responses.ResponseCodeInterpreterToolCall):
                    call_part, _, _ = _map_code_interpreter_tool_call(chunk.item, self.provider_name)

                    args_json = call_part.args_as_json_str()
                    # Drop the final `"}` so that we can add code deltas
                    args_json_delta = args_json[:-2]
                    assert args_json_delta.endswith('"code":"'), f'Expected {args_json_delta!r} to end in `"code":"`'

                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=args_json_delta,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(chunk.item, responses.response_output_item.ImageGenerationCall):
                    call_part, _, _ = _map_image_generation_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-call', part=call_part)
                elif isinstance(chunk.item, responses.response_output_item.McpCall):
                    call_part, _ = _map_mcp_call(chunk.item, self.provider_name)

                    args_json = call_part.args_as_json_str()
                    # Drop the final `{}}` so that we can add tool args deltas
                    args_json_delta = args_json[:-3]
                    assert args_json_delta.endswith('"tool_args":'), (
                        f'Expected {args_json_delta!r} to end in `"tool_args":"`'
                    )

                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=args_json_delta,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(chunk.item, responses.response_output_item.McpListTools):
                    call_part, _ = _map_mcp_list_tools(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-call', part=call_part)
                else:
                    warnings.warn(  # pragma: no cover
                        f'Handling of this item type is not yet implemented. Please report on our GitHub: {chunk}',
                        UserWarning,
                    )

            elif isinstance(chunk, responses.ResponseOutputItemDoneEvent):
                if isinstance(chunk.item, responses.ResponseReasoningItem):
                    if signature := chunk.item.encrypted_content:  # pragma: no branch
                        # Add the signature to the part corresponding to the first summary/raw CoT
                        for event in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=chunk.item.id,
                            id=chunk.item.id,
                            signature=signature,
                            provider_name=self.provider_name,
                        ):
                            yield event
                elif isinstance(chunk.item, responses.ResponseCodeInterpreterToolCall):
                    _, return_part, file_parts = _map_code_interpreter_tool_call(chunk.item, self.provider_name)
                    for i, file_part in enumerate(file_parts):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=f'{chunk.item.id}-file-{i}', part=file_part
                        )
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.ResponseFunctionWebSearch):
                    call_part, return_part = _map_web_search_tool_call(chunk.item, self.provider_name)

                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=call_part.args,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.ResponseFileSearchToolCall):
                    call_part, return_part = _map_file_search_tool_call(chunk.item, self.provider_name)

                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=call_part.args,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.response_output_item.ImageGenerationCall):
                    _, return_part, file_part = _map_image_generation_tool_call(chunk.item, self.provider_name)
                    if file_part:  # pragma: no branch
                        yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-file', part=file_part)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)

                elif isinstance(chunk.item, responses.response_output_item.McpCall):
                    _, return_part = _map_mcp_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.response_output_item.McpListTools):
                    _, return_part = _map_mcp_list_tools(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartAddedEvent):
                # Use same vendor_part_id as raw CoT for first summary (index 0) so they merge into one ThinkingPart
                vendor_id = chunk.item_id if chunk.summary_index == 0 else f'{chunk.item_id}-{chunk.summary_index}'
                for event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id=vendor_id,
                    content=chunk.part.text,
                    id=chunk.item_id,
                ):
                    yield event

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDeltaEvent):
                # Use same vendor_part_id as raw CoT for first summary (index 0) so they merge into one ThinkingPart
                vendor_id = chunk.item_id if chunk.summary_index == 0 else f'{chunk.item_id}-{chunk.summary_index}'
                for event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id=vendor_id,
                    content=chunk.delta,
                    id=chunk.item_id,
                ):
                    yield event

            elif isinstance(chunk, responses.ResponseReasoningTextDeltaEvent):
                # Handle raw CoT from gpt-oss models using callback pattern
                for event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id=chunk.item_id,
                    id=chunk.item_id,
                    provider_details=_make_raw_content_updater(chunk.delta, chunk.content_index),
                ):
                    yield event

            elif isinstance(chunk, responses.ResponseReasoningTextDoneEvent):
                pass  # content already accumulated via delta events

            elif isinstance(chunk, responses.ResponseOutputTextAnnotationAddedEvent):
                # TODO(Marcelo): We should support annotations in the future.
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseTextDeltaEvent):
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id=chunk.item_id, content=chunk.delta, id=chunk.item_id
                ):
                    yield event

            elif isinstance(chunk, responses.ResponseTextDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallSearchingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseAudioDeltaEvent):  # pragma: lax no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCodeDeltaEvent):
                json_args_delta = to_json(chunk.delta).decode()[1:-1]  # Drop the surrounding `"`
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args=json_args_delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCodeDoneEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args='"}',
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallInterpretingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallCompletedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallGeneratingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallPartialImageEvent):
                # Not present on the type, but present on the actual object.
                # See https://github.com/openai/openai-python/issues/2649
                output_format = getattr(chunk, 'output_format', 'png')
                file_part = FilePart(
                    content=BinaryImage(
                        data=base64.b64decode(chunk.partial_image_b64),
                        media_type=f'image/{output_format}',
                    ),
                    id=chunk.item_id,
                )
                yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item_id}-file', part=file_part)

            elif isinstance(chunk, responses.ResponseMcpCallArgumentsDoneEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args='}',
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseMcpCallArgumentsDeltaEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args=chunk.delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseMcpListToolsInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpListToolsCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpListToolsFailedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpCallFailedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseFileSearchCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseFileSearchCallSearchingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseFileSearchCallInProgressEvent):
                pass  # there's nothing we need to do here

            else:  # pragma: no cover
                warnings.warn(
                    f'Handling of this event type is not yet implemented. Please report on our GitHub: {chunk}',
                    UserWarning,
                )

    def _map_usage(self, response: responses.Response) -> usage.RequestUsage:
        return _map_usage(response, self._provider_name, self._provider_url, self.model_name)

    @property
    def model_name(self) -> OpenAIModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def provider_url(self) -> str:
        """Get the provider base URL."""
        return self._provider_url

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


# Responses API-specific helper functions


def _split_combined_tool_call_id(combined_id: str) -> tuple[str, str | None]:
    # When reasoning, the Responses API requires the `ResponseFunctionToolCall` to be returned with both the `call_id` and `id` fields.
    # Before our `ToolCallPart` gained the `id` field alongside `tool_call_id` field, we combined the two fields into a single string stored on `tool_call_id`.
    if '|' in combined_id:
        call_id, id = combined_id.split('|', 1)
        return call_id, id
    else:
        return combined_id, None


def _map_code_interpreter_tool_call(
    item: responses.ResponseCodeInterpreterToolCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart, list[FilePart]]:
    result: dict[str, Any] = {
        'status': item.status,
    }

    file_parts: list[FilePart] = []
    logs: list[str] = []
    if item.outputs:
        for output in item.outputs:
            if isinstance(output, responses.response_code_interpreter_tool_call.OutputImage):
                file_parts.append(
                    FilePart(
                        content=BinaryImage.from_data_uri(output.url),
                        id=item.id,
                    )
                )
            elif isinstance(output, responses.response_code_interpreter_tool_call.OutputLogs):
                logs.append(output.logs)
            else:
                assert_never(output)

    if logs:
        result['logs'] = logs

    return (
        BuiltinToolCallPart(
            tool_name=CodeExecutionTool.kind,
            tool_call_id=item.id,
            args={
                'container_id': item.container_id,
                'code': item.code or '',
            },
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=CodeExecutionTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
        file_parts,
    )


def _map_web_search_tool_call(
    item: responses.ResponseFunctionWebSearch, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    args: dict[str, Any] | None = None

    result = {
        'status': item.status,
    }

    if action := item.action:
        # We need to exclude None values because of https://github.com/pydantic/pydantic-ai/issues/3653
        args = action.model_dump(mode='json', exclude_none=True)

        # To prevent `Unknown parameter: 'input[2].action.sources'` for `ActionSearch`
        if sources := args.pop('sources', None):
            result['sources'] = sources

    return (
        BuiltinToolCallPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            args=args,
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
    )


def _map_file_search_tool_call(
    item: responses.ResponseFileSearchToolCall,
    provider_name: str,
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    args = {'queries': item.queries}

    result: dict[str, Any] = {
        'status': item.status,
    }
    if item.results is not None:
        result['results'] = [r.model_dump(mode='json') for r in item.results]

    return (
        BuiltinToolCallPart(
            tool_name=FileSearchTool.kind,
            tool_call_id=item.id,
            args=args,
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=FileSearchTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
    )


def _map_image_generation_tool_call(
    item: responses.response_output_item.ImageGenerationCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart, FilePart | None]:
    result = {
        'status': item.status,
    }

    # Not present on the type, but present on the actual object.
    # See https://github.com/openai/openai-python/issues/2649
    if background := getattr(item, 'background', None):
        result['background'] = background
    if quality := getattr(item, 'quality', None):
        result['quality'] = quality
    if size := getattr(item, 'size', None):
        result['size'] = size
    if revised_prompt := getattr(item, 'revised_prompt', None):
        result['revised_prompt'] = revised_prompt
    output_format = getattr(item, 'output_format', 'png')

    file_part: FilePart | None = None
    if item.result:
        file_part = FilePart(
            content=BinaryImage(
                data=base64.b64decode(item.result),
                media_type=f'image/{output_format}',
            ),
            id=item.id,
        )

        # For some reason, the streaming API leaves `status` as `generating` even though generation has completed.
        result['status'] = 'completed'

    return (
        BuiltinToolCallPart(
            tool_name=ImageGenerationTool.kind,
            tool_call_id=item.id,
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=ImageGenerationTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
        file_part,
    )


def _map_mcp_list_tools(
    item: responses.response_output_item.McpListTools, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    tool_name = ':'.join([MCPServerTool.kind, item.server_label])
    return (
        BuiltinToolCallPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            provider_name=provider_name,
            args={'action': 'list_tools'},
        ),
        BuiltinToolReturnPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            content=item.model_dump(mode='json', include={'tools', 'error'}),
            provider_name=provider_name,
        ),
    )


def _map_mcp_call(
    item: responses.response_output_item.McpCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    tool_name = ':'.join([MCPServerTool.kind, item.server_label])
    return (
        BuiltinToolCallPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            args={
                'action': 'call_tool',
                'tool_name': item.name,
                'tool_args': json.loads(item.arguments) if item.arguments else {},
            },
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            content={
                'output': item.output,
                'error': item.error,
            },
            provider_name=provider_name,
        ),
    )
