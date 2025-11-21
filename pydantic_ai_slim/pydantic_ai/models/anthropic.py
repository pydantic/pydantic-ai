from __future__ import annotations as _annotations

import io
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, cast, overload

from pydantic import TypeAdapter
from typing_extensions import assert_never

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._run_context import RunContext
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..builtin_tools import CodeExecutionTool, MCPServerTool, MemoryTool, WebSearchTool
from ..exceptions import UserError
from ..messages import (
    BinaryContent,
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
)
from ..profiles import ModelProfileSpec
from ..providers import Provider, infer_provider
from ..providers.anthropic import AsyncAnthropicClient
from ..settings import ModelSettings, merge_model_settings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests, download_item, get_user_agent

_FINISH_REASON_MAP: dict[BetaStopReason, FinishReason] = {
    'end_turn': 'stop',
    'max_tokens': 'length',
    'stop_sequence': 'stop',
    'tool_use': 'tool_call',
    'pause_turn': 'stop',
    'refusal': 'content_filter',
}


try:
    from anthropic import NOT_GIVEN, APIStatusError, AsyncAnthropicBedrock, AsyncStream, omit as OMIT
    from anthropic.types.beta import (
        BetaBase64PDFBlockParam,
        BetaBase64PDFSourceParam,
        BetaCacheControlEphemeralParam,
        BetaCitationsDelta,
        BetaCodeExecutionTool20250522Param,
        BetaCodeExecutionToolResultBlock,
        BetaCodeExecutionToolResultBlockContent,
        BetaCodeExecutionToolResultBlockParam,
        BetaCodeExecutionToolResultBlockParamContentParam,
        BetaContentBlock,
        BetaContentBlockParam,
        BetaImageBlockParam,
        BetaInputJSONDelta,
        BetaMCPToolResultBlock,
        BetaMCPToolUseBlock,
        BetaMCPToolUseBlockParam,
        BetaMemoryTool20250818Param,
        BetaMessage,
        BetaMessageParam,
        BetaMessageTokensCount,
        BetaMetadataParam,
        BetaPlainTextSourceParam,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaRedactedThinkingBlock,
        BetaRedactedThinkingBlockParam,
        BetaRequestMCPServerToolConfigurationParam,
        BetaRequestMCPServerURLDefinitionParam,
        BetaServerToolUseBlock,
        BetaServerToolUseBlockParam,
        BetaSignatureDelta,
        BetaStopReason,
        BetaTextBlock,
        BetaTextBlockParam,
        BetaTextDelta,
        BetaThinkingBlock,
        BetaThinkingBlockParam,
        BetaThinkingConfigParam,
        BetaThinkingDelta,
        BetaToolChoiceParam,
        BetaToolParam,
        BetaToolResultBlockParam,
        BetaToolUnionParam,
        BetaToolUseBlock,
        BetaToolUseBlockParam,
        BetaWebSearchTool20250305Param,
        BetaWebSearchToolResultBlock,
        BetaWebSearchToolResultBlockContent,
        BetaWebSearchToolResultBlockParam,
        BetaWebSearchToolResultBlockParamContentParam,
    )
    from anthropic.types.beta.beta_web_search_tool_20250305_param import UserLocation
    from anthropic.types.model_param import ModelParam

except ImportError as _import_error:
    raise ImportError(
        'Please install `anthropic` to use the Anthropic model, '
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from _import_error

LatestAnthropicModelNames = ModelParam
"""Latest Anthropic models."""

AnthropicModelName = str | LatestAnthropicModelNames
"""Possible Anthropic model names.

Since Anthropic supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models) for a full list.
"""


class AnthropicModelSettings(ModelSettings, total=False):
    """Settings used for an Anthropic model request."""

    # ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    anthropic_metadata: BetaMetadataParam
    """An object describing metadata about the request.

    Contains `user_id`, an external identifier for the user who is associated with the request.
    """

    anthropic_thinking: BetaThinkingConfigParam
    """Determine whether the model should generate a thinking block.

    See [the Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) for more information.
    """

    anthropic_cache_tool_definitions: bool | Literal['5m', '1h']
    """Whether to add `cache_control` to the last tool definition.

    When enabled, the last tool in the `tools` array will have `cache_control` set,
    allowing Anthropic to cache tool definitions and reduce costs.
    If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly.
    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
    """

    anthropic_cache_instructions: bool | Literal['5m', '1h']
    """Whether to add `cache_control` to the last system prompt block.

    When enabled, the last system prompt will have `cache_control` set,
    allowing Anthropic to cache system instructions and reduce costs.
    If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly.
    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
    """


@dataclass(init=False)
class AnthropicModel(Model):
    """A model that uses the Anthropic API.

    Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncAnthropicClient = field(repr=False)

    _model_name: AnthropicModelName = field(repr=False)
    _provider: Provider[AsyncAnthropicClient] = field(repr=False)

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        provider: Literal['anthropic', 'gateway'] | Provider[AsyncAnthropicClient] = 'anthropic',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use. List of model names available
                [here](https://docs.anthropic.com/en/docs/about-claude/models).
            provider: The provider to use for the Anthropic API. Can be either the string 'anthropic' or an
                instance of `Provider[AsyncAnthropicClient]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/anthropic' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> AnthropicModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._messages_create(
            messages, False, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        return model_response

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )

        response = await self._messages_count_tokens(
            messages, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )

        return usage.RequestUsage(input_tokens=response.input_tokens)

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
        response = await self._messages_create(
            messages, True, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        settings = merge_model_settings(self.settings, model_settings)
        if (
            model_request_parameters.output_tools
            and settings
            and (thinking := settings.get('anthropic_thinking'))
            and thinking.get('type') == 'enabled'
        ):
            if model_request_parameters.output_mode == 'auto':
                model_request_parameters = replace(model_request_parameters, output_mode='prompted')
            elif (
                model_request_parameters.output_mode == 'tool' and not model_request_parameters.allow_text_output
            ):  # pragma: no branch
                # This would result in `tool_choice=required`, which Anthropic does not support with thinking.
                raise UserError(
                    'Anthropic does not support thinking and output tools at the same time. Use `output_type=PromptedOutput(...)` instead.'
                )
        return super().prepare_request(model_settings, model_request_parameters)

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[BetaRawMessageStreamEvent]:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessage:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessage | AsyncStream[BetaRawMessageStreamEvent]:
        # standalone function to make it easier to override
        tools = self._get_tools(model_request_parameters, model_settings)
        tools, mcp_servers, beta_features = self._add_builtin_tools(tools, model_request_parameters)

        tool_choice = self._infer_tool_choice(tools, model_settings, model_request_parameters)

        system_prompt, anthropic_messages = await self._map_message(messages, model_request_parameters, model_settings)

        try:
            extra_headers = self._map_extra_headers(beta_features, model_settings)

            return await self.client.beta.messages.create(
                max_tokens=model_settings.get('max_tokens', 4096),
                system=system_prompt or OMIT,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                mcp_servers=mcp_servers or OMIT,
                stream=stream,
                thinking=model_settings.get('anthropic_thinking', OMIT),
                stop_sequences=model_settings.get('stop_sequences', OMIT),
                temperature=model_settings.get('temperature', OMIT),
                top_p=model_settings.get('top_p', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                metadata=model_settings.get('anthropic_metadata', OMIT),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover

    async def _messages_count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessageTokensCount:
        if isinstance(self.client, AsyncAnthropicBedrock):
            raise UserError('AsyncAnthropicBedrock client does not support `count_tokens` api.')

        # standalone function to make it easier to override
        tools = self._get_tools(model_request_parameters, model_settings)
        tools, mcp_servers, beta_features = self._add_builtin_tools(tools, model_request_parameters)

        tool_choice = self._infer_tool_choice(tools, model_settings, model_request_parameters)

        system_prompt, anthropic_messages = await self._map_message(messages, model_request_parameters, model_settings)

        try:
            extra_headers = self._map_extra_headers(beta_features, model_settings)

            return await self.client.beta.messages.count_tokens(
                system=system_prompt or OMIT,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                mcp_servers=mcp_servers or OMIT,
                thinking=model_settings.get('anthropic_thinking', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover

    def _process_response(self, response: BetaMessage) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        items: list[ModelResponsePart] = []
        builtin_tool_calls: dict[str, BuiltinToolCallPart] = {}
        for item in response.content:
            if isinstance(item, BetaTextBlock):
                items.append(TextPart(content=item.text))
            elif isinstance(item, BetaServerToolUseBlock):
                call_part = _map_server_tool_use_block(item, self.system)
                builtin_tool_calls[call_part.tool_call_id] = call_part
                items.append(call_part)
            elif isinstance(item, BetaWebSearchToolResultBlock):
                items.append(_map_web_search_tool_result_block(item, self.system))
            elif isinstance(item, BetaCodeExecutionToolResultBlock):
                items.append(_map_code_execution_tool_result_block(item, self.system))
            elif isinstance(item, BetaRedactedThinkingBlock):
                items.append(
                    ThinkingPart(id='redacted_thinking', content='', signature=item.data, provider_name=self.system)
                )
            elif isinstance(item, BetaThinkingBlock):
                items.append(ThinkingPart(content=item.thinking, signature=item.signature, provider_name=self.system))
            elif isinstance(item, BetaMCPToolUseBlock):
                call_part = _map_mcp_server_use_block(item, self.system)
                builtin_tool_calls[call_part.tool_call_id] = call_part
                items.append(call_part)
            elif isinstance(item, BetaMCPToolResultBlock):
                call_part = builtin_tool_calls.get(item.tool_use_id)
                items.append(_map_mcp_server_result_block(item, call_part, self.system))
            else:
                assert isinstance(item, BetaToolUseBlock), f'unexpected item type {type(item)}'
                items.append(
                    ToolCallPart(
                        tool_name=item.name,
                        args=cast(dict[str, Any], item.input),
                        tool_call_id=item.id,
                    )
                )

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] | None = None
        if raw_finish_reason := response.stop_reason:  # pragma: no branch
            provider_details = {'finish_reason': raw_finish_reason}
            finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response, self._provider.name, self._provider.base_url, self._model_name),
            model_name=response.model,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self, response: AsyncStream[BetaRawMessageStreamEvent], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')  # pragma: no cover

        assert isinstance(first_chunk, BetaRawMessageStartEvent)

        return AnthropicStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.message.model,
            _response=peekable_response,
            _timestamp=_utils.now_utc(),
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
        )

    def _get_tools(
        self, model_request_parameters: ModelRequestParameters, model_settings: AnthropicModelSettings
    ) -> list[BetaToolUnionParam]:
        tools: list[BetaToolUnionParam] = [
            self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()
        ]

        # Add cache_control to the last tool if enabled
        if tools and (cache_tool_defs := model_settings.get('anthropic_cache_tool_definitions')):
            # If True, use '5m'; otherwise use the specified ttl value
            ttl: Literal['5m', '1h'] = '5m' if cache_tool_defs is True else cache_tool_defs
            last_tool = tools[-1]
            last_tool['cache_control'] = BetaCacheControlEphemeralParam(type='ephemeral', ttl=ttl)

        return tools

    def _add_builtin_tools(
        self, tools: list[BetaToolUnionParam], model_request_parameters: ModelRequestParameters
    ) -> tuple[list[BetaToolUnionParam], list[BetaRequestMCPServerURLDefinitionParam], list[str]]:
        beta_features: list[str] = []
        mcp_servers: list[BetaRequestMCPServerURLDefinitionParam] = []
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                user_location = UserLocation(type='approximate', **tool.user_location) if tool.user_location else None
                tools.append(
                    BetaWebSearchTool20250305Param(
                        name='web_search',
                        type='web_search_20250305',
                        max_uses=tool.max_uses,
                        allowed_domains=tool.allowed_domains,
                        blocked_domains=tool.blocked_domains,
                        user_location=user_location,
                    )
                )
            elif isinstance(tool, CodeExecutionTool):  # pragma: no branch
                tools.append(BetaCodeExecutionTool20250522Param(name='code_execution', type='code_execution_20250522'))
                beta_features.append('code-execution-2025-05-22')
            elif isinstance(tool, MemoryTool):  # pragma: no branch
                if 'memory' not in model_request_parameters.tool_defs:
                    raise UserError("Built-in `MemoryTool` requires a 'memory' tool to be defined.")
                # Replace the memory tool definition with the built-in memory tool
                tools = [tool for tool in tools if tool['name'] != 'memory']
                tools.append(BetaMemoryTool20250818Param(name='memory', type='memory_20250818'))
                beta_features.append('context-management-2025-06-27')
            elif isinstance(tool, MCPServerTool) and tool.url:
                mcp_server_url_definition_param = BetaRequestMCPServerURLDefinitionParam(
                    type='url',
                    name=tool.id,
                    url=tool.url,
                )
                if tool.allowed_tools is not None:  # pragma: no branch
                    mcp_server_url_definition_param['tool_configuration'] = BetaRequestMCPServerToolConfigurationParam(
                        enabled=bool(tool.allowed_tools),
                        allowed_tools=tool.allowed_tools,
                    )
                if tool.authorization_token:  # pragma: no cover
                    mcp_server_url_definition_param['authorization_token'] = tool.authorization_token
                mcp_servers.append(mcp_server_url_definition_param)
                beta_features.append('mcp-client-2025-04-04')
            else:  # pragma: no cover
                raise UserError(
                    f'`{tool.__class__.__name__}` is not supported by `AnthropicModel`. If it should be, please file an issue.'
                )
        return tools, mcp_servers, beta_features

    def _infer_tool_choice(
        self,
        tools: list[BetaToolUnionParam],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaToolChoiceParam | None:
        if not tools:
            return None
        else:
            tool_choice: BetaToolChoiceParam

            if not model_request_parameters.allow_text_output:
                tool_choice = {'type': 'any'}
            else:
                tool_choice = {'type': 'auto'}

            if 'parallel_tool_calls' in model_settings:
                tool_choice['disable_parallel_tool_use'] = not model_settings['parallel_tool_calls']

            return tool_choice

    def _map_extra_headers(self, beta_features: list[str], model_settings: AnthropicModelSettings) -> dict[str, str]:
        """Apply beta_features to extra_headers in model_settings."""
        extra_headers = model_settings.get('extra_headers', {})
        extra_headers.setdefault('User-Agent', get_user_agent())
        if beta_features:
            if 'anthropic-beta' in extra_headers:
                beta_features.insert(0, extra_headers['anthropic-beta'])
            extra_headers['anthropic-beta'] = ','.join(beta_features)
        return extra_headers

    async def _map_message(  # noqa: C901
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> tuple[str | list[BetaTextBlockParam], list[BetaMessageParam]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        system_prompt_parts: list[str] = []
        anthropic_messages: list[BetaMessageParam] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                user_content_params: list[BetaContentBlockParam] = []
                for request_part in m.parts:
                    if isinstance(request_part, SystemPromptPart):
                        system_prompt_parts.append(request_part.content)
                    elif isinstance(request_part, UserPromptPart):
                        async for content in self._map_user_prompt(request_part):
                            if isinstance(content, CachePoint):
                                self._add_cache_control_to_last_param(user_content_params, ttl=content.ttl)
                            else:
                                user_content_params.append(content)
                    elif isinstance(request_part, ToolReturnPart):
                        tool_result_block_param = BetaToolResultBlockParam(
                            tool_use_id=_guard_tool_call_id(t=request_part),
                            type='tool_result',
                            content=request_part.model_response_str(),
                            is_error=False,
                        )
                        user_content_params.append(tool_result_block_param)
                    elif isinstance(request_part, RetryPromptPart):  # pragma: no branch
                        if request_part.tool_name is None:
                            text = request_part.model_response()  # pragma: no cover
                            retry_param = BetaTextBlockParam(type='text', text=text)  # pragma: no cover
                        else:
                            retry_param = BetaToolResultBlockParam(
                                tool_use_id=_guard_tool_call_id(t=request_part),
                                type='tool_result',
                                content=request_part.model_response(),
                                is_error=True,
                            )
                        user_content_params.append(retry_param)
                if len(user_content_params) > 0:
                    anthropic_messages.append(BetaMessageParam(role='user', content=user_content_params))
            elif isinstance(m, ModelResponse):
                assistant_content_params: list[
                    BetaTextBlockParam
                    | BetaToolUseBlockParam
                    | BetaServerToolUseBlockParam
                    | BetaWebSearchToolResultBlockParam
                    | BetaCodeExecutionToolResultBlockParam
                    | BetaThinkingBlockParam
                    | BetaRedactedThinkingBlockParam
                    | BetaMCPToolUseBlockParam
                    | BetaMCPToolResultBlock
                ] = []
                for response_part in m.parts:
                    if isinstance(response_part, TextPart):
                        if response_part.content:
                            assistant_content_params.append(BetaTextBlockParam(text=response_part.content, type='text'))
                    elif isinstance(response_part, ToolCallPart):
                        tool_use_block_param = BetaToolUseBlockParam(
                            id=_guard_tool_call_id(t=response_part),
                            type='tool_use',
                            name=response_part.tool_name,
                            input=response_part.args_as_dict(),
                        )
                        assistant_content_params.append(tool_use_block_param)
                    elif isinstance(response_part, ThinkingPart):
                        if (
                            response_part.provider_name == self.system and response_part.signature is not None
                        ):  # pragma: no branch
                            if response_part.id == 'redacted_thinking':
                                assistant_content_params.append(
                                    BetaRedactedThinkingBlockParam(
                                        data=response_part.signature,
                                        type='redacted_thinking',
                                    )
                                )
                            else:
                                assistant_content_params.append(
                                    BetaThinkingBlockParam(
                                        thinking=response_part.content,
                                        signature=response_part.signature,
                                        type='thinking',
                                    )
                                )
                        elif response_part.content:  # pragma: no branch
                            start_tag, end_tag = self.profile.thinking_tags
                            assistant_content_params.append(
                                BetaTextBlockParam(
                                    text='\n'.join([start_tag, response_part.content, end_tag]), type='text'
                                )
                            )
                    elif isinstance(response_part, BuiltinToolCallPart):
                        if response_part.provider_name == self.system:
                            tool_use_id = _guard_tool_call_id(t=response_part)
                            if response_part.tool_name == WebSearchTool.kind:
                                server_tool_use_block_param = BetaServerToolUseBlockParam(
                                    id=tool_use_id,
                                    type='server_tool_use',
                                    name='web_search',
                                    input=response_part.args_as_dict(),
                                )
                                assistant_content_params.append(server_tool_use_block_param)
                            elif response_part.tool_name == CodeExecutionTool.kind:
                                server_tool_use_block_param = BetaServerToolUseBlockParam(
                                    id=tool_use_id,
                                    type='server_tool_use',
                                    name='code_execution',
                                    input=response_part.args_as_dict(),
                                )
                                assistant_content_params.append(server_tool_use_block_param)
                            elif (
                                response_part.tool_name.startswith(MCPServerTool.kind)
                                and (server_id := response_part.tool_name.split(':', 1)[1])
                                and (args := response_part.args_as_dict())
                                and (tool_name := args.get('tool_name'))
                                and (tool_args := args.get('tool_args'))
                            ):  # pragma: no branch
                                mcp_tool_use_block_param = BetaMCPToolUseBlockParam(
                                    id=tool_use_id,
                                    type='mcp_tool_use',
                                    server_name=server_id,
                                    name=tool_name,
                                    input=tool_args,
                                )
                                assistant_content_params.append(mcp_tool_use_block_param)
                    elif isinstance(response_part, BuiltinToolReturnPart):
                        if response_part.provider_name == self.system:
                            tool_use_id = _guard_tool_call_id(t=response_part)
                            if response_part.tool_name in (
                                WebSearchTool.kind,
                                'web_search_tool_result',  # Backward compatibility
                            ) and isinstance(response_part.content, dict | list):
                                assistant_content_params.append(
                                    BetaWebSearchToolResultBlockParam(
                                        tool_use_id=tool_use_id,
                                        type='web_search_tool_result',
                                        content=cast(
                                            BetaWebSearchToolResultBlockParamContentParam,
                                            response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                        ),
                                    )
                                )
                            elif response_part.tool_name in (  # pragma: no branch
                                CodeExecutionTool.kind,
                                'code_execution_tool_result',  # Backward compatibility
                            ) and isinstance(response_part.content, dict):
                                assistant_content_params.append(
                                    BetaCodeExecutionToolResultBlockParam(
                                        tool_use_id=tool_use_id,
                                        type='code_execution_tool_result',
                                        content=cast(
                                            BetaCodeExecutionToolResultBlockParamContentParam,
                                            response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                        ),
                                    )
                                )
                            elif response_part.tool_name.startswith(MCPServerTool.kind) and isinstance(
                                response_part.content, dict
                            ):  # pragma: no branch
                                assistant_content_params.append(
                                    BetaMCPToolResultBlock(
                                        tool_use_id=tool_use_id,
                                        type='mcp_tool_result',
                                        **cast(dict[str, Any], response_part.content),  # pyright: ignore[reportUnknownMemberType]
                                    )
                                )
                    elif isinstance(response_part, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
                        pass
                    else:
                        assert_never(response_part)
                if len(assistant_content_params) > 0:
                    anthropic_messages.append(BetaMessageParam(role='assistant', content=assistant_content_params))
            else:
                assert_never(m)
        if instructions := self._get_instructions(messages, model_request_parameters):
            system_prompt_parts.insert(0, instructions)
        system_prompt = '\n\n'.join(system_prompt_parts)

        # If anthropic_cache_instructions is enabled, return system prompt as a list with cache_control
        if system_prompt and (cache_instructions := model_settings.get('anthropic_cache_instructions')):
            # If True, use '5m'; otherwise use the specified ttl value
            ttl: Literal['5m', '1h'] = '5m' if cache_instructions is True else cache_instructions
            system_prompt_blocks = [
                BetaTextBlockParam(
                    type='text',
                    text=system_prompt,
                    cache_control=BetaCacheControlEphemeralParam(type='ephemeral', ttl=ttl),
                )
            ]
            return system_prompt_blocks, anthropic_messages

        return system_prompt, anthropic_messages

    @staticmethod
    def _add_cache_control_to_last_param(params: list[BetaContentBlockParam], ttl: Literal['5m', '1h'] = '5m') -> None:
        """Add cache control to the last content block param.

        See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
        """
        if not params:
            raise UserError(
                'CachePoint cannot be the first content in a user message - there must be previous content to attach the CachePoint to. '
                'To cache system instructions or tool definitions, use the `anthropic_cache_instructions` or `anthropic_cache_tool_definitions` settings instead.'
            )

        # Only certain types support cache_control
        # See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#what-can-be-cached
        cacheable_types = {'text', 'tool_use', 'server_tool_use', 'image', 'tool_result'}
        # Cast needed because BetaContentBlockParam is a union including response Block types (Pydantic models)
        # that don't support dict operations, even though at runtime we only have request Param types (TypedDicts).
        last_param = cast(dict[str, Any], params[-1])
        if last_param['type'] not in cacheable_types:
            raise UserError(f'Cache control not supported for param type: {last_param["type"]}')

        # Add cache_control to the last param
        last_param['cache_control'] = BetaCacheControlEphemeralParam(type='ephemeral', ttl=ttl)

    @staticmethod
    async def _map_user_prompt(
        part: UserPromptPart,
    ) -> AsyncGenerator[BetaContentBlockParam | CachePoint]:
        if isinstance(part.content, str):
            if part.content:  # Only yield non-empty text
                yield BetaTextBlockParam(text=part.content, type='text')
        else:
            for item in part.content:
                if isinstance(item, str):
                    if item:  # Only yield non-empty text
                        yield BetaTextBlockParam(text=item, type='text')
                elif isinstance(item, CachePoint):
                    yield item
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        yield BetaImageBlockParam(
                            source={'data': io.BytesIO(item.data), 'media_type': item.media_type, 'type': 'base64'},  # type: ignore
                            type='image',
                        )
                    elif item.media_type == 'application/pdf':
                        yield BetaBase64PDFBlockParam(
                            source=BetaBase64PDFSourceParam(
                                data=io.BytesIO(item.data),
                                media_type='application/pdf',
                                type='base64',
                            ),
                            type='document',
                        )
                    else:
                        raise RuntimeError('Only images and PDFs are supported for binary content')
                elif isinstance(item, ImageUrl):
                    yield BetaImageBlockParam(source={'type': 'url', 'url': item.url}, type='image')
                elif isinstance(item, DocumentUrl):
                    if item.media_type == 'application/pdf':
                        yield BetaBase64PDFBlockParam(source={'url': item.url, 'type': 'url'}, type='document')
                    elif item.media_type == 'text/plain':
                        downloaded_item = await download_item(item, data_format='text')
                        yield BetaBase64PDFBlockParam(
                            source=BetaPlainTextSourceParam(
                                data=downloaded_item['data'], media_type=item.media_type, type='text'
                            ),
                            type='document',
                        )
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported media type: {item.media_type}')
                else:
                    raise RuntimeError(f'Unsupported content type: {type(item)}')  # pragma: no cover

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> BetaToolParam:
        return {
            'name': f.name,
            'description': f.description or '',
            'input_schema': f.parameters_json_schema,
        }


def _map_usage(
    message: BetaMessage | BetaRawMessageStartEvent | BetaRawMessageDeltaEvent,
    provider: str,
    provider_url: str,
    model: str,
    existing_usage: usage.RequestUsage | None = None,
) -> usage.RequestUsage:
    if isinstance(message, BetaMessage):
        response_usage = message.usage
    elif isinstance(message, BetaRawMessageStartEvent):
        response_usage = message.message.usage
    elif isinstance(message, BetaRawMessageDeltaEvent):
        response_usage = message.usage
    else:
        assert_never(message)

    # In streaming, usage appears in different events.
    # The values are cumulative, meaning new values should replace existing ones entirely.
    details: dict[str, int] = (existing_usage.details if existing_usage else {}) | {
        key: value for key, value in response_usage.model_dump().items() if isinstance(value, int)
    }

    # Note: genai-prices already extracts cache_creation_input_tokens and cache_read_input_tokens
    # from the Anthropic response and maps them to cache_write_tokens and cache_read_tokens
    return usage.RequestUsage.extract(
        dict(model=model, usage=details),
        provider=provider,
        provider_url=provider_url,
        provider_fallback='anthropic',
        details=details,
    )


@dataclass
class AnthropicStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Anthropic models."""

    _model_name: AnthropicModelName
    _response: AsyncIterable[BetaRawMessageStreamEvent]
    _timestamp: datetime
    _provider_name: str
    _provider_url: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        current_block: BetaContentBlock | None = None

        builtin_tool_calls: dict[str, BuiltinToolCallPart] = {}
        async for event in self._response:
            if isinstance(event, BetaRawMessageStartEvent):
                self._usage = _map_usage(event, self._provider_name, self._provider_url, self._model_name)
                self.provider_response_id = event.message.id

            elif isinstance(event, BetaRawContentBlockStartEvent):
                current_block = event.content_block
                if isinstance(current_block, BetaTextBlock) and current_block.text:
                    maybe_event = self._parts_manager.handle_text_delta(
                        vendor_part_id=event.index, content=current_block.text
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(current_block, BetaThinkingBlock):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        content=current_block.thinking,
                        signature=current_block.signature,
                        provider_name=self.provider_name,
                    )
                elif isinstance(current_block, BetaRedactedThinkingBlock):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        id='redacted_thinking',
                        signature=current_block.data,
                        provider_name=self.provider_name,
                    )
                elif isinstance(current_block, BetaToolUseBlock):
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=event.index,
                        tool_name=current_block.name,
                        args=cast(dict[str, Any], current_block.input) or None,
                        tool_call_id=current_block.id,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(current_block, BetaServerToolUseBlock):
                    call_part = _map_server_tool_use_block(current_block, self.provider_name)
                    builtin_tool_calls[call_part.tool_call_id] = call_part
                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index,
                        part=call_part,
                    )
                elif isinstance(current_block, BetaWebSearchToolResultBlock):
                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index,
                        part=_map_web_search_tool_result_block(current_block, self.provider_name),
                    )
                elif isinstance(current_block, BetaCodeExecutionToolResultBlock):
                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index,
                        part=_map_code_execution_tool_result_block(current_block, self.provider_name),
                    )
                elif isinstance(current_block, BetaMCPToolUseBlock):
                    call_part = _map_mcp_server_use_block(current_block, self.provider_name)
                    builtin_tool_calls[call_part.tool_call_id] = call_part

                    args_json = call_part.args_as_json_str()
                    # Drop the final `{}}` so that we can add tool args deltas
                    args_json_delta = args_json[:-3]
                    assert args_json_delta.endswith('"tool_args":'), (
                        f'Expected {args_json_delta!r} to end in `"tool_args":`'
                    )

                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index, part=replace(call_part, args=None)
                    )
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=event.index,
                        args=args_json_delta,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(current_block, BetaMCPToolResultBlock):
                    call_part = builtin_tool_calls.get(current_block.tool_use_id)
                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index,
                        part=_map_mcp_server_result_block(current_block, call_part, self.provider_name),
                    )

            elif isinstance(event, BetaRawContentBlockDeltaEvent):
                if isinstance(event.delta, BetaTextDelta):
                    maybe_event = self._parts_manager.handle_text_delta(
                        vendor_part_id=event.index, content=event.delta.text
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(event.delta, BetaThinkingDelta):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        content=event.delta.thinking,
                        provider_name=self.provider_name,
                    )
                elif isinstance(event.delta, BetaSignatureDelta):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        signature=event.delta.signature,
                        provider_name=self.provider_name,
                    )
                elif isinstance(event.delta, BetaInputJSONDelta):
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=event.index,
                        args=event.delta.partial_json,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                # TODO(Marcelo): We need to handle citations.
                elif isinstance(event.delta, BetaCitationsDelta):
                    pass

            elif isinstance(event, BetaRawMessageDeltaEvent):
                self._usage = _map_usage(event, self._provider_name, self._provider_url, self._model_name, self._usage)
                if raw_finish_reason := event.delta.stop_reason:  # pragma: no branch
                    self.provider_details = {'finish_reason': raw_finish_reason}
                    self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

            elif isinstance(event, BetaRawContentBlockStopEvent):  # pragma: no branch
                if isinstance(current_block, BetaMCPToolUseBlock):
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=event.index,
                        args='}',
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                current_block = None
            elif isinstance(event, BetaRawMessageStopEvent):  # pragma: no branch
                current_block = None

    @property
    def model_name(self) -> AnthropicModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


def _map_server_tool_use_block(item: BetaServerToolUseBlock, provider_name: str) -> BuiltinToolCallPart:
    if item.name == 'web_search':
        return BuiltinToolCallPart(
            provider_name=provider_name,
            tool_name=WebSearchTool.kind,
            args=cast(dict[str, Any], item.input) or None,
            tool_call_id=item.id,
        )
    elif item.name == 'code_execution':
        return BuiltinToolCallPart(
            provider_name=provider_name,
            tool_name=CodeExecutionTool.kind,
            args=cast(dict[str, Any], item.input) or None,
            tool_call_id=item.id,
        )
    elif item.name in ('web_fetch', 'bash_code_execution', 'text_editor_code_execution'):  # pragma: no cover
        raise NotImplementedError(f'Anthropic built-in tool {item.name!r} is not currently supported.')
    else:
        assert_never(item.name)


web_search_tool_result_content_ta: TypeAdapter[BetaWebSearchToolResultBlockContent] = TypeAdapter(
    BetaWebSearchToolResultBlockContent
)


def _map_web_search_tool_result_block(item: BetaWebSearchToolResultBlock, provider_name: str) -> BuiltinToolReturnPart:
    return BuiltinToolReturnPart(
        provider_name=provider_name,
        tool_name=WebSearchTool.kind,
        content=web_search_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
    )


code_execution_tool_result_content_ta: TypeAdapter[BetaCodeExecutionToolResultBlockContent] = TypeAdapter(
    BetaCodeExecutionToolResultBlockContent
)


def _map_code_execution_tool_result_block(
    item: BetaCodeExecutionToolResultBlock, provider_name: str
) -> BuiltinToolReturnPart:
    return BuiltinToolReturnPart(
        provider_name=provider_name,
        tool_name=CodeExecutionTool.kind,
        content=code_execution_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
    )


def _map_mcp_server_use_block(item: BetaMCPToolUseBlock, provider_name: str) -> BuiltinToolCallPart:
    return BuiltinToolCallPart(
        provider_name=provider_name,
        tool_name=':'.join([MCPServerTool.kind, item.server_name]),
        args={
            'action': 'call_tool',
            'tool_name': item.name,
            'tool_args': cast(dict[str, Any], item.input),
        },
        tool_call_id=item.id,
    )


def _map_mcp_server_result_block(
    item: BetaMCPToolResultBlock, call_part: BuiltinToolCallPart | None, provider_name: str
) -> BuiltinToolReturnPart:
    return BuiltinToolReturnPart(
        provider_name=provider_name,
        tool_name=call_part.tool_name if call_part else MCPServerTool.kind,
        content=item.model_dump(mode='json', include={'content', 'is_error'}),
        tool_call_id=item.tool_use_id,
    )
