from __future__ import annotations

import functools
import typing
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
    ParamSpec,
    assert_never,
    overload,
)

import anyio
import boto3

from .. import result
from ..messages import (
    ArgsDict,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import AgentModel, EitherStreamedResponse, Model

if TYPE_CHECKING:
    from botocore.eventstream import EventStream
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        InferenceConfigurationTypeDef,
        MessageUnionTypeDef,
        ToolTypeDef,
        ToolUseBlockOutputTypeDef,
    )


P = ParamSpec('P')
T = typing.TypeVar('T')


async def run_in_threadpool(func: typing.Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Wrapper around `anyio.to_thread.run_sync`, copied from fastapi."""
    if kwargs:  # pragma: no cover
        # run_sync doesn't accept 'kwargs', so bind them in here
        func = functools.partial(func, **kwargs)
    return await anyio.to_thread.run_sync(func, *args)


def exclude_none(data):
    """Exclude None values from a dictionary."""
    return {k: v for k, v in data.items() if v is not None}


@dataclass(init=False)
class BedrockModel(Model):
    """A model that uses the Bedrock-runtime API."""

    model_name: str
    client: BedrockRuntimeClient

    def __init__(
        self,
        model_name: str,
        *,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_region: str | None = None,
    ):
        self.model_name = model_name
        self.client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region,
        )

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return BedrockAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
            support_tools_choice=True if self.model_name.startswith('anthropic') else False,
        )

    def name(self) -> str:
        return self.model_name

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolTypeDef:
        return {
            'toolSpec': {
                'name': f.name,
                'description': f.description,
                'inputSchema': {'json': f.parameters_json_schema},
            }
        }


@dataclass
class BedrockAgentModel(AgentModel):
    """Implementation of `AgentModel` for Bedrock models."""

    client: BedrockRuntimeClient
    model_name: str
    allow_text_result: bool
    tools: list[ToolTypeDef]

    support_tools_choice: bool

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, result.Usage]:
        response = await self._messages_create(messages, False, model_settings)
        return await self._process_response(response)

    @staticmethod
    async def _process_response(
        response: ConverseResponseTypeDef,
    ) -> tuple[ModelResponse, result.Usage]:
        items: list[ModelResponsePart] = []
        for item in response['output']['message']['content']:
            if item.get('text'):
                items.append(TextPart(item['text']))
            else:
                assert item.get('toolUse')
                items.append(
                    ToolCallPart.from_raw_args(
                        item['toolUse']['name'],
                        item['toolUse']['input'],
                        item['toolUse']['toolUseId'],
                    ),
                )
        usage = result.Usage(
            request_tokens=response['usage']['inputTokens'],
            response_tokens=response['usage']['outputTokens'],
            total_tokens=response['usage']['totalTokens'],
        )
        return ModelResponse(items), usage

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[EitherStreamedResponse]:
        response = await self._messages_create(messages, True, model_settings)
        async with response:
            yield await self._process_streamed_response(response)

    @staticmethod
    async def _process_streamed_response(
        response: EventStream[ConverseStreamOutputTypeDef],
    ) -> EitherStreamedResponse:
        raise NotImplementedError('Streamed responses are not yet supported for this models.')

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: ModelSettings | None,
    ) -> EventStream[ConverseStreamOutputTypeDef]:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: ModelSettings | None,
    ) -> ConverseResponseTypeDef:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: ModelSettings | None,
    ) -> ConverseResponseTypeDef | EventStream[ConverseStreamOutputTypeDef]:
        if not self.tools:
            tool_choice: None = None
        elif not self.allow_text_result and self.support_tools_choice:
            tool_choice = {'tool': {'name': tool_type_def['toolSpec']['name']} for tool_type_def in self.tools}
        else:
            tool_choice = None

        system_prompt, bedrock_messages = self._map_message(messages)
        inference_config = self._map_inference_config(model_settings)
        toolConfig = (
            exclude_none(
                {
                    'tools': self.tools,
                    'toolChoice': tool_choice,
                }
            )
            if self.tools
            else None
        )
        model_settings = model_settings or {}

        params = exclude_none(
            dict(
                modelId=self.model_name,
                messages=bedrock_messages,
                system=[{'text': system_prompt}],
                inferenceConfig=inference_config,
                toolConfig=toolConfig,
            )
        )
        if stream:
            model_response = await run_in_threadpool(self.client.converse_stream, **params)
        else:
            model_response = await run_in_threadpool(self.client.converse, **params)
        return model_response

    @staticmethod
    def _map_inference_config(
        model_settings: ModelSettings | None,
    ) -> InferenceConfigurationTypeDef:
        model_settings = model_settings or {}
        return exclude_none(
            {
                'maxTokens': model_settings.get('max_tokens'),
                'temperature': model_settings.get('temperature'),
                'topP': model_settings.get('top_p'),
                # TODO: This is not included in model_settings yet
                # "stopSequences": model_settings.get('stop_sequences'),
            }
        )

    @staticmethod
    def _map_message(
        messages: list[ModelMessage],
    ) -> tuple[str, list[MessageUnionTypeDef]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        system_prompt: str = ''
        bedrock_messages: list[MessageUnionTypeDef] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        system_prompt += part.content
                    elif isinstance(part, UserPromptPart):
                        bedrock_messages.append(
                            {
                                'role': 'user',
                                'content': [{'text': part.content}],
                            }
                        )
                    elif isinstance(part, ToolReturnPart):
                        bedrock_messages.append(
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        'toolResult': {
                                            'toolUseId': part.tool_call_id,
                                            'content': part.model_response_str(),
                                            'status': 'success',
                                        }
                                    }
                                ],
                            },
                        )
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            bedrock_messages.append(
                                {
                                    'role': 'user',
                                    'content': [{'text': part.content}],
                                }
                            )
                        else:
                            bedrock_messages.append(
                                {
                                    'role': 'user',
                                    'content': [
                                        {
                                            'toolResult': {
                                                'toolUseId': part.tool_call_id,
                                                'content': part.model_response(),
                                                'status': 'error',
                                            }
                                        }
                                    ],
                                }
                            )
            elif isinstance(m, ModelResponse):
                content: list[ContentBlockOutputTypeDef] = []
                for item in m.parts:
                    if isinstance(item, TextPart):
                        content.append({'text': item.content})
                    else:
                        assert isinstance(item, ToolCallPart)
                        content.append(_map_tool_call(item))
                bedrock_messages.append(
                    {
                        'role': 'assistant',
                        'content': content,
                    }
                )
            else:
                assert_never(m)
        return system_prompt, bedrock_messages


def _map_tool_call(t: ToolCallPart) -> ToolUseBlockOutputTypeDef:
    assert isinstance(t.args, ArgsDict), f'Expected ArgsDict, got {t.args}'
    return {
        'toolUseId': t.tool_call_id,
        'name': t.tool_name,
        'input': t.args_as_dict(),
    }
