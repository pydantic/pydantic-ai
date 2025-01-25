from __future__ import annotations

import functools
import typing
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)

import anyio

try:
    import boto3
except ImportError as _import_error:
    raise ImportError(
        'Please install `boto3` to use the Bedrock model, '
        "you can use the `bedrock` optional group — `pip install 'pydantic-ai-slim[bedrock]'`"
    ) from _import_error


from typing_extensions import ParamSpec, assert_never

from .. import result
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import AgentModel, Model, StreamedResponse

if TYPE_CHECKING:
    from botocore.eventstream import EventStream
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamMetadataEventTypeDef,
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
    return await anyio.to_thread.run_sync(func, *args) # type: ignore


def exclude_none(data: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Exclude None values from a dictionary."""
    return {k: v for k, v in data.items() if v is not None}


class AsyncIteratorWrapper:
    def __init__(self, sync_iterator: Iterator[T]):
        self.sync_iterator = iter(sync_iterator)

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        try:
            # Run the synchronous next() call in a thread pool
            item = await anyio.to_thread.run_sync(next, self.sync_iterator) # type: ignore
            return item # type: ignore
        except RuntimeError as e:
            if type(e.__cause__) is StopIteration:
                raise StopAsyncIteration
            else:
                raise e


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class BedrockStreamedResponse(StreamedResponse):
    _event_stream: EventStream[ConverseStreamOutputTypeDef]
    _timestamp: datetime = field(default_factory=now_utc, init=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Return an async iterator of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This method should be implemented by subclasses to translate the vendor-specific stream of events into
        pydantic_ai-format events.
        """
        chunk: ConverseStreamOutputTypeDef
        async for chunk in AsyncIteratorWrapper(self._event_stream):
            if "messageStart" in chunk:
                continue
            if "messageStop" in chunk:
                continue
            if "metadata" in chunk:
                if "usage" in chunk["metadata"]:
                    self._usage += self._map_usage(chunk["metadata"])
                continue
            if "contentBlockStart" in chunk:
                index = chunk["contentBlockStart"]["contentBlockIndex"]
                start = chunk["contentBlockStart"]["start"]
                if "toolUse" in start:
                    tool_use_start = start["toolUse"]
                    tool_id = tool_use_start["toolUseId"]
                    tool_name = tool_use_start["name"]
                    yield self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=index,
                        tool_name=tool_name,
                        args=None,
                        tool_call_id=tool_id,
                    )

            if "contentBlockDelta" in chunk:
                index = chunk["contentBlockDelta"]["contentBlockIndex"]
                delta = chunk["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield self._parts_manager.handle_text_delta(
                        vendor_part_id=index, content=delta["text"]
                    )
                if "toolUse" in delta:
                    tool_use = delta["toolUse"]
                    yield self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=index,
                        tool_name=tool_use.get("name"),
                        args=tool_use.get("input"),
                        tool_call_id=tool_id,
                    )

    def timestamp(self) -> datetime:
        return self._timestamp

    def _map_usage(self, metadata: ConverseStreamMetadataEventTypeDef) -> result.Usage:
        return result.Usage(
            request_tokens=metadata["usage"]["inputTokens"],
            response_tokens=metadata["usage"]["outputTokens"],
            total_tokens=metadata["usage"]["totalTokens"],
        )


@dataclass(init=False)
class BedrockModel(Model):
    """A model that uses the Bedrock-runtime API."""

    model_name: str
    client: BedrockRuntimeClient

    def __init__(
        self,
        model_name: str,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str | None = None,
        bedrock_client: BedrockRuntimeClient | None = None,
    ):
        self.model_name = model_name
        if bedrock_client:
            self.client = bedrock_client
        else:
            self.client = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
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
            support_tools_choice=(True if self.model_name.startswith("anthropic") else False),
        )

    def name(self) -> str:
        return self.model_name

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolTypeDef:
        return {
            "toolSpec": {
                "name": f.name,
                "description": f.description,
                "inputSchema": {"json": f.parameters_json_schema},
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
        for item in response["output"]["message"]["content"]:
            if item.get("text"):
                items.append(TextPart(item["text"]))
            else:
                assert item.get("toolUse")
                items.append(
                    ToolCallPart.from_raw_args(
                        item["toolUse"]["name"],
                        item["toolUse"]["input"],
                        item["toolUse"]["toolUseId"],
                    ),
                )
        usage = result.Usage(
            request_tokens=response["usage"]["inputTokens"],
            response_tokens=response["usage"]["outputTokens"],
            total_tokens=response["usage"]["totalTokens"],
        )
        return ModelResponse(items), usage

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[StreamedResponse]:
        response = await self._messages_create(messages, True, model_settings)
        yield BedrockStreamedResponse(_model_name=self.model_name, _event_stream=response)

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
        if not self.tools or not self.support_tools_choice:
            tool_choice: None = None
        elif not self.allow_text_result:
            tool_choice = {"any": {}}
        else:
            tool_choice = {"auto": {}}

        system_prompt, bedrock_messages = self._map_message(messages)
        inference_config = self._map_inference_config(model_settings)
        toolConfig = (
            exclude_none(
                {
                    "tools": self.tools,
                    "toolChoice": tool_choice,
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
                system=[{"text": system_prompt}],
                inferenceConfig=inference_config,
                toolConfig=toolConfig,
            )
        )
        if stream:
            model_response = await run_in_threadpool(self.client.converse_stream, **params)
            model_response = model_response["stream"]
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
                "maxTokens": model_settings.get("max_tokens"),
                "temperature": model_settings.get("temperature"),
                "topP": model_settings.get("top_p"),
                # TODO: This is not included in model_settings yet
                # "stopSequences": model_settings.get('stop_sequences'),
            }
        )

    @staticmethod
    def _map_message(
        messages: list[ModelMessage],
    ) -> tuple[str, list[MessageUnionTypeDef]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        system_prompt: str = ""
        bedrock_messages: list[MessageUnionTypeDef] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        system_prompt += part.content
                    elif isinstance(part, UserPromptPart):
                        bedrock_messages.append(
                            {
                                "role": "user",
                                "content": [{"text": part.content}],
                            }
                        )
                    elif isinstance(part, ToolReturnPart):
                        bedrock_messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "toolResult": {
                                            "toolUseId": part.tool_call_id,
                                            "content": [{"text": part.model_response_str()}],
                                            "status": "success",
                                        }
                                    }
                                ],
                            },
                        )
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            bedrock_messages.append(
                                {
                                    "role": "user",
                                    "content": [{"text": part.content}],
                                }
                            )
                        else:
                            bedrock_messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "toolResult": {
                                                "toolUseId": part.tool_call_id,
                                                "content": [{"text": part.model_response()}],
                                                "status": "error",
                                            }
                                        }
                                    ],
                                }
                            )
            elif isinstance(m, ModelResponse):
                content: list[ContentBlockOutputTypeDef] = []
                for item in m.parts:
                    if isinstance(item, TextPart):
                        content.append({"text": item.content})
                    else:
                        assert isinstance(item, ToolCallPart)
                        content.append(_map_tool_call(item))  # FIXME: MISSING key
                bedrock_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
            else:
                assert_never(m)
        return system_prompt, bedrock_messages


def _map_tool_call(t: ToolCallPart) -> ToolUseBlockOutputTypeDef:
    assert isinstance(t.args, ArgsDict), f"Expected ArgsDict, got {t.args}"
    return {
        "toolUse": {
            "toolUseId": t.tool_call_id,
            "name": t.tool_name,
            "input": t.args_as_dict(),
        }
    }
