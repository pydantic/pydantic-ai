from __future__ import annotations as _annotations

import json
import os
from collections.abc import Mapping, Sequence, AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from json import JSONDecodeError
from typing import Any, Literal

from _datetime import timezone
from httpx import AsyncClient as AsyncHTTPClient, Response as HTTPResponse

from . import (
    AbstractToolDefinition,
    AgentModel,
    EitherStreamedResponse,
    Model,
    StreamStructuredResponse,
    StreamTextResponse,
    cached_async_http_client,
    check_allow_model_requests, get_user_agent,
)
from .. import UnexpectedModelBehavior, _utils, result, exceptions
from ..messages import (
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    ToolCall,
)
from ..result import Cost

PYDANTIC_AI_API_BASE_URL = os.environ.get("PYDANTIC_AI_API_BASE_URL", "https://api.openai.com/v1")
PYDANTIC_AI_API_KEY = os.environ.get("PYDANTIC_AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
PYDANTIC_AI_MODEL = os.environ.get("PYDANTIC_AI_MODEL")


@dataclass(init=False)
class GeneralModel(Model):
    """A model that calls any LLM API based on input"""

    http_client: AsyncHTTPClient
    base_url: str
    api_key: str
    model_name: str

    def __init__(
            self,
            *,
            http_client: AsyncHTTPClient | None = None,
            base_url: str | None = PYDANTIC_AI_API_BASE_URL,
            api_key: str | None = PYDANTIC_AI_API_KEY,
            model_name: str | None = PYDANTIC_AI_MODEL
    ):
        """Initialize a general model.

        Args:
            base_url: base URL of the API, can be either local or cloud. If not provided,
                the `PYDANTIC_AI_API_BASE_URL` env variable will be used if available
            api_key: Optional API key to use for authentication, If not provided,
                the `PYDANTIC_AI_API_KEY` or `OPENAI_API_KEY` env variable will be used if available
            model_name: The name of the model to use

        """
        self.http_client = http_client or cached_async_http_client()
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

    async def agent_model(
            self,
            function_tools: Mapping[str, AbstractToolDefinition],
            allow_text_result: bool,
            result_tools: Sequence[AbstractToolDefinition] | None,
    ) -> AgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools.values()]
        if result_tools is not None:
            tools += [self._map_tool_definition(r) for r in result_tools]

        return GeneralAgentModel(
            http_client=self.http_client,
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            allow_text_result=allow_text_result,
            tools=tools,
        )

    def name(self) -> str:
        return f'general:{self.model_name}'

    @staticmethod
    def _map_tool_definition(f: AbstractToolDefinition):
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.json_schema,
            },
        }


@dataclass
class GeneralAgentModel(AgentModel):
    """Implementation of `AgentModel` for general models."""

    http_client: AsyncHTTPClient
    base_url: str
    api_key: str
    model_name: str
    allow_text_result: bool
    tools: list

    async def request(self, messages: list[Message]) -> tuple[ModelAnyResponse, result.Cost]:
        async with self._make_request(messages, False) as http_response:
            response = json.loads((await http_response.aread()).decode("utf-8"))
        cost = _map_cost(response)
        return self._process_response(response), cost

    @asynccontextmanager
    async def request_stream(self, messages: list[Message]) -> AsyncIterator[EitherStreamedResponse]:
        async with self._make_request(messages, streamed=True) as r:
            yield await self._process_streamed_response(r.aiter_bytes())

    @asynccontextmanager
    async def _make_request(self, messages: list[Message], streamed: bool) -> HTTPResponse | AsyncIterator[
        HTTPResponse]:
        url = f"{self.base_url}/chat/completions"
        auth_header = f"Bearer {self.api_key}" if self.api_key else None
        headers = {
            "Content-Type": "application/json",
            "User-Agent": get_user_agent(),
        }
        if auth_header:
            headers["Authorization"] = auth_header

        stop = ["<|im_end|>", "<|end|>", "<|eot_id|>"]

        if not self.tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not self.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        body = {
            "messages": [self._message_to_dict(m) for m in messages],
            "model": self.model_name,
            "stop": stop,
            "max_tokens": None,
            "temperature": 0,
            "tools": self.tools or None,
            "tool_choice": tool_choice or None,
            "parallel_tool_calls": True if self.tools else None,
            "stream": streamed,
        }
        json_body = json.dumps(body).encode("utf-8")

        async with self.http_client.stream('POST', url, content=json_body, headers=headers) as r:
            if r.status_code != 200:
                await r.aread()
                raise exceptions.UnexpectedModelBehavior(f'Unexpected response from LLM API {r.status_code}', r.text)
            yield r

    @staticmethod
    def _message_to_dict(m: Message):
        return {
            'role': m.role,
            'content': m.content
        }

    @staticmethod
    def _process_response(response: Any) -> ModelAnyResponse:
        """Process a non-streamed response, and prepare a message to return."""

        message = response["choices"][0]["message"]
        content = message["content"].strip()
        tool_calls = message.get("tool_calls")
        timestamp = datetime.fromtimestamp(response["created"], tz=timezone.utc)

        if tool_calls is not None:
            calls = [ToolCall.from_json(
                c.get('function').get('name'),
                c.get('function').get('arguments'),
                c.get('id'),
            ) for c in tool_calls]
            return ModelStructuredResponse(calls, timestamp=timestamp)

        else:
            return ModelTextResponse(content, timestamp=timestamp)

    @staticmethod
    async def _process_streamed_response(bytes_iterator: AsyncIterator[bytes]) -> EitherStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""

        try:
            first_found = False
            while not first_found:
                first_bytes = await bytes_iterator.__anext__()
                first_str = _clean_str(first_bytes.decode('utf-8'))
                if first_str is not None and len(first_str) > 0:
                    first_chunk = json.loads(first_str)
                    first_found = True

        except StopAsyncIteration as e:
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls') from e
        timestamp = datetime.fromtimestamp(first_chunk.get('created'), tz=timezone.utc)
        delta = first_chunk.get('choices')[0].get('delta')
        start_cost = _map_cost(first_chunk)

        # the first chunk may only contain `role`, so we iterate until we get either `tool_calls` or `content`
        while delta.get('tool_calls') is None and delta.get('content') is None:
            try:
                next_bytes = await bytes_iterator.__anext__()
                next_chunk = _response_bytes_to_dict(next_bytes)
            except StopAsyncIteration as e:
                raise UnexpectedModelBehavior('Streamed response ended without content or tool calls') from e
            delta = next_chunk.get('choices')[0].get('delta')
            start_cost += _map_cost(next_chunk)

        if delta.get('content') is not None:
            return GeneralStreamTextResponse(delta.get('content'), bytes_iterator, timestamp, start_cost)
        else:
            assert delta.tool_calls is not None, f'Expected delta with tool_calls, got {delta}'
            return GeneralStreamStructuredResponse(
                bytes_iterator,
                {c.index: c for c in delta.tool_calls},
                timestamp,
                start_cost,
            )


@dataclass
class GeneralStreamTextResponse(StreamTextResponse):
    """Implementation of `StreamTextResponse` for General models."""

    _first: str | None
    _bytes_iterator: AsyncIterator[bytes]
    _timestamp: datetime
    _cost: result.Cost
    _buffer: list[str] = field(default_factory=list, init=False)

    async def __anext__(self) -> None:
        if self._first is not None:
            self._buffer.append(self._first)
            self._first = None
            return None

        # Find next chunk that can be decoded
        decoded = False
        while not decoded:
            b = await self._bytes_iterator.__anext__()
            try:
                chunk = _response_bytes_to_dict(b)
                decoded = True
            except JSONDecodeError:
                continue

            self._cost += _map_cost(chunk)

        try:
            choice = chunk.get('choices')[0]
        except IndexError:
            raise StopAsyncIteration()

        content = choice.get('delta').get('content')
        if content is not None:
            self._buffer.append(content)

    def get(self, *, final: bool = False) -> Iterable[str]:
        yield from self._buffer
        self._buffer.clear()

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


@dataclass
class GeneralStreamStructuredResponse(StreamStructuredResponse):
    """Implementation of `StreamStructuredResponse` for general models."""

    _bytes_iterator: AsyncIterator[bytes]
    _delta_tool_calls: dict[int, Any]
    _timestamp: datetime
    _cost: result.Cost

    async def __anext__(self) -> None:
        b = await self._bytes_iterator.__anext__()
        chunk = _response_bytes_to_dict(b)
        self._cost += _map_cost(chunk)
        try:
            choice = chunk.get('choices')[0]
        except IndexError:
            raise StopAsyncIteration()

        if choice.get('finish_reason') is not None:
            raise StopAsyncIteration()

        content = choice.get('delta').get('content')
        assert content is None, f'Expected tool calls, got content instead, invalid chunk: {chunk!r}'

        tool_calls = choice.get('delta').get('tool_calls')
        for new in tool_calls or []:
            if current := self._delta_tool_calls.get(new.index):
                if current.get('function') is None:
                    current['function'] = new.get('function')
                elif new.get('function') is not None:
                    current['function']['name'] = _utils.add_optional(current['function'].get('name'),
                                                                      new['function'].get('name'))
                    current['function']['arguments'] = _utils.add_optional(current.get('function').get('arguments'),
                                                                           new['function'].get('arguments'))
            else:
                self._delta_tool_calls[new.index] = new

    def get(self, *, final: bool = False) -> ModelStructuredResponse:
        calls: list[ToolCall] = []
        for c in self._delta_tool_calls.values():
            if f := c.get('function'):
                if f.get('name') is not None and f.get('arguments') is not None:
                    calls.append(ToolCall.from_json(f.get('name'), f.get('arguments'), c.get('id')))

        return ModelStructuredResponse(calls, timestamp=self._timestamp)

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


def _map_cost(response) -> result.Cost:
    usage = response.get('usage')

    if usage is None:
        return result.Cost()
    else:
        details: dict[str, int] = {}
        if usage.get('completion_tokens_details') is not None:
            details.update(usage.get('completion_tokens_details').model_dump(exclude_none=True))
        if usage.get('prompt_tokens_details') is not None:
            details.update(usage.get('prompt_tokens_details').model_dump(exclude_none=True))

        return result.Cost(
            request_tokens=usage.get('prompt_tokens'),
            response_tokens=usage.get('completion_tokens'),
            total_tokens=usage.get('total_tokens'),
            details=details,
        )


def _response_bytes_to_dict(b: bytes) -> dict:
    string = _clean_str(b.decode('utf-8'))
    return json.loads(string)


def _clean_str(string: str) -> str:
    payload = ''

    lines = string.splitlines(True)
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            if line[0] == ":":
                continue
            elif line == "data: [DONE]" or line == "[DONE]":
                break
            elif line:
                part = line
                if line[:6] == "data: ":
                    part = line[6:]

                if part is not None:
                    if len(payload) == 0:
                        payload = part.strip()
                    else:
                        payload += part

    return payload
