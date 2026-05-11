"""OpenResponses-aware model client.

Subclass of [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]
that bypasses the OpenAI Python SDK's closed pydantic union (which rejects
`pydantic_ai:*` extension event types per the [OpenResponses spec](https://www.openresponses.org/specification))
and parses raw SSE with `httpx`. Use this Model to point a Pydantic AI agent at
another agent's `agent.beta.to_responses()` endpoint with full lossless round-trip
of `pydantic_ai:agent_context`, `pydantic_ai:custom_tool_call`, and
`pydantic_ai:custom_tool_call_output` items.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
from typing_extensions import assert_never

from .._utils import is_str_dict, now_utc as _now_utc
from ..exceptions import ModelHTTPError
from ..messages import (
    AgentContextPart,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from ..settings import ModelSettings
from . import ModelRequestParameters, StreamedResponse, check_allow_model_requests, get_user_agent
from .openai import OpenAIResponsesModel

try:
    from httpx_sse import EventSource, ServerSentEvent
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `httpx_sse` package to use `OpenResponsesModel`, '
        'you can use the `responses` optional group — `pip install "pydantic-ai-slim[responses]"`'
    ) from e

__all__ = ['OpenResponsesModel']


_FINISH_REASON_MAP: dict[str, FinishReason] = {
    'completed': 'stop',
    'incomplete': 'length',
    'failed': 'error',
    'cancelled': 'stop',
}


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    """Return only the `dict[str, Any]` elements of an unknown-shape value.

    The Responses wire is loose-typed (`dict[str, Any]` everywhere); this helper localizes
    the structural narrowing so call sites don't repeat `if not is_str_dict(x): continue`.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:  # pyright: ignore[reportUnknownVariableType]
        if is_str_dict(item):
            out.append(item)
    return out


@dataclass(init=False)
class OpenResponsesModel(OpenAIResponsesModel):
    """A Model client that speaks the OpenResponses protocol with extension-item awareness.

    Inherits provider/auth/profile setup, outbound message serialization, and tool definition
    mapping from [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]. Only
    the request transport is overridden — raw `httpx` instead of the OpenAI SDK's closed
    pydantic response union — so `pydantic_ai:`-prefixed extension items round-trip cleanly.

    Pair with another Pydantic AI agent exposed via `agent.beta.to_responses(mode='openresponses')`
    to build layered-agent stacks: an outer agent calls an inner agent over the OpenResponses
    wire, with backend tool calls and `agent_context` items lossless across layers.
    """

    @property
    def system(self) -> str:
        return 'openresponses'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        body = await self._build_request_body(messages, model_settings, model_request_parameters, stream=False)

        try:
            response = await self.client._client.post(  # pyright: ignore[reportPrivateUsage]
                f'{self._base_path}/responses',
                json=body,
                headers=self._build_headers(model_settings),
            )
        except httpx.HTTPError as exc:  # pragma: no cover
            raise ModelHTTPError(status_code=0, model_name=self.model_name, body=str(exc)) from exc

        if response.status_code >= 400:
            raise ModelHTTPError(
                status_code=response.status_code,
                model_name=self.model_name,
                body=response.text,
            )

        payload = response.json()
        if not is_str_dict(payload):
            raise ValueError('OpenResponses /responses endpoint returned a non-object body.')
        return self._process_dict_response(payload)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        body = await self._build_request_body(messages, model_settings, model_request_parameters, stream=True)

        request = self.client._client.build_request(  # pyright: ignore[reportPrivateUsage]
            'POST',
            f'{self._base_path}/responses',
            json=body,
            headers=self._build_headers(model_settings, sse=True),
        )
        response = await self.client._client.send(request, stream=True)  # pyright: ignore[reportPrivateUsage]
        if response.status_code >= 400:
            await response.aread()
            await response.aclose()
            raise ModelHTTPError(
                status_code=response.status_code,
                model_name=self.model_name,
                body=response.text,
            )

        try:
            yield OpenResponsesStreamedResponse(
                model_request_parameters=model_request_parameters,
                _model_name=self.model_name,
                _provider_name=self.system,
                _provider_url=self._base_path,
                _response=response,
            )
        finally:
            await response.aclose()

    @property
    def _base_path(self) -> str:
        """Base URL for the upstream OpenResponses endpoint, with trailing slash stripped."""
        return str(self.client.base_url).rstrip('/')

    def _build_headers(self, model_settings: ModelSettings | None, *, sse: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {'User-Agent': get_user_agent()}
        api_key = getattr(self.client, 'api_key', None)
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        if sse:
            headers['Accept'] = 'text/event-stream'
        if model_settings:
            extra_headers = model_settings.get('extra_headers') or {}
            for key, value in extra_headers.items():
                if isinstance(key, str) and isinstance(value, str):
                    headers[key] = value
        return headers

    async def _build_request_body(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build a Responses-compatible request body, with `pydantic_ai:agent_context` injection."""
        input_items = self._serialize_messages(messages)
        body: dict[str, Any] = {
            'model': self.model_name,
            'input': input_items,
            'stream': stream,
        }
        instructions = self._collect_instructions(messages, model_request_parameters)
        if instructions:
            body['instructions'] = instructions
        tool_defs = list(model_request_parameters.tool_defs.values())
        if tool_defs:
            body['tools'] = [
                {
                    'type': 'function',
                    'name': t.name,
                    'description': t.description,
                    'parameters': t.parameters_json_schema,
                }
                for t in tool_defs
            ]
        if model_settings:
            for key in ('temperature', 'top_p', 'max_tokens'):
                value = model_settings.get(key)
                if value is not None:
                    body['max_output_tokens' if key == 'max_tokens' else key] = value
        return body

    @staticmethod
    def _collect_instructions(
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> str | None:
        # Mirror OpenAIResponsesModel's instructions surfacing: the Responses API expects
        # per-run instructions as a top-level field rather than baked into `input`.
        from ..messages import SystemPromptPart

        parts: list[str] = []
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    parts.append(part.content)
        return '\n\n'.join(parts) if parts else None

    @staticmethod
    def _serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
        """Walk Pydantic AI messages → OpenResponses input items.

        Lean serializer covering the layered-agent demo surface: user/assistant text,
        tool calls/returns, and `pydantic_ai:agent_context` items emitted by an outer
        agent. Reasoning / images / builtin-tool round-trip are deferred to
        `OpenAIResponsesModel` proper for callers that don't need extension lossless.
        """
        from ..messages import SystemPromptPart, UserPromptPart

        items: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        content = part.content
                        if isinstance(content, str):
                            items.append({'role': 'user', 'content': content})
                        else:
                            text_parts = [c if isinstance(c, str) else getattr(c, 'content', '') for c in content]
                            items.append({'role': 'user', 'content': ''.join(text_parts)})
                    elif isinstance(part, ToolReturnPart):
                        items.append(
                            {
                                'type': 'function_call_output',
                                'call_id': part.tool_call_id,
                                'output': part.model_response_str(),
                            }
                        )
                    elif isinstance(part, SystemPromptPart):
                        # System prompts are surfaced via `instructions` (top-level), not as input items.
                        continue
            elif isinstance(message, ModelResponse):
                for item in message.parts:
                    if isinstance(item, TextPart):
                        items.append({'role': 'assistant', 'content': item.content})
                    elif isinstance(item, ToolCallPart):
                        items.append(
                            {
                                'type': 'function_call',
                                'call_id': item.tool_call_id,
                                'name': item.tool_name,
                                'arguments': item.args_as_json_str(),
                            }
                        )
                    elif isinstance(item, AgentContextPart):
                        items.append(
                            {
                                'type': 'pydantic_ai:agent_context',
                                'from_agent': item.from_agent,
                                'role': item.role,
                                'content': item.content,
                            }
                        )
            else:
                assert_never(message)
        return items

    def _process_dict_response(self, payload: dict[str, Any]) -> ModelResponse:
        """Dispatch a non-streaming `Response` JSON body to a `ModelResponse`."""
        parts: list[ModelResponsePart] = []
        tool_call_names: dict[str, str] = {}
        for raw_item in _iter_dicts(payload.get('output')):
            self._dispatch_output_item(raw_item, parts, tool_call_names)

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] = {}
        status = payload.get('status')
        if isinstance(status, str):
            provider_details['finish_reason'] = status
            finish_reason = _FINISH_REASON_MAP.get(status)

        response_id_raw = payload.get('id')
        return ModelResponse(
            parts=parts,
            model_name=str(payload.get('model') or self.model_name),
            provider_response_id=response_id_raw if isinstance(response_id_raw, str) else None,
            timestamp=_now_utc(),
            provider_name=self.system,
            provider_url=self._base_path,
            finish_reason=finish_reason,
            provider_details=provider_details or None,
        )

    @staticmethod
    def _dispatch_output_item(
        item: dict[str, Any],
        parts: list[ModelResponsePart],
        tool_call_names: dict[str, str],
    ) -> None:
        item_type = item.get('type')
        if item_type == 'message':
            for chunk in _iter_dicts(item.get('content')):
                if chunk.get('type') == 'output_text':
                    text = chunk.get('text')
                    if isinstance(text, str):
                        parts.append(TextPart(content=text))
        elif item_type == 'function_call':
            call_id = item.get('call_id') or item.get('id')
            name = item.get('name')
            if isinstance(call_id, str) and isinstance(name, str):
                tool_call_names[call_id] = name
                parts.append(
                    ToolCallPart(
                        tool_name=name,
                        tool_call_id=call_id,
                        args=item.get('arguments'),
                    )
                )
        elif item_type == 'pydantic_ai:custom_tool_call':
            call_id = item.get('call_id') or item.get('id')
            name = item.get('name')
            if isinstance(call_id, str) and isinstance(name, str):
                tool_call_names[call_id] = name
                # Backend tool L2 ran server-side: surface to L1 as builtin so the consumer
                # doesn't try to re-execute. Round-trip back to L2 happens via the adapter's
                # `pydantic_ai:custom_tool_call` input-item parse, which restores `ToolCallPart`.
                parts.append(
                    BuiltinToolCallPart(
                        tool_name=name,
                        tool_call_id=call_id,
                        args=item.get('arguments'),
                    )
                )
        elif item_type == 'pydantic_ai:custom_tool_call_output':
            call_id = item.get('call_id')
            if isinstance(call_id, str):
                tool_name = tool_call_names.get(call_id, '')
                parts.append(
                    BuiltinToolReturnPart(
                        tool_name=tool_name,
                        tool_call_id=call_id,
                        content=str(item.get('output', '')),
                    )
                )
        elif item_type == 'pydantic_ai:agent_context':
            from_agent = item.get('from_agent', '')
            role = item.get('role', 'context')
            content = item.get('content', '')
            if (
                isinstance(from_agent, str)
                and isinstance(content, str)
                and content
                and role in ('developer', 'context', 'observation')
            ):
                parts.append(
                    AgentContextPart(
                        content=content,
                        from_agent=from_agent,
                        role=role,
                    )
                )


@dataclass
class OpenResponsesStreamedResponse(StreamedResponse):
    """Streaming response wrapping raw SSE from an OpenResponses endpoint."""

    _model_name: str = ''
    _provider_name: str = ''
    _provider_url: str = ''
    _response: httpx.Response = field(default=None)  # pyright: ignore[reportAssignmentType]
    _timestamp: datetime = field(default_factory=_now_utc)
    _tool_call_names: dict[str, str] = field(default_factory=dict[str, str])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str | None:
        return self._provider_name

    @property
    def provider_url(self) -> str | None:
        return self._provider_url

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    async def close_stream(self) -> None:
        await self._response.aclose()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        event_source = EventSource(self._response)
        async for sse in event_source.aiter_sse():
            async for ev in self._handle_sse(sse):
                yield ev

    async def _handle_sse(self, sse: ServerSentEvent) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        if sse.event == 'done' or not sse.data:
            return
        try:
            payload = json.loads(sse.data)
        except json.JSONDecodeError:  # pragma: no cover
            return
        if not is_str_dict(payload):  # pragma: no cover
            return

        event_type = payload.get('type', sse.event)
        if event_type == 'response.created':
            response = payload.get('response')
            if is_str_dict(response):
                response_id = response.get('id')
                if isinstance(response_id, str):
                    self.provider_response_id = response_id
        elif event_type == 'response.output_text.delta':
            delta = payload.get('delta')
            item_id = payload.get('item_id')
            if isinstance(delta, str) and isinstance(item_id, str):
                for ev in self._parts_manager.handle_text_delta(vendor_part_id=item_id, content=delta):
                    yield ev
        elif event_type == 'response.output_item.added':
            item = payload.get('item')
            if is_str_dict(item):
                async for ev in self._dispatch_item_added(item):
                    yield ev
        elif event_type == 'response.output_item.done':
            item = payload.get('item')
            if is_str_dict(item):
                async for ev in self._dispatch_item_done(item):
                    yield ev
        elif event_type == 'response.completed':
            response = payload.get('response')
            if is_str_dict(response):
                status = response.get('status')
                if isinstance(status, str):
                    self.finish_reason = _FINISH_REASON_MAP.get(status)
                    self.provider_details = {**(self.provider_details or {}), 'finish_reason': status}

    async def _dispatch_item_added(self, item: dict[str, Any]) -> AsyncIterator[ModelResponseStreamEvent]:
        item_type = item.get('type')
        item_id = item.get('id')
        if item_type == 'message':
            return
        if item_type == 'function_call':
            call_id = item.get('call_id') or item.get('id')
            name = item.get('name')
            if isinstance(call_id, str) and isinstance(name, str):
                self._tool_call_names[call_id] = name
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=call_id,
                    tool_name=name,
                    args=item.get('arguments') or '',
                    tool_call_id=call_id,
                )
        elif item_type == 'pydantic_ai:custom_tool_call':
            call_id = item.get('call_id') or item.get('id')
            name = item.get('name')
            if isinstance(call_id, str) and isinstance(name, str):
                self._tool_call_names[call_id] = name
                yield self._parts_manager.handle_part(
                    vendor_part_id=call_id,
                    part=BuiltinToolCallPart(
                        tool_name=name,
                        tool_call_id=call_id,
                        args=item.get('arguments'),
                    ),
                )
        elif item_type == 'pydantic_ai:agent_context':
            from_agent = item.get('from_agent', '')
            role = item.get('role', 'context')
            content = item.get('content', '')
            if (
                isinstance(from_agent, str)
                and isinstance(content, str)
                and content
                and role in ('developer', 'context', 'observation')
                and isinstance(item_id, str)
            ):
                yield self._parts_manager.handle_part(
                    vendor_part_id=item_id,
                    part=AgentContextPart(content=content, from_agent=from_agent, role=role, id=item_id),
                )

    async def _dispatch_item_done(self, item: dict[str, Any]) -> AsyncIterator[ModelResponseStreamEvent]:
        item_type = item.get('type')
        if item_type == 'pydantic_ai:custom_tool_call_output':
            call_id = item.get('call_id')
            if isinstance(call_id, str):
                tool_name = self._tool_call_names.get(call_id, '')
                yield self._parts_manager.handle_part(
                    vendor_part_id=f'{call_id}-output',
                    part=BuiltinToolReturnPart(
                        tool_name=tool_name,
                        tool_call_id=call_id,
                        content=str(item.get('output', '')),
                    ),
                )

    def get_stream_cancel_errors(self) -> tuple[type[BaseException], ...]:
        return (httpx.StreamError, httpx.TransportError)


# Re-export AgentContextPart so callers don't need to import it from messages directly.
__all__.append('AgentContextPart')
__all__.append('OpenResponsesStreamedResponse')
