"""OpenAI Responses protocol event stream transformer for Pydantic AI agents."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    from openai.types.responses import (
        Response as ResponseObject,
        ResponseCodeInterpreterCallCompletedEvent,
        ResponseCodeInterpreterCallInProgressEvent,
        ResponseCodeInterpreterToolCall,
        ResponseCompletedEvent,
        ResponseContentPartAddedEvent,
        ResponseContentPartDoneEvent,
        ResponseCreatedEvent,
        ResponseFileSearchCallCompletedEvent,
        ResponseFileSearchCallInProgressEvent,
        ResponseFileSearchToolCall,
        ResponseFunctionCallArgumentsDeltaEvent,
        ResponseFunctionCallArgumentsDoneEvent,
        ResponseFunctionToolCall,
        ResponseFunctionWebSearch,
        ResponseImageGenCallCompletedEvent,
        ResponseImageGenCallInProgressEvent,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
        ResponseUsage,
        ResponseWebSearchCallCompletedEvent,
        ResponseWebSearchCallInProgressEvent,
    )
    from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
    from openai.types.responses.response_function_web_search import ActionSearch
    from openai.types.responses.response_output_item import ImageGenerationCall
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Responses integration, '
        'you can use the `responses` optional group — `pip install "pydantic-ai-slim[responses]"`'
    ) from e

from ...messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from .. import UIEventStream

__all__ = ['ResponsesEventStream']


_BuiltinItem = (
    ResponseFunctionWebSearch | ResponseCodeInterpreterToolCall | ResponseFileSearchToolCall | ImageGenerationCall
)


@dataclass
class ResponsesEventStream(UIEventStream[ResponseCreateParamsStreaming, Any, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the OpenAI Responses protocol."""

    frontend_tool_names: frozenset[str] = field(default_factory=frozenset[str])
    """Names of tools the client declared in the request `tools` array.

    `ToolCallPart`s for tools NOT in this set are server-executed (agent-registered)
    and are suppressed from the wire — vanilla SDK clients should not see them as
    `function_call` items, since the SDK contract treats those as requests for the
    client to act and the agent already ran them.
    """

    _response_id: str = ''
    _text_item_id: str = ''
    _output_index: int = 0
    _message_item_added: bool = False
    _sequence_number: int = 0
    _error: bool = False
    _accumulated_text: list[str] = field(default_factory=list[str])
    _open_function_calls: dict[str, tuple[ResponseFunctionToolCall, int]] = field(
        default_factory=dict[str, tuple[ResponseFunctionToolCall, int]]
    )
    _open_builtin_calls: dict[str, tuple[_BuiltinItem, int]] = field(
        default_factory=dict[str, tuple[_BuiltinItem, int]]
    )

    @property
    def content_type(self) -> str:
        return 'text/event-stream'

    def encode_event(self, event: Any) -> str:
        event_type = getattr(event, 'type', 'message')
        event_data = event.model_dump_json(exclude_unset=True)
        return f'event: {event_type}\ndata: {event_data}\n\n'

    async def encode_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[str]:
        async for event in stream:
            yield self.encode_event(event)
        yield 'event: done\ndata: [DONE]\n\n'

    def _next_seq(self) -> int:
        n = self._sequence_number
        self._sequence_number += 1
        return n

    def _model_name(self) -> str:
        model = self.run_input.get('model')
        return str(model) if model is not None else 'unknown'

    def _instructions(self) -> str | None:
        instructions = self.run_input.get('instructions')
        return instructions if isinstance(instructions, str) else None

    def _empty_usage(self) -> ResponseUsage:
        return ResponseUsage(
            input_tokens=0,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        )

    def _build_response(
        self,
        *,
        status: Literal['in_progress', 'completed', 'failed'],
        usage: ResponseUsage,
    ) -> ResponseObject:
        # `tool_choice` and `tools` come from the request-side TypedDict union, which is
        # structurally compatible with but distinct from the response-side pydantic union;
        # the runtime accepts both, the static unions don't intersect.
        return ResponseObject(
            id=self._response_id,
            object='response',
            created_at=time.time(),
            model=self._model_name(),
            output=[],
            status=status,
            usage=usage,
            instructions=self._instructions(),
            parallel_tool_calls=bool(self.run_input.get('parallel_tool_calls', True)),
            tool_choice=self.run_input.get('tool_choice') or 'auto',  # pyright: ignore[reportArgumentType]
            tools=self.run_input.get('tools') or [],  # pyright: ignore[reportArgumentType]
        )

    async def before_stream(self) -> AsyncIterator[Any]:
        self._response_id = self.new_message_id()
        self._text_item_id = self.new_message_id()
        yield ResponseCreatedEvent(
            type='response.created',
            response=self._build_response(status='in_progress', usage=self._empty_usage()),
            sequence_number=self._next_seq(),
        )

    async def after_stream(self) -> AsyncIterator[Any]:
        # Close any still-open content/output items so the spec ordering is preserved.
        async for ev in self._close_open_message_item():
            yield ev

        usage = self._empty_usage()
        if self._result is not None:
            run_usage = self._result.usage()
            usage = ResponseUsage(
                input_tokens=run_usage.input_tokens or 0,
                input_tokens_details=InputTokensDetails(cached_tokens=run_usage.cache_read_tokens or 0),
                output_tokens=run_usage.output_tokens or 0,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=run_usage.total_tokens or 0,
            )

        status = 'failed' if self._error else 'completed'
        yield ResponseCompletedEvent(
            type='response.completed',
            response=self._build_response(status=status, usage=usage),
            sequence_number=self._next_seq(),
        )

    async def on_error(self, error: Exception) -> AsyncIterator[Any]:
        self._error = True
        return
        yield  # Make this an async generator

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[Any]:
        if not self._message_item_added:
            yield ResponseOutputItemAddedEvent(
                type='response.output_item.added',
                item=ResponseOutputMessage(
                    type='message',
                    id=self._text_item_id,
                    role='assistant',
                    status='in_progress',
                    content=[],
                ),
                output_index=self._output_index,
                sequence_number=self._next_seq(),
            )
            yield ResponseContentPartAddedEvent(
                type='response.content_part.added',
                item_id=self._text_item_id,
                output_index=self._output_index,
                content_index=0,
                part=ResponseOutputText(type='output_text', text='', annotations=[]),
                sequence_number=self._next_seq(),
            )
            self._message_item_added = True

        if part.content:
            self._accumulated_text.append(part.content)
            yield ResponseTextDeltaEvent(
                type='response.output_text.delta',
                item_id=self._text_item_id,
                output_index=self._output_index,
                content_index=0,
                delta=part.content,
                logprobs=[],
                sequence_number=self._next_seq(),
            )

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[Any]:
        if not delta.content_delta:
            return
        self._accumulated_text.append(delta.content_delta)
        yield ResponseTextDeltaEvent(
            type='response.output_text.delta',
            item_id=self._text_item_id,
            output_index=self._output_index,
            content_index=0,
            delta=delta.content_delta,
            logprobs=[],
            sequence_number=self._next_seq(),
        )

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[Any]:
        if followed_by_text:
            return
        async for ev in self._close_open_message_item():
            yield ev

    async def _close_open_message_item(self) -> AsyncIterator[Any]:
        if not self._message_item_added:
            return

        full_text = ''.join(self._accumulated_text)
        yield ResponseTextDoneEvent(
            type='response.output_text.done',
            item_id=self._text_item_id,
            output_index=self._output_index,
            content_index=0,
            text=full_text,
            logprobs=[],
            sequence_number=self._next_seq(),
        )
        yield ResponseContentPartDoneEvent(
            type='response.content_part.done',
            item_id=self._text_item_id,
            output_index=self._output_index,
            content_index=0,
            part=ResponseOutputText(type='output_text', text=full_text, annotations=[]),
            sequence_number=self._next_seq(),
        )
        yield ResponseOutputItemDoneEvent(
            type='response.output_item.done',
            item=ResponseOutputMessage(
                type='message',
                id=self._text_item_id,
                role='assistant',
                status='completed',
                content=[ResponseOutputText(type='output_text', text=full_text, annotations=[])],
            ),
            output_index=self._output_index,
            sequence_number=self._next_seq(),
        )

        self._message_item_added = False
        self._accumulated_text = []
        self._text_item_id = self.new_message_id()
        self._output_index += 1

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[Any]:
        if part.tool_name not in self.frontend_tool_names:
            # Backend (agent-registered) tools are server-executed; emitting a `function_call`
            # output item would mislead vanilla SDK clients into thinking they owe a
            # `function_call_output` follow-up. Suppress entirely.
            return

        async for ev in self._close_open_message_item():
            yield ev

        item_id = self.new_message_id()
        item = ResponseFunctionToolCall(
            id=item_id,
            type='function_call',
            call_id=part.tool_call_id,
            name=part.tool_name,
            arguments='',
            status='in_progress',
        )
        output_index = self._output_index
        self._open_function_calls[part.tool_call_id] = (item, output_index)

        yield ResponseOutputItemAddedEvent(
            type='response.output_item.added',
            item=item,
            output_index=output_index,
            sequence_number=self._next_seq(),
        )

        if part.args:
            initial = part.args_as_json_str()
            item.arguments = initial
            yield ResponseFunctionCallArgumentsDeltaEvent(
                type='response.function_call_arguments.delta',
                item_id=item_id,
                output_index=output_index,
                delta=initial,
                sequence_number=self._next_seq(),
            )

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[Any]:
        tool_call_id = delta.tool_call_id
        if not tool_call_id or tool_call_id not in self._open_function_calls:
            return
        delta_text = delta.args_delta if isinstance(delta.args_delta, str) else ''
        if not delta_text:
            return
        item, output_index = self._open_function_calls[tool_call_id]
        item.arguments += delta_text
        yield ResponseFunctionCallArgumentsDeltaEvent(
            type='response.function_call_arguments.delta',
            item_id=item.id or '',
            output_index=output_index,
            delta=delta_text,
            sequence_number=self._next_seq(),
        )

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[Any]:
        async for ev in self._finalize_function_call(part.tool_call_id, final_args=part.args_as_json_str()):
            yield ev

    async def _finalize_function_call(self, tool_call_id: str, *, final_args: str) -> AsyncIterator[Any]:
        if tool_call_id not in self._open_function_calls:
            return
        item, output_index = self._open_function_calls.pop(tool_call_id)
        item.arguments = final_args
        item.status = 'completed'

        yield ResponseFunctionCallArgumentsDoneEvent(
            type='response.function_call_arguments.done',
            item_id=item.id or '',
            name=item.name,
            output_index=output_index,
            arguments=final_args,
            sequence_number=self._next_seq(),
        )
        yield ResponseOutputItemDoneEvent(
            type='response.output_item.done',
            item=item,
            output_index=output_index,
            sequence_number=self._next_seq(),
        )
        self._output_index += 1

    async def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[Any]:
        item = self._build_builtin_item(part, status='in_progress')
        if item is None:
            return

        async for ev in self._close_open_message_item():
            yield ev

        output_index = self._output_index
        self._open_builtin_calls[part.tool_call_id] = (item, output_index)

        yield ResponseOutputItemAddedEvent(
            type='response.output_item.added',
            item=item,
            output_index=output_index,
            sequence_number=self._next_seq(),
        )
        async for ev in self._builtin_in_progress_event(part.tool_name, item, output_index):
            yield ev

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[Any]:
        # `BuiltinToolReturnPart` carries the result; closing the item is deferred to that handler.
        return
        yield  # Make this an async generator

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[Any]:
        if part.tool_call_id not in self._open_builtin_calls:
            return

        item, output_index = self._open_builtin_calls.pop(part.tool_call_id)
        item.status = 'completed'

        async for ev in self._builtin_completed_event(part.tool_name, item, output_index):
            yield ev
        yield ResponseOutputItemDoneEvent(
            type='response.output_item.done',
            item=item,
            output_index=output_index,
            sequence_number=self._next_seq(),
        )
        self._output_index += 1

    @staticmethod
    def _build_builtin_item(
        part: BuiltinToolCallPart, *, status: Literal['in_progress', 'completed']
    ) -> _BuiltinItem | None:
        item_id = part.tool_call_id
        match part.tool_name:
            case 'web_search':
                return ResponseFunctionWebSearch(
                    id=item_id,
                    type='web_search_call',
                    status=status,
                    action=ActionSearch(type='search', query=''),
                )
            case 'code_execution':
                return ResponseCodeInterpreterToolCall(
                    id=item_id,
                    type='code_interpreter_call',
                    status=status,
                    container_id='',
                )
            case 'file_search':
                return ResponseFileSearchToolCall(
                    id=item_id,
                    type='file_search_call',
                    status=status,
                    queries=[],
                )
            case 'image_generation':
                return ImageGenerationCall(
                    id=item_id,
                    type='image_generation_call',
                    status=status,
                )
            case _:
                return None

    async def _builtin_in_progress_event(
        self, tool_name: str, item: _BuiltinItem, output_index: int
    ) -> AsyncIterator[Any]:
        item_id = item.id
        seq = self._next_seq()
        match tool_name:
            case 'web_search':
                yield ResponseWebSearchCallInProgressEvent(
                    type='response.web_search_call.in_progress',
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
            case 'code_execution':
                yield ResponseCodeInterpreterCallInProgressEvent(
                    type='response.code_interpreter_call.in_progress',
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
            case 'file_search':
                yield ResponseFileSearchCallInProgressEvent(
                    type='response.file_search_call.in_progress',
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
            case _:
                # 'image_generation' is the only remaining mapped kind; `_build_builtin_item`
                # returns None for any other name so this default arm is unreachable for
                # those — and reachable only for image_generation here.
                yield ResponseImageGenCallInProgressEvent(
                    type='response.image_generation_call.in_progress',
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=seq,
                )

    async def _builtin_completed_event(
        self, tool_name: str, item: _BuiltinItem, output_index: int
    ) -> AsyncIterator[Any]:
        item_id = item.id
        seq = self._next_seq()
        match tool_name:
            case 'web_search':
                yield ResponseWebSearchCallCompletedEvent(
                    type='response.web_search_call.completed',
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
            case 'code_execution':
                yield ResponseCodeInterpreterCallCompletedEvent(
                    type='response.code_interpreter_call.completed',
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
            case 'file_search':
                yield ResponseFileSearchCallCompletedEvent(
                    type='response.file_search_call.completed',
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
            case _:
                yield ResponseImageGenCallCompletedEvent(
                    type='response.image_generation_call.completed',
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=seq,
                )
