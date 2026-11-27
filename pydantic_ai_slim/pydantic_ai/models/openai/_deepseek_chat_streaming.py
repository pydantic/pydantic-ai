"""Stream DeepSeek's server-side web tool events through the Chat Completions parts manager."""

from collections.abc import Iterator
from dataclasses import dataclass, field

from openai.types.chat import ChatCompletionChunk, chat_completion_chunk

from ...messages import ModelResponseStreamEvent, NativeToolReturnPart
from ._chat_streaming import OpenAIStreamedResponse
from ._deepseek_web_tools import (
    CHAT_STEP_KEY,
    CHAT_STEPS_KEY,
    ChatWebCall,
    ChatWebEvent,
    chat_web_step_metadata,
    chat_web_steps,
    check_chat_web_error,
)


@dataclass
class DeepSeekChatStreamedResponse(OpenAIStreamedResponse):
    _deepseek_started_calls: dict[str, ChatWebCall] = field(default_factory=dict[str, ChatWebCall], init=False)

    def _map_top_level_fields(self, chunk: ChatCompletionChunk) -> None:
        super()._map_top_level_fields(chunk)
        check_chat_web_error((chunk.model_extra or {}).get('error'), self.model_name, in_stream=True)

    def _map_part_delta(self, choice: chat_completion_chunk.Choice) -> Iterator[ModelResponseStreamEvent]:
        yield from self._map_deepseek_web_chunk(choice)
        yield from super()._map_part_delta(choice)

    def _map_deepseek_web_chunk(self, choice: chat_completion_chunk.Choice) -> Iterator[ModelResponseStreamEvent]:
        if raw_event := (choice.model_extra or {}).get('server_tool_event'):
            yield from self._map_deepseek_web_progress(ChatWebEvent.model_validate(raw_event))
        delta = choice.delta.model_extra or {}
        if not (raw_steps := delta.get('server_tool_calls')):
            return
        steps = chat_web_steps.validate_python(raw_steps)
        self.provider_details = {
            **(self.provider_details or {}),
            CHAT_STEPS_KEY: chat_web_step_metadata(steps),
        }
        for step in steps:
            for call in step.calls:
                call_id = ('deepseek_call', call.id)
                if self._parts_manager.get_part_by_vendor_id(call_id) is None:
                    yield self._parts_manager.handle_part(
                        vendor_part_id=call_id, part=call.call_part(self.provider_name)
                    )
                yield from self._map_deepseek_web_result(call)

    def _map_deepseek_web_progress(self, event: ChatWebEvent) -> Iterator[ModelResponseStreamEvent]:
        if event.type == 'step_reasoning_done':
            key = ('reasoning', 'reasoning_content', self._reasoning_generation)
            if self._parts_manager.get_part_by_vendor_id(key) is not None:
                yield from self._parts_manager.handle_thinking_delta(
                    vendor_part_id=key, provider_details={CHAT_STEP_KEY: event.step}
                )
            self._advance_part_boundary('tool')
        elif event.type == 'call_started':
            call = ChatWebCall.model_validate(event.call)
            self._deepseek_started_calls[call.id] = call
            yield self._parts_manager.handle_part(
                vendor_part_id=('deepseek_call', call.id), part=call.call_part(self.provider_name)
            )
        elif event.call and isinstance(call_id := event.call.get('id'), str):
            if started_call := self._deepseek_started_calls.pop(call_id, None):
                # Progress completion provides result fields but omits the original action and signature.
                completed_call = ChatWebCall.model_validate(started_call.model_dump(exclude_unset=True) | event.call)
                yield from self._map_deepseek_web_result(completed_call)

    def _map_deepseek_web_result(self, call: ChatWebCall) -> Iterator[ModelResponseStreamEvent]:
        result_id = ('deepseek_result', call.id)
        result = call.return_part(self.provider_name)
        previous = self._parts_manager.get_part_by_vendor_id(result_id)
        if not isinstance(previous, NativeToolReturnPart) or (previous.content, previous.outcome) != (
            result.content,
            result.outcome,
        ):
            yield self._parts_manager.handle_part(vendor_part_id=result_id, part=result)
