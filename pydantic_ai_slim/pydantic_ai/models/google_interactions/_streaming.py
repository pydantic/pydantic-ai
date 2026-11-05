"""Step-indexed Google Interactions streams."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from google.genai import interactions as gi
from pydantic import JsonValue, TypeAdapter

from ... import _utils
from ...exceptions import UnexpectedModelBehavior
from ...messages import ModelResponseStreamEvent, TextPart
from ...native_tools import MCPServerTool
from .. import StreamedResponse
from ._response import JSON_OBJECT, build_response, map_step, mark_step

_STEP = TypeAdapter(gi.Step)
_OBJECT_LIST = TypeAdapter(list[dict[str, JsonValue]])


@dataclass
class GoogleInteractionsStreamedResponse(StreamedResponse):
    """A streamed Interactions response retaining complete steps for stateless replay."""

    _response: AsyncIterator[gi.InteractionSSEEvent]
    _model_name: str
    _provider_name: str
    _provider_url: str
    _timestamp: datetime = field(default_factory=_utils.now_utc)
    _steps: dict[int, dict[str, JsonValue]] = field(default_factory=dict[int, dict[str, JsonValue]])
    _arguments: dict[int, str] = field(default_factory=dict[int, str])
    _completed: bool = False
    _server_calls: set[str] = field(default_factory=set[str])
    _media_steps: set[int] = field(init=False, default_factory=set[int])
    _deferred_events: list[gi.InteractionSSEEvent] = field(init=False, default_factory=list[gi.InteractionSSEEvent])

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in self._response:
            for value in self._ready_events(event):
                for result in await self._map_event(value):
                    yield result
        if not self._completed:
            raise UnexpectedModelBehavior('Google Interactions stream ended before `interaction.completed`.')

    def _ready_events(self, event: gi.InteractionSSEEvent) -> Iterable[gi.InteractionSSEEvent]:
        if isinstance(event, gi.ErrorEvent) or not (self._deferred_events or self._is_mcp_candidate(event)):
            return (event,)
        # A namespace alone does not prove execution. Retain event order until results establish it.
        self._deferred_events.append(event)
        if not isinstance(event, gi.InteractionCompletedEvent):
            return ()
        self._server_calls.update(
            value.step.call_id
            for value in self._deferred_events
            if isinstance(value, gi.StepStart) and isinstance(value.step, gi.FunctionResultStep)
        )
        events = self._deferred_events
        self._deferred_events = []
        return events

    def _is_mcp_candidate(self, event: gi.InteractionSSEEvent) -> bool:
        return (
            isinstance(event, gi.StepStart)
            and isinstance(event.step, gi.FunctionCallStep)
            and event.step.name not in self.model_request_parameters.declared_tool_defs
            and any(
                isinstance(tool, MCPServerTool) and event.step.name.startswith(f'{tool.id}:')
                for tool in self.model_request_parameters.native_tools
            )
        )

    async def _map_event(self, event: gi.InteractionSSEEvent) -> Iterable[ModelResponseStreamEvent]:
        if isinstance(event, gi.StepStart):
            return await self._map_start(event)
        elif isinstance(event, gi.StepDelta):
            return self._map_delta(event)
        elif isinstance(event, gi.StepStop):
            return await self._map_stop(event)
        elif isinstance(event, gi.InteractionCompletedEvent):
            return self._map_completed(event)
        elif isinstance(event, (gi.InteractionCreatedEvent, gi.InteractionStatusUpdate)):
            return ()
        elif isinstance(event, gi.ErrorEvent):
            raise UnexpectedModelBehavior(f'Google Interactions stream failed: {event}')
        else:
            raise UnexpectedModelBehavior(f'Unexpected Google Interactions stream event: {event}')

    async def _map_start(self, event: gi.StepStart) -> list[ModelResponseStreamEvent]:
        step = event.step
        self._steps[event.index] = JSON_OBJECT.validate_python(
            step.model_dump(mode='json', by_alias=True, exclude_none=True)
        )
        if isinstance(step, gi.FunctionCallStep):
            if step.id in self._server_calls:
                return []
            return [
                self._parts_manager.handle_tool_call_part(
                    vendor_part_id=event.index,
                    tool_name=step.name,
                    args=json.dumps(step.arguments) if step.arguments else '',
                    tool_call_id=step.id,
                    provider_name=self.provider_name,
                )
            ]
        elif isinstance(step, gi.ThoughtStep):
            return list(
                self._parts_manager.handle_thinking_delta(
                    vendor_part_id=event.index,
                    content=''.join(item.text for item in step.summary or [] if isinstance(item, gi.TextContent)),
                    signature=step.signature,
                    provider_name=self.provider_name,
                )
            )
        elif isinstance(step, gi.ModelOutputStep):
            return [
                self._parts_manager.handle_part(vendor_part_id=(event.index, index), part=part)
                for index, part in enumerate(
                    mark_step(
                        await map_step(step, self.provider_name, server_calls=frozenset(self._server_calls)),
                        event.index,
                    )
                )
            ]
        return []

    async def _map_stop(self, event: gi.StepStop) -> list[ModelResponseStreamEvent]:
        if event.index in self._arguments:
            self._steps[event.index]['arguments'] = JSON_OBJECT.validate_json(self._arguments[event.index])
        step = _STEP.validate_python(self._steps[event.index])
        details = {'google_interactions_step_index': event.index, 'google_interactions_content_index': 0}
        if isinstance(step, gi.ThoughtStep):
            return list(
                self._parts_manager.handle_thinking_delta(
                    vendor_part_id=event.index, provider_name=self.provider_name, provider_details=details
                )
            )
        elif isinstance(step, gi.FunctionCallStep) and step.id not in self._server_calls:
            return list(
                self._parts_manager.handle_tool_call_delta_events(
                    vendor_part_id=event.index, provider_name=self.provider_name, provider_details=details
                )
            )
        events: list[ModelResponseStreamEvent] = []
        for index, part in enumerate(
            mark_step(await map_step(step, self.provider_name, server_calls=frozenset(self._server_calls)), event.index)
        ):
            if (
                isinstance(step, gi.ModelOutputStep)
                and isinstance(part, TextPart)
                and self._parts_manager.get_part_by_vendor_id((event.index, index)) is not None
            ):
                events.extend(
                    self._parts_manager.handle_text_delta(
                        vendor_part_id=(event.index, index),
                        content='',
                        citations=part.citations,
                        provider_name=self.provider_name,
                        provider_details=part.provider_details,
                    )
                )
            else:
                events.append(self._parts_manager.handle_part(vendor_part_id=(event.index, index), part=part))
        return events

    def _map_completed(self, event: gi.InteractionCompletedEvent) -> Iterable[ModelResponseStreamEvent]:
        raw = JSON_OBJECT.validate_python(event.interaction.model_dump(mode='json', by_alias=True, exclude_none=True))
        raw['steps'] = [self._steps[index] for index in sorted(self._steps)]
        response = build_response(
            gi.Interaction.model_validate(raw),
            [],
            self.model_name,
            self.provider_name,
            self.provider_url,
            self.canonical_system,
        )
        self.provider_details = response.provider_details
        self.provider_response_id = response.provider_response_id
        self.finish_reason = response.finish_reason
        self._usage = response.usage
        self._timestamp = response.timestamp
        self._completed = True
        return ()

    def _map_delta(self, event: gi.StepDelta) -> Iterable[ModelResponseStreamEvent]:
        delta = event.delta
        if isinstance(delta, gi.TextDelta):
            return self._map_text(event.index, delta)
        elif isinstance(delta, gi.ThoughtSignatureDelta):
            return self._map_signature(event.index, delta)
        elif isinstance(delta, gi.ThoughtSummaryDelta):
            return self._map_summary(event.index, delta)
        elif isinstance(delta, gi.ArgumentsDelta):
            return self._map_arguments(event.index, delta)
        elif isinstance(delta, (gi.ImageDelta, gi.AudioDelta, gi.VideoDelta, gi.DocumentDelta)):
            self._map_media(event.index, delta)
        elif isinstance(delta, gi.TextAnnotationDelta):
            self._map_annotations(event.index, delta)
        else:
            values = JSON_OBJECT.validate_python(delta.model_dump(mode='json', by_alias=True, exclude_none=True))
            values.pop('type', None)
            self._steps[event.index].update(values)
        return ()

    def _map_text(self, index: int, delta: gi.TextDelta) -> Iterable[ModelResponseStreamEvent]:
        step = self._steps[index]
        content = _OBJECT_LIST.validate_python(step.get('content', []))
        if content and content[-1].get('type') == 'text':
            content[-1]['text'] = str(content[-1].get('text', '')) + delta.text
        else:
            content.append({'type': 'text', 'text': delta.text})
        step['content'] = [value for value in content]
        if index in self._media_steps:
            return ()
        return self._parts_manager.handle_text_delta(
            vendor_part_id=(index, len(content) - 1), content=delta.text, provider_name=self.provider_name
        )

    def _map_signature(self, index: int, delta: gi.ThoughtSignatureDelta) -> Iterable[ModelResponseStreamEvent]:
        signature = str(self._steps[index].get('signature', '')) + (delta.signature or '')
        self._steps[index]['signature'] = signature
        return self._parts_manager.handle_thinking_delta(
            vendor_part_id=index, signature=signature, provider_name=self.provider_name
        )

    def _map_summary(self, index: int, delta: gi.ThoughtSummaryDelta) -> Iterable[ModelResponseStreamEvent]:
        if not isinstance(delta.content, gi.TextContent):
            raise UnexpectedModelBehavior('Expected text in a Google thought summary.')
        summary = _OBJECT_LIST.validate_python(self._steps[index].get('summary', []))
        summary.append({'type': 'text', 'text': delta.content.text})
        self._steps[index]['summary'] = [value for value in summary]
        return self._parts_manager.handle_thinking_delta(
            vendor_part_id=index, content=delta.content.text, provider_name=self.provider_name
        )

    def _map_arguments(self, index: int, delta: gi.ArgumentsDelta) -> Iterable[ModelResponseStreamEvent]:
        self._arguments[index] = self._arguments.get(index, '') + (delta.arguments or '')
        if self._steps[index].get('id') in self._server_calls:
            return ()
        return self._parts_manager.handle_tool_call_delta_events(
            vendor_part_id=index, args=delta.arguments, provider_name=self.provider_name
        )

    def _map_media(self, index: int, delta: gi.ImageDelta | gi.AudioDelta | gi.VideoDelta | gi.DocumentDelta) -> None:
        # Files have no deltas: defer later content until the complete file can be emitted in order.
        self._media_steps.add(index)
        content = _OBJECT_LIST.validate_python(self._steps[index].get('content', []))
        value = JSON_OBJECT.validate_python(delta.model_dump(mode='json', by_alias=True, exclude_none=True))
        if content and content[-1].get('type') == delta.type:
            previous = content[-1]
            if delta.data and isinstance(previous.get('data'), str):
                value['data'] = base64.b64encode(
                    base64.b64decode(str(previous['data'])) + base64.b64decode(delta.data)
                ).decode()
            previous.update(value)
        else:
            content.append(value)
        self._steps[index]['content'] = [value for value in content]

    def _map_annotations(self, index: int, delta: gi.TextAnnotationDelta) -> None:
        content = _OBJECT_LIST.validate_python(self._steps[index].get('content', []))
        if not content or content[-1].get('type') != 'text':
            raise UnexpectedModelBehavior('Google emitted citations without a text content block.')
        previous = TypeAdapter(list[JsonValue]).validate_python(content[-1].get('annotations', []))
        previous.extend(
            JSON_OBJECT.validate_python(annotation.model_dump(mode='json', by_alias=True, exclude_none=True))
            for annotation in delta.annotations or []
        )
        content[-1]['annotations'] = previous
        self._steps[index]['content'] = [value for value in content]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def provider_url(self) -> str:
        return self._provider_url

    @property
    def timestamp(self) -> datetime:
        return self._timestamp
