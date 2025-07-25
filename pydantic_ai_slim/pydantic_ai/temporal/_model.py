from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow

from .._run_context import RunContext
from ..agent import EventStreamHandler
from ..exceptions import UserError
from ..messages import (
    FinalResultEvent,
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
)
from ..models import Model, ModelRequestParameters, StreamedResponse
from ..settings import ModelSettings
from ..usage import Usage
from ._settings import TemporalSettings


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any


class _TemporalStreamedResponse(StreamedResponse):
    def __init__(self, response: ModelResponse):
        self.response = response

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        return
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        """Build a [`ModelResponse`][pydantic_ai.messages.ModelResponse] from the data received from the stream so far."""
        return self.response

    def usage(self) -> Usage:
        """Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
        return self.response.usage

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self.response.model_name or ''

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self.response.timestamp


def temporalize_model(  # noqa: C901
    model: Model,
    settings: TemporalSettings | None = None,
    event_stream_handler: EventStreamHandler | None = None,
) -> list[Callable[..., Any]]:
    """Temporalize a model.

    Args:
        model: The model to temporalize.
        settings: The temporal settings to use.
        event_stream_handler: The event stream handler to use.
    """
    if activities := getattr(model, '__temporal_activities', None):
        return activities

    settings = settings or TemporalSettings()

    original_request = model.request
    original_request_stream = model.request_stream

    @activity.defn(name='model_request')
    async def request_activity(params: _RequestParams) -> ModelResponse:
        return await original_request(params.messages, params.model_settings, params.model_request_parameters)

    @activity.defn(name='model_request_stream')
    async def request_stream_activity(params: _RequestParams) -> ModelResponse:
        run_context = settings.deserialize_run_context(params.serialized_run_context)
        async with original_request_stream(
            params.messages, params.model_settings, params.model_request_parameters, run_context
        ) as streamed_response:
            tool_defs = {
                tool_def.name: tool_def
                for tool_def in [
                    *params.model_request_parameters.output_tools,
                    *params.model_request_parameters.function_tools,
                ]
            }

            async def aiter():
                def _get_final_result_event(e: ModelResponseStreamEvent) -> FinalResultEvent | None:
                    """Return an appropriate FinalResultEvent if `e` corresponds to a part that will produce a final result."""
                    if isinstance(e, PartStartEvent):
                        new_part = e.part
                        if (
                            isinstance(new_part, TextPart) and params.model_request_parameters.allow_text_output
                        ):  # pragma: no branch
                            return FinalResultEvent(tool_name=None, tool_call_id=None)
                        elif isinstance(new_part, ToolCallPart) and (tool_def := tool_defs.get(new_part.tool_name)):
                            if tool_def.kind == 'output':
                                return FinalResultEvent(
                                    tool_name=new_part.tool_name, tool_call_id=new_part.tool_call_id
                                )
                            elif tool_def.kind == 'deferred':
                                return FinalResultEvent(tool_name=None, tool_call_id=None)

                # TODO: usage_checking_stream = _get_usage_checking_stream_response(
                #     self._raw_stream_response, self._usage_limits, self.usage
                # )
                async for event in streamed_response:
                    yield event
                    if (final_result_event := _get_final_result_event(event)) is not None:
                        yield final_result_event
                        break

                # If we broke out of the above loop, we need to yield the rest of the events
                # If we didn't, this will just be a no-op
                async for event in streamed_response:
                    yield event

            assert event_stream_handler is not None
            await event_stream_handler(run_context, aiter())

            async for _ in streamed_response:
                pass
        return streamed_response.get()

    async def request(
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=request_activity,
            arg=_RequestParams(
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
                serialized_run_context=None,
            ),
            **settings.execute_activity_options,
        )

    @asynccontextmanager
    async def request_stream(
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        if event_stream_handler is None:
            raise UserError('Streaming with Temporal requires `Agent` to have an `event_stream_handler`')
        if run_context is None:
            raise UserError('Streaming with Temporal requires `request_stream` to be called with a `run_context`')

        serialized_run_context = settings.serialize_run_context(run_context)
        response = await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=request_stream_activity,
            arg=_RequestParams(
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
                serialized_run_context=serialized_run_context,
            ),
            **settings.execute_activity_options,
        )
        yield _TemporalStreamedResponse(response)

    model.request = request
    model.request_stream = request_stream

    activities = [request_activity, request_stream_activity]
    setattr(model, '__temporal_activities', activities)
    return activities
