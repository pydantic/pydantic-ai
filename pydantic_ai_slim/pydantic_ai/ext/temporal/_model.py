from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage

from ._run_context import TemporalRunContext


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any


class _TemporalStreamedResponse(StreamedResponse):
    def __init__(self, model_request_parameters: ModelRequestParameters, response: ModelResponse):
        super().__init__(model_request_parameters)
        self.response = response

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        return
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        return self.response

    def usage(self) -> Usage:
        return self.response.usage

    @property
    def model_name(self) -> str:
        return self.response.model_name or ''

    @property
    def timestamp(self) -> datetime:
        return self.response.timestamp


class TemporalModel(WrapperModel):
    def __init__(
        self,
        model: Model,
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        run_context_type: type[TemporalRunContext] = TemporalRunContext,
        event_stream_handler: EventStreamHandler[Any] | None = None,
    ):
        super().__init__(model)
        self.activity_config = activity_config
        self.run_context_type = run_context_type
        self.event_stream_handler = event_stream_handler

        @activity.defn(name=f'{activity_name_prefix}__model_request')
        async def request_activity(params: _RequestParams) -> ModelResponse:
            return await self.wrapped.request(params.messages, params.model_settings, params.model_request_parameters)

        self.request_activity = request_activity

        @activity.defn(name=f'{activity_name_prefix}__model_request_stream')
        async def request_stream_activity(params: _RequestParams) -> ModelResponse:
            # An error is raised in `request_stream` if no `event_stream_handler` is set.
            assert self.event_stream_handler is not None

            run_context = self.run_context_type.deserialize_run_context(params.serialized_run_context)
            async with self.wrapped.request_stream(
                params.messages, params.model_settings, params.model_request_parameters, run_context
            ) as streamed_response:
                await self.event_stream_handler(run_context, streamed_response)

                async for _ in streamed_response:
                    pass
            return streamed_response.get()

        self.request_stream_activity = request_stream_activity

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return [self.request_activity, self.request_stream_activity]

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if not workflow.in_workflow():
            return await super().request(messages, model_settings, model_request_parameters)

        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=self.request_activity,
            arg=_RequestParams(
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
                serialized_run_context=None,
            ),
            **self.activity_config,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        if self.event_stream_handler is None:
            raise UserError('Streaming with Temporal requires `Agent` to have an `event_stream_handler` set.')

        if run_context is None:
            raise UserError('Streaming with Temporal requires `request_stream` to be called with a `run_context`')

        serialized_run_context = self.run_context_type.serialize_run_context(run_context)
        response = await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=self.request_stream_activity,
            arg=_RequestParams(
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
                serialized_run_context=serialized_run_context,
            ),
            **self.activity_config,
        )
        yield _TemporalStreamedResponse(model_request_parameters, response)
