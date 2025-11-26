from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
)
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RequestUsage

from ._run_context import TemporalRunContext


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    messages: list[ModelMessage]
    # `model_settings` can't be a `ModelSettings` because Temporal would end up dropping fields only defined on its subclasses.
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any


class TemporalStreamedResponse(StreamedResponse):
    def __init__(self, model_request_parameters: ModelRequestParameters, response: ModelResponse):
        super().__init__(model_request_parameters)
        self.response = response

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        return
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        return self.response

    def usage(self) -> RequestUsage:
        return self.response.usage  # pragma: no cover

    @property
    def model_name(self) -> str:
        return self.response.model_name or ''  # pragma: no cover

    @property
    def provider_name(self) -> str:
        return self.response.provider_name or ''  # pragma: no cover

    @property
    def timestamp(self) -> datetime:
        return self.response.timestamp  # pragma: no cover


class TemporalModel(WrapperModel):
    def __init__(
        self,
        model: Model,
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
        event_stream_handler: EventStreamHandler[Any] | None = None,
    ):
        super().__init__(model)
        self.activity_config = activity_config
        self.run_context_type = run_context_type
        self.event_stream_handler = event_stream_handler

        @activity.defn(name=f'{activity_name_prefix}__model_request')
        async def request_activity(params: _RequestParams) -> ModelResponse:
            return await self.wrapped.request(
                params.messages,
                cast(ModelSettings | None, params.model_settings),
                params.model_request_parameters,
            )

        self.request_activity = request_activity

        async def request_stream_activity(params: _RequestParams, deps: AgentDepsT) -> ModelResponse:
            # An error is raised in `request_stream` if no `event_stream_handler` is set.
            assert self.event_stream_handler is not None

            run_context = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)
            async with self.wrapped.request_stream(
                params.messages,
                cast(ModelSettings | None, params.model_settings),
                params.model_request_parameters,
                run_context,
            ) as streamed_response:
                await self.event_stream_handler(run_context, streamed_response)

                async for _ in streamed_response:
                    pass
            return streamed_response.get()

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        request_stream_activity.__annotations__['deps'] = deps_type

        self.request_stream_activity = activity.defn(name=f'{activity_name_prefix}__model_request_stream')(
            request_stream_activity
        )

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

        self._validate_model_request_parameters(model_request_parameters)

        return await workflow.execute_activity(
            activity=self.request_activity,
            arg=_RequestParams(
                messages=messages,
                model_settings=cast(dict[str, Any] | None, model_settings),
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
        if not workflow.in_workflow():
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                yield streamed_response
                return

        if run_context is None:
            raise UserError(
                'A Temporal model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        # We can never get here without an `event_stream_handler`, as `TemporalAgent.run_stream` and `TemporalAgent.iter` raise an error saying to use `TemporalAgent.run` instead,
        # and that only calls `request_stream` if `event_stream_handler` is set.
        assert self.event_stream_handler is not None

        self._validate_model_request_parameters(model_request_parameters)

        serialized_run_context = self.run_context_type.serialize_run_context(run_context)
        response = await workflow.execute_activity(
            activity=self.request_stream_activity,
            args=[
                _RequestParams(
                    messages=messages,
                    model_settings=cast(dict[str, Any] | None, model_settings),
                    model_request_parameters=model_request_parameters,
                    serialized_run_context=serialized_run_context,
                ),
                run_context.deps,
            ],
            **self.activity_config,
        )
        yield TemporalStreamedResponse(model_request_parameters, response)

    def _validate_model_request_parameters(self, model_request_parameters: ModelRequestParameters) -> None:
        if model_request_parameters.allow_image_output:
            raise UserError('Image output is not supported with Temporal because of the 2MB payload size limit.')
