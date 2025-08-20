from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from dbos import DBOS, DBOSConfiguredInstance
from typing_extensions import TypedDict

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage


class StepConfig(TypedDict, total=False):
    """Configuration for a step in the DBOS workflow."""

    retries_allowed: bool
    interval_seconds: float
    max_attempts: int
    backoff_rate: float


class DBOSStreamedResponse(StreamedResponse):
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
    def timestamp(self) -> datetime:
        return self.response.timestamp  # pragma: no cover


@DBOS.dbos_class()
class DBOSModel(WrapperModel, DBOSConfiguredInstance):
    def __init__(
        self,
        model: Model,
        *,
        step_name_prefix: str,
        step_config: StepConfig,
        event_stream_handler: EventStreamHandler[Any] | None = None,
    ):
        super().__init__(model)
        self.step_config = step_config
        self.event_stream_handler = event_stream_handler
        self._step_name_prefix = step_name_prefix

        DBOSConfiguredInstance.__init__(self, f'{step_name_prefix}__model')

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        print(f'Running DBOSModel request with step config: {self.step_config}')

        # Wrap the request in a DBOS step.
        @DBOS.step(
            name=f'{self._step_name_prefix}__model.request',
            retries_allowed=self.step_config.get('retries_allowed', False),
        )
        async def wrapped_request_step(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return await super(DBOSModel, self).request(messages, model_settings, model_request_parameters)

        return await wrapped_request_step(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        # If not in a workflow (could be in a step), just call the wrapped request_stream method.
        if DBOS.workflow_id is None or DBOS.step_id is not None:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                yield streamed_response
                return

        # TODO (Qian): this might not be the right way to handle stream.
        @DBOS.step(
            name=f'{self._step_name_prefix}__model.request_stream',
            retries_allowed=self.step_config.get('retries_allowed', False),
        )
        async def wrapped_request_stream_step(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any] | None = None,
        ) -> ModelResponse:
            # TODO(Qian): do we need to check event stream handler here?
            # assert self.event_stream_handler is not None, (
            #     'Event stream handler must be set for streamed requests when using DBOS model.'
            # )
            assert run_context is not None, 'Run context must be provided for streamed requests when using DBOS model.'

            async with super(DBOSModel, self).request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                if self.event_stream_handler is not None:
                    await self.event_stream_handler(run_context, streamed_response)

                async for _ in streamed_response:
                    pass
            return streamed_response.get()

        response = await wrapped_request_stream_step(messages, model_settings, model_request_parameters, run_context)
        yield DBOSStreamedResponse(model_request_parameters, response)
