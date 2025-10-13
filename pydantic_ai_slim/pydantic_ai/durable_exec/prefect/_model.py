from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
)
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from ._types import TaskConfig, default_task_config


class PrefectStreamedResponse(StreamedResponse):
    """A non-streaming response wrapper for Prefect tasks.

    When a model request is executed inside a Prefect flow, the entire stream
    is consumed within the task, and this wrapper is returned containing the
    final response.
    """

    def __init__(self, model_request_parameters: ModelRequestParameters, response: ModelResponse):
        super().__init__(model_request_parameters)
        self.response = response

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Return an empty iterator since the stream has already been consumed."""
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


class PrefectModel(WrapperModel):
    """A wrapper for Model that integrates with Prefect, turning request and request_stream into Prefect tasks."""

    def __init__(
        self,
        model: Any,
        *,
        task_config: TaskConfig,
        event_stream_handler: EventStreamHandler[Any] | None = None,
    ):
        super().__init__(model)
        self.task_config = default_task_config | (task_config or {})
        self.event_stream_handler = event_stream_handler

        @task
        async def wrapped_request(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            response = await super(PrefectModel, self).request(messages, model_settings, model_request_parameters)
            return response

        self._wrapped_request = wrapped_request

        @task
        async def request_stream_task(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            ctx: RunContext[Any] | None,
        ) -> ModelResponse:
            async with super(PrefectModel, self).request_stream(
                messages, model_settings, model_request_parameters, ctx
            ) as streamed_response:
                if self.event_stream_handler is not None:
                    assert ctx is not None, (
                        'A Prefect model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. '
                        'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
                    )
                    await self.event_stream_handler(ctx, streamed_response)

                # Consume the entire stream
                async for _ in streamed_response:
                    pass
            response = streamed_response.get()
            return response

        self._wrapped_request_stream = request_stream_task

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a model request, wrapped as a Prefect task when in a flow."""
        return await self._wrapped_request.with_options(
            name=f'Model Request: {self.wrapped.model_name}', **self.task_config
        )(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming model request.

        When inside a Prefect flow, the stream is consumed within a task and
        a non-streaming response is returned. When not in a flow, behaves normally.
        """
        # Check if we're in a flow context
        flow_run_context = FlowRunContext.get()

        # If not in a flow, just call the wrapped request_stream method
        if flow_run_context is None:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                yield streamed_response
                return

        # If in a flow, consume the stream in a task and return the final response
        response = await self._wrapped_request_stream.with_options(
            name=f'Model Request (Streaming): {self.wrapped.model_name}', **self.task_config
        )(messages, model_settings, model_request_parameters, run_context)
        yield PrefectStreamedResponse(model_request_parameters, response)
