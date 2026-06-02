from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dbos import DBOS

from pydantic_ai import (
    ModelMessage,
    ModelResponse,
)
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import CompletedStreamedResponse, WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from ._utils import StepConfig


class DBOSModel(WrapperModel):
    """A wrapper for Model that integrates with DBOS, turning request and request_stream to DBOS steps."""

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

        # Wrap the request in a DBOS step.
        @DBOS.step(
            name=f'{self._step_name_prefix}__model.request',
            **self.step_config,
        )
        async def wrapped_request_step(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return await super(DBOSModel, self).request(messages, model_settings, model_request_parameters)

        self._dbos_wrapped_request_step = wrapped_request_step

        # Wrap the request_stream in a DBOS step.
        @DBOS.step(
            name=f'{self._step_name_prefix}__model.request_stream',
            **self.step_config,
        )
        async def wrapped_request_stream_step(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any] | None = None,
        ) -> ModelResponse:
            async with super(DBOSModel, self).request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                if self.event_stream_handler is not None:
                    assert run_context is not None, (
                        'A DBOS model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
                    )
                    await self.event_stream_handler(run_context, streamed_response)

                async for _ in streamed_response:
                    pass
            return streamed_response.get()

        self._dbos_wrapped_request_stream_step = wrapped_request_stream_step

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await self._dbos_wrapped_request_step(messages, model_settings, model_request_parameters)

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

        response = await self._dbos_wrapped_request_stream_step(
            messages, model_settings, model_request_parameters, run_context
        )
        yield CompletedStreamedResponse(model_request_parameters, response)
