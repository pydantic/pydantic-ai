from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

from dbos import DBOS

from pydantic_ai import (
    ModelMessage,
    ModelResponse,
)
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
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
        get_event_stream_handler: Callable[[], EventStreamHandler[Any] | None],
    ):
        super().__init__(model)
        self.step_config = step_config
        # Resolve the effective event stream handler lazily inside the step so that a per-run
        # handler (set on a `ContextVar` by `DBOSAgent`) is picked up without rebuilding the model
        # and re-registering its DBOS steps.
        self._get_event_stream_handler = get_event_stream_handler
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
            event_stream_handler = self._get_event_stream_handler()
            async with super(DBOSModel, self).request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                if event_stream_handler is not None:
                    assert run_context is not None, (
                        'A DBOS model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
                    )
                    await event_stream_handler(run_context, streamed_response)

                async for _ in streamed_response:
                    pass
            return streamed_response.get()

        self._dbos_wrapped_request_stream_step = wrapped_request_stream_step

        # Wrap the server-side suspended/background response teardown in a DBOS step. It performs a
        # raw HTTP call to the provider to cancel the job, so it must run as a step (durable,
        # retried, recorded) rather than inline in the workflow.
        @DBOS.step(
            name=f'{self._step_name_prefix}__model.cancel_suspended_response',
            **self.step_config,
        )
        async def wrapped_cancel_suspended_response_step(response: ModelResponse) -> None:
            await super(DBOSModel, self).cancel_suspended_response(response)

        self._dbos_wrapped_cancel_suspended_response_step = wrapped_cancel_suspended_response_step

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await self._dbos_wrapped_request_step(messages, model_settings, model_request_parameters)

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        await self._dbos_wrapped_cancel_suspended_response_step(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
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
        # Without an `event_stream_handler`, the step drained and discarded the real stream's events
        # (e.g. `agent.iter` inside a workflow, where the caller drives the workflow-side stream via
        # `node.stream(...)`/`stream_text()`). Replay the response's parts as events so that stream
        # produces content. With a handler, events were already delivered inside the step, so the
        # workflow-side stream stays empty to avoid delivering them twice.
        yield CompletedStreamedResponse(
            response,
            model_request_parameters=model_request_parameters,
            replay_events=self._get_event_stream_handler() is None,
        )
