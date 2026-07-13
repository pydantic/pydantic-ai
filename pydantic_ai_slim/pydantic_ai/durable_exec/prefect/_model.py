from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai import (
    ModelMessage,
    ModelResponse,
)
from pydantic_ai._utils import fill_run_metadata
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import CompletedStreamedResponse, WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from ._types import TaskConfig, default_task_config


def _stamp_response_provenance(response: ModelResponse, messages: list[ModelMessage]) -> None:
    """Stamp the producing run's `run_id`/`conversation_id` on the response before Prefect persists it.

    The agent graph only fills these after the task returns, so without this the cached payload has
    them unset and a cache replay in a different conversation would be re-stamped as if it were
    produced there. Server-side state guards (e.g. OpenAI `openai_conversation_id='auto'`) rely on a
    replayed response keeping its original `conversation_id` to avoid continuing another
    conversation's provider-side state.
    """
    if messages:  # pragma: no branch
        final_request = messages[-1]
        fill_run_metadata(response, run_id=final_request.run_id, conversation_id=final_request.conversation_id)


class PrefectModel(WrapperModel):
    """A wrapper for Model that integrates with Prefect, turning request and request_stream into Prefect tasks."""

    def __init__(
        self,
        model: Any,
        *,
        task_config: TaskConfig,
        get_event_stream_handler: Callable[[], EventStreamHandler[Any] | None],
    ):
        super().__init__(model)
        self.task_config = default_task_config | (task_config or {})
        # Resolve the effective event stream handler lazily inside the task so that a per-run
        # handler (set on a `ContextVar` by `PrefectAgent`) is picked up without rebuilding the model
        # and re-registering its Prefect tasks.
        self._get_event_stream_handler = get_event_stream_handler

        @task
        async def wrapped_request(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            response = await super(PrefectModel, self).request(messages, model_settings, model_request_parameters)
            _stamp_response_provenance(response, messages)
            return response

        self._wrapped_request = wrapped_request

        @task
        async def request_stream_task(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            ctx: RunContext[Any] | None,
        ) -> ModelResponse:
            event_stream_handler = self._get_event_stream_handler()
            async with super(PrefectModel, self).request_stream(
                messages, model_settings, model_request_parameters, ctx
            ) as streamed_response:
                if event_stream_handler is not None:
                    assert ctx is not None, (
                        'A Prefect model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. '
                        'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
                    )
                    await event_stream_handler(ctx, streamed_response)

                # Consume the entire stream
                async for _ in streamed_response:
                    pass
            response = streamed_response.get()
            _stamp_response_provenance(response, messages)
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
    ) -> AsyncGenerator[StreamedResponse]:
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
        yield CompletedStreamedResponse(model_request_parameters, response)
