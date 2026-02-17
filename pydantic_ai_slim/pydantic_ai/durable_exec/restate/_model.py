from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from pydantic.errors import PydanticUserError

from pydantic_ai.agent.abstract import EventStreamHandler
from pydantic_ai.durable_exec.restate._serde import PydanticTypeAdapter
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from ._restate_types import Context, RunOptions, TerminalError

MODEL_RESPONSE_SERDE = PydanticTypeAdapter(ModelResponse)


class RestateStreamedResponse(StreamedResponse):
    def __init__(self, model_request_parameters: ModelRequestParameters, response: ModelResponse):
        super().__init__(model_request_parameters)
        self.response = response

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        # Restate wraps a completed `ModelResponse`; streamed events are not replayed.
        return
        yield  # pragma: no cover

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
    def provider_url(self) -> str | None:
        return self.response.provider_url  # pragma: no cover

    @property
    def timestamp(self) -> datetime:
        return self.response.timestamp  # pragma: no cover


class RestateModelWrapper(WrapperModel):
    def __init__(
        self,
        wrapped: Model,
        context: Context,
        event_stream_handler: EventStreamHandler[Any] | None = None,
        max_attempts: int | None = None,
    ):
        super().__init__(wrapped)
        self._options = RunOptions[ModelResponse](serde=MODEL_RESPONSE_SERDE, max_attempts=max_attempts)
        self._context = context
        self._event_stream_handler = event_stream_handler

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        async def request_run() -> ModelResponse:
            try:
                return await self.wrapped.request(messages, model_settings, model_request_parameters)
            except (UserError, PydanticUserError) as e:
                raise TerminalError(str(e)) from e

        return await self._context.run_typed('Model call', request_run, self._options)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        if run_context is None:
            raise TerminalError(
                'A model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        fn = self._event_stream_handler
        if fn is None:
            raise TerminalError(
                'A Restate model requires an `event_stream_handler` to be set on `RestateAgent` at creation time. '
                'Set `event_stream_handler=...` on the agent and use `agent.run()` instead.'
            )

        async def request_stream_run() -> ModelResponse:
            try:
                # `run_context` (and therefore `deps`) are captured in this closure.
                # Unlike Temporal, Restate does not require serializing the run context across durable boundaries —
                # only the returned `ModelResponse` is serialized via `MODEL_RESPONSE_SERDE`.
                async with self.wrapped.request_stream(
                    messages,
                    model_settings,
                    model_request_parameters,
                    run_context,
                ) as streamed_response:
                    # Run the handler inside the durable step so any handler side effects are recorded/deduplicated
                    # by Restate. The handler should still be written to be idempotent since this step may be retried
                    # if it fails before completion.
                    # Note: `streamed_response` yields `ModelResponseStreamEvent`, which is part of `AgentStreamEvent`.
                    await fn(run_context, streamed_response)

                    async for _ in streamed_response:
                        pass
                return streamed_response.get()
            except (UserError, PydanticUserError) as e:
                raise TerminalError(str(e)) from e

        response = await self._context.run_typed('Model stream call', request_stream_run, self._options)

        yield RestateStreamedResponse(model_request_parameters, response)
