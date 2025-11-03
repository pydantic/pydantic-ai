from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from restate import Context, RunOptions

from pydantic_ai.agent.abstract import EventStreamHandler
from pydantic_ai.durable_exec.restate._serde import PydanticTypeAdapter
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RequestUsage

MODEL_RESPONSE_SERDE = PydanticTypeAdapter(ModelResponse)


class RestateStreamedResponse(StreamedResponse):
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


class RestateModelWrapper(WrapperModel):
    def __init__(
        self,
        wrapped: Model,
        context: Context,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        max_attempts: int | None = None,
    ):
        super().__init__(wrapped)
        self._options = RunOptions(serde=MODEL_RESPONSE_SERDE, max_attempts=max_attempts)
        self._context = context
        self._event_stream_handler = event_stream_handler

    async def request(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return await self._context.run_typed('Model call', self.wrapped.request, self._options, *args, **kwargs)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[AgentDepsT] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        if run_context is None:
            raise UserError(
                'A model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        # We can never get here without an `event_stream_handler`, as `TemporalAgent.run_stream` and `TemporalAgent.iter` raise an error saying to use `TemporalAgent.run` instead,
        # and that only calls `request_stream` if `event_stream_handler` is set.
        fn = self._event_stream_handler
        assert fn is not None

        async def request_stream_run():
            async with self.wrapped.request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as streamed_response:
                await fn(run_context, streamed_response)

                async for _ in streamed_response:
                    pass
            return streamed_response.get()

        response = await self._context.run_typed('Model stream call', request_stream_run, self._options)

        yield RestateStreamedResponse(model_request_parameters, response)
