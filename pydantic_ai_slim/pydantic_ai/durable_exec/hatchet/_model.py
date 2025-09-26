from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any

from hatchet_sdk import Context, Hatchet
from pydantic import BaseModel, ConfigDict

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
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RequestUsage

from ._run_context import HatchetRunContext
from ._utils import TaskConfig


class ModelInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters


class ModelStreamInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any
    deps_type_name: str


class HatchetModel(WrapperModel):
    """A wrapper for Model that integrates with Hatchet, turning request and request_stream to Hatchet tasks."""

    def __init__(
        self,
        model: Model,
        *,
        task_name_prefix: str,
        task_config: TaskConfig,
        hatchet: Hatchet,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        deps_type: type[AgentDepsT] | None = None,
        run_context_type: type[HatchetRunContext[AgentDepsT]] = HatchetRunContext[AgentDepsT],
    ):
        super().__init__(model)
        self.task_config = task_config
        self.hatchet = hatchet
        self._task_name_prefix = task_name_prefix
        self.event_stream_handler = event_stream_handler
        self.deps_type = deps_type
        self.run_context_type = run_context_type

        @hatchet.task(
            name=f'{self._task_name_prefix}__model__request',
            description=self.task_config.description,
            input_validator=ModelInput,
            version=self.task_config.version,
            sticky=self.task_config.sticky,
            default_priority=self.task_config.default_priority,
            concurrency=self.task_config.concurrency,
            schedule_timeout=self.task_config.schedule_timeout,
            execution_timeout=self.task_config.execution_timeout,
            retries=self.task_config.retries,
            rate_limits=self.task_config.rate_limits,
            desired_worker_labels=self.task_config.desired_worker_labels,
            backoff_factor=self.task_config.backoff_factor,
            backoff_max_seconds=self.task_config.backoff_max_seconds,
            default_filters=self.task_config.default_filters,
        )
        async def wrapped_request_task(
            input: ModelInput,
            _ctx: Context,
        ) -> ModelResponse:
            return await super(HatchetModel, self).request(
                input.messages, input.model_settings, input.model_request_parameters
            )

        self.hatchet_wrapped_request_task = wrapped_request_task

        @hatchet.task(
            name=f'{self._task_name_prefix}__model__request_stream',
            description=self.task_config.description,
            input_validator=ModelStreamInput,
            version=self.task_config.version,
            sticky=self.task_config.sticky,
            default_priority=self.task_config.default_priority,
            concurrency=self.task_config.concurrency,
            schedule_timeout=self.task_config.schedule_timeout,
            execution_timeout=self.task_config.execution_timeout,
            retries=self.task_config.retries,
            rate_limits=self.task_config.rate_limits,
            desired_worker_labels=self.task_config.desired_worker_labels,
            backoff_factor=self.task_config.backoff_factor,
            backoff_max_seconds=self.task_config.backoff_max_seconds,
            default_filters=self.task_config.default_filters,
        )
        async def wrapped_request_stream_task(
            input: ModelStreamInput,
            ctx: Context,
        ) -> ModelResponse:
            assert self.event_stream_handler

            run_context = self.run_context_type.deserialize_run_context(
                input.serialized_run_context, deps=input.serialized_run_context, hatchet_context=ctx
            )

            async with self.wrapped.request_stream(
                input.messages, input.model_settings, input.model_request_parameters, run_context
            ) as streamed_response:
                async for s in streamed_response:
                    print('streamed chunk', s)
                    serialized = json.dumps(asdict(s), default=str)

                    await ctx.aio_put_stream(serialized)

            return streamed_response.get()

        self.hatchet_wrapped_request_stream_task = wrapped_request_stream_task

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await self.hatchet_wrapped_request_task.aio_run(
            ModelInput(
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ):
        if self.hatchet.is_in_task_run:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                yield streamed_response
                return

        if run_context is None:
            raise UserError(
                'A Hatchet model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        assert self.event_stream_handler is not None

        res = await self.hatchet_wrapped_request_stream_task.aio_run(
            input=ModelStreamInput(
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
                serialized_run_context=self.run_context_type.serialize_run_context(run_context),
                deps_type_name=self.deps_type.__name__ if self.deps_type else '',
            )
        )

        yield HatchetStreamedResponse(
            model_request_parameters=model_request_parameters,
            response=res,
        )


class HatchetStreamedResponse(StreamedResponse):
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
