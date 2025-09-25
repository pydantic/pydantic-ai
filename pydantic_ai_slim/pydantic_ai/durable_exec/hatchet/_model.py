from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from hatchet_sdk import Context, Hatchet
from pydantic import BaseModel, ConfigDict

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from ._utils import TaskConfig


class ModelInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters


class HatchetModel(WrapperModel):
    """A wrapper for Model that integrates with Hatchet, turning request and request_stream to Hatchet tasks."""

    def __init__(
        self,
        model: Model,
        *,
        task_name_prefix: str,
        task_config: TaskConfig,
        hatchet: Hatchet,
        event_stream_handler: Any = None,
    ):
        super().__init__(model)
        self.task_config = task_config
        self.hatchet = hatchet
        self._task_name_prefix = task_name_prefix
        self.event_stream_handler = event_stream_handler

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
                if self.event_stream_handler is not None and run_context is not None:
                    await self.event_stream_handler(run_context, streamed_response)
                    async for _ in streamed_response:
                        pass
                yield streamed_response
        else:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                yield streamed_response
