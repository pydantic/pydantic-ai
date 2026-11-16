"""Lifecycle-tracking models shared by the durable-execution test suites.

Temporal, DBOS, Prefect, and the common durability capability tests all need to
prove which durable unit owns a rebuilt model's context-manager lifecycle. This
module keeps the common tracking and failure behavior in one test double, while
engine-specific subclasses can add hooks with their existing event vocabulary.
"""

from __future__ import annotations

from types import TracebackType

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings


class LifecycleTrackingModel(TestModel):
    """Record model lifecycle and request events for durable-execution tests."""

    def __init__(
        self,
        events: list[str],
        *,
        event_prefix: str = '',
        include_exit_exception: bool = True,
        fail: bool = False,
        fail_exit: bool = False,
        suppress_exit: bool = False,
        model_name: str = 'lifecycle',
        custom_output_text: str = 'ok',
    ) -> None:
        super().__init__(custom_output_text=custom_output_text, model_name=model_name)
        self.events = events
        self.event_prefix = event_prefix
        self.include_exit_exception = include_exit_exception
        self.fail = fail
        self.fail_exit = fail_exit
        self.suppress_exit = suppress_exit

    async def __aenter__(self) -> LifecycleTrackingModel:
        self.events.append(f'{self.event_prefix}enter')
        return await super().__aenter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        exception_name = exc_type.__name__ if exc_type is not None else 'none'
        exit_event = f'{self.event_prefix}exit'
        if self.include_exit_exception:
            exit_event = f'{exit_event}:{exception_name}'
        self.events.append(exit_event)
        await super().__aexit__(exc_type, exc_val, exc_tb)
        if self.fail_exit:
            raise ValueError('exit failed')
        return self.suppress_exit

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.events.append('request')
        if self.fail:
            raise RuntimeError('request failed')
        return await super().request(messages, model_settings, model_request_parameters)
