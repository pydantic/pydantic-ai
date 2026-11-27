"""Tells a streamed model activity that its workflow waits for what it streamed when the run cancels it.

A run opts in to keeping a cancelled stream's partial response by configuring the model activity with
`ActivityCancellationType.WAIT_CANCELLATION_COMPLETED`, which the activity can't see. The workflow marks the activity
it schedules for an opted-in stream with a header, so that only then does the activity return what it streamed for a
cancellation. Every other activity is scheduled with the same input and headers as before, and fails on a cancellation
as before.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from temporalio import workflow
from temporalio.api.common.v1 import Payload
from temporalio.worker import (
    ActivityInboundInterceptor,
    ExecuteActivityInput,
    Interceptor,
    StartActivityInput,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    WorkflowOutboundInterceptor,
)

_HEADER = 'pydantic-ai-keep-partial-on-cancel'
"""The activity header marking a streamed model activity whose workflow waits for its partial on a cancellation."""

_MARK = Payload(metadata={'encoding': b'json/plain'}, data=b'true')

_marking: ContextVar[bool] = ContextVar('pydantic_ai.temporal.marking_keep_partial', default=False)
"""Whether the activity the workflow is scheduling now gets the header."""

_marked: ContextVar[bool] = ContextVar('pydantic_ai.temporal.keeps_partial', default=False)
"""Whether the activity running in this context was scheduled with the header."""


@contextmanager
def marking_keep_partial() -> Generator[None]:
    """Schedule the activities started inside with the header that has them keep their partial on a cancellation."""
    token = _marking.set(True)
    try:
        yield
    finally:
        _marking.reset(token)


def keeps_partial_on_cancel() -> bool:
    """Whether the running activity was scheduled to return what it streamed when the run cancels it."""
    return _marked.get()


class KeepPartialInterceptor(Interceptor):
    """Carries the opt-in from the workflow that schedules a streamed model activity to the activity."""

    def intercept_activity(self, next: ActivityInboundInterceptor) -> ActivityInboundInterceptor:
        return _ActivityInbound(next)

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor] | None:
        return _WorkflowInbound


class _WorkflowInbound(WorkflowInboundInterceptor):
    def init(self, outbound: WorkflowOutboundInterceptor) -> None:
        super().init(_WorkflowOutbound(outbound))


class _WorkflowOutbound(WorkflowOutboundInterceptor):
    def start_activity(self, input: StartActivityInput) -> workflow.ActivityHandle[Any]:
        if _marking.get():
            input.headers = {**input.headers, _HEADER: _MARK}
        return super().start_activity(input)


class _ActivityInbound(ActivityInboundInterceptor):
    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        token = _marked.set(_HEADER in input.headers)
        try:
            return await super().execute_activity(input)
        finally:
            _marked.reset(token)
