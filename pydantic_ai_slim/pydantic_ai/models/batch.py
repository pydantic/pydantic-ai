"""Send an agent's model requests through a provider batch API, for batch pricing in exchange for latency."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from typing_extensions import Literal, TypeAlias, TypedDict  # noqa: UP035

from .._run_context import RunContext
from ..exceptions import UserError
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from . import Model, ModelRequestParameters, StreamedResponse
from .wrapper import WrapperModel

__all__ = (
    'BatchModel',
    'BatchRequest',
    'BatchResult',
    'BatchResultStatus',
    'BatchSettings',
    'BatchStatus',
)

BatchStatus: TypeAlias = Literal['in_progress', 'canceling', 'ended']
"""The job's processing state; an ended job can contain mixed per-request outcomes."""

BatchResultStatus: TypeAlias = Literal['succeeded', 'errored', 'expired', 'canceled']
"""How the provider finished one request of a batch."""


class BatchSettings(TypedDict, total=False):
    """When a durable batch queue submits the requests it has collected.

    A batch is submitted when it holds `max_requests` requests, when its oldest request has waited
    `max_wait`, or when it reaches the provider's or engine's size limit, whichever comes first.
    """

    max_requests: int
    """The most requests one batch holds. Defaults to 100; the provider's own limit caps it."""

    max_wait: timedelta
    """How long the oldest queued request waits for others before its batch is submitted.

    Counted from when the queue durably accepted it, so later arrivals never extend it. Provider
    processing time comes on top. Defaults to 30 seconds.
    """

    poll_interval: timedelta
    """How often a submitted batch is checked for results. Defaults to one minute."""


@dataclass(frozen=True, kw_only=True)
class BatchRequest:
    """One model request encoded for a provider batch API."""

    custom_id: str
    """The ID that matches the request to its result within the batch."""

    group: str
    """Everything requests must share to go into one batch, such as endpoint, credentials and beta flags.

    It identifies the batch queue, so it is persisted: it must not contain secrets.
    """

    body: str
    """The provider-native request parameters, as JSON."""


@dataclass(frozen=True, kw_only=True)
class BatchResult:
    """The provider's result for one request of a batch."""

    custom_id: str
    """The ID of the request this result belongs to."""

    status: BatchResultStatus
    """How the provider finished the request."""

    body: str | None = None
    """The provider-native result, as JSON, for a request that succeeded or errored."""


@dataclass(init=False)
class BatchModel(WrapperModel):
    """Send a model's requests through its provider's batch API.

    Batch APIs cost less, typically half, but a result can take up to a day. The agent runs as usual:
    each request is queued, submitted in a batch with others and collected for you. That takes a durable
    execution context that can wait that long, such as a Temporal workflow with
    [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]; anywhere else a
    request raises a [`UserError`][pydantic_ai.exceptions.UserError] before anything is queued.

    Requests aren't streamed. Register the model with the durability capability, as the agent's model
    or in `models=`, since the queue rebuilds it on the worker from its registered name.
    """

    max_requests: int
    """The resolved number of requests that triggers submission."""

    max_wait: timedelta
    """The resolved maximum age of the oldest queued request before submission."""

    poll_interval: timedelta
    """The resolved interval between checks for provider results."""

    def __init__(self, wrapped: Model, *, batch_settings: BatchSettings | None = None) -> None:
        super().__init__(wrapped)
        if self.wrapped.batch_max_requests is None:
            raise UserError(f'{type(self.wrapped).__name__} {self.wrapped.model_name!r} has no batch API.')
        settings: BatchSettings = batch_settings or {}
        self.max_requests = settings.get('max_requests', 100)
        self.max_wait = settings.get('max_wait', timedelta(seconds=30))
        self.poll_interval = settings.get('poll_interval', timedelta(minutes=1))
        if self.max_requests <= 0:
            raise UserError('`BatchSettings.max_requests` must be positive.')
        if self.max_wait <= timedelta(0):
            raise UserError('`BatchSettings.max_wait` must be positive.')
        if self.poll_interval <= timedelta(0):
            raise UserError('`BatchSettings.poll_interval` must be positive.')

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise UserError(
            '`BatchModel` requests are queued for a provider batch API and can wait hours for their result, '
            'so they need a durable execution context with batch support, such as a Temporal workflow with '
            '`TemporalDurability`. Run the agent there, or use the wrapped model directly.'
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        raise UserError('`BatchModel` does not support streaming: a batch result arrives all at once.')
        # Required to type the context manager; streaming always raises.
        yield  # pragma: no cover
