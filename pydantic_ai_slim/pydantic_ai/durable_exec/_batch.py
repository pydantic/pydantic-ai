from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from typing_extensions import Protocol

from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.batch import BatchModel
from pydantic_ai.models.wrapper import WrapperModel

from ._operation import ModelRequestParams


@dataclass(frozen=True, kw_only=True)
class BatchTicket:
    """An accepted request whose durable result is still outstanding."""

    item_id: str
    """The stable request ID shared by enqueue retries and result delivery."""
    queue_id: str
    """The durable backend's queue identifier."""
    group: str
    """The provider compatibility key, containing no secrets."""
    accepted_at: datetime
    """When the queue durably accepted the request."""


def find_batch_model(model: Model) -> BatchModel | None:
    """The `BatchModel` among `model`'s wrapper layers, if any."""
    while isinstance(model, WrapperModel):
        if isinstance(model, BatchModel):
            return model
        model = model.wrapped
    return None


class DurableBatchBinding(Protocol):
    """An engine's durable batch queue, which `BaseDurabilityCapability` routes `BatchModel` requests through.

    Called from container code: each method must be deterministic on replay and do its I/O in durable units.
    """

    async def enqueue(self, params: ModelRequestParams) -> BatchTicket:
        """Durably queue a request under an ID that stays the same when the enqueue is retried.

        If it raises, including on cancellation, nothing may stay queued: a late enqueue of the same ID
        must be rejected.
        """
        ...

    async def wait(self, ticket: BatchTicket, params: ModelRequestParams) -> ModelResponse | None:
        """Wait a bounded time for the ticket's result, and return it completed, or `None` if it hasn't arrived yet.

        `params` carries the request's settings and parameters for decoding, without its messages. A
        provider error for the request is raised as the model error it would have been on its own.
        """
        ...

    async def cancel(self, ticket: BatchTicket) -> None:
        """Abandon the ticket's request, without affecting other requests in its batch."""
        ...
