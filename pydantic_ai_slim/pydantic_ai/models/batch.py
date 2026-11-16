"""The data types of a provider's batch API, for batch pricing in exchange for latency.

The [`BatchModelRequests`][pydantic_ai.capabilities.BatchModelRequests] capability lets an agent use a batch API
without changing how it runs, and [`model_request_batch`][pydantic_ai.direct.model_request_batch] sends a list of
requests as one batch. The `batch_*` methods on [`Model`][pydantic_ai.models.Model] are the provider-level primitives
both are built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from typing_extensions import Literal, TypeAlias  # noqa: UP035

from ..messages import ModelMessage
from ..settings import ModelSettings
from . import ModelRequestParameters

__all__ = (
    'Batch',
    'BatchRequest',
    'BatchResult',
    'BatchResultStatus',
    'BatchStatus',
    'EncodedBatchRequest',
)

BatchStatus: TypeAlias = Literal['in_progress', 'canceling', 'ended']
"""The processing state of a batch job. An ended job can hold a mix of per-request outcomes."""

BatchResultStatus: TypeAlias = Literal['succeeded', 'errored', 'expired', 'canceled']
"""How the provider finished one request of a batch."""


@dataclass(kw_only=True)
class BatchRequest:
    """One model request to send as part of a batch.

    Every aspect of the request can differ between the requests of a batch, as long as the provider accepts them in
    one job: [`Model.batch_encode`][pydantic_ai.models.Model.batch_encode] reports which ones it can combine.
    """

    custom_id: str
    """The ID that matches the request to its result. It must be unique within a batch."""

    messages: list[ModelMessage]
    """The messages to send to the model."""

    model_settings: ModelSettings | None = None
    """Settings for this request."""

    model_request_parameters: ModelRequestParameters = field(default_factory=ModelRequestParameters)
    """Tools, output schema and other request parameters for this request."""


@dataclass(frozen=True, kw_only=True)
class EncodedBatchRequest:
    """A [`BatchRequest`][pydantic_ai.models.batch.BatchRequest] encoded for one provider's batch API."""

    custom_id: str
    """The ID that matches the request to its result within the batch."""

    group: str
    """Everything requests must share to be submitted in one batch, such as the endpoint, credentials and model.

    It is stored on the [`Batch`][pydantic_ai.models.batch.Batch] handle, so it never contains secrets.
    """

    body: str
    """The provider-native request, as JSON."""


@dataclass(kw_only=True)
class Batch:
    """A handle to a submitted batch job.

    It holds only plain data, so you can store it, for example with a Pydantic `TypeAdapter`, and use it from another
    process to check on the job, retrieve its results or cancel it.
    """

    id: str
    """The provider's ID for the batch job."""

    model_name: str
    """The name of the model the batch was submitted to."""

    provider_name: str
    """The name of the provider the batch was submitted to."""

    group: str
    """The [`group`][pydantic_ai.models.batch.EncodedBatchRequest.group] the batch's requests share.

    Used to check that a later call uses a compatible client configuration.
    """

    status: BatchStatus
    """The job's processing state when this handle was last updated."""

    created_at: datetime
    """When the provider created the job."""

    request_count: int
    """The number of requests in the batch."""

    provider_details: dict[str, Any] | None = None
    """Provider-specific job details, such as the files a provider writes results to."""


@dataclass(frozen=True, kw_only=True)
class BatchResult:
    """The provider's result for one request of a batch.

    Turn it into a [`ModelResponse`][pydantic_ai.messages.ModelResponse] with
    [`Model.batch_decode`][pydantic_ai.models.Model.batch_decode].
    """

    custom_id: str
    """The ID of the request this result belongs to."""

    status: BatchResultStatus
    """How the provider finished the request."""

    body: str | None = None
    """The provider-native result or error, as JSON, when the request `'succeeded'` or `'errored'`."""
