"""The provider-independent steps of the batch codecs in `_anthropic_batch.py`, `_openai_batch.py` and `_google_batch.py`."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic_core import to_json

from ..exceptions import ModelAPIError, ModelHTTPError, UserError
from ..messages import ModelResponse
from ..settings import ModelSettings
from .batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest

if TYPE_CHECKING:
    from . import Model, ModelRequestParameters


class BatchJobError(ModelHTTPError):
    """The error of a batch job that ended without results for its requests, which it doesn't name, so it's theirs.

    `model_request_batch` reports it for each of the job's requests, as other jobs' results are still valid.
    """


def shared_group(requests: Sequence[EncodedBatchRequest], provider: str, causes: str) -> str:
    """The `group` all `requests` share, which a batch must have at least one of.

    `causes` names what makes this provider's groups differ, for the error.
    """
    if not requests:
        raise UserError('A batch must contain at least one request.')
    group = requests[0].group
    if any(request.group != group for request in requests):
        raise UserError(f'Requests in one {provider} batch must share a `group`; these came from different {causes}.')
    return group


def require_ended(batch: Batch) -> None:
    if batch.status != 'ended':
        raise UserError(f'Batch {batch.id!r} has not ended yet; check its status with `batch_status()` first.')


def refuse_http_client_auth(http_client: object, provider: str) -> None:
    """Refuse an HTTP client with an `auth` of its own, which can send any credential, so its batch group can't tell it apart."""
    if getattr(http_client, 'auth', None) is not None:
        raise UserError(f'Batching requires the credential on the {provider} client, not an `auth` on its HTTP client.')


def credential_hash(headers: Mapping[str, object], query: object) -> str:
    """A hash of the headers and query a client sends with every request, which include its credentials."""
    return hashlib.sha256(
        to_json(
            {
                'headers': {
                    name: value
                    for name, value in headers.items()
                    if isinstance(value, str)
                    and name.lower() != 'user-agent'
                    and not name.lower().startswith('x-stainless-')
                },
                'query': query,
            }
        )
    ).hexdigest()


def prepare_decode(model: Model, request: BatchRequest) -> tuple[ModelSettings | None, ModelRequestParameters]:
    """Prepare `request` as `batch_encode` did, to process its result."""
    with warnings.catch_warnings():
        # `batch_encode` already warned about any setting the request dropped.
        warnings.simplefilter('ignore')
        return model.prepare_request(request.model_settings, request.model_request_parameters)


def batch_priced(response: ModelResponse) -> ModelResponse:
    """Mark `response` as processed in a batch, which selects batch prices when `genai-prices` calculates its cost."""
    response.provider_details = {**(response.provider_details or {}), 'processing': 'batch'}
    return response


def unrun_error(model_name: str, result: BatchResult) -> ModelAPIError:
    """The error for a request the provider ended without running."""
    return ModelAPIError(model_name, f'The batch request was {result.status} before the model ran it.')
