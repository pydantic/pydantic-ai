"""An in-memory batch API for test models, which runs a batch's requests as soon as it's created."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import TypeAdapter
from pydantic_core import to_json

from .._utils import now_utc
from ..exceptions import ModelHTTPError, UserError
from ..messages import ModelResponse
from . import Model
from ._batch_codec import unrun_error
from .batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest

_RESPONSE_ADAPTER = TypeAdapter(ModelResponse)
_ERROR_ADAPTER = TypeAdapter(dict[str, object])


@dataclass
class _State:
    group: str = field(default_factory=lambda: f'in-memory:{uuid.uuid4().hex}')
    """Requests and batches live on the instance that encoded them, so each instance is a group of its own."""
    requests: dict[str, BatchRequest] = field(default_factory=dict[str, BatchRequest])
    batches: dict[str, tuple[Batch, list[BatchResult]]] = field(
        default_factory=dict[str, tuple[Batch, list[BatchResult]]]
    )
    errors: dict[str, Exception] = field(default_factory=dict[str, Exception])
    """Errors other than HTTP ones that requests raised, by the token their result names, to raise from `batch_decode`."""


class InMemoryBatchModel(Model):
    """A model whose batch API runs a batch's requests through `request()` when the batch is created.

    Batches only exist in the process and on the instance that created them.
    """

    @property
    def batch_max_requests(self) -> int | None:
        return 10_000

    @property
    def _batch_state(self) -> _State:
        # Created on first use, which runs on several threads at once when one model serves runs on several event loops,
        # as `cached_property` doesn't lock: `setdefault` keeps the first state any of them stored.
        state: _State | None = vars(self).get('_in_memory_batch_state')
        if state is not None:
            return state
        return vars(self).setdefault('_in_memory_batch_state', _State())

    async def batch_encode(self, request: BatchRequest) -> EncodedBatchRequest:
        token = uuid.uuid4().hex
        self._batch_state.requests[token] = request
        return EncodedBatchRequest(custom_id=request.custom_id, group=self._batch_state.group, body=token)

    async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
        state = self._batch_state
        results: list[BatchResult] = []
        for encoded in requests:
            request = state.requests.pop(encoded.body)
            try:
                response = await self.request(
                    request.messages, request.model_settings, request.model_request_parameters
                )
            except ModelHTTPError as error:
                body = to_json({'status_code': error.status_code, 'body': error.body}).decode()
                results.append(BatchResult(custom_id=encoded.custom_id, status='errored', body=body))
            except Exception as error:
                # Fails only this request, as it would fail only its own run without batching.
                state.errors[encoded.body] = error
                body = to_json({'error': encoded.body}).decode()
                results.append(BatchResult(custom_id=encoded.custom_id, status='errored', body=body))
            else:
                body = _RESPONSE_ADAPTER.dump_json(response).decode()
                results.append(BatchResult(custom_id=encoded.custom_id, status='succeeded', body=body))
        batch = Batch(
            id=f'batch_{uuid.uuid4().hex}',
            model_name=self.model_name,
            provider_name=self.system,
            group=state.group,
            status='ended',
            created_at=now_utc(),
            request_count=len(requests),
        )
        state.batches[batch.id] = (batch, results)
        return batch

    async def batch_status(self, batch: Batch) -> Batch:
        return self._stored_batch(batch)[0]

    async def batch_results(self, batch: Batch) -> list[BatchResult]:
        return list(self._stored_batch(batch)[1])

    async def batch_cancel(self, batch: Batch) -> Batch:
        return self._stored_batch(batch)[0]

    def batch_decode(self, result: BatchResult, request: BatchRequest) -> ModelResponse:
        if result.status == 'succeeded' and result.body is not None:
            return _RESPONSE_ADAPTER.validate_json(result.body)
        if result.status == 'errored' and result.body is not None:
            error = _ERROR_ADAPTER.validate_json(result.body)
            if isinstance(token := error.get('error'), str):
                raise self._batch_state.errors[token]
            status_code = error['status_code']
            assert isinstance(status_code, int)
            raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=error['body'])
        raise unrun_error(self.model_name, result)

    def _stored_batch(self, batch: Batch) -> tuple[Batch, list[BatchResult]]:
        if (stored := self._batch_state.batches.get(batch.id)) is None:
            raise UserError(f'Batch {batch.id!r} was not created by this model instance.')
        return stored
