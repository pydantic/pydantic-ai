"""The `StreamedResponse` both models return for a `ModelResponse` built without streaming."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from ...messages import ModelResponse, ModelResponseStreamEvent
from .. import StreamedResponse


@dataclass
class ModelResponseStreamedResponse(StreamedResponse):
    """`StreamedResponse` wrapper for pre-built `ModelResponse` objects."""

    _model_response: ModelResponse

    def __post_init__(self) -> None:
        self._usage = self._model_response.usage
        self.provider_response_id = self._model_response.provider_response_id
        self.provider_details = self._model_response.provider_details
        self.canonical_system = self._model_response.canonical_system
        self.finish_reason = self._model_response.finish_reason
        self.state = self._model_response.state
        self.metadata = self._model_response.metadata
        for index, part in enumerate(self._model_response.parts):
            self._parts_manager.handle_part(vendor_part_id=index, part=part)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if False:  # pragma: no cover
            yield cast(ModelResponseStreamEvent, None)

    async def close_stream(self) -> None:
        # No live connection to tear down: this wraps an already-retrieved `ModelResponse` (e.g. a
        # cursor-less background resume). `cancel()` and the continuation composite's teardown call this,
        # so it must no-op rather than inherit the base `NotImplementedError`; a server-side background
        # job is cancelled via `cancel_suspended_response`, not here.
        pass

    @property
    def model_name(self) -> str:
        # model_name is always set when ModelResponseStreamedResponse is constructed
        assert self._model_response.model_name is not None
        return self._model_response.model_name

    @property
    def provider_name(self) -> str | None:
        return self._model_response.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._model_response.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._model_response.timestamp
