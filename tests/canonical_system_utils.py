"""A scripted model whose streams carry a `canonical_system`, shared by the UI adapters' stream round-trip tests.

`FunctionModel`'s streams have no `provider_name`, so a response's `canonical_system` never applies to their parts.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    ThinkingPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings

PROVIDER_NAME = 'stamped'
TIMESTAMP = datetime(2024, 1, 1, tzinfo=UTC)

StampedScript = Callable[[list[ModelMessage]], list[ModelResponsePart]]


@dataclass
class StampedStreamedResponse(StreamedResponse):
    """Streams scripted parts as a provider named `PROVIDER_NAME` would.

    A thinking part's signature and `provider_name` arrive in a delta after the part starts, as real providers
    send them, so its `PartStartEvent` can't tell yet whether the response's `canonical_system` applies to it.
    """

    _parts: list[ModelResponsePart] = field(default_factory=list[ModelResponsePart])

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for index, part in enumerate(self._parts):
            if isinstance(part, ThinkingPart):
                for event in self._parts_manager.handle_thinking_delta(vendor_part_id=index, content=part.content):
                    yield event
                for event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id=index, signature=part.signature, provider_name=part.provider_name
                ):
                    yield event
            else:
                yield self._parts_manager.handle_part(vendor_part_id=index, part=part)

    async def close_stream(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return PROVIDER_NAME

    @property
    def provider_name(self) -> str:
        return PROVIDER_NAME

    @property
    def provider_url(self) -> None:
        return None

    @property
    def timestamp(self) -> datetime:
        return TIMESTAMP


class StampedModel(Model):
    """A streaming-only model answering with `script`, whose responses carry `canonical_system`."""

    def __init__(self, script: StampedScript, *, canonical_system: str | None) -> None:
        super().__init__()
        self.script = script
        self.stamp = canonical_system

    @property
    def model_name(self) -> str:
        return PROVIDER_NAME

    @property
    def system(self) -> str:
        return PROVIDER_NAME

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:  # pragma: no cover
        raise AssertionError('UI adapters stream')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        response = StampedStreamedResponse(model_request_parameters, _parts=self.script(messages))
        response.canonical_system = self.stamp
        yield response
