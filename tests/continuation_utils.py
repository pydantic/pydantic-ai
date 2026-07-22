"""Scripted continuation models shared by the durable-execution test suites.

The Temporal/DBOS/Prefect durability tests need a model that produces a
`suspended → complete` continuation chain (Anthropic `pause_turn`, OpenAI background
mode) so they can assert that each segment resolves in its own durable boundary
(activity/step/task). `FunctionModel` can't emit a suspended *streaming* segment,
so this module provides a small scripted model, mirroring the one in
`tests/models/test_streamed_continuation.py` (the core regression net for the
continuation loop), extended with call counting, scripted request errors, and a
`reset()` so module-level agents (required by Temporal workflows) can be re-scripted
per test.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponseStreamEvent, TextPart
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

TIMESTAMP = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass
class StreamSegment:
    """One scripted `request_stream` call: the text parts it streams and the segment's metadata."""

    texts: list[str]
    state: Literal['suspended', 'complete']
    provider_response_id: str
    input_tokens: int
    output_tokens: int


@dataclass
class ScriptedStreamedResponse(StreamedResponse):
    """Streams a segment's text parts through the parts manager, exposing scripted metadata via `get()`."""

    _segment: StreamSegment = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        segment = self._segment
        # Populate metadata up front so the continuation composite can resolve replace-vs-accumulate
        # on the first reindexable event, mirroring a real provider stream that knows its id from the start.
        self.provider_response_id = segment.provider_response_id
        self.state = segment.state
        self._usage = RequestUsage(input_tokens=segment.input_tokens, output_tokens=segment.output_tokens)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for index, text in enumerate(self._segment.texts):
            # An empty delta opens the part (`PartStartEvent`); the content follows (`PartDeltaEvent`).
            for event in self._parts_manager.handle_text_delta(vendor_part_id=index, content=''):
                yield event
            for event in self._parts_manager.handle_text_delta(vendor_part_id=index, content=text):
                yield event

    async def close_stream(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return 'scripted'

    @property
    def provider_name(self) -> str:
        return 'scripted'

    @property
    def provider_url(self) -> str:
        return 'scripted'

    @property
    def timestamp(self) -> datetime:
        return TIMESTAMP


class ScriptedContinuationModel(Model):
    """A `Model` whose `request`/`request_stream` replay scripted `suspended → complete` chains.

    `responses` drives the non-streaming path (an `Exception` entry is raised instead of
    returned); `segments` drives the streaming path. `request_calls`/`request_stream_calls`
    count provider calls and `cancelled` records `cancel_suspended_response` teardowns, so
    tests can assert how many segments crossed the durable boundary and that a leaked
    server-side job was cancelled there.
    """

    def __init__(
        self,
        *,
        responses: list[ModelResponse | Exception] | None = None,
        segments: list[StreamSegment] | None = None,
    ) -> None:
        super().__init__()
        self.responses = responses or []
        self.segments = segments or []
        self.request_calls = 0
        self.request_stream_calls = 0
        self.cancelled: list[ModelResponse] = []

    def reset(
        self,
        *,
        responses: list[ModelResponse | Exception] | None = None,
        segments: list[StreamSegment] | None = None,
    ) -> None:
        """Re-script the model and clear recorded calls, for module-level reuse across tests."""
        self.responses = responses or []
        self.segments = segments or []
        self.request_calls = 0
        self.request_stream_calls = 0
        self.cancelled = []

    @property
    def model_name(self) -> str:
        return 'scripted'

    @property
    def system(self) -> str:
        return 'scripted'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.request_calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        self.request_stream_calls += 1
        segment = self.segments.pop(0)
        yield ScriptedStreamedResponse(model_request_parameters, _segment=segment)

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        self.cancelled.append(response)


def scripted_response(
    *,
    texts: list[str],
    state: Literal['suspended', 'complete'] = 'complete',
    provider_response_id: str,
    input_tokens: int,
    output_tokens: int,
) -> ModelResponse:
    """Build a `ModelResponse` for a `ScriptedContinuationModel` script (or a resume-seed history tail)."""
    return ModelResponse(
        parts=[TextPart(content=t) for t in texts],
        model_name='scripted',
        provider_name='scripted',
        provider_response_id=provider_response_id,
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        state=state,
        timestamp=TIMESTAMP,
    )
