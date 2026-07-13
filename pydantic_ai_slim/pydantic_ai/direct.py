"""Methods for making imperative requests to language models with minimal abstraction.

These methods allow you to make requests to LLMs where the only abstraction is input and output schema
translation so you can use all models with the same API.

These methods are thin wrappers around [`Model`][pydantic_ai.models.Model] implementations.
"""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType

from pydantic_ai.usage import RequestUsage
from pydantic_graph._utils import run_until_complete as _run_until_complete

from . import agent, messages, models, settings
from ._sync_stream import SyncStreamBridge
from .models import StreamedResponse, instrumented as instrumented_models

__all__ = (
    'model_request',
    'model_request_sync',
    'model_request_stream',
    'model_request_stream_sync',
    'StreamedResponseSync',
)


def _ensure_instruction_parts(
    msgs: Sequence[messages.ModelMessage],
    model_request_parameters: models.ModelRequestParameters,
) -> models.ModelRequestParameters:
    """Populate instruction_parts from message history if not already set.

    When using the direct API, users set `instructions` on `ModelRequest` but may not set
    `instruction_parts` on `ModelRequestParameters`. This bridges the gap so models that
    read `instruction_parts` directly still see the instructions.
    """
    if model_request_parameters.instruction_parts is not None:
        return model_request_parameters
    for message in reversed(msgs):
        if isinstance(message, messages.ModelRequest) and message.instructions is not None:
            return dataclasses.replace(
                model_request_parameters,
                instruction_parts=[messages.InstructionPart(content=message.instructions)],
            )
    return model_request_parameters


async def model_request(
    model: models.Model | models.KnownModelName | str,
    messages: Sequence[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> messages.ModelResponse:
    """Make a non-streamed request to a model.

    ```py title="model_request_example.py"
    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request


    async def main():
        model_response = await model_request(
            'anthropic:claude-haiku-4-5',
            [ModelRequest.user_text_prompt('What is the capital of France?')]  # (1)!
        )
        print(model_response)
        '''
        ModelResponse(
            parts=[TextPart(content='The capital of France is Paris.')],
            usage=RequestUsage(input_tokens=56, output_tokens=7),
            model_name='claude-haiku-4-5',
            timestamp=datetime.datetime(...),
        )
        '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        The model response and token usage associated with the request.
    """
    model_instance = _prepare_model(model, instrument)
    mrp = _ensure_instruction_parts(messages, model_request_parameters or models.ModelRequestParameters())
    return await model_instance.request(
        list(messages),
        model_settings,
        mrp,
    )


def model_request_sync(
    model: models.Model | models.KnownModelName | str,
    messages: Sequence[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> messages.ModelResponse:
    """Make a Synchronous, non-streamed request to a model.

    This is a convenience method that wraps [`model_request`][pydantic_ai.direct.model_request] with
    `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

    ```py title="model_request_sync_example.py"
    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request_sync

    model_response = model_request_sync(
        'anthropic:claude-haiku-4-5',
        [ModelRequest.user_text_prompt('What is the capital of France?')]  # (1)!
    )
    print(model_response)
    '''
    ModelResponse(
        parts=[TextPart(content='The capital of France is Paris.')],
        usage=RequestUsage(input_tokens=56, output_tokens=7),
        model_name='claude-haiku-4-5',
        timestamp=datetime.datetime(...),
    )
    '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        The model response and token usage associated with the request.
    """
    return _run_until_complete(
        model_request(
            model,
            list(messages),
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
            instrument=instrument,
        )
    )


def model_request_stream(
    model: models.Model | models.KnownModelName | str,
    messages: Sequence[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> AbstractAsyncContextManager[models.StreamedResponse]:
    """Make a streamed async request to a model.

    ```py {title="model_request_stream_example.py"}

    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request_stream


    async def main():
        messages = [ModelRequest.user_text_prompt('Who was Albert Einstein?')]  # (1)!
        async with model_request_stream('openai:gpt-5-mini', messages) as stream:
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            print(chunks)
            '''
            [
                PartStartEvent(index=0, part=TextPart(content='Albert Einstein was ')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(
                    index=0, delta=TextPartDelta(content_delta='a German-born theoretical ')
                ),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='physicist.')),
                PartEndEvent(
                    index=0,
                    part=TextPart(
                        content='Albert Einstein was a German-born theoretical physicist.'
                    ),
                ),
            ]
            '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        A [stream response][pydantic_ai.models.StreamedResponse] async context manager.
    """
    model_instance = _prepare_model(model, instrument)
    mrp = _ensure_instruction_parts(messages, model_request_parameters or models.ModelRequestParameters())
    return model_instance.request_stream(
        list(messages),
        model_settings,
        mrp,
    )


def model_request_stream_sync(
    model: models.Model | models.KnownModelName | str,
    messages: Sequence[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> StreamedResponseSync:
    """Make a streamed synchronous request to a model.

    This is the synchronous version of [`model_request_stream`][pydantic_ai.direct.model_request_stream].
    It uses threading to run the asynchronous stream in the background while providing a synchronous iterator interface.

    ```py {title="model_request_stream_sync_example.py"}

    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request_stream_sync

    messages = [ModelRequest.user_text_prompt('Who was Albert Einstein?')]
    with model_request_stream_sync('openai:gpt-5-mini', messages) as stream:
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        print(chunks)
        '''
        [
            PartStartEvent(index=0, part=TextPart(content='Albert Einstein was ')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=0, delta=TextPartDelta(content_delta='a German-born theoretical ')
            ),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='physicist.')),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content='Albert Einstein was a German-born theoretical physicist.'
                ),
            ),
        ]
        '''
    ```

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        A [sync stream response][pydantic_ai.direct.StreamedResponseSync] context manager.
    """
    async_stream_cm = model_request_stream(
        model=model,
        messages=list(messages),
        model_settings=model_settings,
        model_request_parameters=model_request_parameters,
        instrument=instrument,
    )

    return StreamedResponseSync(async_stream_cm)


def _prepare_model(
    model: models.Model | models.KnownModelName | str,
    instrument: instrumented_models.InstrumentationSettings | bool | None,
) -> models.Model:
    model_instance = models.infer_model(model)

    if instrument is None:
        instrument = agent.Agent._instrument_default  # pyright: ignore[reportPrivateUsage]

    return instrumented_models.instrument_model(model_instance, instrument)


@dataclass
class StreamedResponseSync:
    """Synchronous wrapper for an async streaming response, running the whole stream on a dedicated event-loop thread.

    The stream runs on an [`anyio` blocking portal][anyio.from_thread.BlockingPortal] (see
    [`SyncStreamBridge`][pydantic_ai._sync_stream.SyncStreamBridge]) rather than pumping a shared event loop
    from a background thread, so exiting the `with` block cancels the underlying request promptly and closes
    the connection instead of waiting for the whole response to arrive.

    This class must be used as a context manager with the `with` statement.
    """

    _async_stream_cm: AbstractAsyncContextManager[StreamedResponse]
    _bridge: SyncStreamBridge[StreamedResponse] | None = field(default=None, init=False)
    _context_entered: bool = field(default=False, init=False)

    def __enter__(self) -> StreamedResponseSync:
        self._context_entered = True
        self._bridge = SyncStreamBridge(self._async_stream_cm, async_alternative='`model_request_stream`')
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._bridge is not None, '`__exit__` is only reachable after `__enter__` sets `_bridge`'
        self._bridge.shutdown((exc_type, exc_val, exc_tb))

    def __iter__(self) -> Iterator[messages.ModelResponseStreamEvent]:
        """Stream the response as an iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
        bridge = self._ensure_bridge()
        return bridge.stream_sync(lambda: aiter(bridge.stream))

    def __repr__(self) -> str:
        if self._bridge is not None:
            return repr(self._bridge.stream)
        else:
            return f'{self.__class__.__name__}(context_entered={self._context_entered})'

    __str__ = __repr__

    def _ensure_bridge(self) -> SyncStreamBridge[StreamedResponse]:
        if self._bridge is None:
            raise RuntimeError(
                'StreamedResponseSync must be used as a context manager. '
                'Use: `with model_request_stream_sync(...) as stream:`'
            )
        return self._bridge

    @property
    def response(self) -> messages.ModelResponse:
        """Get the current state of the response."""
        bridge = self._ensure_bridge()
        return bridge.call(bridge.stream.get)

    @property
    def usage(self) -> RequestUsage:
        """Get the usage of the response so far."""
        bridge = self._ensure_bridge()
        return bridge.call(lambda: bridge.stream.usage)

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        bridge = self._ensure_bridge()
        return bridge.call(lambda: bridge.stream.model_name)

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        bridge = self._ensure_bridge()
        return bridge.call(lambda: bridge.stream.timestamp)
