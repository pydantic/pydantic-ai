"""Methods for making imperative requests to language models with minimal abstraction.

These methods allow you to make requests to LLMs where the only abstraction is input and output schema
translation so you can use all models with the same API.

These methods are thin wrappers around [`Model`][pydantic_ai.models.Model] implementations.
"""

from __future__ import annotations as _annotations

import queue
import threading
from collections.abc import Iterator, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType

from pydantic_ai.usage import RequestUsage
from pydantic_graph._utils import get_event_loop as _get_event_loop

from . import agent, messages, models, settings
from .models import StreamedResponse, instrumented as instrumented_models

__all__ = (
    'model_request',
    'model_request_sync',
    'model_request_stream',
    'model_request_stream_sync',
    'StreamedResponseSync',
)

STREAM_INITIALIZATION_TIMEOUT = 30


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
    return await model_instance.request(
        list(messages),
        model_settings,
        model_request_parameters or models.ModelRequestParameters(),
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
    return _get_event_loop().run_until_complete(
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
        async with model_request_stream('openai:gpt-4.1-mini', messages) as stream:
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
    return model_instance.request_stream(
        list(messages),
        model_settings,
        model_request_parameters or models.ModelRequestParameters(),
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
    with model_request_stream_sync('openai:gpt-4.1-mini', messages) as stream:
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
    """Synchronous wrapper to async streaming responses by running the async producer in a background thread and providing a synchronous iterator.

    This class must be used as a context manager with the `with` statement.
    """

    _async_stream_cm: AbstractAsyncContextManager[StreamedResponse]
    _queue: queue.Queue[messages.ModelResponseStreamEvent | Exception | None] = field(
        default_factory=queue.Queue, init=False
    )
    _thread: threading.Thread | None = field(default=None, init=False)
    _stream_response: StreamedResponse | None = field(default=None, init=False)
    _exception: Exception | None = field(default=None, init=False)
    _context_entered: bool = field(default=False, init=False)
    _stream_ready: threading.Event = field(default_factory=threading.Event, init=False)

    def __enter__(self) -> StreamedResponseSync:
        self._context_entered = True
        self._start_producer()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        self._cleanup()

    def __iter__(self) -> Iterator[messages.ModelResponseStreamEvent]:
        """Stream the response as an iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
        self._check_context_manager_usage()

        while True:
            item = self._queue.get()
            if item is None:  # End of stream
                break
            elif isinstance(item, Exception):
                raise item
            else:
                yield item

    def __repr__(self) -> str:
        if self._stream_response:
            return repr(self._stream_response)
        else:
            return f'{self.__class__.__name__}(context_entered={self._context_entered})'

    __str__ = __repr__

    def _check_context_manager_usage(self) -> None:
        if not self._context_entered:
            raise RuntimeError(
                'StreamedResponseSync must be used as a context manager. '
                'Use: `with model_request_stream_sync(...) as stream:`'
            )

    def _ensure_stream_ready(self) -> StreamedResponse:
        self._check_context_manager_usage()

        if self._stream_response is None:
            # Wait for the background thread to signal that the stream is ready
            if not self._stream_ready.wait(timeout=STREAM_INITIALIZATION_TIMEOUT):
                raise RuntimeError('Stream failed to initialize within timeout')

            if self._stream_response is None:  # pragma: no cover
                raise RuntimeError('Stream failed to initialize')

        return self._stream_response

    def _start_producer(self):
        self._thread = threading.Thread(target=self._async_producer, daemon=True)
        self._thread.start()

    def _async_producer(self):
        async def _consume_async_stream():
            try:
                async with self._async_stream_cm as stream:
                    self._stream_response = stream
                    # Signal that the stream is ready
                    self._stream_ready.set()
                    async for event in stream:
                        self._queue.put(event)
            except Exception as e:
                # Signal ready even on error so waiting threads don't hang
                self._stream_ready.set()
                self._queue.put(e)
            finally:
                self._queue.put(None)  # Signal end

        _get_event_loop().run_until_complete(_consume_async_stream())

    def _cleanup(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()

    # TODO (v2): Drop in favor of `response` property
    def get(self) -> messages.ModelResponse:
        """Build a ModelResponse from the data received from the stream so far."""
        return self._ensure_stream_ready().get()

    @property
    def response(self) -> messages.ModelResponse:
        """Get the current state of the response."""
        return self.get()

    # TODO (v2): Make this a property
    def usage(self) -> RequestUsage:
        """Get the usage of the response so far."""
        return self._ensure_stream_ready().usage()

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._ensure_stream_ready().model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._ensure_stream_ready().timestamp
