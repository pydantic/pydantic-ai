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
    # High-level batch processing
    'model_request_batch',
    'model_request_batch_sync',
    # Low-level batch processing functions (advanced)
    'batch_create',
    'batch_create_sync',
    'batch_status',
    'batch_status_sync',
    'batch_results',
    'batch_results_sync',
    'batch_cancel',
    'batch_cancel_sync',
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


# --- High-Level Batch Processing ---


async def model_request_batch(
    model: models.Model | models.KnownModelName | str,
    requests: Sequence[
        tuple[str, Sequence[messages.ModelMessage]]
        | tuple[str, Sequence[messages.ModelMessage], models.ModelRequestParameters]
    ],
    *,
    model_request_parameters: models.ModelRequestParameters | None = None,
    model_settings: settings.ModelSettings | None = None,
    poll_interval: float = 60.0,
    timeout: float | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> list[models.BatchResult]:
    """Submit a batch and wait for all results.

    This is the recommended high-level function for batch processing.
    It handles the complete batch lifecycle: create, poll, and retrieve results.

    Batch processing typically offers 50% cost reduction compared to regular API calls.

    **Cancellation**: To cancel a running batch, use `asyncio.create_task(...).cancel()`.
    The function will attempt to cancel the batch on the API before raising `CancelledError`.

    ```py title="model_request_batch_example.py" test="skip"
    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request_batch


    async def main():
        requests = [
            ('question-1', [ModelRequest.user_text_prompt('What is 2+2?')]),
            ('question-2', [ModelRequest.user_text_prompt('What is 3+3?')]),
        ]

        results = await model_request_batch('openai:gpt-5-mini', requests, poll_interval=30.0)

        for result in results:
            if result.is_successful:
                print(f'{result.custom_id}: {result.response.parts[0].content}')
            else:
                print(f'{result.custom_id} failed: {result.error}')
    ```

    Args:
        model: The model to make batch requests to.
        requests: List of tuples, either:
            - (custom_id, messages) - uses model_request_parameters default
            - (custom_id, messages, parameters) - per-request parameters
        model_request_parameters: Default parameters for requests that don't specify their own.
        model_settings: Settings applied to all requests in the batch.
        poll_interval: How often to poll for completion (in seconds). Default: 60s.
        timeout: Maximum time to wait for completion (in seconds). None means no timeout.
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        List of BatchResult objects, one per request in the batch.

    Raises:
        NotImplementedError: If the model doesn't support batch processing.
        asyncio.CancelledError: If the task is cancelled (batch cancellation attempted).
        asyncio.TimeoutError: If timeout is exceeded before completion.
    """
    import asyncio

    model_instance = _prepare_model(model, instrument)
    default_params = model_request_parameters or models.ModelRequestParameters()

    # Normalize requests to 3-tuples
    requests_list: list[tuple[str, list[messages.ModelMessage], models.ModelRequestParameters]] = []
    for req in requests:
        if len(req) == 2:
            custom_id, msgs = req
            requests_list.append((custom_id, list(msgs), default_params))
        else:
            custom_id, msgs, params = req
            requests_list.append((custom_id, list(msgs), params))

    # Create the batch
    batch = await model_instance.batch_create(requests_list, model_settings)

    # Track start time for timeout
    start_time: float | None = None
    if timeout is not None:
        import time

        start_time = time.monotonic()

    try:
        # Poll until complete
        while not batch.is_complete:
            await asyncio.sleep(poll_interval)

            # Check timeout
            if timeout is not None and start_time is not None:  # pragma: no cover
                import time

                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    # Attempt to cancel before raising timeout
                    try:
                        await model_instance.batch_cancel(batch)
                    except Exception:
                        pass  # Best effort cancellation
                    raise asyncio.TimeoutError(f'Batch {batch.id} timed out after {elapsed:.1f}s')

            batch = await model_instance.batch_status(batch)

    except asyncio.CancelledError:  # pragma: no cover
        # On cancellation, attempt to cancel the batch
        try:
            await model_instance.batch_cancel(batch)
        except Exception:
            pass  # Best effort cancellation
        raise

    # Retrieve results
    return await model_instance.batch_results(batch)


def model_request_batch_sync(
    model: models.Model | models.KnownModelName | str,
    requests: Sequence[
        tuple[str, Sequence[messages.ModelMessage]]
        | tuple[str, Sequence[messages.ModelMessage], models.ModelRequestParameters]
    ],
    *,
    model_request_parameters: models.ModelRequestParameters | None = None,
    model_settings: settings.ModelSettings | None = None,
    poll_interval: float = 60.0,
    timeout: float | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> list[models.BatchResult]:
    """Submit a batch and wait for all results (synchronous version).

    This is a convenience method that wraps [`model_request_batch`][pydantic_ai.direct.model_request_batch]
    with `loop.run_until_complete(...)`.

    See [`model_request_batch`][pydantic_ai.direct.model_request_batch] for full documentation.

    Args:
        model: The model to make batch requests to.
        requests: List of tuples, either (custom_id, messages) or (custom_id, messages, parameters).
        model_request_parameters: Default parameters for requests that don't specify their own.
        model_settings: Settings applied to all requests in the batch.
        poll_interval: How often to poll for completion (in seconds). Default: 60s.
        timeout: Maximum time to wait for completion (in seconds). None means no timeout.
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        List of BatchResult objects, one per request in the batch.
    """
    return _get_event_loop().run_until_complete(
        model_request_batch(
            model,
            requests,
            model_request_parameters=model_request_parameters,
            model_settings=model_settings,
            poll_interval=poll_interval,
            timeout=timeout,
            instrument=instrument,
        )
    )


# --- Low-Level Batch Processing Functions ---
# These functions provide a thin wrapper around Model batch methods.
# For most use cases, prefer model_request_batch() above.


async def batch_create(
    model: models.Model | models.KnownModelName | str,
    requests: Sequence[
        tuple[str, Sequence[messages.ModelMessage]]
        | tuple[str, Sequence[messages.ModelMessage], models.ModelRequestParameters]
    ],
    *,
    model_request_parameters: models.ModelRequestParameters | None = None,
    model_settings: settings.ModelSettings | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> models.Batch:
    """Submit a batch of requests for asynchronous processing.

    Batch processing allows submitting multiple requests together,
    typically at reduced cost (e.g., 50% discount on OpenAI).

    See the [OpenAI Batch API docs](https://platform.openai.com/docs/guides/batch) for details.

    ```py title="batch_create_example.py" test="skip"
    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import batch_create


    async def main():
        # Simple usage - just (custom_id, messages) tuples
        requests = [
            ('req-1', [ModelRequest.user_text_prompt('What is 2+2?')]),
            ('req-2', [ModelRequest.user_text_prompt('What is 3+3?')]),
        ]
        batch = await batch_create('openai:gpt-5-mini', requests)
        print(f'Batch {batch.id} created with status: {batch.status}')
    ```

    Args:
        model: The model to make batch requests to.
        requests: List of tuples, either:
            - (custom_id, messages) - uses model_request_parameters default
            - (custom_id, messages, parameters) - per-request parameters
        model_request_parameters: Default parameters for requests that don't specify their own.
        model_settings: Settings applied to all requests in the batch.
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        Batch object with job ID and initial status.

    Raises:
        NotImplementedError: If the model doesn't support batch processing.
    """
    model_instance = _prepare_model(model, instrument)
    default_params = model_request_parameters or models.ModelRequestParameters()

    # Normalize requests to 3-tuples
    requests_list: list[tuple[str, list[messages.ModelMessage], models.ModelRequestParameters]] = []
    for req in requests:
        if len(req) == 2:
            custom_id, msgs = req
            requests_list.append((custom_id, list(msgs), default_params))
        else:
            custom_id, msgs, params = req
            requests_list.append((custom_id, list(msgs), params))

    return await model_instance.batch_create(requests_list, model_settings)


def batch_create_sync(
    model: models.Model | models.KnownModelName | str,
    requests: Sequence[
        tuple[str, Sequence[messages.ModelMessage]]
        | tuple[str, Sequence[messages.ModelMessage], models.ModelRequestParameters]
    ],
    *,
    model_request_parameters: models.ModelRequestParameters | None = None,
    model_settings: settings.ModelSettings | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> models.Batch:
    """Submit a batch of requests for asynchronous processing (synchronous version).

    This is a convenience method that wraps [`batch_create`][pydantic_ai.direct.batch_create] with
    `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

    Args:
        model: The model to make batch requests to.
        requests: List of tuples, either (custom_id, messages) or (custom_id, messages, parameters).
        model_request_parameters: Default parameters for requests that don't specify their own.
        model_settings: Settings applied to all requests in the batch.
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        Batch object with job ID and initial status.
    """
    return _get_event_loop().run_until_complete(
        batch_create(
            model,
            requests,
            model_request_parameters=model_request_parameters,
            model_settings=model_settings,
            instrument=instrument,
        )
    )


async def batch_status(
    model: models.Model | models.KnownModelName | str,
    batch: models.Batch,
    *,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> models.Batch:
    """Get current status of a batch job.

    ```py title="batch_status_example.py"
    import asyncio

    from pydantic_ai.direct import batch_status


    async def poll_until_complete(model, batch):
        while not batch.is_complete:
            await asyncio.sleep(60)  # Poll every minute
            batch = await batch_status(model, batch)
            print(f'Status: {batch.status}, completed: {batch.completed_count}/{batch.request_count}')
        return batch
    ```

    Args:
        model: The model the batch was created with.
        batch: Batch object from batch_create() or previous batch_status().
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        Updated Batch object with current status.
    """
    model_instance = _prepare_model(model, instrument)
    return await model_instance.batch_status(batch)


def batch_status_sync(
    model: models.Model | models.KnownModelName | str,
    batch: models.Batch,
    *,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> models.Batch:
    """Get current status of a batch job (synchronous version).

    This is a convenience method that wraps [`batch_status`][pydantic_ai.direct.batch_status] with
    `loop.run_until_complete(...)`.

    Args:
        model: The model the batch was created with.
        batch: Batch object from batch_create() or previous batch_status().
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        Updated Batch object with current status.
    """
    return _get_event_loop().run_until_complete(batch_status(model, batch, instrument=instrument))


async def batch_results(
    model: models.Model | models.KnownModelName | str,
    batch: models.Batch,
    *,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> list[models.BatchResult]:
    """Retrieve results from a completed batch.

    ```py title="batch_results_example.py"
    from pydantic_ai.direct import batch_results


    async def process_results(model, batch):
        results = await batch_results(model, batch)
        for result in results:
            if result.is_successful:
                print(f'{result.custom_id}: {result.response}')
            else:
                print(f'{result.custom_id} failed: {result.error}')
    ```

    Args:
        model: The model the batch was created with.
        batch: Batch object that has is_complete=True.
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        List of BatchResult objects, one per request in the batch.

    Raises:
        ValueError: If batch is not complete.
    """
    model_instance = _prepare_model(model, instrument)
    return await model_instance.batch_results(batch)


def batch_results_sync(
    model: models.Model | models.KnownModelName | str,
    batch: models.Batch,
    *,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> list[models.BatchResult]:
    """Retrieve results from a completed batch (synchronous version).

    This is a convenience method that wraps [`batch_results`][pydantic_ai.direct.batch_results] with
    `loop.run_until_complete(...)`.

    Args:
        model: The model the batch was created with.
        batch: Batch object that has is_complete=True.
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        List of BatchResult objects, one per request in the batch.
    """
    return _get_event_loop().run_until_complete(batch_results(model, batch, instrument=instrument))


async def batch_cancel(
    model: models.Model | models.KnownModelName | str,
    batch: models.Batch,
    *,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> models.Batch:
    """Cancel a pending or in-progress batch job.

    ```py title="batch_cancel_example.py"
    from pydantic_ai.direct import batch_cancel


    async def cancel_if_too_slow(model, batch, timeout_seconds):
        import asyncio
        await asyncio.sleep(timeout_seconds)
        if not batch.is_complete:
            batch = await batch_cancel(model, batch)
            print(f'Cancelled batch: {batch.status}')
    ```

    Args:
        model: The model the batch was created with.
        batch: Batch object to cancel.
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        Updated Batch object with cancellation status.
    """
    model_instance = _prepare_model(model, instrument)
    return await model_instance.batch_cancel(batch)


def batch_cancel_sync(
    model: models.Model | models.KnownModelName | str,
    batch: models.Batch,
    *,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> models.Batch:
    """Cancel a pending or in-progress batch job (synchronous version).

    This is a convenience method that wraps [`batch_cancel`][pydantic_ai.direct.batch_cancel] with
    `loop.run_until_complete(...)`.

    Args:
        model: The model the batch was created with.
        batch: Batch object to cancel.
        instrument: Whether to instrument with OpenTelemetry/Logfire.

    Returns:
        Updated Batch object with cancellation status.
    """
    return _get_event_loop().run_until_complete(batch_cancel(model, batch, instrument=instrument))
