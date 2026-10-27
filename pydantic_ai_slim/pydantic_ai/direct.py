"""Methods for making imperative requests to language models with minimal abstraction.

These methods allow you to make requests to LLMs where the only abstraction is input and output schema
translation so you can use all models with the same API.

These methods are thin wrappers around [`Model`][pydantic_ai.models.Model] implementations.
"""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import TracebackType

import anyio

from pydantic_ai.usage import RequestUsage

from . import agent, messages, models, settings
from ._sync_stream import SyncStreamBridge
from ._utils import run_until_complete as _run_until_complete
from .exceptions import ModelAPIError, UserError
from .models import StreamedResponse, batch as batch_models, instrumented as instrumented_models
from .models._batch_codec import BatchJobError
from .models._batch_model import check_resumable, refuse_pending_batch_request
from .models._batch_settings import (
    DEFAULT_MAX_POLL_INTERVAL,
    DEFAULT_MAX_RESUBMISSIONS,
    DEFAULT_POLL_INTERVAL,
    check_batch_settings,
    check_failure_tolerated,
    missing_result_error,
    poll_delay,
    resubmittable,
    transient,
)
from .models._request_timeout import enforce_request_deadline, open_request_stream, start_request_deadline

__all__ = (
    'model_request',
    'model_request_batch',
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


def _prepare_messages_carrying_a_retry(
    model: models.Model,
    msgs: Sequence[messages.ModelMessage],
    model_request_parameters: models.ModelRequestParameters,
) -> list[messages.ModelMessage]:
    """Run the model's message preparation, for the one history shape that cannot go out without it.

    A [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart] is kept model-neutrally and only
    becomes something a model can send at
    [`prepare_messages`][pydantic_ai.models.Model.prepare_messages] time, so an untranslated one
    reaching an adapter raises a `UserError` naming that step. A deprecated
    [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] would reach a model in the shape it had
    before deprecation, so a history holding one is prepared too, to send it in the current shape:
    a tool-less one in the voice its cause calls for, rather than as bare user text.

    A [`ModelSettingsDeltaPart`][pydantic_ai.messages.ModelSettingsDeltaPart] an agent run recorded is removed
    first: these helpers send the `model_settings` they're given at request level, and no run records what
    this request changes, so an earlier change replayed in place would override them.

    Preparation does more than translate retries, though — it wraps mid-conversation system prompts,
    synthesizes tool-search exchanges, announces tool-availability deltas, converts realtime speech
    parts — and these helpers are documented as thin wrappers around
    [`Model.request`][pydantic_ai.models.Model.request], sending the history they are handed. Scoping
    preparation to histories holding a retry leaves every other caller's request exactly as it was.
    """
    msgs = models._without_model_settings_deltas(list(msgs))  # pyright: ignore[reportPrivateUsage]
    # TODO(v3): remove `RetryPromptPart`
    if any(
        isinstance(part, messages.RetryFeedbackPart | messages.RetryPromptPart)  # pyright: ignore[reportDeprecated]
        for message in msgs
        if isinstance(message, messages.ModelRequest)
        for part in message.parts
    ):
        return model.prepare_messages(msgs, model_request_parameters)
    return msgs


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
    check_resumable(model_instance, messages)
    mrp = _ensure_instruction_parts(messages, model_request_parameters or models.ModelRequestParameters())
    deadline = start_request_deadline(model_instance, model_settings)
    with enforce_request_deadline(deadline):
        return await model_instance.request(
            _prepare_messages_carrying_a_retry(model_instance, messages, mrp),
            model_settings,
            mrp,
        )


async def model_request_batch(
    model: models.Model | models.KnownModelName | str,
    requests: Sequence[batch_models.BatchRequest],
    *,
    max_requests: int | None = None,
    max_resubmissions: int = DEFAULT_MAX_RESUBMISSIONS,
    poll_interval: timedelta = DEFAULT_POLL_INTERVAL,
    max_poll_interval: timedelta = DEFAULT_MAX_POLL_INTERVAL,
) -> list[messages.ModelResponse | ModelAPIError]:
    """Send requests through the model's batch API, and wait for their responses.

    Batch APIs cost less, typically half, but can take up to a day to finish. The requests are submitted in as few
    batches as the provider accepts, or `max_requests` allows, and each batch is checked on with a
    growing interval until it ends. Cancelling the call, for example with `asyncio.timeout`, cancels the batches that
    haven't ended. A request's `request_timeout` setting raises a [`UserError`][pydantic_ai.exceptions.UserError],
    since one request in a batch can't be timed out separately.

    Requests that expired before the provider ran them, that a batch ended without a result for, or that failed with
    a rate limit, timeout or server error are submitted again in new batches, up to `max_resubmissions` times, as a
    provider SDK retries a live request. A cancelled request isn't.

    A request's own error is returned in its place. An error creating a batch, checking on one or downloading its
    results is raised from the call instead, after cancelling the batches that haven't ended. A check that fails the way
    a live request is retried on, like a dropped connection or a server error, is first tried again at the next poll,
    up to 3 failures in a row, as with [`BatchModelRequests`][pydantic_ai.capabilities.BatchModelRequests].

    ```py title="model_request_batch_example.py"
    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request_batch
    from pydantic_ai.models.batch import BatchRequest


    async def main():
        questions = {'france': 'What is the capital of France?', 'italy': 'What is the capital of Italy?'}
        responses = await model_request_batch(
            'anthropic:claude-haiku-5-5',
            [
                BatchRequest(custom_id=key, messages=[ModelRequest.user_text_prompt(question)])
                for key, question in questions.items()
            ],
        )
        for key, response in zip(questions, responses):
            print(key, response if isinstance(response, Exception) else response.text)
            #> france The capital of France is Paris.
            #> italy The capital of Italy is Rome.
    ```

    To submit requests now and collect their results in another process, use the `batch_*` methods of
    [`Model`][pydantic_ai.models.Model] directly; see [the low-level `Model` API](../batch.md#submit-now-collect-later).

    Unlike [`model_request`][pydantic_ai.direct.model_request], the requests aren't instrumented with
    OpenTelemetry/Logfire.

    Args:
        model: The model to send the requests to. We allow `str` here since the actual list of allowed models changes frequently.
        requests: The requests to send, each with its own messages, settings and parameters. Their `custom_id`s must be unique.
        max_requests: The most requests per batch. Defaults to the provider's limit, which also caps it.
        max_resubmissions: How many times a request that expired, or that failed with a transient error, is submitted
            again in a new batch. `0` returns the error instead.
        poll_interval: How long to wait before first checking a batch. The interval grows by half after every check,
            up to `max_poll_interval`.
        max_poll_interval: The longest wait between checks of a batch.

    Returns:
        One item per request, in the same order: the request's response, or the
            [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] it failed with.
    """
    model_instance = models.infer_model(model)
    batch_size = check_batch_settings(
        model_instance,
        max_requests=max_requests,
        max_resubmissions=max_resubmissions,
        poll_interval=poll_interval,
        max_poll_interval=max_poll_interval,
    )
    if len({request.custom_id for request in requests}) != len(requests):
        raise UserError('Each batch request needs a unique `custom_id`.')
    for request in requests:
        refuse_pending_batch_request(request.messages)
    if any(
        'request_timeout' in (settings.merge_model_settings(model_instance.settings, request.model_settings) or {})
        for request in requests
    ):
        raise UserError(
            '`model_request_batch` does not support `request_timeout`: a batch can take up to a day to finish. '
            'To stop waiting, cancel the call, which cancels its batches.'
        )

    requests = [
        dataclasses.replace(
            request,
            model_request_parameters=_ensure_instruction_parts(request.messages, request.model_request_parameters),
        )
        for request in requests
    ]
    outcomes: dict[str, messages.ModelResponse | ModelAPIError] = {}
    unanswered = requests
    resubmissions = 0
    while unanswered:
        results = await _run_batches(
            model_instance,
            unanswered,
            batch_size=batch_size,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
        )
        retry: list[batch_models.BatchRequest] = []
        for request in unanswered:
            result = results[request.custom_id]
            try:
                if isinstance(result, ModelAPIError):
                    raise result
                outcomes[request.custom_id] = model_instance.batch_decode(result, request)
            except ModelAPIError as error:
                outcomes[request.custom_id] = error
                if resubmissions < max_resubmissions and _resubmittable(result, error):
                    retry.append(request)
        unanswered = retry
        resubmissions += 1
    return [outcomes[request.custom_id] for request in requests]


async def _run_batches(
    model: models.Model,
    requests: Sequence[batch_models.BatchRequest],
    *,
    batch_size: int,
    poll_interval: timedelta,
    max_poll_interval: timedelta,
) -> dict[str, batch_models.BatchResult | ModelAPIError]:
    """Submit `requests` in as few batches as allowed, wait for them all, and return each request's result.

    That's an error for the requests of a job that ended without results, and for a request a batch ended without a
    result for.
    """
    groups: dict[str, list[batch_models.EncodedBatchRequest]] = {}
    for request in requests:
        encoded = await model.batch_encode(request)
        groups.setdefault(encoded.group, []).append(encoded)

    batches: list[batch_models.Batch] = []
    batch_custom_ids: list[list[str]] = []
    try:
        for group in groups.values():
            for chunk in _chunks(model, group, batch_size):
                batches.append(await model.batch_create(chunk))
                batch_custom_ids.append([encoded.custom_id for encoded in chunk])
        for index, batch in enumerate(batches):
            batches[index] = await _wait_for_batch(
                model,
                batch,
                poll_interval=poll_interval,
                max_poll_interval=max_poll_interval,
                # Later batches ran while the earlier ones were waited for, so they're checked straight away.
                check_now=index > 0,
            )
        results: dict[str, batch_models.BatchResult | ModelAPIError] = {}
        for batch, custom_ids in zip(batches, batch_custom_ids, strict=True):
            try:
                batch_results = {result.custom_id: result for result in await model.batch_results(batch)}
            except BatchJobError as error:
                # The job ended without results, so its error is each of its requests' outcome.
                results.update(dict.fromkeys(custom_ids, error))
            else:
                unlisted = model._batch_unlisted_status(batch)  # pyright: ignore[reportPrivateUsage]
                for custom_id in custom_ids:
                    results[custom_id] = batch_results.get(custom_id) or (
                        missing_result_error(model.model_name, batch.id)
                        if unlisted is None
                        else batch_models.BatchResult(custom_id=custom_id, status=unlisted)
                    )
    except BaseException:
        await _cancel_batches(model, batches)
        raise
    return results


def _chunks(
    model: models.Model, requests: Sequence[batch_models.EncodedBatchRequest], max_requests: int
) -> Iterator[list[batch_models.EncodedBatchRequest]]:
    """Split requests into batches of at most `max_requests` requests and the model's `batch_max_bytes` bytes."""
    max_bytes = model.batch_max_bytes
    chunk: list[batch_models.EncodedBatchRequest] = []
    size = 0
    for request in requests:
        request_size = model.batch_request_size(request)
        if chunk and (len(chunk) == max_requests or (max_bytes is not None and size + request_size > max_bytes)):
            yield chunk
            chunk, size = [], 0
        chunk.append(request)
        size += request_size
    # `model_request_batch` only splits groups it put a request in.
    yield chunk


def _resubmittable(result: batch_models.BatchResult | ModelAPIError, error: ModelAPIError) -> bool:
    """Whether a request that failed with `error` is submitted again: see `resubmittable`."""
    if isinstance(result, BatchJobError):
        return transient(result)
    # Otherwise an error is the batch's missing result for the request.
    return isinstance(result, ModelAPIError) or resubmittable(result, error)


async def _wait_for_batch(
    model: models.Model,
    batch: batch_models.Batch,
    *,
    poll_interval: timedelta,
    max_poll_interval: timedelta,
    check_now: bool,
) -> batch_models.Batch:
    """Check on `batch` with a growing interval until it has ended, and return its final state.

    With `check_now`, it's checked once before the first wait. A check that fails the way a live request is retried
    on is tried again at the next poll, as `BatchModelRequests` does.
    """
    polls = failed_checks = 0
    while check_now or batch.status != 'ended':
        if not check_now:
            await anyio.sleep(poll_delay(polls, poll_interval, max_poll_interval))
            polls += 1
        check_now = False
        try:
            batch = await model.batch_status(batch)
        except ModelAPIError as error:
            failed_checks += 1
            if not check_failure_tolerated(error, failed_checks):
                raise
        else:
            failed_checks = 0
    return batch


async def _cancel_batches(model: models.Model, batches: Sequence[batch_models.Batch]) -> None:
    """Cancel the batches that haven't ended, shielded so the cancellation that triggered it can't interrupt it."""
    with anyio.CancelScope(shield=True):
        async with anyio.create_task_group() as tg:
            for batch in batches:
                if batch.status != 'ended':
                    tg.start_soon(_cancel_batch, model, batch)


async def _cancel_batch(model: models.Model, batch: batch_models.Batch) -> None:
    # Best effort: a failed cancel must not replace the error or cancellation that stopped the call.
    with suppress(Exception):
        await model.batch_cancel(batch)


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
    check_resumable(model_instance, messages)
    mrp = _ensure_instruction_parts(messages, model_request_parameters or models.ModelRequestParameters())
    return open_request_stream(
        model_instance,
        _prepare_messages_carrying_a_retry(model_instance, messages, mrp),
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
    It drives the asynchronous stream on the caller's event loop while providing a synchronous iterator interface.
    The returned context manager must be used and closed on the thread where the synchronous stream is created.

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
    """Synchronous wrapper for an async streaming response, running the whole stream on the caller's event loop.

    The stream uses the internal `SyncStreamBridge` to keep context-manager
    and iterator lifecycles in stable tasks. Exiting the `with` block cancels the underlying request promptly
    and closes the connection instead of waiting for the whole response to arrive.

    This class must be used as a context manager with the `with` statement. The synchronous stream is created
    when the `with` block is entered and must be used and closed on that thread.
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
        # The pump task retains this factory until it exits. Capture the stream rather than the bridge
        # to avoid a reference cycle that would delay the bridge's finalizer.
        stream = bridge.stream
        return bridge.stream_sync(lambda: aiter(stream))

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
