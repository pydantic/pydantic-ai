"""BatchModel wrapper for queueing agent.run() calls and submitting as a batch."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .._run_context import RunContext
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from . import Batch, BatchResult, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from . import KnownModelName


@dataclass
class _QueuedRequest:
    """Internal representation of a queued batch request."""

    custom_id: str
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters
    future: asyncio.Future[ModelResponse]


@dataclass
class BatchModel(Model):
    """A model wrapper that queues requests and submits them as a batch.

    This enables using the Agent API with batch processing. Requests made via
    `agent.run()` are queued until either:
    - `batch_size` is reached (if set), triggering auto-submit
    - `submit()` is called manually
    - The context manager exits

    The responses are resolved asynchronously when the batch completes.

    Example usage pattern::

        async with BatchModel('openai:gpt-5-mini', batch_size=100) as batch_model:
            agent = Agent(model=batch_model)
            tasks = [agent.run(prompt) for prompt in prompts]
            await batch_model.submit()
            results = await asyncio.gather(*tasks)

    For complete runnable examples, see the [Batch Processing documentation](https://ai.pydantic.dev/batch-processing/).

    Note:
        Streaming is not supported with batch processing since batches
        are processed asynchronously by the provider.
    """

    wrapped: Model = field(repr=False)
    """The underlying model to use for batch requests."""

    batch_size: int | None = None
    """Optional batch size threshold. When the queue reaches this size, auto-submit is triggered."""

    should_submit: Callable[[list[_QueuedRequest]], bool] | None = None
    """Optional callback to determine if batch should be submitted. Takes the current queue."""

    poll_interval: float = 60.0
    """Interval in seconds between status checks when waiting for batch completion."""

    _queue: list[_QueuedRequest] = field(default_factory=list, repr=False, init=False)
    """Internal queue of pending requests."""

    _pending_batch: Batch | None = field(default=None, repr=False, init=False)
    """Currently processing batch, if any."""

    _batch_task: asyncio.Task[list[BatchResult]] | None = field(default=None, repr=False, init=False)
    """Task waiting for batch completion."""

    def __init__(
        self,
        model: Model | KnownModelName | str,
        *,
        batch_size: int | None = None,
        should_submit: Callable[[list[_QueuedRequest]], bool] | None = None,
        poll_interval: float = 60.0,
        settings: ModelSettings | None = None,
        profile: Any = None,
    ) -> None:
        """Initialize the BatchModel wrapper.

        Args:
            model: The underlying model to use for batch requests.
                Can be a Model instance or a model name string.
            batch_size: Optional batch size threshold for auto-submit.
            should_submit: Optional callback to determine if batch should be submitted.
            poll_interval: Interval in seconds between status checks.
            settings: Model-specific settings that will be used as defaults.
            profile: The model profile to use.
        """
        super().__init__(settings=settings, profile=profile)

        if isinstance(model, str):
            self.wrapped = infer_model(model)
        else:
            self.wrapped = model

        self.batch_size = batch_size
        self.should_submit = should_submit
        self.poll_interval = poll_interval
        self._queue = []
        self._pending_batch = None
        self._batch_task = None

    async def __aenter__(self) -> BatchModel:
        """Enter the context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager, submitting any remaining requests."""
        if self._queue and not self._pending_batch:
            await self.submit()

        # Wait for any pending batch to complete
        if self._batch_task:
            try:
                await self._batch_task
            except Exception as e:
                # Errors are propagated through individual futures.
                # If we reach here, it means the task failed in a way that
                # couldn't be propagated to futures (e.g., setup error).
                # Log it for debugging but don't re-raise since futures may
                # already have the error.
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f'Batch task completed with exception: {e}', exc_info=True)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Queue a request and return a future that resolves when the batch completes.

        This method doesn't immediately make a request to the model. Instead, it
        queues the request and returns a future. The actual request is made when
        the batch is submitted (either via `submit()` or auto-submit).

        Args:
            messages: The message history for this request.
            model_settings: Settings for this request.
            model_request_parameters: Tool definitions, output schema, etc.

        Returns:
            The model response (when the batch completes).

        Raises:
            RuntimeError: If a batch is already being processed.
        """
        if self._batch_task and not self._batch_task.done():
            raise RuntimeError(
                'Cannot queue new requests while a batch is being processed. '
                'Wait for the current batch to complete or create a new BatchModel.'
            )

        # Create a future for this request
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ModelResponse] = loop.create_future()

        # Generate a unique custom_id
        custom_id = f'req_{uuid.uuid4().hex[:12]}'

        # Queue the request
        queued = _QueuedRequest(
            custom_id=custom_id,
            messages=messages,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
            future=future,
        )
        self._queue.append(queued)

        # Check if we should auto-submit
        should_submit = False
        if self.batch_size and len(self._queue) >= self.batch_size:
            should_submit = True
        elif self.should_submit and self.should_submit(self._queue):
            should_submit = True

        if should_submit:
            # Don't await - let submit run in background
            asyncio.create_task(self.submit())

        # Return the future - caller will await it
        return await future

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Streaming is not supported with batch processing.

        Raises:
            NotImplementedError: Always, as batch processing is asynchronous.
        """
        raise NotImplementedError(
            'Streaming is not supported with BatchModel. Batch processing is asynchronous and cannot stream responses.'
        )
        yield  # pragma: no cover

    async def submit(self) -> Batch:
        """Submit the queued requests as a batch.

        This method submits all queued requests to the provider's batch API,
        then polls for completion and resolves all pending futures with their
        corresponding responses.

        Returns:
            The Batch object representing the submitted job.

        Raises:
            RuntimeError: If the queue is empty or a batch is already processing.
        """
        if not self._queue:
            raise RuntimeError('Cannot submit an empty batch.')

        if self._batch_task and not self._batch_task.done():  # pragma: no cover
            raise RuntimeError('A batch is already being processed.')

        # Prepare batch requests with per-request settings embedded
        requests: list[tuple[str, list[ModelMessage], ModelRequestParameters]] = []
        for req in self._queue:
            # Embed per-request settings into params if present
            params = req.model_request_parameters
            if req.model_settings is not None:
                params = replace(params, model_settings=req.model_settings)
            requests.append((req.custom_id, req.messages, params))

        # Use BatchModel's own settings as batch-wide fallback
        batch_wide_settings = self.settings

        # Create a copy of the queue and clear it
        pending_requests = list(self._queue)
        self._queue.clear()

        # Submit the batch - if this fails, propagate error to all futures
        try:
            self._pending_batch = await self.wrapped.batch_create(requests, batch_wide_settings)
        except Exception as e:
            # batch_create failed - fail all queued futures
            for req in pending_requests:
                if not req.future.done():  # pragma: no cover
                    req.future.set_exception(e)
            raise

        # Start background task to wait for completion
        self._batch_task = asyncio.create_task(self._wait_and_resolve(pending_requests))

        return self._pending_batch

    async def _wait_and_resolve(self, pending_requests: list[_QueuedRequest]) -> list[BatchResult]:
        """Wait for batch completion and resolve all pending futures."""
        assert self._pending_batch is not None

        batch = self._pending_batch
        try:
            # Poll until complete
            while not batch.is_complete:
                await asyncio.sleep(self.poll_interval)
                batch = await self.wrapped.batch_status(batch)

            # Get results
            results = await self.wrapped.batch_results(batch)

            # Create a map from custom_id to result
            result_map: dict[str, BatchResult] = {r.custom_id: r for r in results}

            # Resolve each future
            for req in pending_requests:
                result = result_map.get(req.custom_id)
                if result is None:  # pragma: no cover
                    req.future.set_exception(RuntimeError(f'No result found for request {req.custom_id}'))
                elif result.is_successful and result.response is not None:
                    req.future.set_result(result.response)
                else:
                    error_msg = f'{result.error.code}: {result.error.message}' if result.error else 'Unknown error'
                    req.future.set_exception(RuntimeError(f'Batch request failed: {error_msg}'))

            return results

        except Exception as e:
            # If batch processing fails, propagate error to all futures
            for req in pending_requests:
                if not req.future.done():  # pragma: lax no cover
                    req.future.set_exception(e)
            raise

        finally:
            self._pending_batch = None

    @property
    def queue_size(self) -> int:
        """Return the current number of queued requests."""
        return len(self._queue)

    @property
    def is_processing(self) -> bool:
        """Return True if a batch is currently being processed."""
        return self._batch_task is not None and not self._batch_task.done()

    @property
    def model_name(self) -> str:
        """The model name of the wrapped model."""
        return self.wrapped.model_name

    @property
    def system(self) -> str:
        """The system of the wrapped model."""
        return self.wrapped.system

    @property
    def base_url(self) -> str | None:
        """The base URL of the wrapped model."""
        return self.wrapped.base_url
