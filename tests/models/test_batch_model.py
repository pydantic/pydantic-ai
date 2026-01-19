"""Tests for the BatchModel wrapper class."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import Batch, BatchError, BatchResult, BatchStatus, ModelRequestParameters
from pydantic_ai.models.batch import BatchModel
from pydantic_ai.usage import RequestUsage


def create_mock_model() -> MagicMock:
    """Create a mock model with batch methods."""
    mock = MagicMock()
    mock.model_name = 'mock-model'
    mock.system = 'mock'
    mock.base_url = 'https://api.mock.com'
    return mock


def create_mock_batch(
    batch_id: str = 'batch_123',
    status: BatchStatus = BatchStatus.IN_PROGRESS,
    request_count: int = 2,
    completed_count: int = 0,
    failed_count: int = 0,
) -> Batch:
    """Create a mock Batch object."""
    return Batch(
        id=batch_id,
        status=status,
        created_at=datetime.now(timezone.utc),
        request_count=request_count,
        completed_count=completed_count,
        failed_count=failed_count,
    )


def create_mock_result(
    custom_id: str,
    content: str = 'Test response',
    is_successful: bool = True,
    error: BatchError | None = None,
) -> BatchResult:
    """Create a mock BatchResult object."""
    if is_successful:
        return BatchResult(
            custom_id=custom_id,
            response=ModelResponse(
                parts=[TextPart(content=content)],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='mock-model',
                timestamp=datetime.now(timezone.utc),
            ),
            error=None,
        )
    else:
        return BatchResult(
            custom_id=custom_id,
            response=None,
            error=error or BatchError(code='error', message='Test error'),
        )


class TestBatchModelBasics:
    """Test basic BatchModel functionality."""

    def test_init_with_model_string(self) -> None:
        """Test initialization with a model name string."""
        with patch('pydantic_ai.models.batch.infer_model') as mock_infer:
            mock_model = create_mock_model()
            mock_infer.return_value = mock_model

            batch_model = BatchModel('openai:gpt-5-mini', batch_size=10)

            mock_infer.assert_called_once_with('openai:gpt-5-mini')
            assert batch_model.wrapped == mock_model
            assert batch_model.batch_size == 10

    def test_init_with_model_instance(self) -> None:
        """Test initialization with a Model instance."""
        mock_model = create_mock_model()

        batch_model = BatchModel(mock_model, batch_size=5, poll_interval=30.0)

        assert batch_model.wrapped == mock_model
        assert batch_model.batch_size == 5
        assert batch_model.poll_interval == 30.0

    def test_model_name_property(self) -> None:
        """Test that model_name is delegated to wrapped model."""
        mock_model = create_mock_model()
        batch_model = BatchModel(mock_model)

        assert batch_model.model_name == 'mock-model'

    def test_system_property(self) -> None:
        """Test that system is delegated to wrapped model."""
        mock_model = create_mock_model()
        batch_model = BatchModel(mock_model)

        assert batch_model.system == 'mock'

    def test_base_url_property(self) -> None:
        """Test that base_url is delegated to wrapped model."""
        mock_model = create_mock_model()
        batch_model = BatchModel(mock_model)

        assert batch_model.base_url == 'https://api.mock.com'


class TestBatchModelQueue:
    """Test BatchModel queue management."""

    def test_queue_starts_empty(self) -> None:
        """Test that queue starts empty."""
        mock_model = create_mock_model()
        batch_model = BatchModel(mock_model)

        assert batch_model.queue_size == 0
        assert not batch_model.is_processing


class TestBatchModelRequest:
    """Test BatchModel.request() method."""

    @pytest.mark.anyio
    async def test_request_queues_without_submit(self) -> None:
        """Test that request() queues the request without submitting."""
        mock_model = create_mock_model()
        batch_model = BatchModel(mock_model)

        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        params = ModelRequestParameters()

        # Start the request but don't await it
        task = asyncio.create_task(batch_model.request(messages, None, params))

        # Give a moment for the task to queue
        await asyncio.sleep(0.01)

        assert batch_model.queue_size == 1
        assert not task.done()

        # Cancel the task since we won't be completing it
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.anyio
    async def test_request_auto_submit_on_batch_size(self) -> None:
        """Test that request() auto-submits when batch_size is reached."""
        mock_model = create_mock_model()

        # Setup batch completion
        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)
        batch_completed = create_mock_batch(status=BatchStatus.COMPLETED, completed_count=2)

        # Capture submitted IDs to create matching results
        submitted_ids: list[str] = []

        async def capture_batch_create(requests: list[tuple[str, Any, Any]], settings: Any) -> Batch:
            for custom_id, _, _ in requests:
                submitted_ids.append(custom_id)
            return batch_in_progress

        async def get_results(batch: Batch) -> list[BatchResult]:
            return [
                create_mock_result(submitted_ids[0], 'Response 1'),
                create_mock_result(submitted_ids[1], 'Response 2'),
            ]

        mock_model.batch_create = AsyncMock(side_effect=capture_batch_create)
        mock_model.batch_status = AsyncMock(return_value=batch_completed)
        mock_model.batch_results = AsyncMock(side_effect=get_results)

        batch_model = BatchModel(mock_model, batch_size=2, poll_interval=0.01)

        messages1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello 1')]
        messages2: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello 2')]
        params = ModelRequestParameters()

        # Queue two requests - should trigger auto-submit
        task1 = asyncio.create_task(batch_model.request(messages1, None, params))
        await asyncio.sleep(0.01)

        # The first request should be queued but not submitted yet
        assert batch_model.queue_size == 1

        task2 = asyncio.create_task(batch_model.request(messages2, None, params))
        await asyncio.sleep(0.05)

        # Both requests should complete
        response1 = await asyncio.wait_for(task1, timeout=1.0)
        response2 = await asyncio.wait_for(task2, timeout=1.0)

        assert isinstance(response1.parts[0], TextPart)
        assert response1.parts[0].content == 'Response 1'
        assert isinstance(response2.parts[0], TextPart)
        assert response2.parts[0].content == 'Response 2'
        mock_model.batch_create.assert_called_once()


class TestBatchModelSubmit:
    """Test BatchModel.submit() method."""

    @pytest.mark.anyio
    async def test_submit_empty_queue_raises(self) -> None:
        """Test that submit() raises when queue is empty."""
        mock_model = create_mock_model()
        batch_model = BatchModel(mock_model)

        with pytest.raises(RuntimeError, match='Cannot submit an empty batch'):
            await batch_model.submit()

    @pytest.mark.anyio
    async def test_submit_creates_batch_and_resolves_futures(self) -> None:
        """Test that submit() creates batch and resolves all futures."""
        mock_model = create_mock_model()

        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)
        batch_completed = create_mock_batch(status=BatchStatus.COMPLETED, completed_count=2)

        # Use side_effect to track the custom_ids that are submitted
        submitted_ids: list[str] = []

        async def capture_batch_create(requests: list[tuple[str, Any, Any]], settings: Any) -> Batch:
            for custom_id, _, _ in requests:
                submitted_ids.append(custom_id)
            return batch_in_progress

        async def get_results(batch: Batch) -> list[BatchResult]:
            return [create_mock_result(cid, f'Response for {cid}') for cid in submitted_ids]

        mock_model.batch_create = AsyncMock(side_effect=capture_batch_create)
        mock_model.batch_status = AsyncMock(return_value=batch_completed)
        mock_model.batch_results = AsyncMock(side_effect=get_results)

        batch_model = BatchModel(mock_model, poll_interval=0.01)

        # Queue some requests
        messages1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello 1')]
        messages2: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello 2')]
        params = ModelRequestParameters()

        task1 = asyncio.create_task(batch_model.request(messages1, None, params))
        task2 = asyncio.create_task(batch_model.request(messages2, None, params))

        # Give time for requests to queue
        await asyncio.sleep(0.01)

        # Submit the batch
        batch = await batch_model.submit()

        assert batch.id == 'batch_123'

        # Wait for results
        response1 = await asyncio.wait_for(task1, timeout=1.0)
        response2 = await asyncio.wait_for(task2, timeout=1.0)

        assert isinstance(response1.parts[0], TextPart)
        assert 'Response for' in response1.parts[0].content
        assert isinstance(response2.parts[0], TextPart)
        assert 'Response for' in response2.parts[0].content


class TestBatchModelContextManager:
    """Test BatchModel as async context manager."""

    @pytest.mark.anyio
    async def test_context_manager_submits_on_exit(self) -> None:
        """Test that exiting context manager submits remaining requests."""
        mock_model = create_mock_model()

        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)
        batch_completed = create_mock_batch(status=BatchStatus.COMPLETED, completed_count=1)

        submitted_ids: list[str] = []

        async def capture_batch_create(requests: list[tuple[str, Any, Any]], settings: Any) -> Batch:
            for custom_id, _, _ in requests:
                submitted_ids.append(custom_id)
            return batch_in_progress

        async def get_results(batch: Batch) -> list[BatchResult]:
            return [create_mock_result(cid, f'Response for {cid}') for cid in submitted_ids]

        mock_model.batch_create = AsyncMock(side_effect=capture_batch_create)
        mock_model.batch_status = AsyncMock(return_value=batch_completed)
        mock_model.batch_results = AsyncMock(side_effect=get_results)

        async with BatchModel(mock_model, poll_interval=0.01) as batch_model:
            messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
            params = ModelRequestParameters()

            task = asyncio.create_task(batch_model.request(messages, None, params))
            await asyncio.sleep(0.01)

            # Don't manually submit - let context manager do it

        # After context exits, the task should complete
        response = await asyncio.wait_for(task, timeout=1.0)
        assert isinstance(response.parts[0], TextPart)
        assert 'Response for' in response.parts[0].content
        mock_model.batch_create.assert_called_once()

    @pytest.mark.anyio
    async def test_context_manager_waits_for_pending_batch(self) -> None:
        """Test that exiting context manager waits for pending batch to complete."""
        mock_model = create_mock_model()

        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)
        batch_completed = create_mock_batch(status=BatchStatus.COMPLETED, completed_count=1)

        submitted_ids: list[str] = []

        async def capture_batch_create(requests: list[tuple[str, Any, Any]], settings: Any) -> Batch:
            for custom_id, _, _ in requests:
                submitted_ids.append(custom_id)
            return batch_in_progress

        async def get_results(batch: Batch) -> list[BatchResult]:
            return [create_mock_result(cid, f'Response for {cid}') for cid in submitted_ids]

        mock_model.batch_create = AsyncMock(side_effect=capture_batch_create)
        mock_model.batch_status = AsyncMock(return_value=batch_completed)
        mock_model.batch_results = AsyncMock(side_effect=get_results)

        batch_model = BatchModel(mock_model, poll_interval=0.01)

        # Enter context, submit, then exit - should wait for batch
        async with batch_model:
            messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
            params = ModelRequestParameters()
            task = asyncio.create_task(batch_model.request(messages, None, params))
            await asyncio.sleep(0.01)
            await batch_model.submit()  # Manually submit
            # Exit with pending batch

        # Task should have completed during __aexit__
        response = await asyncio.wait_for(task, timeout=1.0)
        assert isinstance(response.parts[0], TextPart)
        assert 'Response for' in response.parts[0].content

    @pytest.mark.anyio
    async def test_context_manager_handles_batch_exception(self) -> None:
        """Test that exiting context manager handles exceptions from batch processing."""
        mock_model = create_mock_model()

        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)

        async def capture_batch_create(requests: list[tuple[str, Any, Any]], settings: Any) -> Batch:
            return batch_in_progress

        mock_model.batch_create = AsyncMock(side_effect=capture_batch_create)
        mock_model.batch_status = AsyncMock(side_effect=RuntimeError('API Error'))

        batch_model = BatchModel(mock_model, poll_interval=0.01)

        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        params = ModelRequestParameters()

        # Enter context, submit, then exit - should catch exception in __aexit__
        async with batch_model:
            task = asyncio.create_task(batch_model.request(messages, None, params))
            await asyncio.sleep(0.01)
            await batch_model.submit()
            await asyncio.sleep(0.02)  # Let the batch task fail

        # The task should fail with the API error
        with pytest.raises(RuntimeError, match='API Error'):
            await task


class TestBatchModelProperties:
    """Test BatchModel property methods."""

    @pytest.mark.anyio
    async def test_is_processing_property(self) -> None:
        """Test that is_processing returns correct value during batch processing."""
        mock_model = create_mock_model()

        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)
        batch_completed = create_mock_batch(status=BatchStatus.COMPLETED, completed_count=1)

        submitted_ids: list[str] = []
        status_check_count = 0

        async def capture_batch_create(requests: list[tuple[str, Any, Any]], settings: Any) -> Batch:
            for custom_id, _, _ in requests:
                submitted_ids.append(custom_id)
            return batch_in_progress

        async def batch_status_delayed(batch: Batch) -> Batch:
            nonlocal status_check_count
            status_check_count += 1
            # Return completed on second check
            if status_check_count >= 2:
                return batch_completed
            return batch_in_progress

        async def get_results(batch: Batch) -> list[BatchResult]:
            return [create_mock_result(cid, f'Response for {cid}') for cid in submitted_ids]

        mock_model.batch_create = AsyncMock(side_effect=capture_batch_create)
        mock_model.batch_status = AsyncMock(side_effect=batch_status_delayed)
        mock_model.batch_results = AsyncMock(side_effect=get_results)

        batch_model = BatchModel(mock_model, poll_interval=0.01)

        # Initially not processing
        assert batch_model.is_processing is False

        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        params = ModelRequestParameters()

        task = asyncio.create_task(batch_model.request(messages, None, params))
        await asyncio.sleep(0.01)
        await batch_model.submit()

        # Now should be processing
        assert batch_model.is_processing is True

        # Wait for completion
        await task

        # No longer processing
        assert batch_model.is_processing is False


class TestBatchModelStreaming:
    """Test that BatchModel doesn't support streaming."""

    @pytest.mark.anyio
    async def test_request_stream_raises(self) -> None:
        """Test that request_stream raises NotImplementedError."""
        mock_model = create_mock_model()
        batch_model = BatchModel(mock_model)

        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        params = ModelRequestParameters()

        with pytest.raises(NotImplementedError, match='Streaming is not supported'):
            async with batch_model.request_stream(messages, None, params):
                pass


class TestBatchModelErrorHandling:
    """Test BatchModel error handling."""

    @pytest.mark.anyio
    async def test_batch_error_propagates_to_futures(self) -> None:
        """Test that batch errors are propagated to all pending futures."""
        mock_model = create_mock_model()

        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)
        batch_failed = create_mock_batch(status=BatchStatus.FAILED)

        mock_model.batch_create = AsyncMock(return_value=batch_in_progress)
        mock_model.batch_status = AsyncMock(return_value=batch_failed)
        mock_model.batch_results = AsyncMock(side_effect=RuntimeError('Batch failed'))

        batch_model = BatchModel(mock_model, poll_interval=0.01)

        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        params = ModelRequestParameters()

        task = asyncio.create_task(batch_model.request(messages, None, params))
        await asyncio.sleep(0.01)

        await batch_model.submit()

        with pytest.raises(RuntimeError, match='Batch failed'):
            await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.anyio
    async def test_individual_request_error_in_result(self) -> None:
        """Test that individual request errors are propagated correctly."""
        mock_model = create_mock_model()

        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)
        batch_completed = create_mock_batch(status=BatchStatus.COMPLETED, completed_count=1, failed_count=1)

        submitted_ids: list[str] = []

        async def capture_batch_create(requests: list[tuple[str, Any, Any]], settings: Any) -> Batch:
            for custom_id, _, _ in requests:
                submitted_ids.append(custom_id)
            return batch_in_progress

        async def get_results(batch: Batch) -> list[BatchResult]:
            # First request succeeds, second fails
            return [
                create_mock_result(submitted_ids[0], 'Success'),
                create_mock_result(
                    submitted_ids[1],
                    is_successful=False,
                    error=BatchError(code='rate_limit', message='Too many requests'),
                ),
            ]

        mock_model.batch_create = AsyncMock(side_effect=capture_batch_create)
        mock_model.batch_status = AsyncMock(return_value=batch_completed)
        mock_model.batch_results = AsyncMock(side_effect=get_results)

        batch_model = BatchModel(mock_model, poll_interval=0.01)

        messages1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello 1')]
        messages2: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello 2')]
        params = ModelRequestParameters()

        task1 = asyncio.create_task(batch_model.request(messages1, None, params))
        task2 = asyncio.create_task(batch_model.request(messages2, None, params))

        await asyncio.sleep(0.01)
        await batch_model.submit()

        # First should succeed
        response1 = await asyncio.wait_for(task1, timeout=1.0)
        assert isinstance(response1.parts[0], TextPart)
        assert response1.parts[0].content == 'Success'

        # Second should fail
        with pytest.raises(RuntimeError, match='rate_limit'):
            await asyncio.wait_for(task2, timeout=1.0)

    @pytest.mark.anyio
    async def test_cannot_queue_while_processing(self) -> None:
        """Test that queuing new requests while processing raises."""
        mock_model = create_mock_model()

        # Batch that never completes (for testing)
        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)

        mock_model.batch_create = AsyncMock(return_value=batch_in_progress)
        mock_model.batch_status = AsyncMock(return_value=batch_in_progress)

        batch_model = BatchModel(mock_model, poll_interval=0.5)

        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        params = ModelRequestParameters()

        # Queue first request and submit
        task1 = asyncio.create_task(batch_model.request(messages, None, params))
        await asyncio.sleep(0.01)
        await batch_model.submit()

        # Try to queue another request while batch is processing
        with pytest.raises(RuntimeError, match='Cannot queue new requests'):
            await batch_model.request(messages, None, params)

        # Clean up
        task1.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass


class TestBatchModelShouldSubmitCallback:
    """Test the should_submit callback functionality."""

    @pytest.mark.anyio
    async def test_should_submit_callback_triggers_submit(self) -> None:
        """Test that should_submit callback can trigger auto-submit."""
        mock_model = create_mock_model()

        batch_in_progress = create_mock_batch(status=BatchStatus.IN_PROGRESS)
        batch_completed = create_mock_batch(status=BatchStatus.COMPLETED, completed_count=1)

        submitted_ids: list[str] = []

        async def capture_batch_create(requests: list[tuple[str, Any, Any]], settings: Any) -> Batch:
            for custom_id, _, _ in requests:
                submitted_ids.append(custom_id)
            return batch_in_progress

        async def get_results(batch: Batch) -> list[BatchResult]:
            return [create_mock_result(cid, f'Response for {cid}') for cid in submitted_ids]

        mock_model.batch_create = AsyncMock(side_effect=capture_batch_create)
        mock_model.batch_status = AsyncMock(return_value=batch_completed)
        mock_model.batch_results = AsyncMock(side_effect=get_results)

        # Custom should_submit that triggers on any queue
        def always_submit(queue: list[Any]) -> bool:
            return len(queue) >= 1

        batch_model = BatchModel(mock_model, should_submit=always_submit, poll_interval=0.01)

        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        params = ModelRequestParameters()

        # Queue a request - should trigger auto-submit via callback
        response = await asyncio.wait_for(batch_model.request(messages, None, params), timeout=1.0)

        assert isinstance(response.parts[0], TextPart)
        assert 'Response for' in response.parts[0].content
        mock_model.batch_create.assert_called_once()
