"""Tests for Anthropic batch processing functionality."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from pydantic_ai import (
    BatchStatus,
    ModelMessage,
    ModelRequest,
    ModelRequestParameters,
    TextPart,
)

from ..conftest import try_import

with try_import() as imports_successful:
    from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage

    from pydantic_ai import ModelAPIError, ModelHTTPError
    from pydantic_ai.models.anthropic import AnthropicBatch, AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]


# --- Helper functions ---


def create_beta_message(text: str = 'Hello there!') -> BetaMessage:
    """Create a proper BetaMessage with text content."""
    return BetaMessage(
        id='msg_123',
        content=[BetaTextBlock(text=text, type='text')],
        model='claude-sonnet-4-5',
        role='assistant',
        stop_reason='end_turn',
        type='message',
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )


# --- Mock Classes ---


@dataclass
class MockMessageBatchResponse:
    """Mock Anthropic MessageBatch response."""

    id: str = 'msgbatch_abc123'
    type: str = 'message_batch'
    processing_status: str = 'in_progress'
    created_at: str = '2024-01-01T00:00:00Z'
    ended_at: str | None = None
    expires_at: str = '2024-01-02T00:00:00Z'
    cancel_initiated_at: str | None = None
    results_url: str | None = None
    request_counts: Any = field(default_factory=lambda: MockRequestCounts())


@dataclass
class MockRequestCounts:
    """Mock request counts."""

    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    canceled: int = 0
    expired: int = 0


@dataclass
class MockBatchResultEntry:
    """Mock entry in batch results stream."""

    custom_id: str
    result: Any


@dataclass
class MockSucceededResult:
    """Mock succeeded result."""

    type: str = 'succeeded'
    message: BetaMessage | None = None


@dataclass
class MockErroredResult:
    """Mock errored result."""

    type: str = 'errored'
    error: Any = None


@dataclass
class MockCanceledResult:
    """Mock canceled result."""

    type: str = 'canceled'


@dataclass
class MockExpiredResult:
    """Mock expired result."""

    type: str = 'expired'


@dataclass
class MockError:
    """Mock error object."""

    type: str = 'invalid_request_error'
    message: str = 'Error message'


@dataclass
class MockAnthropicBatchClient:
    """Mock for Anthropic batch-related operations."""

    batches_list: list[MockMessageBatchResponse] = field(default_factory=list)
    batch_create_calls: list[dict[str, Any]] = field(default_factory=list)
    results_entries: list[MockBatchResultEntry] = field(default_factory=list)
    base_url: str = 'https://api.anthropic.com'

    def create_model(self) -> AnthropicModel:
        """Create an AnthropicModel using this mock as the client."""
        return AnthropicModel(
            'claude-sonnet-4-5',
            provider=AnthropicProvider(anthropic_client=cast(AsyncAnthropic, self)),
        )

    @classmethod
    def create_mock(cls) -> AsyncAnthropic:  # pragma: no cover
        """Create a mock Anthropic client with proper type casting."""
        return cast(AsyncAnthropic, cls())

    @property
    def messages(self) -> Any:
        return self._MessagesNamespace(self)  # pragma: no cover

    @property
    def beta(self) -> Any:
        return self._BetaNamespace(self)

    @dataclass
    class _MessagesNamespace:
        parent: MockAnthropicBatchClient

        @property
        def batches(self) -> Any:
            return MockAnthropicBatchClient._BatchesNamespace(self.parent)  # pragma: no cover

    @dataclass
    class _BetaNamespace:
        parent: MockAnthropicBatchClient

        @property
        def messages(self) -> Any:
            return MockAnthropicBatchClient._BetaMessagesNamespace(self.parent)

    @dataclass
    class _BetaMessagesNamespace:
        parent: MockAnthropicBatchClient

        async def create(self, **kwargs: Any) -> BetaMessage:
            """Mock regular message creation (for prepare_request)."""
            return create_beta_message()  # pragma: no cover

        async def count_tokens(self, **kwargs: Any) -> Any:
            """Mock token counting."""
            mock = MagicMock()  # pragma: no cover
            mock.input_tokens = 10  # pragma: no cover
            return mock  # pragma: no cover

        @property
        def batches(self) -> Any:
            """Expose batches under beta.messages for batch processing."""
            return MockAnthropicBatchClient._BatchesNamespace(self.parent)

    @dataclass
    class _BatchesNamespace:
        parent: MockAnthropicBatchClient

        async def create(self, **kwargs: Any) -> MockMessageBatchResponse:
            self.parent.batch_create_calls.append(kwargs)
            if self.parent.batches_list:  # pragma: no cover
                return self.parent.batches_list[0]
            return MockMessageBatchResponse()

        async def retrieve(self, batch_id: str) -> MockMessageBatchResponse:
            for batch in self.parent.batches_list:
                if batch.id == batch_id:  # pragma: no branch
                    return batch
            return MockMessageBatchResponse(id=batch_id)  # pragma: no cover

        async def cancel(self, batch_id: str) -> MockMessageBatchResponse:
            return MockMessageBatchResponse(
                id=batch_id,
                processing_status='canceling',
            )

        async def results(self, batch_id: str) -> Any:
            return MockAsyncIterator(self.parent.results_entries)


class MockAsyncIterator:
    """Mock async iterator for results stream."""

    def __init__(self, items: list[Any]):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


# --- AnthropicBatch Dataclass Tests ---


class TestAnthropicBatch:
    """Tests for AnthropicBatch dataclass."""

    def test_anthropic_batch_creation(self):
        """Test AnthropicBatch extends Batch with Anthropic-specific fields."""
        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            results_url='https://api.anthropic.com/v1/messages/batches/msgbatch_123/results',
            expires_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            processing_count=0,
            succeeded_count=5,
            errored_count=0,
            canceled_count=0,
            expired_count=0,
        )

        assert batch.id == 'msgbatch_123'
        assert batch.results_url is not None
        assert batch.expires_at is not None
        assert batch.succeeded_count == 5
        assert batch.is_complete is True

    def test_anthropic_batch_status_mapping(self):
        """Test AnthropicBatch status mapping for terminal states."""
        batch_id = 'msgbatch_123'
        created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Completed batch
        completed = AnthropicBatch(id=batch_id, created_at=created_at, status=BatchStatus.COMPLETED)
        assert completed.is_complete is True
        assert completed.is_successful is True

        # Failed batch
        failed = AnthropicBatch(id=batch_id, created_at=created_at, status=BatchStatus.FAILED)
        assert failed.is_complete is True
        assert failed.is_successful is False

        # Cancelled batch
        cancelled = AnthropicBatch(id=batch_id, created_at=created_at, status=BatchStatus.CANCELLED)
        assert cancelled.is_complete is True
        assert cancelled.is_successful is False

        # Expired batch
        expired = AnthropicBatch(id=batch_id, created_at=created_at, status=BatchStatus.EXPIRED)
        assert expired.is_complete is True
        assert expired.is_successful is False


# --- AnthropicModel Batch Method Tests ---


class TestAnthropicModelBatch:
    """Tests for AnthropicModel batch methods."""

    async def test_batch_create_builds_requests_array(self, allow_model_requests: None):
        """Test that batch_create builds proper requests array."""
        mock_client = MockAnthropicBatchClient()

        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        batch = await model.batch_create(requests)

        # Verify batch was created with correct params
        assert len(mock_client.batch_create_calls) == 1
        batch_call = mock_client.batch_create_calls[0]
        assert 'requests' in batch_call
        assert len(batch_call['requests']) == 2
        assert batch_call['requests'][0]['custom_id'] == 'req-1'
        assert batch_call['requests'][1]['custom_id'] == 'req-2'

        # Verify each request has proper params structure
        for req in batch_call['requests']:
            assert 'params' in req
            assert 'model' in req['params']
            assert 'max_tokens' in req['params']
            assert 'messages' in req['params']

        # Verify returned batch
        assert batch.id == 'msgbatch_abc123'
        assert batch.status == BatchStatus.IN_PROGRESS

    async def test_batch_create_minimum_requests(self, allow_model_requests: None):
        """Test that batch_create requires at least 2 requests."""
        mock_client = MockAnthropicBatchClient()
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
        ]

        with pytest.raises(ValueError, match='at least 2 requests'):
            await model.batch_create(requests)

    async def test_batch_status(self, allow_model_requests: None):
        """Test batch_status retrieves updated batch info."""
        mock_client = MockAnthropicBatchClient()
        mock_client.batches_list = [
            MockMessageBatchResponse(
                id='msgbatch_123',
                processing_status='ended',
                ended_at='2024-01-01T12:00:00Z',
                request_counts=MockRequestCounts(succeeded=5),
            )
        ]

        model = mock_client.create_model()

        initial_batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await model.batch_status(initial_batch)

        assert updated.status == BatchStatus.COMPLETED
        assert updated.succeeded_count == 5

    async def test_batch_results_success(self, allow_model_requests: None):
        """Test batch_results parses results correctly."""
        mock_client = MockAnthropicBatchClient()
        mock_client.results_entries = [
            MockBatchResultEntry(
                custom_id='req-1',
                result=MockSucceededResult(message=create_beta_message()),
            ),
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        results = await model.batch_results(batch)

        assert len(results) == 1
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True
        assert results[0].response is not None
        assert len(results[0].response.parts) == 1
        assert isinstance(results[0].response.parts[0], TextPart)
        assert results[0].response.parts[0].content == 'Hello there!'

    async def test_batch_results_with_errors(self, allow_model_requests: None):
        """Test batch_results handles error responses."""
        mock_client = MockAnthropicBatchClient()
        mock_client.results_entries = [
            MockBatchResultEntry(
                custom_id='req-1',
                result=MockSucceededResult(message=create_beta_message()),
            ),
            MockBatchResultEntry(
                custom_id='req-2',
                result=MockErroredResult(error=MockError(type='rate_limit_error', message='Rate limit exceeded')),
            ),
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        results = await model.batch_results(batch)

        assert len(results) == 2

        # First result should be successful
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True

        # Second result should have error
        assert results[1].custom_id == 'req-2'
        assert results[1].is_successful is False
        assert results[1].error is not None
        assert results[1].error.code == 'rate_limit_error'
        assert results[1].error.message == 'Rate limit exceeded'

    async def test_batch_results_with_canceled(self, allow_model_requests: None):
        """Test batch_results handles canceled responses."""
        mock_client = MockAnthropicBatchClient()
        mock_client.results_entries = [
            MockBatchResultEntry(
                custom_id='req-1',
                result=MockCanceledResult(),
            ),
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        results = await model.batch_results(batch)

        assert len(results) == 1
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is False
        assert results[0].error is not None
        assert results[0].error.code == 'canceled'

    async def test_batch_results_with_expired(self, allow_model_requests: None):
        """Test batch_results handles expired responses."""
        mock_client = MockAnthropicBatchClient()
        mock_client.results_entries = [
            MockBatchResultEntry(
                custom_id='req-1',
                result=MockExpiredResult(),
            ),
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        results = await model.batch_results(batch)

        assert len(results) == 1
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is False
        assert results[0].error is not None
        assert results[0].error.code == 'expired'

    async def test_batch_results_incomplete_raises(self, allow_model_requests: None):
        """Test batch_results raises if batch not complete."""
        mock_client = MockAnthropicBatchClient()
        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(ValueError, match='not complete'):
            await model.batch_results(batch)

    async def test_batch_cancel(self, allow_model_requests: None):
        """Test batch_cancel sends cancel request."""
        mock_client = MockAnthropicBatchClient()
        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        cancelled = await model.batch_cancel(batch)

        assert cancelled.status == BatchStatus.CANCELLING


# --- Status Mapping Tests ---


class TestAnthropicBatchStatusMapping:
    """Tests for Anthropic batch status mapping logic."""

    async def test_status_in_progress(self, allow_model_requests: None):
        """Test in_progress status maps correctly."""
        mock_client = MockAnthropicBatchClient()
        mock_client.batches_list = [
            MockMessageBatchResponse(
                id='msgbatch_123',
                processing_status='in_progress',
                request_counts=MockRequestCounts(processing=5),
            )
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.PENDING,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.IN_PROGRESS

    async def test_status_canceling(self, allow_model_requests: None):
        """Test canceling status maps correctly."""
        mock_client = MockAnthropicBatchClient()
        mock_client.batches_list = [
            MockMessageBatchResponse(
                id='msgbatch_123',
                processing_status='canceling',
            )
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.CANCELLING

    async def test_status_ended_all_succeeded(self, allow_model_requests: None):
        """Test ended status with all succeeded maps to COMPLETED."""
        mock_client = MockAnthropicBatchClient()
        mock_client.batches_list = [
            MockMessageBatchResponse(
                id='msgbatch_123',
                processing_status='ended',
                ended_at='2024-01-01T12:00:00Z',
                request_counts=MockRequestCounts(succeeded=5),
            )
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.COMPLETED

    async def test_status_ended_all_errored(self, allow_model_requests: None):
        """Test ended status with all errored maps to FAILED."""
        mock_client = MockAnthropicBatchClient()
        mock_client.batches_list = [
            MockMessageBatchResponse(
                id='msgbatch_123',
                processing_status='ended',
                ended_at='2024-01-01T12:00:00Z',
                request_counts=MockRequestCounts(errored=5),
            )
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.FAILED

    async def test_status_ended_all_canceled(self, allow_model_requests: None):
        """Test ended status with all canceled maps to CANCELLED."""
        mock_client = MockAnthropicBatchClient()
        mock_client.batches_list = [
            MockMessageBatchResponse(
                id='msgbatch_123',
                processing_status='ended',
                ended_at='2024-01-01T12:00:00Z',
                request_counts=MockRequestCounts(canceled=5),
            )
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.CANCELLED

    async def test_status_ended_all_expired(self, allow_model_requests: None):
        """Test ended status with all expired maps to EXPIRED."""
        mock_client = MockAnthropicBatchClient()
        mock_client.batches_list = [
            MockMessageBatchResponse(
                id='msgbatch_123',
                processing_status='ended',
                ended_at='2024-01-01T12:00:00Z',
                request_counts=MockRequestCounts(expired=5),
            )
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.EXPIRED

    async def test_status_ended_mixed_results_completed(self, allow_model_requests: None):
        """Test ended status with mixed results (some succeeded) maps to COMPLETED."""
        mock_client = MockAnthropicBatchClient()
        mock_client.batches_list = [
            MockMessageBatchResponse(
                id='msgbatch_123',
                processing_status='ended',
                ended_at='2024-01-01T12:00:00Z',
                request_counts=MockRequestCounts(succeeded=3, errored=2),
            )
        ]

        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await model.batch_status(batch)
        # Mixed results with some succeeded = COMPLETED (partial success)
        assert updated.status == BatchStatus.COMPLETED
        assert updated.succeeded_count == 3
        assert updated.errored_count == 2


# --- API Error Handling Tests ---


class TestAnthropicBatchAPIErrorHandling:
    """Tests for batch API error handling."""

    @dataclass
    class MockAnthropicWithAPIError:
        """Mock Anthropic client that raises API errors."""

        error_type: str = 'status'  # 'status' or 'connection'
        error_on: str = 'batches.create'
        base_url: str = 'https://api.anthropic.com'

        def create_model(self) -> AnthropicModel:
            """Create an AnthropicModel using this mock as the client."""
            return AnthropicModel(
                'claude-sonnet-4-5',
                provider=AnthropicProvider(anthropic_client=cast(AsyncAnthropic, self)),
            )

        @property
        def messages(self) -> Any:
            return self._MessagesNamespace(self)  # pragma: no cover

        @property
        def beta(self) -> Any:
            return self._BetaNamespace(self)

        def _raise_error(self) -> None:
            if self.error_type == 'status':
                raise APIStatusError(
                    message='Rate limit exceeded',
                    response=MagicMock(status_code=429),
                    body={'error': {'message': 'Rate limit exceeded'}},
                )
            else:
                raise APIConnectionError(message='Connection failed', request=MagicMock())

        @dataclass
        class _MessagesNamespace:
            parent: TestAnthropicBatchAPIErrorHandling.MockAnthropicWithAPIError

            @property
            def batches(self) -> Any:
                return TestAnthropicBatchAPIErrorHandling.MockAnthropicWithAPIError._BatchesNamespace(
                    self.parent
                )  # pragma: no cover

        @dataclass
        class _BetaNamespace:
            parent: TestAnthropicBatchAPIErrorHandling.MockAnthropicWithAPIError

            @property
            def messages(self) -> Any:
                return TestAnthropicBatchAPIErrorHandling.MockAnthropicWithAPIError._BetaMessagesNamespace(self.parent)

        @dataclass
        class _BetaMessagesNamespace:
            parent: TestAnthropicBatchAPIErrorHandling.MockAnthropicWithAPIError

            async def create(self, **kwargs: Any) -> BetaMessage:
                return create_beta_message()  # pragma: no cover

            @property
            def batches(self) -> Any:
                """Expose batches under beta.messages for batch processing."""
                return TestAnthropicBatchAPIErrorHandling.MockAnthropicWithAPIError._BatchesNamespace(self.parent)

        @dataclass
        class _BatchesNamespace:
            parent: TestAnthropicBatchAPIErrorHandling.MockAnthropicWithAPIError

            async def create(self, **kwargs: Any) -> MockMessageBatchResponse:
                if self.parent.error_on == 'batches.create':
                    self.parent._raise_error()
                return MockMessageBatchResponse()  # pragma: no cover

            async def retrieve(self, batch_id: str) -> MockMessageBatchResponse:
                if self.parent.error_on == 'batches.retrieve':
                    self.parent._raise_error()
                return MockMessageBatchResponse(id=batch_id)  # pragma: no cover

            async def cancel(self, batch_id: str) -> MockMessageBatchResponse:
                if self.parent.error_on == 'batches.cancel':
                    self.parent._raise_error()
                return MockMessageBatchResponse(id=batch_id)  # pragma: no cover

            async def results(self, batch_id: str) -> Any:
                if self.parent.error_on == 'batches.results':
                    self.parent._raise_error()
                return MockAsyncIterator([])  # pragma: no cover

    async def test_batch_create_api_status_error(self, allow_model_requests: None):
        """Test batch_create raises ModelHTTPError on APIStatusError."""
        mock_client = self.MockAnthropicWithAPIError(error_type='status', error_on='batches.create')
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_create(requests)

        assert exc_info.value.status_code == 429

    async def test_batch_create_api_connection_error(self, allow_model_requests: None):
        """Test batch_create raises ModelAPIError on APIConnectionError."""
        mock_client = self.MockAnthropicWithAPIError(error_type='connection', error_on='batches.create')
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        with pytest.raises(ModelAPIError) as exc_info:
            await model.batch_create(requests)

        assert 'Connection failed' in str(exc_info.value)

    async def test_batch_status_api_status_error(self, allow_model_requests: None):
        """Test batch_status raises ModelHTTPError on APIStatusError."""
        mock_client = self.MockAnthropicWithAPIError(error_type='status', error_on='batches.retrieve')
        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_status(batch)

        assert exc_info.value.status_code == 429

    async def test_batch_status_api_connection_error(self, allow_model_requests: None):
        """Test batch_status raises ModelAPIError on APIConnectionError."""
        mock_client = self.MockAnthropicWithAPIError(error_type='connection', error_on='batches.retrieve')
        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(ModelAPIError) as exc_info:
            await model.batch_status(batch)

        assert 'Connection failed' in str(exc_info.value)

    async def test_batch_cancel_api_status_error(self, allow_model_requests: None):
        """Test batch_cancel raises ModelHTTPError on APIStatusError."""
        mock_client = self.MockAnthropicWithAPIError(error_type='status', error_on='batches.cancel')
        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_cancel(batch)

        assert exc_info.value.status_code == 429

    async def test_batch_cancel_api_connection_error(self, allow_model_requests: None):
        """Test batch_cancel raises ModelAPIError on APIConnectionError."""
        mock_client = self.MockAnthropicWithAPIError(error_type='connection', error_on='batches.cancel')
        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(ModelAPIError) as exc_info:
            await model.batch_cancel(batch)

        assert 'Connection failed' in str(exc_info.value)

    async def test_batch_results_api_status_error(self, allow_model_requests: None):
        """Test batch_results raises ModelHTTPError on APIStatusError."""
        mock_client = self.MockAnthropicWithAPIError(error_type='status', error_on='batches.results')
        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_results(batch)

        assert exc_info.value.status_code == 429

    async def test_batch_results_api_connection_error(self, allow_model_requests: None):
        """Test batch_results raises ModelAPIError on APIConnectionError."""
        mock_client = self.MockAnthropicWithAPIError(error_type='connection', error_on='batches.results')
        model = mock_client.create_model()

        batch = AnthropicBatch(
            id='msgbatch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(ModelAPIError) as exc_info:
            await model.batch_results(batch)

        assert 'Connection failed' in str(exc_info.value)

    async def test_batch_create_per_request_settings(self, allow_model_requests: None):
        """Test that per-request settings override batch-wide settings."""
        mock_client = MockAnthropicBatchClient()
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]

        # Create requests with different per-request settings
        params_1 = replace(ModelRequestParameters(), model_settings={'temperature': 0.2, 'max_tokens': 100})
        params_2 = replace(ModelRequestParameters(), model_settings={'temperature': 0.9, 'max_tokens': 200})

        requests = [
            ('req-1', messages_1, params_1),
            ('req-2', messages_2, params_2),
        ]

        # Call with batch-wide settings that should be overridden
        await model.batch_create(requests, model_settings={'temperature': 0.5, 'max_tokens': 150})

        # Verify batch was created
        assert len(mock_client.batch_create_calls) == 1
        batch_call = mock_client.batch_create_calls[0]
        assert len(batch_call['requests']) == 2

        # Verify first request has temperature 0.2 and max_tokens 100 (per-request settings)
        req_1_params = batch_call['requests'][0]['params']
        assert req_1_params['temperature'] == 0.2
        assert req_1_params['max_tokens'] == 100

        # Verify second request has temperature 0.9 and max_tokens 200 (per-request settings)
        req_2_params = batch_call['requests'][1]['params']
        assert req_2_params['temperature'] == 0.9
        assert req_2_params['max_tokens'] == 200

    async def test_batch_create_batch_wide_fallback(self, allow_model_requests: None):
        """Test that batch-wide settings are used when per-request settings are absent."""
        mock_client = MockAnthropicBatchClient()
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]

        # Create requests without per-request settings
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        # Call with batch-wide settings
        await model.batch_create(requests, model_settings={'temperature': 0.7, 'max_tokens': 500})

        # Verify batch was created
        assert len(mock_client.batch_create_calls) == 1
        batch_call = mock_client.batch_create_calls[0]

        # Both requests should use batch-wide settings
        assert batch_call['requests'][0]['params']['temperature'] == 0.7
        assert batch_call['requests'][0]['params']['max_tokens'] == 500
        assert batch_call['requests'][1]['params']['temperature'] == 0.7
        assert batch_call['requests'][1]['params']['max_tokens'] == 500

    async def test_batch_create_settings_merging(self, allow_model_requests: None):
        """Test that per-request settings are merged with batch-wide settings."""
        mock_client = MockAnthropicBatchClient()
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]

        # Per-request setting only overrides temperature, not max_tokens
        params_1 = replace(ModelRequestParameters(), model_settings={'temperature': 0.9})

        requests = [
            ('req-1', messages_1, params_1),
            ('req-2', messages_1, ModelRequestParameters()),  # No per-request settings
        ]

        # Batch-wide has both temperature and max_tokens
        await model.batch_create(requests, model_settings={'temperature': 0.5, 'max_tokens': 300})

        batch_call = mock_client.batch_create_calls[0]

        # First request: temperature from per-request (0.9), max_tokens from batch-wide (300)
        req_1_params = batch_call['requests'][0]['params']
        assert req_1_params['temperature'] == 0.9
        assert req_1_params['max_tokens'] == 300

        # Second request: both from batch-wide
        req_2_params = batch_call['requests'][1]['params']
        assert req_2_params['temperature'] == 0.5
        assert req_2_params['max_tokens'] == 300
