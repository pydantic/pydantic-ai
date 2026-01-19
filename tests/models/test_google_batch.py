"""Tests for Google batch processing functionality."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

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
    from google.genai import Client
    from google.genai.types import Candidate, Content, GenerateContentResponse, Part

    from pydantic_ai import ModelHTTPError
    from pydantic_ai.models.google import GoogleBatch, GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
]


# --- Mock Classes ---


@dataclass
class MockBatchResponse:
    """Mock Google Batch response."""

    name: str = 'projects/test/locations/us-central1/batches/batch_abc123'
    state: str = 'JOB_STATE_RUNNING'
    create_time: str = '2024-01-01T00:00:00Z'
    end_time: str | None = None
    display_name: str | None = None
    request_counts: Any = None
    responses: list[Any] | None = None


@dataclass
class MockRequestCounts:
    """Mock request counts."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0


@dataclass
class MockBatchResultItem:
    """Mock item in batch results.

    Google's InlinedResponse format only has 'response' and 'error' fields.
    Unlike OpenAI/Anthropic, there is no custom_id - responses are matched by position.
    """

    response: Any | None = None
    error: Any | None = None


@dataclass
class MockError:
    """Mock error object."""

    code: str = 'invalid_request'
    message: str = 'Error message'


@dataclass
class MockAPIError(Exception):
    """Mock Google API error."""

    code: int = 400
    details: dict[str, Any] = field(default_factory=dict)
    message: str = 'API Error'


@dataclass
class MockHttpOptions:
    """Mock HTTP options."""

    base_url: str = 'https://generativelanguage.googleapis.com'


@dataclass
class MockAPIClient:
    """Mock API client for provider access."""

    vertexai: bool = False
    _http_options: MockHttpOptions = field(default_factory=MockHttpOptions)


@dataclass
class MockGoogleBatchClient:
    """Mock for Google batch-related operations."""

    batches_list: list[MockBatchResponse] = field(default_factory=list)
    batch_create_calls: list[dict[str, Any]] = field(default_factory=list)
    _api_client: MockAPIClient = field(default_factory=MockAPIClient)

    def create_model(self) -> GoogleModel:
        """Create a GoogleModel using this mock as the client."""
        return GoogleModel(
            'gemini-2.0-flash',
            provider=GoogleProvider(client=cast(Client, self)),
        )

    @property
    def aio(self) -> Any:
        return self._AioNamespace(self)

    @dataclass
    class _AioNamespace:
        parent: MockGoogleBatchClient

        @property
        def batches(self) -> Any:
            return MockGoogleBatchClient._BatchesNamespace(self.parent)

        @property
        def models(self) -> Any:
            return MockGoogleBatchClient._ModelsNamespace(self.parent)  # pragma: no cover

    @dataclass
    class _ModelsNamespace:
        parent: MockGoogleBatchClient

        async def generate_content(self, **kwargs: Any) -> GenerateContentResponse:
            """Mock regular content generation."""
            return create_mock_response()  # pragma: no cover

    @dataclass
    class _BatchesNamespace:
        parent: MockGoogleBatchClient

        async def create(self, **kwargs: Any) -> MockBatchResponse:
            self.parent.batch_create_calls.append(kwargs)
            if self.parent.batches_list:  # pragma: no cover
                return self.parent.batches_list[0]
            return MockBatchResponse()

        async def get(self, name: str) -> MockBatchResponse:
            for batch in self.parent.batches_list:
                if batch.name == name or name in batch.name:  # pragma: no branch
                    return batch
            return MockBatchResponse(name=name)  # pragma: no cover

        async def cancel(self, name: str) -> MockBatchResponse:
            return MockBatchResponse(
                name=name,
                state='JOB_STATE_CANCELLING',
            )


def create_mock_response(text: str = 'Hello there!') -> GenerateContentResponse:
    """Create a proper response using real SDK types."""
    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(parts=[Part(text=text)]),
            )
        ],
        model_version='gemini-2.0-flash',
    )


# --- GoogleBatch Dataclass Tests ---


class TestGoogleBatch:
    """Tests for GoogleBatch dataclass."""

    def test_google_batch_creation(self):
        """Test GoogleBatch extends Batch with Google-specific fields."""
        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
            display_name='Test Batch',
            state='JOB_STATE_SUCCEEDED',
        )

        assert batch.id == 'batch_123'
        assert batch.name is not None
        assert batch.display_name == 'Test Batch'
        assert batch.state == 'JOB_STATE_SUCCEEDED'
        assert batch.is_complete is True

    def test_google_batch_status_mapping(self):
        """Test GoogleBatch status mapping for terminal states."""
        batch_id = 'batch_123'
        created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Completed batch
        completed = GoogleBatch(id=batch_id, created_at=created_at, status=BatchStatus.COMPLETED)
        assert completed.is_complete is True
        assert completed.is_successful is True

        # Failed batch
        failed = GoogleBatch(id=batch_id, created_at=created_at, status=BatchStatus.FAILED)
        assert failed.is_complete is True
        assert failed.is_successful is False

        # Cancelled batch
        cancelled = GoogleBatch(id=batch_id, created_at=created_at, status=BatchStatus.CANCELLED)
        assert cancelled.is_complete is True
        assert cancelled.is_successful is False

        # Expired batch
        expired = GoogleBatch(id=batch_id, created_at=created_at, status=BatchStatus.EXPIRED)
        assert expired.is_complete is True
        assert expired.is_successful is False


# --- GoogleModel Batch Method Tests ---


class TestGoogleModelBatch:
    """Tests for GoogleModel batch methods."""

    async def test_batch_create_builds_requests_array(self, allow_model_requests: None):
        """Test that batch_create builds proper requests array."""
        mock_client = MockGoogleBatchClient()

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
        assert 'src' in batch_call
        assert 'model' in batch_call
        assert batch_call['model'] == 'gemini-2.0-flash'
        assert len(batch_call['src']) == 2

        # Verify each request has proper InlinedRequest structure (contents, config)
        # Google's format does NOT include custom_id or 'request' wrapper
        for req in batch_call['src']:
            assert 'contents' in req
            assert 'config' in req
            # Verify no custom_id in request (tracked separately)
            assert 'custom_id' not in req
            assert 'request' not in req

        # Verify returned batch tracks custom_ids separately
        assert 'batch_abc123' in batch.id
        assert batch.status == BatchStatus.IN_PROGRESS
        assert batch.custom_ids == ['req-1', 'req-2']

    async def test_batch_create_minimum_requests(self, allow_model_requests: None):
        """Test that batch_create requires at least 2 requests."""
        mock_client = MockGoogleBatchClient()
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
        ]

        with pytest.raises(ValueError, match='at least 2 requests'):
            await model.batch_create(requests)

    async def test_batch_status(self, allow_model_requests: None):
        """Test batch_status retrieves updated batch info."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_SUCCEEDED',
                end_time='2024-01-01T12:00:00Z',
                request_counts=MockRequestCounts(total=5, succeeded=5),
            )
        ]

        model = mock_client.create_model()

        initial_batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        updated = await model.batch_status(initial_batch)

        assert updated.status == BatchStatus.COMPLETED
        assert updated.request_count == 5
        assert updated.completed_count == 5

    async def test_batch_results_success(self, allow_model_requests: None):
        """Test batch_results parses results correctly."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_SUCCEEDED',
                responses=[
                    # Google returns responses in order, no custom_id in response
                    MockBatchResultItem(response=create_mock_response()),
                ],
            )
        ]

        model = mock_client.create_model()

        # GoogleBatch stores custom_ids for position-based matching
        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
            custom_ids=['req-1'],  # custom_ids tracked separately
        )

        results = await model.batch_results(batch)

        assert len(results) == 1
        # custom_id is matched by position from batch.custom_ids
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True
        assert results[0].response is not None
        assert len(results[0].response.parts) == 1
        assert isinstance(results[0].response.parts[0], TextPart)
        assert results[0].response.parts[0].content == 'Hello there!'

    async def test_batch_results_with_errors(self, allow_model_requests: None):
        """Test batch_results handles error responses."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_SUCCEEDED',
                responses=[
                    # Google responses are ordered, no custom_id field
                    MockBatchResultItem(response=create_mock_response()),
                    MockBatchResultItem(error=MockError(code='rate_limit', message='Rate limit exceeded')),
                ],
            )
        ]

        model = mock_client.create_model()

        # GoogleBatch tracks custom_ids for position-based matching
        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
            custom_ids=['req-1', 'req-2'],  # matched by position
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
        assert results[1].error.code == 'rate_limit'
        assert results[1].error.message == 'Rate limit exceeded'

    async def test_batch_results_incomplete_raises(self, allow_model_requests: None):
        """Test batch_results raises if batch not complete."""
        mock_client = MockGoogleBatchClient()
        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        with pytest.raises(ValueError, match='not complete'):
            await model.batch_results(batch)

    async def test_batch_cancel(self, allow_model_requests: None):
        """Test batch_cancel sends cancel request."""
        mock_client = MockGoogleBatchClient()
        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        cancelled = await model.batch_cancel(batch)

        assert cancelled.status == BatchStatus.CANCELLING


# --- Status Mapping Tests ---


class TestGoogleBatchStatusMapping:
    """Tests for Google batch status mapping logic."""

    async def test_status_pending(self, allow_model_requests: None):
        """Test JOB_STATE_PENDING maps correctly."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_PENDING',
            )
        ]

        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.PENDING,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.PENDING

    async def test_status_running(self, allow_model_requests: None):
        """Test JOB_STATE_RUNNING maps correctly."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_RUNNING',
            )
        ]

        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.PENDING,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.IN_PROGRESS

    async def test_status_succeeded(self, allow_model_requests: None):
        """Test JOB_STATE_SUCCEEDED maps correctly."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_SUCCEEDED',
                end_time='2024-01-01T12:00:00Z',
            )
        ]

        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.COMPLETED

    async def test_status_failed(self, allow_model_requests: None):
        """Test JOB_STATE_FAILED maps correctly."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_FAILED',
                end_time='2024-01-01T12:00:00Z',
            )
        ]

        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.FAILED

    async def test_status_cancelled(self, allow_model_requests: None):
        """Test JOB_STATE_CANCELLED maps correctly."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_CANCELLED',
            )
        ]

        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.CANCELLED

    async def test_status_cancelling(self, allow_model_requests: None):
        """Test JOB_STATE_CANCELLING maps correctly."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_CANCELLING',
            )
        ]

        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        updated = await model.batch_status(batch)
        assert updated.status == BatchStatus.CANCELLING


# --- API Error Handling Tests ---


class TestGoogleBatchAPIErrorHandling:
    """Tests for batch API error handling."""

    @dataclass
    class MockGoogleWithAPIError:
        """Mock Google client that raises API errors."""

        error_type: str = 'status'  # 'status' or 'connection'
        error_on: str = 'batches.create'
        _api_client: MockAPIClient = field(default_factory=MockAPIClient)

        def create_model(self) -> GoogleModel:
            """Create a GoogleModel using this mock as the client."""
            return GoogleModel(
                'gemini-2.0-flash',
                provider=GoogleProvider(client=cast(Client, self)),
            )

        @property
        def aio(self) -> Any:
            return self._AioNamespace(self)

        def _raise_error(self) -> None:
            from google.genai import errors

            raise errors.APIError(
                code=429,
                response_json={'error': {'message': 'Rate limit exceeded'}},
            )

        @dataclass
        class _AioNamespace:
            parent: TestGoogleBatchAPIErrorHandling.MockGoogleWithAPIError

            @property
            def batches(self) -> Any:
                return TestGoogleBatchAPIErrorHandling.MockGoogleWithAPIError._BatchesNamespace(self.parent)

            @property
            def models(self) -> Any:
                return TestGoogleBatchAPIErrorHandling.MockGoogleWithAPIError._ModelsNamespace(self.parent)  # pragma: no cover

        @dataclass
        class _ModelsNamespace:
            parent: TestGoogleBatchAPIErrorHandling.MockGoogleWithAPIError

            async def generate_content(self, **kwargs: Any) -> GenerateContentResponse:
                return create_mock_response()  # pragma: no cover

        @dataclass
        class _BatchesNamespace:
            parent: TestGoogleBatchAPIErrorHandling.MockGoogleWithAPIError

            async def create(self, **kwargs: Any) -> MockBatchResponse:
                if self.parent.error_on == 'batches.create':
                    self.parent._raise_error()
                return MockBatchResponse()  # pragma: no cover

            async def get(self, name: str) -> MockBatchResponse:
                if self.parent.error_on == 'batches.get':
                    self.parent._raise_error()
                return MockBatchResponse(name=name)  # pragma: no cover

            async def cancel(self, name: str) -> MockBatchResponse:
                if self.parent.error_on == 'batches.cancel':
                    self.parent._raise_error()
                return MockBatchResponse(name=name)  # pragma: no cover

    async def test_batch_create_api_error(self, allow_model_requests: None):
        """Test batch_create raises ModelHTTPError on API error."""
        mock_client = self.MockGoogleWithAPIError(error_on='batches.create')
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

    async def test_batch_status_api_error(self, allow_model_requests: None):
        """Test batch_status raises ModelHTTPError on API error."""
        mock_client = self.MockGoogleWithAPIError(error_on='batches.get')
        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_status(batch)

        assert exc_info.value.status_code == 429

    async def test_batch_cancel_api_error(self, allow_model_requests: None):
        """Test batch_cancel raises ModelHTTPError on API error."""
        mock_client = self.MockGoogleWithAPIError(error_on='batches.cancel')
        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_cancel(batch)

        assert exc_info.value.status_code == 429

    async def test_batch_results_api_error(self, allow_model_requests: None):
        """Test batch_results raises ModelHTTPError on API error."""
        mock_client = self.MockGoogleWithAPIError(error_on='batches.get')
        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_results(batch)

        assert exc_info.value.status_code == 429


# --- Empty Results Tests ---


class TestGoogleBatchEmptyResults:
    """Tests for handling empty batch results."""

    async def test_batch_results_empty_responses(self, allow_model_requests: None):
        """Test batch_results handles empty responses."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_SUCCEEDED',
                responses=[],
            )
        ]

        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        results = await model.batch_results(batch)

        assert results == []

    async def test_batch_results_no_responses_attr(self, allow_model_requests: None):
        """Test batch_results handles missing responses attribute."""
        mock_client = MockGoogleBatchClient()
        mock_client.batches_list = [
            MockBatchResponse(
                name='projects/test/locations/us-central1/batches/batch_123',
                state='JOB_STATE_SUCCEEDED',
                responses=None,
            )
        ]

        model = mock_client.create_model()

        batch = GoogleBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name='projects/test/locations/us-central1/batches/batch_123',
        )

        results = await model.batch_results(batch)

        assert results == []
