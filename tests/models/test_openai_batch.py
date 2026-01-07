"""Tests for OpenAI batch processing functionality."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from pydantic_ai import (
    Batch,
    BatchError,
    BatchResult,
    BatchStatus,
    ModelMessage,
    ModelRequest,
    ModelRequestParameters,
    ModelResponse,
    TextPart,
)
from pydantic_ai.direct import (
    batch_cancel,
    batch_cancel_sync,
    batch_create,
    batch_create_sync,
    batch_results,
    batch_status,
    batch_status_sync,
)
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types import Batch as OpenAIBatchResponse
    from openai.types.batch_request_counts import BatchRequestCounts

    from pydantic_ai.models.openai import OpenAIBatch, OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


# --- Unit Tests for Batch Abstractions ---


class TestBatchStatus:
    """Tests for BatchStatus enum."""

    def test_batch_status_values(self):
        """Verify all expected status values exist."""
        assert BatchStatus.PENDING == 'pending'
        assert BatchStatus.VALIDATING == 'validating'
        assert BatchStatus.IN_PROGRESS == 'in_progress'
        assert BatchStatus.FINALIZING == 'finalizing'
        assert BatchStatus.COMPLETED == 'completed'
        assert BatchStatus.FAILED == 'failed'
        assert BatchStatus.EXPIRED == 'expired'
        assert BatchStatus.CANCELLING == 'cancelling'
        assert BatchStatus.CANCELLED == 'cancelled'

    def test_batch_status_is_string_enum(self):
        """BatchStatus should be usable as strings."""
        assert BatchStatus.COMPLETED.value == 'completed'
        assert BatchStatus.PENDING.value == 'pending'
        # Can compare with string values
        assert BatchStatus.COMPLETED == 'completed'


class TestBatch:
    """Tests for Batch dataclass."""

    def test_batch_creation(self):
        """Test basic batch creation."""
        batch = Batch(
            id='batch_123',
            status=BatchStatus.PENDING,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert batch.id == 'batch_123'
        assert batch.status == BatchStatus.PENDING
        assert batch.request_count == 0
        assert batch.completed_count == 0
        assert batch.failed_count == 0
        assert batch.completed_at is None

    def test_batch_is_complete_property(self):
        """Test is_complete property for various states."""
        base_id = 'batch_123'
        base_created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Not complete states
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.PENDING).is_complete is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.VALIDATING).is_complete is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.IN_PROGRESS).is_complete is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.FINALIZING).is_complete is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.CANCELLING).is_complete is False

        # Complete states
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.COMPLETED).is_complete is True
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.FAILED).is_complete is True
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.EXPIRED).is_complete is True
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.CANCELLED).is_complete is True

    def test_batch_is_successful_property(self):
        """Test is_successful property."""
        base_id = 'batch_123'
        base_created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Only COMPLETED is successful
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.COMPLETED).is_successful is True
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.FAILED).is_successful is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.EXPIRED).is_successful is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.CANCELLED).is_successful is False


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_successful_result(self):
        """Test successful batch result."""
        response = ModelResponse(
            parts=[TextPart(content='Hello')],
            model_name='gpt-4o-mini',
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )
        result = BatchResult(custom_id='req-1', response=response)

        assert result.custom_id == 'req-1'
        assert result.is_successful is True
        assert result.response == response
        assert result.error is None

    def test_failed_result(self):
        """Test failed batch result."""
        error = BatchError(code='rate_limit', message='Rate limit exceeded')
        result = BatchResult(custom_id='req-2', error=error)

        assert result.custom_id == 'req-2'
        assert result.is_successful is False
        assert result.response is None
        assert result.error == error


class TestBatchError:
    """Tests for BatchError dataclass."""

    def test_batch_error_creation(self):
        """Test batch error creation."""
        error = BatchError(code='invalid_request', message='Invalid JSON in request body')

        assert error.code == 'invalid_request'
        assert error.message == 'Invalid JSON in request body'


# --- OpenAI Batch Implementation Tests ---


@dataclass
class MockOpenAIBatch:
    """Mock for OpenAI batch-related operations."""

    batches_list: list[OpenAIBatchResponse] = field(default_factory=list)
    files_list: list[dict[str, Any]] = field(default_factory=list)
    batch_create_calls: list[dict[str, Any]] = field(default_factory=list)
    file_create_calls: list[dict[str, Any]] = field(default_factory=list)
    file_content_responses: dict[str, str] = field(default_factory=dict)
    base_url: str = 'https://api.openai.com/v1'

    @classmethod
    def create_mock(cls) -> AsyncOpenAI:
        """Create a mock OpenAI client with proper type casting for Pyright."""
        return cast(AsyncOpenAI, cls())

    @property
    def batches(self) -> Any:
        return self._BatchesNamespace(self)

    @property
    def files(self) -> Any:
        return self._FilesNamespace(self)

    @dataclass
    class _BatchesNamespace:
        parent: MockOpenAIBatch

        async def create(self, **kwargs: Any) -> OpenAIBatchResponse:
            self.parent.batch_create_calls.append(kwargs)
            if self.parent.batches_list:
                return self.parent.batches_list[0]
            # Return a default batch response
            return OpenAIBatchResponse(
                id='batch_abc123',
                completion_window='24h',
                created_at=1704067200,  # 2024-01-01
                endpoint='/v1/chat/completions',
                input_file_id=kwargs.get('input_file_id', 'file_xyz'),
                object='batch',
                status='validating',
            )

        async def retrieve(self, batch_id: str) -> OpenAIBatchResponse:
            for batch in self.parent.batches_list:
                if batch.id == batch_id:
                    return batch
            # Return a default batch response
            return OpenAIBatchResponse(
                id=batch_id,
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_xyz',
                object='batch',
                status='completed',
                output_file_id='file_output',
            )

        async def cancel(self, batch_id: str) -> OpenAIBatchResponse:
            return OpenAIBatchResponse(
                id=batch_id,
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_xyz',
                object='batch',
                status='cancelling',
            )

    @dataclass
    class _FilesNamespace:
        parent: MockOpenAIBatch

        async def create(self, **kwargs: Any) -> MagicMock:
            self.parent.file_create_calls.append(kwargs)
            mock_file = MagicMock()
            mock_file.id = 'file_uploaded_123'
            return mock_file

        async def content(self, file_id: str) -> MagicMock:
            mock_response = MagicMock()
            content = self.parent.file_content_responses.get(file_id, '')
            mock_response.text = content
            mock_response.read.return_value = content.encode('utf-8')
            return mock_response


class TestOpenAIBatch:
    """Tests for OpenAIBatch dataclass."""

    def test_openai_batch_creation(self):
        """Test OpenAIBatch extends Batch with OpenAI-specific fields."""
        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
            error_file_id=None,
            endpoint='/v1/chat/completions',
            completion_window='24h',
            metadata={'key': 'value'},
        )

        assert batch.id == 'batch_123'
        assert batch.input_file_id == 'file_input'
        assert batch.output_file_id == 'file_output'
        assert batch.endpoint == '/v1/chat/completions'
        assert batch.is_complete is True


class TestOpenAIChatModelBatch:
    """Tests for OpenAIChatModel batch methods."""

    async def test_batch_create_builds_jsonl(self, allow_model_requests: None):
        """Test that batch_create builds proper JSONL and uploads."""
        mock_client = MockOpenAIBatch()

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        batch = await model.batch_create(requests)

        # Verify file was created
        assert len(mock_client.file_create_calls) == 1
        file_call = mock_client.file_create_calls[0]
        assert file_call['purpose'] == 'batch'

        # Verify batch was created with correct params
        assert len(mock_client.batch_create_calls) == 1
        batch_call = mock_client.batch_create_calls[0]
        assert batch_call['endpoint'] == '/v1/chat/completions'
        assert batch_call['completion_window'] == '24h'
        assert batch_call['input_file_id'] == 'file_uploaded_123'

        # Verify returned batch
        assert batch.id == 'batch_abc123'
        assert batch.status == BatchStatus.VALIDATING

    async def test_batch_create_minimum_requests(self, allow_model_requests: None):
        """Test that batch_create requires at least 2 requests."""
        mock_client = MockOpenAIBatch()
        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
        ]

        with pytest.raises(ValueError, match='at least 2 requests'):
            await model.batch_create(requests)

    async def test_batch_status(self, allow_model_requests: None):
        """Test batch_status retrieves updated batch info."""
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_123',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
                output_file_id='file_output',
                request_counts=BatchRequestCounts(completed=5, failed=0, total=5),
            )
        ]

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        initial_batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        updated = await model.batch_status(initial_batch)

        assert updated.status == BatchStatus.COMPLETED
        assert updated.request_count == 5
        assert updated.completed_count == 5
        assert updated.failed_count == 0

    async def test_batch_results_success(self, allow_model_requests: None):
        """Test batch_results parses output file correctly."""
        # Create JSONL output content
        output_line = {
            'id': 'resp_123',
            'custom_id': 'req-1',
            'response': {
                'status_code': 200,
                'body': {
                    'id': 'chatcmpl-123',
                    'object': 'chat.completion',
                    'created': 1704067200,
                    'model': 'gpt-4o-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'Hello there!'},
                            'finish_reason': 'stop',
                        }
                    ],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                },
            },
        }
        output_content = json.dumps(output_line)

        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = output_content

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = await model.batch_results(batch)

        assert len(results) == 1
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True
        assert results[0].response is not None
        assert len(results[0].response.parts) == 1
        assert isinstance(results[0].response.parts[0], TextPart)
        assert results[0].response.parts[0].content == 'Hello there!'

    async def test_batch_results_incomplete_raises(self, allow_model_requests: None):
        """Test batch_results raises if batch not complete."""
        mock_client = MockOpenAIBatch()
        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        with pytest.raises(ValueError, match='not complete'):
            await model.batch_results(batch)

    async def test_batch_cancel(self, allow_model_requests: None):
        """Test batch_cancel sends cancel request."""
        mock_client = MockOpenAIBatch()
        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        cancelled = await model.batch_cancel(batch)

        assert cancelled.status == BatchStatus.CANCELLING


# --- Direct API Batch Function Tests ---


class TestDirectAPIBatchFunctions:
    """Tests for Direct API batch functions."""

    async def test_batch_create_direct_api(self, allow_model_requests: None):
        """Test batch_create Direct API function."""
        mock_client = MockOpenAIBatch()
        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        batch = await batch_create(model, requests)

        assert batch.id == 'batch_abc123'
        assert batch.status == BatchStatus.VALIDATING

    async def test_batch_create_direct_api_simple_tuples(self, allow_model_requests: None):
        """Test batch_create Direct API with simple 2-tuples (no explicit params)."""
        mock_client = MockOpenAIBatch()
        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        # Use simple 2-tuples - parameters should default
        requests: list[tuple[str, list[ModelMessage]]] = [
            ('req-1', [ModelRequest.user_text_prompt('Hello')]),
            ('req-2', [ModelRequest.user_text_prompt('World')]),
        ]

        batch = await batch_create(model, requests)

        assert batch.id == 'batch_abc123'
        assert batch.status == BatchStatus.VALIDATING

    async def test_batch_status_direct_api(self, allow_model_requests: None):
        """Test batch_status Direct API function."""
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_123',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
                output_file_id='file_output',
            )
        ]

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        initial_batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        updated = await batch_status(model, initial_batch)

        assert updated.status == BatchStatus.COMPLETED

    async def test_batch_results_direct_api(self, allow_model_requests: None):
        """Test batch_results Direct API function."""
        output_line = {
            'id': 'resp_123',
            'custom_id': 'req-1',
            'response': {
                'status_code': 200,
                'body': {
                    'id': 'chatcmpl-123',
                    'object': 'chat.completion',
                    'created': 1704067200,
                    'model': 'gpt-4o-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'Test response'},
                            'finish_reason': 'stop',
                        }
                    ],
                },
            },
        }

        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = json.dumps(output_line)

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = await batch_results(model, batch)

        assert len(results) == 1
        assert results[0].is_successful is True

    async def test_batch_cancel_direct_api(self, allow_model_requests: None):
        """Test batch_cancel Direct API function."""
        mock_client = MockOpenAIBatch()

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        cancelled = await batch_cancel(model, batch)

        assert cancelled.status == BatchStatus.CANCELLING

    def test_batch_create_sync(self, allow_model_requests: None):
        """Test synchronous batch_create."""
        mock_client = MockOpenAIBatch()
        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        batch = batch_create_sync(model, requests)

        assert batch.id == 'batch_abc123'

    def test_batch_status_sync(self, allow_model_requests: None):
        """Test synchronous batch_status."""
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_123',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
            )
        ]
        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        initial_batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        updated = batch_status_sync(model, initial_batch)

        assert updated.status == BatchStatus.COMPLETED

    def test_batch_cancel_sync(self, allow_model_requests: None):
        """Test synchronous batch_cancel."""
        mock_client = MockOpenAIBatch()

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        cancelled = batch_cancel_sync(model, batch)

        assert cancelled.status == BatchStatus.CANCELLING

    def test_batch_results_sync(self, allow_model_requests: None):
        """Test synchronous batch_results."""
        from pydantic_ai.direct import batch_results_sync

        # Create response content
        response_line = {
            'id': 'resp_1',
            'custom_id': 'req-1',
            'response': {
                'status_code': 200,
                'body': {
                    'id': 'chatcmpl-1',
                    'object': 'chat.completion',
                    'created': 1704067200,
                    'model': 'gpt-4o-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'Hello!'},
                            'finish_reason': 'stop',
                        }
                    ],
                },
            },
        }

        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = json.dumps(response_line)

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = batch_results_sync(model, batch)

        assert len(results) == 1
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful


# --- Model Base Class Tests ---


class TestModelBatchMethodsNotImplemented:
    """Test that Model base class batch methods raise NotImplementedError."""

    async def test_batch_create_not_implemented(self, allow_model_requests: None):
        """Test batch_create raises NotImplementedError for unsupported models."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        with pytest.raises(NotImplementedError, match='does not support batch processing'):
            await model.batch_create(requests)

    async def test_batch_status_not_implemented(self, allow_model_requests: None):
        """Test batch_status raises NotImplementedError for unsupported models."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        batch = Batch(
            id='batch_123',
            status=BatchStatus.PENDING,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(NotImplementedError, match='does not support batch processing'):
            await model.batch_status(batch)

    async def test_batch_results_not_implemented(self, allow_model_requests: None):
        """Test batch_results raises NotImplementedError for unsupported models."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        batch = Batch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(NotImplementedError, match='does not support batch processing'):
            await model.batch_results(batch)

    async def test_batch_cancel_not_implemented(self, allow_model_requests: None):
        """Test batch_cancel raises NotImplementedError for unsupported models."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        batch = Batch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(NotImplementedError, match='does not support batch processing'):
            await model.batch_cancel(batch)


# --- Integration Tests ---


class TestBatchErrorHandling:
    """Tests for batch error handling scenarios."""

    async def test_batch_results_with_errors(self, allow_model_requests: None):
        """Test batch_results handles error responses."""
        # Success response
        success_line = {
            'id': 'resp_1',
            'custom_id': 'req-1',
            'response': {
                'status_code': 200,
                'body': {
                    'id': 'chatcmpl-1',
                    'object': 'chat.completion',
                    'created': 1704067200,
                    'model': 'gpt-4o-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'Success!'},
                            'finish_reason': 'stop',
                        }
                    ],
                },
            },
        }

        # Error response
        error_line = {
            'id': 'resp_2',
            'custom_id': 'req-2',
            'error': {
                'code': 'rate_limit_exceeded',
                'message': 'Rate limit exceeded',
            },
        }

        output_content = '\n'.join([json.dumps(success_line), json.dumps(error_line)])

        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = output_content

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = await model.batch_results(batch)

        assert len(results) == 2

        # First result should be successful
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True
        assert results[0].response is not None

        # Second result should have error
        assert results[1].custom_id == 'req-2'
        assert results[1].is_successful is False
        assert results[1].error is not None
        assert results[1].error.code == 'rate_limit_exceeded'
        assert results[1].error.message == 'Rate limit exceeded'

    async def test_batch_results_empty_output(self, allow_model_requests: None):
        """Test batch_results handles empty output file."""
        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = ''

        model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)))

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = await model.batch_results(batch)

        assert results == []
