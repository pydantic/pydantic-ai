from __future__ import annotations as _annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .. import ModelHTTPError
from ..messages import UserPromptPart
from ..models import KnownModelName, Model, check_allow_model_requests
from ..models.openai import OpenAIModel
from ..models.wrapper import WrapperModel

try:
    from openai import APIStatusError, AsyncOpenAI
except ImportError as _import_error:
    raise ImportError('Please install "pydantic-ai-slim[openai]"`') from _import_error


def create_chat_request(
    custom_id: str,
    prompt: str | UserPromptPart | list[UserPromptPart],
    model: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    system_prompt: str | None = None,
) -> BatchRequest:
    """Create a chat completion batch request with pydantic-ai style parameters.

    Args:
        custom_id: Unique identifier for this request
        prompt: User prompt (string or UserPromptPart)
        model: Model name (e.g., "gpt-4o-mini")
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: Optional system prompt

    Returns:
        BatchRequest: Configured batch request

    Example:
        ```python
        from pydantic_ai.batches.openai import create_chat_request

        requests = [
            create_chat_request("req-1", "What is 2+2?", "gpt-4o-mini", max_tokens=50),
            create_chat_request("req-2", "Write a haiku", "gpt-4o-mini", max_tokens=100),
        ]
        ```
    """
    # Build messages list
    messages: list[dict[str, Any]] = []

    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    # Handle different prompt types
    if isinstance(prompt, str):
        messages.append({'role': 'user', 'content': prompt})
    elif isinstance(prompt, UserPromptPart):
        # Convert UserPromptPart to message format (simplified)
        messages.append({'role': 'user', 'content': str(prompt)})
    else:
        # Handle list of UserPromptParts (prompt is list[UserPromptPart])
        content_parts: list[str] = []
        for part in prompt:
            content_parts.append(str(part))
        messages.append({'role': 'user', 'content': ' '.join(content_parts)})

    # Build request body
    body: dict[str, Any] = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
    }

    if temperature is not None:
        body['temperature'] = temperature

    return BatchRequest(custom_id=custom_id, body=body)


__all__ = (
    'BatchRequest',
    'BatchJob',
    'BatchResult',
    'OpenAIBatchModel',
    'create_chat_request',
)


@dataclass
class BatchRequest:
    """Single request for batch processing."""

    custom_id: str
    method: str = 'POST'
    url: str = '/v1/chat/completions'
    body: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class BatchJob:
    """Batch job information returned by OpenAI."""

    id: str
    object: str
    endpoint: str
    errors: dict[str, Any] | None
    input_file_id: str
    completion_window: str
    status: Literal[
        'validating',
        'failed',
        'in_progress',
        'finalizing',
        'completed',
        'expired',
        'cancelling',
        'cancelled',
    ]
    output_file_id: str | None
    error_file_id: str | None
    created_at: int
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: dict[str, int] | None = None
    metadata: dict[str, str] | None = None


@dataclass
class BatchResult:
    """Single result from a batch job."""

    id: str
    custom_id: str | None
    response: dict[str, Any] | None
    error: dict[str, Any] | None


@dataclass(init=False)
class OpenAIBatchModel(WrapperModel):
    """A wrapper that adds batch processing capabilities to OpenAI models.

    This model wraps any OpenAI model and adds batch processing methods while preserving
    all the original functionality. Provides 50% cost savings compared to synchronous
    API calls with a 24-hour processing window.

    Example:
        ```python
        import asyncio
        from pydantic_ai.batches.openai import OpenAIBatchModel, BatchRequest
        from pydantic_ai.messages import UserPrompt

        async def main():
            # Create directly from model name
            batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

            # Use as regular model
            messages = [UserPrompt("Hello")]
            response = await batch_model.request(messages)

            # Or use batch functionality
            requests = [BatchRequest(custom_id="1", body={"model": "gpt-4o-mini", "messages": []})]
            batch_id = await batch_model.batch_create_job(requests)
        ```
    """

    def __init__(self, wrapped: Model | KnownModelName):
        """Initialize OpenAI batch model.

        Args:
            wrapped: OpenAI model to wrap, or model name string like 'openai:gpt-4o'

        Raises:
            ValueError: If the wrapped model is not an OpenAI model
        """
        super().__init__(wrapped)

        # Verify this is an OpenAI model that has a client
        if not isinstance(self.wrapped, OpenAIModel):
            raise ValueError(
                f'OpenAIBatchModel requires an OpenAI model, got {type(self.wrapped).__name__}. '
                f"Use models from pydantic_ai.models.openai or model strings like 'openai:gpt-4o'."
            )

    @property
    def client(self) -> AsyncOpenAI:
        """Get the OpenAI client from the wrapped model."""
        return cast(OpenAIModel, self.wrapped).client

    async def batch_create_job(
        self,
        requests: list[BatchRequest],
        endpoint: str = '/v1/chat/completions',
        completion_window: str = '24h',
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Create a batch job with multiple requests.

        Args:
            requests: List of batch requests to process
            endpoint: OpenAI API endpoint (default: "/v1/chat/completions")
            completion_window: Processing window (default: "24h")
            metadata: Optional metadata for the batch

        Returns:
            batch_id: The ID of the created batch job

        Raises:
            ModelHTTPError: If the API request fails
        """
        check_allow_model_requests()

        # Convert requests to JSONL format
        jsonl_lines: list[str] = []
        for req in requests:
            jsonl_lines.append(
                json.dumps(
                    {
                        'custom_id': req.custom_id,
                        'method': req.method,
                        'url': req.url,
                        'body': req.body,
                    }
                )
            )

        jsonl_content = '\n'.join(jsonl_lines)

        try:
            # Upload file
            batch_file = await self.client.files.create(file=io.BytesIO(jsonl_content.encode('utf-8')), purpose='batch')

            # Create batch job
            batch_job = await self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint=cast(Any, endpoint),  # OpenAI SDK has strict Literal types
                completion_window=cast(Any, completion_window),
                metadata=metadata or {},
            )

            return batch_job.id

        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: no cover

    async def batch_retrieve_job(self, batch_id: str) -> BatchJob:
        """Retrieve the status and details of a batch job.

        Args:
            batch_id: The ID of the batch job

        Returns:
            BatchJob: Complete batch job information

        Raises:
            ModelHTTPError: If the API request fails
        """
        check_allow_model_requests()

        try:
            batch = await self.client.batches.retrieve(batch_id)

            return BatchJob(
                id=batch.id,
                object=batch.object,
                endpoint=batch.endpoint,
                errors=batch.errors.model_dump() if batch.errors else None,
                input_file_id=batch.input_file_id,
                completion_window=batch.completion_window,
                status=batch.status,
                output_file_id=batch.output_file_id,
                error_file_id=batch.error_file_id,
                created_at=batch.created_at,
                in_progress_at=batch.in_progress_at,
                expires_at=batch.expires_at,
                finalizing_at=batch.finalizing_at,
                completed_at=batch.completed_at,
                failed_at=batch.failed_at,
                expired_at=batch.expired_at,
                cancelling_at=batch.cancelling_at,
                cancelled_at=batch.cancelled_at,
                request_counts=(batch.request_counts.model_dump() if batch.request_counts else None),
                metadata=batch.metadata,
            )

        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: no cover

    async def batch_get_results(self, batch_id: str) -> list[BatchResult]:
        """Get the results of a completed batch job.

        Args:
            batch_id: The ID of the batch job

        Returns:
            list[BatchResult]: List of batch results

        Raises:
            ValueError: If batch is not completed or has no output file
            ModelHTTPError: If the API request fails
        """
        check_allow_model_requests()

        # First check if batch is completed
        batch_info = await self.batch_retrieve_job(batch_id)

        if batch_info.status != 'completed':
            raise ValueError(f'Batch {batch_id} is not completed. Status: {batch_info.status}')

        if batch_info.output_file_id is None:
            raise ValueError(f'Batch {batch_id} has no output file')

        try:
            # Download and parse results
            file_response = await self.client.files.content(batch_info.output_file_id)
            file_content = file_response.read()

            results: list[BatchResult] = []
            for line in file_content.decode('utf-8').strip().split('\n'):
                if line.strip():
                    result_data = json.loads(line)
                    results.append(
                        BatchResult(
                            id=result_data.get('id', ''),
                            custom_id=result_data.get('custom_id'),
                            response=result_data.get('response'),
                            error=result_data.get('error'),
                        )
                    )

            return results

        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: no cover

    async def batch_cancel_job(self, batch_id: str) -> BatchJob:
        """Cancel a batch job.

        Args:
            batch_id: The ID of the batch job to cancel

        Returns:
            BatchJob: Updated batch job information

        Raises:
            ModelHTTPError: If the API request fails
        """
        check_allow_model_requests()

        try:
            batch = await self.client.batches.cancel(batch_id)

            return BatchJob(
                id=batch.id,
                object=batch.object,
                endpoint=batch.endpoint,
                errors=batch.errors.model_dump() if batch.errors else None,
                input_file_id=batch.input_file_id,
                completion_window=batch.completion_window,
                status=batch.status,
                output_file_id=batch.output_file_id,
                error_file_id=batch.error_file_id,
                created_at=batch.created_at,
                in_progress_at=batch.in_progress_at,
                expires_at=batch.expires_at,
                finalizing_at=batch.finalizing_at,
                completed_at=batch.completed_at,
                failed_at=batch.failed_at,
                expired_at=batch.expired_at,
                cancelling_at=batch.cancelling_at,
                cancelled_at=batch.cancelled_at,
                request_counts=(batch.request_counts.model_dump() if batch.request_counts else None),
                metadata=batch.metadata,
            )

        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: no cover

    async def batch_list_jobs(self, limit: int = 20) -> list[BatchJob]:
        """List all batch jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            list[BatchJob]: List of batch jobs

        Raises:
            ModelHTTPError: If the API request fails
        """
        check_allow_model_requests()

        try:
            batches = await self.client.batches.list(limit=limit)

            jobs: list[BatchJob] = []
            for batch in batches.data:
                jobs.append(
                    BatchJob(
                        id=batch.id,
                        object=batch.object,
                        endpoint=batch.endpoint,
                        errors=batch.errors.model_dump() if batch.errors else None,
                        input_file_id=batch.input_file_id,
                        completion_window=batch.completion_window,
                        status=batch.status,
                        output_file_id=batch.output_file_id,
                        error_file_id=batch.error_file_id,
                        created_at=batch.created_at,
                        in_progress_at=batch.in_progress_at,
                        expires_at=batch.expires_at,
                        finalizing_at=batch.finalizing_at,
                        completed_at=batch.completed_at,
                        failed_at=batch.failed_at,
                        expired_at=batch.expired_at,
                        cancelling_at=batch.cancelling_at,
                        cancelled_at=batch.cancelled_at,
                        request_counts=(batch.request_counts.model_dump() if batch.request_counts else None),
                        metadata=batch.metadata,
                    )
                )

            return jobs

        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: no cover
