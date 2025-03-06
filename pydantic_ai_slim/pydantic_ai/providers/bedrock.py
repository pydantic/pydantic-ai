from __future__ import annotations as _annotations

from pydantic_ai.providers import Provider

try:
    import boto3
    from botocore.client import BaseClient
except ImportError as _import_error:
    raise ImportError(
        'Please install `boto3` to use the Bedrock provider, '
        "you can use the `bedrock` optional group — `pip install 'pydantic-ai-slim[bedrock]'`"
    ) from _import_error


class BedrockProvider(Provider[BaseClient]):
    """Provider for AWS Bedrock."""

    @property
    def name(self) -> str:
        return 'bedrock'

    @property
    def base_url(self) -> str:
        return self._client.meta.endpoint_url

    @property
    def client(self) -> BaseClient:
        return self._client

    def __init__(self, bedrock_client: BaseClient | None = None):
        self._client = bedrock_client or boto3.client('bedrock-runtime')  # type: ignore[reportUnknownMemberType]
