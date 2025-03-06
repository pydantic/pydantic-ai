from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, cast

from pydantic_ai.providers import Provider

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient

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
    def client(self) -> BedrockRuntimeClient:
        return cast('BedrockRuntimeClient', self._client)

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str | None = None,
        bedrock_client: BedrockRuntimeClient | None = None,
    ):
        if bedrock_client:
            self._client = bedrock_client
        else:
            self._client = boto3.client(  # type: ignore[reportUnknownMemberType]
                'bedrock-runtime',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
