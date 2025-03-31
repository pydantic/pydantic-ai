from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider

try:
    import boto3
    from botocore.client import BaseClient
    from botocore.config import Config
    from botocore.exceptions import NoRegionError
except ImportError as _import_error:
    raise ImportError(
        'Please install the `boto3` package to use the Bedrock provider, '
        'you can use the `bedrock` optional group — `pip install "pydantic-ai-slim[bedrock]"`'
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

    @overload
    def __init__(self, *, bedrock_client: BaseClient) -> None: ...

    @overload
    def __init__(
        self,
        *,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        bedrock_client: BaseClient | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Args:
            bedrock_client: A boto3 client for Bedrock Runtime. If provided, other arguments are ignored.
            region_name: The AWS region name.
            aws_access_key_id: The AWS access key ID.
            aws_secret_access_key: The AWS secret access key.
            aws_session_token: The AWS session token.
            aws_read_timeout: The read timeout for Bedrock client.
            aws_connect_timeout: The connect timeout for Bedrock client.
        """
        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            try:
                read_timeout = aws_read_timeout or float(os.getenv('AWS_READ_TIMEOUT', 300))
                connect_timeout = aws_connect_timeout or float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
                self._client = boto3.client(  # type: ignore[reportUnknownMemberType]
                    'bedrock-runtime',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=region_name,
                    config=Config(read_timeout=read_timeout, connect_timeout=connect_timeout),
                )
            except NoRegionError as exc:  # pragma: no cover
                raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') from exc
