from __future__ import annotations as _annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Literal, overload

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
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


@dataclass
class BedrockModelProfile(ModelProfile):
    """Profile for models used with BedrockModel.

    ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    bedrock_supports_tool_choice: bool = True
    bedrock_tool_result_format: Literal['text', 'json'] = 'text'


_provider_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
    'anthropic': lambda model_name: BedrockModelProfile(bedrock_supports_tool_choice=False).update(
        anthropic_model_profile(model_name)
    ),
    'mistral': lambda model_name: BedrockModelProfile(bedrock_tool_result_format='json').update(
        mistral_model_profile(model_name)
    ),
    'cohere': cohere_model_profile,
    'amazon': amazon_model_profile,
    'meta': meta_model_profile,
    'deepseek': deepseek_model_profile,
}


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

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Get the model profile for a Bedrock model.

        This function parses the provider and model name from Bedrock model names and delegates to the appropriate
        provider's model profile function. For example:
        - "us.anthropic.claude-3-sonnet-20240229-v1:0" -> provider="anthropic", model_name="claude-3-sonnet-20240229"
        - "mistral.mistral-7b-instruct-v0:2" -> provider="mistral", model_name="mistral-7b-instruct"
        - "cohere.command-text-v14" -> provider="cohere", model_name="command-text"
        - "amazon.titan-text-express-v1" -> provider="amazon", model_name="titan-text-express"
        - "meta.llama3-8b-instruct-v1:0" -> provider="meta", model_name="llama3-8b-instruct"
        """
        # Split the model name into parts
        parts = model_name.split('.', 2)

        # Handle regional prefixes (e.g. "us.")
        if len(parts) > 2 and len(parts[0]) == 2:
            parts = parts[1:]

        if len(parts) < 2:
            return None

        provider = parts[0]
        model_name_with_version = parts[1]

        # Remove version suffix if it matches the format (e.g. "-v1:0" or "-v14")
        version_match = re.match(r'(.+)-v\d+(?::\d+)?$', model_name_with_version)
        if version_match:
            model_name = version_match.group(1)
        else:
            model_name = model_name_with_version

        if provider in _provider_to_profile:
            return _provider_to_profile[provider](model_name)

        return None

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
        profile_name: str | None = None,
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
        profile_name: str | None = None,
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
            profile_name: The AWS profile name.
            aws_read_timeout: The read timeout for Bedrock client.
            aws_connect_timeout: The connect timeout for Bedrock client.
        """
        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            try:
                read_timeout = aws_read_timeout or float(os.getenv('AWS_READ_TIMEOUT', 300))
                connect_timeout = aws_connect_timeout or float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=region_name,
                    profile_name=profile_name,
                )
                self._client = session.client(  # type: ignore[reportUnknownMemberType]
                    'bedrock-runtime',
                    config=Config(read_timeout=read_timeout, connect_timeout=connect_timeout),
                )
            except NoRegionError as exc:  # pragma: no cover
                raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') from exc
