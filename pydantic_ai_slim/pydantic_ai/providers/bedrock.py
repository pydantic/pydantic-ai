from __future__ import annotations as _annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.exceptions import UserError
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
    from botocore.session import Session
    from botocore.tokens import FrozenAuthToken
except ImportError as _import_error:
    raise ImportError(
        'Please install the `boto3` package to use the Bedrock provider, '
        'you can use the `bedrock` optional group — `pip install "pydantic-ai-slim[bedrock]"`'
    ) from _import_error


@dataclass(kw_only=True)
class BedrockModelProfile(ModelProfile):
    """Profile for models used with BedrockModel.

    ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    bedrock_supports_tool_choice: bool = False
    bedrock_tool_result_format: Literal['text', 'json'] = 'text'
    bedrock_send_back_thinking_parts: bool = False
    bedrock_supports_prompt_caching: bool = False
    bedrock_supports_tool_caching: bool = False


def bedrock_amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model used via Bedrock."""
    profile = _without_builtin_tools(amazon_model_profile(model_name))
    if 'nova' in model_name:
        profile = BedrockModelProfile(
            bedrock_supports_tool_choice=True,
            bedrock_supports_prompt_caching=True,
        ).update(profile)

    if 'nova-2' in model_name:
        profile.supported_builtin_tools = frozenset({CodeExecutionTool})

    return profile


def bedrock_deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model used via Bedrock."""
    profile = deepseek_model_profile(model_name)
    if 'r1' in model_name:
        return BedrockModelProfile(bedrock_send_back_thinking_parts=True).update(profile)
    return profile  # pragma: no cover


# Known geo prefixes for cross-region inference profile IDs
BEDROCK_GEO_PREFIXES: tuple[str, ...] = ('us', 'eu', 'apac', 'jp', 'au', 'ca', 'global', 'us-gov')


def remove_bedrock_geo_prefix(model_name: str) -> str:
    """Remove inference geographic prefix from model ID if present.

    Bedrock supports cross-region inference using geographic prefixes like
    'us.', 'eu.', 'apac.', etc. This function strips those prefixes.

    Example:
        'us.amazon.titan-embed-text-v2:0' -> 'amazon.titan-embed-text-v2:0'
        'amazon.titan-embed-text-v2:0' -> 'amazon.titan-embed-text-v2:0'
    """
    for prefix in BEDROCK_GEO_PREFIXES:
        if model_name.startswith(f'{prefix}.'):
            return model_name.removeprefix(f'{prefix}.')
    return model_name


def _without_builtin_tools(profile: ModelProfile | None) -> ModelProfile:
    return replace(profile or BedrockModelProfile(), supported_builtin_tools=frozenset())


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
        provider_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
            'anthropic': lambda model_name: BedrockModelProfile(
                bedrock_supports_tool_choice=True,
                bedrock_send_back_thinking_parts=True,
                bedrock_supports_prompt_caching=True,
                bedrock_supports_tool_caching=True,
            ).update(_without_builtin_tools(anthropic_model_profile(model_name))),
            'mistral': lambda model_name: BedrockModelProfile(bedrock_tool_result_format='json').update(
                _without_builtin_tools(mistral_model_profile(model_name))
            ),
            'cohere': lambda model_name: _without_builtin_tools(cohere_model_profile(model_name)),
            'amazon': bedrock_amazon_model_profile,
            'meta': lambda model_name: _without_builtin_tools(meta_model_profile(model_name)),
            'deepseek': lambda model_name: _without_builtin_tools(bedrock_deepseek_model_profile(model_name)),
        }

        # Split the model name into parts
        parts = model_name.split('.', 2)

        # Handle regional prefixes
        if len(parts) > 2 and parts[0] in BEDROCK_GEO_PREFIXES:
            parts = parts[1:]

        # required format is provider.model-name-with-version
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

        if provider in provider_to_profile:
            return provider_to_profile[provider](model_name)

        return None

    @overload
    def __init__(self, *, bedrock_client: BaseClient) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        base_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        bedrock_client: BaseClient | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        base_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        api_key: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Args:
            bedrock_client: A boto3 client for Bedrock Runtime. If provided, other arguments are ignored.
            aws_access_key_id: The AWS access key ID. If not set, the `AWS_ACCESS_KEY_ID` environment variable will be used if available.
            aws_secret_access_key: The AWS secret access key. If not set, the `AWS_SECRET_ACCESS_KEY` environment variable will be used if available.
            aws_session_token: The AWS session token. If not set, the `AWS_SESSION_TOKEN` environment variable will be used if available.
            api_key: The API key for Bedrock client. Can be used instead of `aws_access_key_id`, `aws_secret_access_key`, and `aws_session_token`. If not set, the `AWS_BEARER_TOKEN_BEDROCK` environment variable will be used if available.
            base_url: The base URL for the Bedrock client.
            region_name: The AWS region name. If not set, the `AWS_DEFAULT_REGION` environment variable will be used if available.
            profile_name: The AWS profile name.
            aws_read_timeout: The read timeout for Bedrock client.
            aws_connect_timeout: The connect timeout for Bedrock client.
        """
        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            read_timeout = aws_read_timeout or float(os.getenv('AWS_READ_TIMEOUT', 300))
            connect_timeout = aws_connect_timeout or float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
            config: dict[str, Any] = {
                'read_timeout': read_timeout,
                'connect_timeout': connect_timeout,
            }
            api_key = api_key or os.getenv('AWS_BEARER_TOKEN_BEDROCK')
            try:
                if api_key is not None:
                    session = boto3.Session(
                        botocore_session=_BearerTokenSession(api_key),
                        region_name=region_name,
                        profile_name=profile_name,
                    )
                    config['signature_version'] = 'bearer'
                else:  # pragma: lax no cover
                    session = boto3.Session(
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        aws_session_token=aws_session_token,
                        region_name=region_name,
                        profile_name=profile_name,
                    )
                self._client = session.client(  # type: ignore[reportUnknownMemberType]
                    'bedrock-runtime',
                    config=Config(**config),
                    endpoint_url=base_url,
                )
            except NoRegionError as exc:  # pragma: no cover
                raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') from exc


class _BearerTokenSession(Session):
    def __init__(self, token: str):
        super().__init__()
        self.token = token

    def get_auth_token(self, **_kwargs: Any) -> FrozenAuthToken:
        return FrozenAuthToken(self.token)

    def get_credentials(self) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        return None
