from __future__ import annotations as _annotations

import os
from typing import Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.spark import spark_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the iFLYTEK Spark provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )

SparkModelName = Literal[
    '4.0Ultra',
    'generalv3.5',
    'max-32k',
    'generalv3',
    'pro-128k',
    'lite',
]


class SparkProvider(_OpenAICompatibleProvider):
    """Provider for the iFLYTEK Spark (讯飞星火) API."""

    @property
    def name(self) -> str:
        return 'spark'

    @property
    def base_url(self) -> str:
        # OpenAI-compatible HTTP endpoint, authenticated with a Bearer API Password.
        return 'https://spark-api-open.xf-yun.com/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = spark_model_profile(model_name)

        # As the Spark API is OpenAI-compatible, assume OpenAIJsonSchemaTransformer unless
        # json_schema_transformer is set explicitly. Spark supports JSON mode
        # (response_format={'type': 'json_object'}).
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            OpenAIModelProfile(supports_json_object_output=True),
        )

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('SPARK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `SPARK_API_KEY` environment variable or pass it via '
                '`SparkProvider(api_key=...)` to use the iFLYTEK Spark provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
