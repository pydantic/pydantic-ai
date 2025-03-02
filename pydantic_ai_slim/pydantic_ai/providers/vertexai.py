from __future__ import annotations as _annotations

import functools

from pydantic_ai._utils import run_in_executor

run_in_executor

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal

import anyio.to_thread
import httpx

from pydantic_ai.exceptions import UserError

from ..models import cached_async_http_client
from . import Provider

try:
    import google.auth
    from google.auth.credentials import Credentials as BaseCredentials
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-auth` to use the VertexAI model, '
        "you can use the `vertexai` optional group — `pip install 'pydantic-ai-slim[vertexai]'`"
    ) from _import_error


class VertexAIProvider(Provider[httpx.AsyncClient]):
    """Provider for Vertex AI API."""

    @property
    def name(self) -> str:
        return 'vertexai'

    @property
    def base_url(self) -> str:
        return 'https://{region}-aiplatform.googleapis.com/v1'

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    def __init__(
        self,
        service_account_file: str,
        project_id: str,
        region: str,
        model_publisher: str = 'google',
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = http_client or cached_async_http_client()
        self.service_account_file = service_account_file
        self.project_id = project_id
        self.region = region
        self.model_publisher = model_publisher


class VertexAIAuth(httpx.Auth):
    """Auth class for Vertex AI API."""

    def __init__(
        self,
        service_account_file: Path | str | None = None,
        project_id: str | None = None,
        region: VertexAiRegion = 'us-central1',
    ) -> None:
        self.service_account_file = service_account_file
        self.project_id = project_id
        self.region = region

        self.credentials = None

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if self.credentials is None:
            self.credentials = await self._get_credentials()
        yield request

    async def _get_credentials(self) -> BaseCredentials | ServiceAccountCredentials:
        if self.service_account_file is not None:
            creds: BaseCredentials | ServiceAccountCredentials = await _creds_from_file(self.service_account_file)
        else:
            creds, creds_project_id = await _async_google_auth()
            creds_source = '`google.auth.default()`'

        if self.project_id is None:
            if creds_project_id is None:
                raise UserError(f'No project_id provided and none found in {creds_source}')
            project_id = creds_project_id
        else:
            project_id = self.project_id

        return creds


async def _async_google_auth() -> tuple[BaseCredentials, str | None]:
    return await anyio.to_thread.run_sync(google.auth.default, ['https://www.googleapis.com/auth/cloud-platform'])  # type: ignore


async def _creds_from_file(service_account_file: str | Path) -> ServiceAccountCredentials:
    service_account_credentials_from_file = functools.partial(
        ServiceAccountCredentials.from_service_account_file,  # type: ignore[reportUnknownMemberType]
        scopes=['https://www.googleapis.com/auth/cloud-platform'],
    )
    return await anyio.to_thread.run_sync(service_account_credentials_from_file, str(service_account_file))


VertexAiRegion = Literal[
    'us-central1',
    'us-east1',
    'us-east4',
    'us-south1',
    'us-west1',
    'us-west2',
    'us-west3',
    'us-west4',
    'us-east5',
    'europe-central2',
    'europe-north1',
    'europe-southwest1',
    'europe-west1',
    'europe-west2',
    'europe-west3',
    'europe-west4',
    'europe-west6',
    'europe-west8',
    'europe-west9',
    'europe-west12',
    'africa-south1',
    'asia-east1',
    'asia-east2',
    'asia-northeast1',
    'asia-northeast2',
    'asia-northeast3',
    'asia-south1',
    'asia-southeast1',
    'asia-southeast2',
    'australia-southeast1',
    'australia-southeast2',
    'me-central1',
    'me-central2',
    'me-west1',
    'northamerica-northeast1',
    'northamerica-northeast2',
    'southamerica-east1',
    'southamerica-west1',
]
"""Regions available for Vertex AI.

More details [here](https://cloud.google.com/vertex-ai/docs/reference/rest#rest_endpoints).
"""
