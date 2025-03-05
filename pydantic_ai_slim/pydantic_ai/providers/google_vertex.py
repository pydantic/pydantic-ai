from __future__ import annotations as _annotations

import functools
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
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
        'Please install `google-auth` to use the Google Vertex AI provider, '
        "you can use the `vertexai` optional group — `pip install 'pydantic-ai-slim[vertexai]'`"
    ) from _import_error


__all__ = ('GoogleVertexProvider',)

# default expiry is 3600 seconds
MAX_TOKEN_AGE = timedelta(seconds=3000)


class GoogleVertexProvider(Provider[httpx.AsyncClient]):
    """Provider for Vertex AI API."""

    @property
    def name(self) -> str:
        return 'google-vertex'

    @property
    def base_url(self) -> str:
        return (
            f'https://{self.region}-aiplatform.googleapis.com/v1'
            f'/projects/{self.project_id}'
            f'/locations/{self.region}'
            f'/publishers/{self.model_publisher}/models/'
        )

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    def __init__(
        self,
        service_account_file: Path | str | None = None,
        project_id: str | None = None,
        region: VertexAiRegion = 'us-central1',
        model_publisher: str = 'google',
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Vertex AI provider.

        Args:
            service_account_file: Path to a service account file.
                If not provided, the default environment credentials will be used.
            project_id: The project ID to use, if not provided it will be taken from the credentials.
            region: The region to make requests to.
            model_publisher: The model publisher to use, I couldn't find a good list of available publishers,
                and from trial and error it seems non-google models don't work with the `generateContent` and
                `streamGenerateContent` functions, hence only `google` is currently supported.
                Please create an issue or PR if you know how to use other publishers.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self._client = http_client or cached_async_http_client()
        self.service_account_file = service_account_file
        self.project_id = project_id
        self.region = region
        self.model_publisher = model_publisher

        self._client.auth = _VertexAIAuth(service_account_file, project_id, region)
        self._client.base_url = self.base_url


class _VertexAIAuth(httpx.Auth):
    """Auth class for Vertex AI API."""

    credentials: BaseCredentials | ServiceAccountCredentials | None

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
        self.token_created: datetime | None = None

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if self.credentials is None:
            self.credentials = await self._get_credentials()
        if self.credentials.token is None or self._token_expired():  # type: ignore[reportUnknownMemberType]
            await anyio.to_thread.run_sync(self._refresh_token)
            self.token_created = datetime.now()
        request.headers['Authorization'] = f'Bearer {self.credentials.token}'  # type: ignore[reportUnknownMemberType]

        # NOTE: This workaround is in place because we might get the project_id from the credentials.
        request.url = httpx.URL(str(request.url).replace('projects/None', f'projects/{self.project_id}'))
        yield request

    async def _get_credentials(self) -> BaseCredentials | ServiceAccountCredentials:
        if self.service_account_file is not None:
            creds = await _creds_from_file(self.service_account_file)
            assert creds.project_id is None or isinstance(creds.project_id, str)  # type: ignore[reportUnknownMemberType]
            creds_project_id: str | None = creds.project_id
            creds_source = 'service account file'
        else:
            creds, creds_project_id = await _async_google_auth()
            creds_source = '`google.auth.default()`'

        if self.project_id is None:
            if creds_project_id is None:
                raise UserError(f'No project_id provided and none found in {creds_source}')
            self.project_id = creds_project_id
        return creds

    def _token_expired(self) -> bool:
        if self.token_created is None:
            return True
        else:
            return (datetime.now() - self.token_created) > MAX_TOKEN_AGE

    def _refresh_token(self) -> str:  # pragma: no cover
        assert self.credentials is not None
        self.credentials.refresh(Request())  # type: ignore[reportUnknownMemberType]
        assert isinstance(self.credentials.token, str), f'Expected token to be a string, got {self.credentials.token}'  # type: ignore[reportUnknownMemberType]
        return self.credentials.token


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
