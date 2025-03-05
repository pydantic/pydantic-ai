from __future__ import annotations as _annotations

import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import deprecated

from .. import usage
from .._utils import run_in_executor
from ..exceptions import UserError
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from . import ModelRequestParameters, StreamedResponse, cached_async_http_client
from .gemini import GeminiModel, GeminiModelName

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

VERTEX_AI_URL_TEMPLATE = (
    'https://{region}-aiplatform.googleapis.com/v1'
    '/projects/{project_id}'
    '/locations/{region}'
    '/publishers/{model_publisher}'
    '/models/{model}'
    ':'
)
"""URL template for Vertex AI.

See
[`generateContent` docs](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/generateContent)
and
[`streamGenerateContent` docs](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/streamGenerateContent)
for more information.

The template is used thus:

* `region` is substituted with the `region` argument,
  see [available regions][pydantic_ai.models.vertexai.VertexAiRegion]
* `model_publisher` is substituted with the `model_publisher` argument
* `model` is substituted with the `model_name` argument
* `project_id` is substituted with the `project_id` from auth/credentials
* `function` (`generateContent` or `streamGenerateContent`) is added to the end of the URL
"""


@deprecated('Please use `GeminiModel(provider=GoogleVertexProvider(...))` instead.')
@dataclass(init=False)
class VertexAIModel(GeminiModel):
    """A model that uses Gemini via the `*-aiplatform.googleapis.com` VertexAI API."""

    service_account_file: Path | str | None
    project_id: str | None
    region: VertexAiRegion
    model_publisher: Literal['google']
    url_template: str

    _model_name: GeminiModelName = field(repr=False)
    _system: str | None = field(default='google-vertex', repr=False)

    # TODO __init__ can be removed once we drop 3.9 and we can set kw_only correctly on the dataclass
    def __init__(
        self,
        model_name: GeminiModelName,
        *,
        service_account_file: Path | str | None = None,
        project_id: str | None = None,
        region: VertexAiRegion = 'us-central1',
        model_publisher: Literal['google'] = 'google',
        http_client: AsyncHTTPClient | None = None,
        url_template: str = VERTEX_AI_URL_TEMPLATE,
    ):
        """Initialize a Vertex AI Gemini model.

        Args:
            model_name: The name of the model to use. I couldn't find a list of supported Google models, in VertexAI
                so for now this uses the same models as the [Gemini model][pydantic_ai.models.gemini.GeminiModel].
            service_account_file: Path to a service account file.
                If not provided, the default environment credentials will be used.
            project_id: The project ID to use, if not provided it will be taken from the credentials.
            region: The region to make requests to.
            model_publisher: The model publisher to use, I couldn't find a good list of available publishers,
                and from trial and error it seems non-google models don't work with the `generateContent` and
                `streamGenerateContent` functions, hence only `google` is currently supported.
                Please create an issue or PR if you know how to use other publishers.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
            url_template: URL template for Vertex AI, see
                [`VERTEX_AI_URL_TEMPLATE` docs][pydantic_ai.models.vertexai.VERTEX_AI_URL_TEMPLATE]
                for more information.
        """
        self._model_name = model_name
        self.service_account_file = service_account_file
        self.project_id = project_id
        self.region = region
        self.model_publisher = model_publisher
        self.client = http_client or cached_async_http_client()
        self.url_template = url_template

        self._auth = None
        self._url = None
        warnings.warn(
            'VertexAIModel is deprecated, please use `GeminiModel(provider=GoogleVertexProvider(...))` instead.',
            DeprecationWarning,
        )
        self._provider = None

    async def ainit(self) -> None:
        """Initialize the model, setting the URL and auth.

        This will raise an error if authentication fails.
        """
        if self._url is not None and self._auth is not None:
            return

        if self.service_account_file is not None:
            creds: BaseCredentials | ServiceAccountCredentials = _creds_from_file(self.service_account_file)
            assert creds.project_id is None or isinstance(creds.project_id, str)
            creds_project_id: str | None = creds.project_id
            creds_source = 'service account file'
        else:
            creds, creds_project_id = await _async_google_auth()
            creds_source = '`google.auth.default()`'

        if self.project_id is None:
            if creds_project_id is None:
                raise UserError(f'No project_id provided and none found in {creds_source}')
            project_id = creds_project_id
        else:
            project_id = self.project_id

        self._url = self.url_template.format(
            region=self.region,
            project_id=project_id,
            model_publisher=self.model_publisher,
            model=self._model_name,
        )
        self._auth = BearerTokenAuth(creds)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        await self.ainit()
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        await self.ainit()
        async with super().request_stream(messages, model_settings, model_request_parameters) as value:
            yield value

    @property
    def model_name(self) -> GeminiModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str | None:
        """The system / model provider."""
        return self._system


# pyright: reportUnknownMemberType=false
def _creds_from_file(service_account_file: str | Path) -> ServiceAccountCredentials:
    return ServiceAccountCredentials.from_service_account_file(
        str(service_account_file), scopes=['https://www.googleapis.com/auth/cloud-platform']
    )


# pyright: reportReturnType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
async def _async_google_auth() -> tuple[BaseCredentials, str | None]:
    return await run_in_executor(google.auth.default, scopes=['https://www.googleapis.com/auth/cloud-platform'])


# default expiry is 3600 seconds
MAX_TOKEN_AGE = timedelta(seconds=3000)


@dataclass
class BearerTokenAuth:
    """Authentication using a bearer token generated by google-auth."""

    credentials: BaseCredentials | ServiceAccountCredentials
    token_created: datetime | None = field(default=None, init=False)

    async def headers(self) -> dict[str, str]:
        if self.credentials.token is None or self._token_expired():
            await run_in_executor(self._refresh_token)
            self.token_created = datetime.now()
        return {'Authorization': f'Bearer {self.credentials.token}'}

    def _token_expired(self) -> bool:
        if self.token_created is None:
            return True
        else:
            return (datetime.now() - self.token_created) > MAX_TOKEN_AGE

    def _refresh_token(self) -> str:
        self.credentials.refresh(Request())
        assert isinstance(self.credentials.token, str), f'Expected token to be a string, got {self.credentials.token}'
        return self.credentials.token


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
