"""Custom interface to the `*-aiplatform.googleapis.com` API for Gemini models.

This model inherits from [`GeminiModel`][pydantic_ai.models.gemini], it relies on the VertexAI
[`generateContent` function endpoint](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/generateContent)
and `streamGenerateContent` function endpoint
having the same schemas as the equivalent [Gemini endpoints][pydantic_ai.models.gemini.GeminiModel].

There are three advantages of using this API over the `generativelanguage.googleapis.com` API which
[`GeminiModel`][pydantic_ai.models.gemini.GeminiModel] uses, and one big disadvantage.

Advantages:

1. The VertexAI API seems to be less flakey, less likely to occasionally return a 503 response.
2. You can
  [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput)
  with VertexAI.
3. If you're running PydanticAI inside GCP, service account authorization means you don't need an API key.

Disadvantage:

1. Service account authorization is much more painful to set up than an API key.

## Example Usage

```py title="vertex_example.py"
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.vertexai import VertexAIModel

model = VertexAIModel(
    'gemini-1.5-flash',
    auth=Path('path/to/service-account.json'),
)
agent = Agent(model)
result = agent.run_sync('Tell me a joke.')
print(result.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
```
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from httpx import AsyncClient as AsyncHTTPClient

from .._utils import run_in_executor
from . import cached_async_http_client
from .gemini import GeminiModel, GeminiModelName

try:
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials
except ImportError as e:
    raise ImportError(
        'Please install `google-auth` to use the VertexAI model, '
        "you can use the `vertexai` optional group — `pip install 'pydantic-ai[vertexai]'`"
    ) from e

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


class VertexAIModel(GeminiModel):
    """A model that uses Gemini via the `*-aiplatform.googleapis.com` API.

    This is implemented by inherits from [`GeminiModel`][pydantic_ai.models.gemini] but using different endpoints
    and authentication.
    """

    # noinspection PyMissingConstructor
    def __init__(
        self,
        model_name: GeminiModelName,
        auth: Path | Credentials,
        *,
        region: VertexAiRegion = 'us-central1',
        model_publisher: Literal['google'] = 'google',
        http_client: AsyncHTTPClient | None = None,
        url_template: str = VERTEX_AI_URL_TEMPLATE,
    ):
        """Initialize a Vertex AI Gemini model.

        Args:
            model_name: The name of the model to use. I couldn't find a list of supported google models,
            auth: Path to a service account file or a `google.auth.Credentials` object.
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
        self.model_name = model_name
        if isinstance(auth, Credentials):
            credentials = auth
        else:
            credentials = _creds_from_file(auth)
        self.auth = BearerTokenAuth(credentials)
        self.http_client = http_client or cached_async_http_client()

        project_id: Any = credentials.project_id
        assert isinstance(project_id, str), f'Expected project_id to be a string, got {project_id}'
        self.url = url_template.format(
            region=region, project_id=project_id, model_publisher=model_publisher, model=model_name
        )

    def name(self) -> str:
        return f'vertexai:{super().name()}'


# pyright: reportUnknownMemberType=false
def _creds_from_file(service_account_file: str | Path) -> Credentials:
    return Credentials.from_service_account_file(
        str(service_account_file), scopes=['https://www.googleapis.com/auth/cloud-platform']
    )


# default expiry is 3600 seconds
MAX_TOKEN_AGE = timedelta(seconds=3000)


@dataclass
class BearerTokenAuth:
    credentials: Credentials
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
