from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from httpx import AsyncClient as AsyncHTTPClient

from .._utils import run_in_executor
from . import cached_async_http_client
from .gemini import GeminiModel, GeminiModelName

VERTEX_AI_URL_TEMPLATE = (
    'https://{region}-aiplatform.googleapis.com/v1'
    '/projects/{project_id}'
    '/locations/{region}'
    '/publishers/google'
    '/models/{model}'
    ':{function}'
)


class VertexAIModel(GeminiModel):
    def __init__(
        self,
        model_name: GeminiModelName,
        auth: str | Path | Credentials,
        *,
        region: str = 'us-central1',
        http_client: AsyncHTTPClient | None = None,
        url_template: str = VERTEX_AI_URL_TEMPLATE,
    ):
        self.model_name = model_name
        if isinstance(auth, Credentials):
            credentials = auth
        else:
            credentials = _creds_from_file(auth)
        self.auth = BearerTokenAuth(credentials)
        self.http_client = http_client or cached_async_http_client()
        # use replace, not format since we don't want to replace `{model}` or `{function}` yet
        project_id: Any = credentials.project_id
        assert isinstance(project_id, str), f'Expected project_id to be a string, got {project_id}'
        self.url_template = url_template.replace('{region}', region).replace('{project_id}', project_id)

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
