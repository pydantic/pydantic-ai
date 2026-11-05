from __future__ import annotations

from pydantic import JsonValue

from ...conftest import try_import
from ..oauth_helpers import make_jwt

with try_import():
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentials


def make_credentials(*, exp: float | None = None, access_token: str = 'access-old') -> OpenAICodexCredentials:
    token = access_token if exp is None else make_jwt({'exp': exp})
    return OpenAICodexCredentials(access_token=token, refresh_token='refresh-1', account_id='acc-1')


class FakeCredentialSource:
    """In-memory stand-in for an application's credential store."""

    def __init__(self, credentials: OpenAICodexCredentials | None = None):
        self.credentials = credentials if credentials is not None else make_credentials(access_token='access-v1')
        self.loads = 0
        self.saves: list[str] = []

    async def load(self) -> OpenAICodexCredentials:
        self.loads += 1
        return self.credentials

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        self.credentials = credentials
        self.saves.append(credentials.access_token)


TOKEN_RESPONSE: dict[str, JsonValue] = {
    'access_token': 'access-new',
    'refresh_token': 'refresh-2',
    'id_token': make_jwt({'https://api.openai.com/auth': {'chatgpt_account_id': 'acc-9'}}),
}
CODEX_URL = 'https://chatgpt.com/backend-api/codex/x'
