"""GitHub device login and model discovery through Pydantic AI's Copilot provider."""

import os
import time

import httpx2
from anyio import to_thread
from openai import APIError
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.github_copilot import GitHubCopilotModel
from pydantic_ai.providers.github_copilot import GitHubCopilotCredentials, GitHubCopilotOAuthFlow, GitHubCopilotProvider
from rich.console import Console

from .credential_store import credentials_path, load_codex_credentials, save_codex_credentials


class Connection(BaseModel):
    """Keep issuance time with core's token lifetime metadata."""

    credentials: GitHubCopilotCredentials
    issued_at: float = Field(ge=0, allow_inf_nan=False)


async def login(*, console: Console) -> str:
    """Display core's device challenge and save only a completed authorization."""
    client_id = os.getenv('GITHUB_COPILOT_CLIENT_ID', '').strip() or 'Iv1.b507a08c87ecfe98'
    flow = GitHubCopilotOAuthFlow(client_id=client_id, scope='read:user')
    authorization = await flow.start()
    console.print(f'Open {authorization.verification_uri}', markup=False, highlight=False)
    console.print(f'Enter code: {authorization.user_code}', markup=False, highlight=False)
    console.print('Approve only the code shown here. Ctrl-C cancels. You can open the link on another device.')
    credentials = await flow.wait_for_authorization()
    connection = Connection(credentials=credentials, issued_at=time.time())
    await to_thread.run_sync(
        lambda: save_codex_credentials(account='github-copilot', value=connection.model_dump_json())
    )
    path = credentials_path(account='github-copilot')
    storage = (
        f'No OS keyring is available; credentials are saved in plaintext at {path}.'
        if path.exists()
        else 'Credentials saved in the OS credential store.'
    )
    return f'GitHub login saved. {storage} Use /add_model > github-copilot to check Copilot access and choose a model.'


def token() -> str:
    """Prefer the saved login; retain Copilot-specific environment tokens when no login exists."""
    raw = load_codex_credentials(account='github-copilot')
    if raw is not None:
        try:
            connection = Connection.model_validate_json(raw)
        except ValidationError:
            raise UserError('Stored Copilot credentials are invalid. Run /login github-copilot.') from None
        lifetime = connection.credentials.expires_in
        if lifetime is not None and time.time() >= connection.issued_at + lifetime:
            raise UserError('Copilot credentials expired. Run /login github-copilot.')
        return connection.credentials.access_token
    for name in ('GITHUB_COPILOT_API_KEY', 'GITHUB_COPILOT_API_TOKEN', 'COPILOT_GITHUB_TOKEN'):
        if value := os.getenv(name):
            return value
    raise UserError('Copilot is not connected. Run /login github-copilot or set GITHUB_COPILOT_API_KEY.')


def model(name: str) -> GitHubCopilotModel:
    """Use core's Copilot model profiles and request semantics."""
    return GitHubCopilotModel(name.removeprefix('github-copilot:'), provider=GitHubCopilotProvider(api_key=token()))


class ServedModel(BaseModel):
    """Copilot's endpoint allowlist determines which models core can use."""

    id: str = Field(min_length=1)
    supported_endpoints: list[str] = Field(default_factory=list)
    model_picker_enabled: bool = False


class ModelList(BaseModel):
    """Validate discovery before presenting identifiers for selection."""

    data: list[ServedModel]


async def discover(*, transport: httpx2.AsyncBaseTransport | None = None) -> list[str]:
    """Use core's endpoint, authentication and headers, without following redirects."""
    async with httpx2.AsyncClient(transport=transport, timeout=20, follow_redirects=False) as client:
        copilot = GitHubCopilotProvider(api_key=await to_thread.run_sync(token), http_client=client)
        try:
            response = await copilot.client.with_options(max_retries=0).models.with_raw_response.list()
        except APIError:
            raise UserError(
                'Copilot model discovery failed. Check connectivity, your subscription and organization policy, '
                'or run /login github-copilot again.'
            ) from None
        try:
            models = ModelList.model_validate_json(response.content)
        except ValidationError:
            raise UserError('Copilot returned an invalid model list.') from None
    names = sorted(
        {
            model.id
            for model in models.data
            if model.model_picker_enabled and '/chat/completions' in model.supported_endpoints
        }
    )
    if not names:
        raise UserError('Copilot returned no Chat Completions models for this account.')
    return names


async def ensure_login(*, console: Console) -> None:
    """Request device authorization when saved or environment credentials cannot be used."""
    try:
        await to_thread.run_sync(token)
    except UserError:
        console.print(await login(console=console), markup=False)
