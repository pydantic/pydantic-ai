"""Connect to a trusted vLLM server using core's OpenAI-compatible provider."""

import asyncio
import json

import httpx
from prompt_toolkit import PromptSession
from pydantic import BaseModel, Field, HttpUrl, SecretStr, TypeAdapter, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vllm import VLLMProvider
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]

from .api_keys import KeyReference, prompt_api_key, resolve_key, save_key_connection
from .command_context import CommandContext
from .credential_store import load_codex_credentials
from .menu_worker import menu_key, run_worker


class Connection(BaseModel):
    """Saved connection with a named-key reference or a legacy inline credential."""

    url: str
    token: SecretStr | KeyReference = Field(default_factory=lambda: SecretStr(''))


class ServedModel(BaseModel):
    """One OpenAI-compatible discovery result."""

    id: str = Field(min_length=1)


class ModelList(BaseModel):
    """Validated discovery response."""

    data: list[ServedModel]


def api_url(value: str) -> str:
    """Accept a server root or API root, but not credentials or query parameters."""
    try:
        url = TypeAdapter(HttpUrl).validate_python(value.strip())
    except ValidationError:
        raise UserError('Enter a valid HTTP(S) server URL.') from None
    if url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an HTTP(S) server URL without credentials, query, or fragment.')
    root = str(url).rstrip('/')
    return root if root.endswith('/v1') else root + '/v1'


async def discover(connection: Connection, *, transport: httpx.AsyncBaseTransport | None = None) -> list[str]:
    """Query only the requested endpoint; do not forward credentials across redirects."""
    token = await asyncio.to_thread(resolve_key, token=connection.token)
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    async with httpx.AsyncClient(transport=transport, timeout=20, follow_redirects=False, trust_env=False) as client:
        try:
            response = await client.get(f'{api_url(connection.url)}/models', headers=headers)
            response.raise_for_status()
        except httpx.HTTPError:
            raise UserError('Model discovery failed. Check the server URL, token, and connectivity.') from None
    try:
        names = sorted({model.id for model in ModelList.model_validate_json(response.content).data})
    except ValidationError:
        raise UserError('The server returned an invalid model list.') from None
    if not names:
        raise UserError('The server returned no models.')
    return names


def save_connection(connection: Connection) -> None:
    """Keep credentials out of command history and SQLite."""
    value = connection.model_dump(mode='json')
    if isinstance(connection.token, SecretStr):
        value['token'] = connection.token.get_secret_value()
    save_key_connection(value=json.dumps(value), account='vllm', token=connection.token)


def model(name: str) -> OpenAIChatModel:
    """Resolve a saved vLLM selection through core, without global API-key fallbacks."""
    raw = load_codex_credentials(account='vllm')
    if raw is None:
        raise UserError('Connect first through /add_model > vllm.')
    try:
        connection = Connection.model_validate_json(raw)
    except ValidationError:
        raise UserError('Stored connection is invalid. Reconfigure through /add_model > vllm.') from None
    provider = VLLMProvider(
        base_url=api_url(connection.url), api_key=resolve_key(token=connection.token) or 'not-required'
    )
    return OpenAIChatModel(name.removeprefix('vllm:'), provider=provider)


def choose(names: list[str]) -> str | None:  # pragma: no cover -- terminal ownership.
    """Pick one discovered model in Termflow."""
    result = (
        MenuBuilder('vLLM models')
        .items([MenuItem(name, value=name) for name in names])
        .searchable()
        .key_source(menu_key)
        .build()
        .run()
    )
    return result.item.value if not result.cancelled and result.item and isinstance(result.item.value, str) else None


async def connect(context: CommandContext, args: list[str]) -> str:
    """Prompt privately, discover models, then persist only after selection."""
    if args:
        raise ValueError('Usage: /vllm (URL and optional token are prompted separately)')
    try:
        raw = await asyncio.to_thread(load_codex_credentials, account='vllm')
        connection = Connection.model_validate_json(raw) if raw else None
    except (ValidationError, UserError):
        connection = None
    if connection is not None:
        action = await run_worker(connection_action)
        if action is None:
            return 'Connection cancelled.'
        if action == 'configure':
            connection = None
    if connection is None:
        connection = await prompt_connection()
    if connection is None:
        return 'Connection cancelled.'
    names = await discover(connection)
    selected = await run_worker(lambda: choose(names))
    if selected is None:
        return 'Connection cancelled.'
    await asyncio.to_thread(save_connection, connection)
    return context.set_setting(['model', f'vllm:{selected}'])


async def prompt_connection() -> Connection | None:
    """Collect connection details without recording them in history."""
    prompt: PromptSession[str] = PromptSession()
    try:
        url = api_url(await prompt.prompt_async('vLLM server URL: '))
        token = await prompt_api_key(prompt=prompt, label='Token (optional, Enter for none): ', optional=True)
    except (EOFError, KeyboardInterrupt):
        return None
    if token is None:
        return None
    return Connection(url=url, token=token if isinstance(token, KeyReference) else SecretStr(token))


def connection_action() -> str | None:  # pragma: no cover -- real terminal.
    """Reuse saved authentication or replace it from the provider menu."""
    result = (
        MenuBuilder('vllm connection')
        .items([MenuItem('Browse models', value='browse'), MenuItem('Reconfigure connection', value='configure')])
        .key_source(menu_key)
        .build()
        .run()
    )
    return result.item.value if not result.cancelled and result.item and isinstance(result.item.value, str) else None
