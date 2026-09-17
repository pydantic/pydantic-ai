"""Authenticate and discover models through the native OpenRouter provider."""

import asyncio
import json

import httpx
from prompt_toolkit import PromptSession
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from rich.console import Console
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu  # pyright: ignore[reportMissingTypeStubs]

from ._rendering import markdown_style
from .command_context import CommandContext
from .credential_store import load_codex_credentials, save_codex_credentials
from .menu_worker import menu_key, run_worker
from .openrouter_auth import OpenRouterAuth


class Connection(BaseModel):
    """Endpoint and optional credential, stored together in keyring."""

    token: SecretStr = Field(min_length=1)


class ServedModel(BaseModel):
    """One OpenAI-compatible discovery result."""

    id: str = Field(min_length=1)


class ModelList(BaseModel):
    """Validated discovery response."""

    data: list[ServedModel]


async def discover(connection: Connection, *, transport: httpx.AsyncBaseTransport | None = None) -> list[str]:
    """Query only the requested endpoint; do not forward credentials across redirects."""
    token = connection.token.get_secret_value()
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    async with httpx.AsyncClient(transport=transport, timeout=20, follow_redirects=False) as client:
        try:
            authentication = await client.get('https://openrouter.ai/api/v1/key', headers=headers)
            authentication.raise_for_status()
            response = await client.get('https://openrouter.ai/api/v1/models', headers=headers)
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
    value = connection.model_dump()
    value['token'] = connection.token.get_secret_value()
    save_codex_credentials(value=json.dumps(value), account='openrouter')


def model(name: str) -> OpenRouterModel:
    """Resolve a saved OpenRouter selection through core, without global API-key fallbacks."""
    raw = load_codex_credentials(account='openrouter')
    if raw is None:
        raise UserError('Connect first through /model > openrouter.')
    try:
        connection = Connection.model_validate_json(raw)
    except ValidationError:
        raise UserError('Stored connection is invalid. Reconfigure through /model > openrouter.') from None
    provider = OpenRouterProvider(api_key=connection.token.get_secret_value())
    return OpenRouterModel(name.removeprefix('openrouter:'), provider=provider)


def choose(names: list[str]) -> str | None:  # pragma: no cover -- terminal ownership.
    """Pick one discovered model in Termflow."""
    result = (
        MenuBuilder('OpenRouter models')
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
        raise ValueError('Usage: /openrouter (choose browser login or enter an API key privately)')
    try:
        raw = await asyncio.to_thread(load_codex_credentials, account='openrouter')
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
    return context.set_setting(['model', f'openrouter:{selected}'])


async def prompt_connection() -> Connection | None:
    """Collect connection details without recording them in history."""
    method = await run_worker(lambda: authentication_menu().run())
    if method.cancelled or method.item is None:
        return None
    if method.item.value == 'browser':
        return Connection(token=await OpenRouterAuth(console=Console()).login())
    prompt: PromptSession[str] = PromptSession()
    try:
        token = await prompt.prompt_async('OpenRouter API key (https://openrouter.ai/keys): ', is_password=True)
    except (EOFError, KeyboardInterrupt):
        return None
    if not token.strip():
        raise ValueError('An OpenRouter API key is required.')
    return Connection(token=SecretStr(token.strip()))


def connection_action() -> str | None:  # pragma: no cover -- real terminal.
    """Reuse saved authentication or replace it from the provider menu."""
    result = (
        MenuBuilder('openrouter connection')
        .items([MenuItem('Browse models', value='browse'), MenuItem('Reconfigure connection', value='configure')])
        .key_source(menu_key)
        .build()
        .run()
    )
    return result.item.value if not result.cancelled and result.item and isinstance(result.item.value, str) else None


def authentication_menu() -> Menu:
    """Choose browser authorization or the existing masked API-key prompt."""
    return (
        MenuBuilder('OpenRouter authentication')
        .style(markdown_style())
        .items(
            [
                MenuItem('Sign in with browser', value='browser'),
                MenuItem('Enter API key', value='key'),
            ]
        )
        .preview(
            lambda item: (
                'Authorize CLAI on openrouter.ai, or enter an existing API key. Credentials stay out of history.'
            )
        )
        .footer_hint('Enter selects - Esc closes')
        .key_source(menu_key)
        .build()
    )
