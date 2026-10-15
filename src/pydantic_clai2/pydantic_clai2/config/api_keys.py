"""Named API keys and a shared, name-only picker for credential prompts."""

import json
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from anyio import to_thread
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError
from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu

from pydantic_ai.exceptions import UserError
from pydantic_clai2.config.credential_store import (
    credential_lock,
    credentials_path,
    delete_credentials,
    load_codex_credentials,
    profile_accounts,
    save_codex_credentials,
)
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.rendering._rendering import markdown_style


class KeyReference(BaseModel):
    """A name resolved from the credential store, not a cached secret."""

    model_config = ConfigDict(extra='forbid')

    # The pattern `normalize_name` gives every `/keys` label; it also rejects most pasted tokens, such as `ghp_...`.
    name: str = Field(pattern=r'^[A-Z_][A-Z0-9_]*$')


def resolve_key(*, token: SecretStr | KeyReference) -> str:
    """Resolve at use time and fail closed when a referenced key was deleted."""
    if isinstance(token, SecretStr):
        return token.get_secret_value()
    keys = load_keys()
    if token.name not in keys:
        raise UserError(
            f'Saved API key {token.name} is missing. Restore it in /keys or reconfigure the connection that uses it.'
        )
    return keys[token.name].get_secret_value()


@dataclass(frozen=True, kw_only=True)
class SavedKey:
    """A capability's `auth` function: the named key's current value, looked up on every run.

    Plugins keep only the name in their settings. Replacing the key in `/keys` reaches the next
    run of every plugin that names it; deleting it fails that run closed with `setup` as the fix.
    """

    name: str
    setup: str

    def __call__(self, ctx: object, /) -> str:
        """Resolve now, so a stale value is never reused."""
        keys = load_keys()
        if self.name not in keys:
            raise UserError(f'Saved API key {self.name} is missing. {self.setup}')
        return keys[self.name].get_secret_value()


def save_key_connection(*, account: str, token: SecretStr | KeyReference, value: str) -> None:
    """Validate references and save atomically with respect to key renames and deletions."""
    with key_transaction():
        if isinstance(token, KeyReference) and token.name not in _load_keys():
            raise UserError(
                f'The selected API key no longer exists. Select a saved key again through {_setup(account)}.'
            )
        save_codex_credentials(account=account, value=value)


def forget_connection(*, account: str) -> None:
    """Drop a saved connection, and with it any key reference it held; the keys themselves stay.

    The account's own lock is held too, so a refresh in flight cannot save the login back afterwards;
    see `replace_credentials`.
    """
    busy = f'Another CLAI is updating the {account} login. Try again.'
    with key_transaction(), credential_lock(account=account, busy=busy):
        delete_credentials(account=account)


class SecretPrompt(Protocol):
    """The masked input used by provider connection flows."""

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        """Read a value without recording it in command history."""
        ...


def normalize_name(*, name: str) -> str:
    """Use uppercase environment-style labels, without exporting environment variables."""
    name = name.strip().upper()
    if re.fullmatch(r'[A-Z_][A-Z0-9_]*', name) is None:
        raise ValueError('Use letters, numbers, and underscores; start with a letter or underscore.')
    return name


def key_transaction() -> AbstractContextManager[None]:
    """Serialize bundle access across processes."""
    return credential_lock(
        account='api-keys', busy='Cannot lock API keys. Close other key editors and check the credential directory.'
    )


def load_keys() -> dict[str, SecretStr]:
    """Read a complete bundle without racing keyring chunk replacement."""
    with key_transaction():
        return _load_keys()


def _load_keys() -> dict[str, SecretStr]:
    raw = load_codex_credentials(account='api-keys')
    if raw is None:
        return {}
    try:
        return TypeAdapter(dict[str, SecretStr]).validate_json(raw)
    except ValidationError:
        raise UserError('Stored API keys are invalid. Repair the api-keys credential bundle.') from None


class KeyExistsError(ValueError):
    """`save_key(replace=False)` found a key of that name, which other connections may share."""


def save_key(*, name: str, value: str, replace: bool = True) -> str:
    """Save one key without touching unrelated credentials or SQLite.

    `replace=False` checks and saves under one lock, so a key another process just created is not overwritten.
    """
    name = normalize_name(name=name)
    value = value.strip()
    if not value:
        raise ValueError('An API key is required.')
    with key_transaction():
        keys = _load_keys()
        if not replace and name in keys:
            raise KeyExistsError(f'{name} is already saved.')
        replaced = name in keys
        keys[name] = SecretStr(value)
        _save_keys(keys=keys)
    telemetry.record('key saved', key_name=name, replaced=replaced)
    path = credentials_path(account='api-keys')
    if path.is_file():
        return f'Saved {name}. No OS keyring is available; keys are stored in plaintext at {path}.'
    return f'Saved {name} in the OS keyring.'


def _save_keys(*, keys: dict[str, SecretStr]) -> None:
    save_codex_credentials(
        account='api-keys', value=json.dumps({label: key.get_secret_value() for label, key in keys.items()})
    )


_KEY_CONSUMERS = {
    'vllm': '/model add',
    'openrouter': '/model add',
    'google-workspace': '/google_workspace',
    'pylon': '/pylon',
    'ordinal': '/ordinal',
    'notion': '/plugins configure notion',
    'slack': '/plugins configure slack',
    'posthog': '/plugins configure posthog',
    'grain': '/grain key',
    'linear': '/plugins configure linear',
}
"""Credential-store accounts that may reference a saved key, and the command that reconfigures each.

Auth profiles (`PROVIDER@PROFILE` accounts) may too; `/login PROVIDER@PROFILE` reconfigures them.
"""


def _setup(account: str) -> str:
    return _KEY_CONSUMERS.get(account, f'/login {account}')


class _Credential(BaseModel):
    token: SecretStr | KeyReference = Field(default_factory=lambda: SecretStr(''))


def key_users(*, name: str) -> list[str]:
    """Find saved provider and plugin references without exposing their inline credentials."""
    users: list[str] = []
    for account in [*_KEY_CONSUMERS, *profile_accounts()]:
        command = _setup(account)
        raw = load_codex_credentials(account=account)
        if raw is not None:
            try:
                credential = _Credential.model_validate_json(raw)
            except ValidationError:
                raise UserError(f'Reconfigure the invalid {account} connection through {command} first.') from None
            if isinstance(credential.token, KeyReference) and credential.token.name == name:
                users.append(account)
    return users


def rename_key(*, name: str, new_name: str) -> str:
    """Rename unused keys; do not strand saved connections or overwrite another key."""
    new_name = normalize_name(name=new_name)
    with key_transaction():
        keys = _load_keys()
        if name not in keys:
            raise ValueError('The saved key no longer exists.')
        if new_name == name:
            return 'API key unchanged.'
        if new_name in keys:
            raise ValueError('That key name already exists.')
        users = key_users(name=name)
        if users:
            raise ValueError(f'Key is used by {", ".join(users)}. Reconfigure those connections before renaming.')
        keys[new_name] = keys.pop(name)
        _save_keys(keys=keys)
    telemetry.record('key renamed', key_name=name, new_key_name=new_name)
    return f'Renamed {name} to {new_name}.'


def delete_key(*, name: str) -> str:
    """Remove a named secret; references will fail closed on their next use."""
    with key_transaction():
        keys = _load_keys()
        keys.pop(name, None)
        _save_keys(keys=keys)
    telemetry.record('key deleted', key_name=name)
    return f'Deleted {name}. Connections referencing it can no longer authenticate.'


async def set_api_key(*, args: list[str]) -> str:
    """Ask for a name and masked value, never accepting secrets as command arguments."""
    if args:
        raise ValueError('Usage: /set api_key (name and key are prompted privately)')
    prompt: PromptSession[str] = PromptSession()
    try:
        name = normalize_name(name=await prompt.prompt_async('API key name (automatically uppercased): '))
        keys = await to_thread.run_sync(load_keys, abandon_on_cancel=True)
        if name in keys:
            answer = await prompt.prompt_async(f'Replace {name}? [y/N]: ')
            if answer.strip().lower() != 'y':
                return 'API key unchanged.'
        value = await prompt.prompt_async(f'API key value for {name}: ', is_password=True)
    except (EOFError, KeyboardInterrupt):
        return 'API key entry cancelled.'
    return await to_thread.run_sync(partial(save_key, name=name, value=value), abandon_on_cancel=True)


def build_key_menu(*, names: list[str], label: str, optional: bool) -> Menu:
    """Build a picker that never receives secret values."""
    items = [MenuItem(name, value=name) for name in sorted(names)]
    items.append(MenuItem('Enter a different API key', value='enter'))
    if optional:
        items.append(MenuItem('No API key', value='none'))
    return (
        MenuBuilder(label)
        .style(markdown_style())
        .items(items)
        .searchable()
        .preview(lambda item: 'Choose a saved key for this connection. Only key names are displayed.')
        .footer_hint('Enter selects - Esc closes')
        .key_source(menu_key)
        .build()
    )


async def prompt_api_key(*, prompt: SecretPrompt, label: str, optional: bool = False) -> str | KeyReference | None:
    """Return a saved-key reference or a masked inline value; None means cancellation."""
    with telemetry.span('key prompt', label=label) as span:
        choice = await _prompt_api_key(prompt=prompt, label=label, optional=optional)
        span.set('answer', _answer(choice))
        return choice


def _answer(choice: str | KeyReference | None) -> str:
    """What kind of answer the key prompt got; never the key's value."""
    if choice is None:
        return 'cancelled'
    if isinstance(choice, KeyReference):
        return 'saved key'
    return 'typed' if choice else 'no key'


async def _prompt_api_key(*, prompt: SecretPrompt, label: str, optional: bool) -> str | KeyReference | None:
    keys = await to_thread.run_sync(load_keys, abandon_on_cancel=True)
    if keys:
        menu = build_key_menu(names=list(keys), label=label, optional=optional)
        result = await run_worker(lambda: menu.run())
        if result.cancelled or result.item is None:
            return None
        selected = result.item.value
        if selected == 'none':
            return ''
        if isinstance(selected, str) and selected in keys:
            return KeyReference(name=selected)
    try:
        return await prompt.prompt_async(label, is_password=True)
    except (EOFError, KeyboardInterrupt):
        return None
