"""Profiles for providers that sign in with an API key or a connection, not a subscription.

`/login openrouter@work` and `/login vllm@lab` save a connection the way `/model add` does, under the
profile's account. `/login openai@work` saves a key (or a `/keys` reference) for any Pydantic AI
provider that takes `api_key`; `openai@work:gpt-5` then runs with it. Models without a profile keep
their existing setup: `/model add` connections, or core's environment variables.
"""

import inspect
import json
import re
from functools import partial
from typing import Annotated, Protocol, TypeGuard

from anyio import to_thread
from prompt_toolkit import PromptSession
from pydantic import BaseModel, Field, SecretStr, ValidationError
from rich.console import Console

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider, infer_provider_class
from pydantic_clai2.config.api_keys import KeyReference, prompt_api_key, resolve_key, save_key_connection
from pydantic_clai2.config.credential_store import credentials_path, load_codex_credentials
from pydantic_clai2.models.profiles import account, parse_model
from pydantic_clai2.plugins.keys import choose_key
from pydantic_clai2.plugins.menus import Dialogs

CONNECTIONS = ('openrouter', 'vllm')
"""Providers whose profile is a full connection, signed in the way `/model add` connects them."""


class KeyConnection(BaseModel):
    """A profile's key: a `/keys` reference, or a key typed at sign-in."""

    token: Annotated[SecretStr, Field(min_length=1)] | KeyReference
    """Required and never empty: an empty key would let the provider fall back to the environment's key."""


URL_ARGUMENTS = {
    'azure': 'azure_endpoint',
    'azure-responses': 'azure_endpoint',
    'xai': 'api_host',
    'litellm': 'api_base',
}
"""Providers whose constructor names its endpoint something other than `base_url`."""


class KeyedProvider(Protocol):
    """A provider class that can be built from string options, such as an API key alone."""

    def __call__(self, **options: str) -> Provider[object]:
        """Build the provider."""
        ...


def _accepts(provider_class: type[Provider[object]], *options: str) -> TypeGuard[KeyedProvider]:
    # `__init__`, as the signature of a `@deprecated` class is its wrapped `__new__` before Python 3.13.
    return inspect.signature(provider_class.__init__).parameters.keys() >= set(options)


def keyed_provider(provider: str, *options: str) -> KeyedProvider | None:
    """The class Pydantic AI runs `provider:` models with, when it accepts `options`, by default `api_key`.

    A known provider whose SDK is missing raises core's `ImportError`, which names the extra to install.
    """
    try:
        provider_class = infer_provider_class(provider)
    except ValueError:
        return None
    return provider_class if _accepts(provider_class, *(options or ('api_key',))) else None


def supports(provider: str, *, profile: str | None) -> bool:
    """Whether `/login PROVIDER[@PROFILE]` is handled here; core providers need a profile."""
    return provider in CONNECTIONS or (profile is not None and keyed_provider(provider) is not None)


async def login(*, provider: str, profile: str | None, console: Console, dialogs: Dialogs | None = None) -> str:
    """Ask for the connection or key privately and save it under the profile's account.

    With `dialogs`, the key is a `/keys` entry, picked or typed there and saved under a name
    such as `OPENAI_WORK`; a connection is set up in the terminal only, printing through `console`.
    """
    name = account(provider, profile)
    if provider in CONNECTIONS:
        if dialogs is not None:
            raise ValueError(f'Connect {name} in the terminal: /login {name}.')
        saved = await _connect(provider=provider, name=name, console=console)
    elif dialogs is not None:
        token = await choose_key(name=_key_name(name), label=f'{provider} API key for {name}', dialogs=dialogs)
        saved = token is not None and await _save_token(provider=provider, name=name, token=token)
    else:
        saved = await _save_key(provider=provider, name=name)
    if not saved:
        return 'Sign-in cancelled.'
    path = credentials_path(account=name)
    storage = (
        f'saved in plaintext at {path}' if await to_thread.run_sync(path.exists) else 'saved in the OS credential store'
    )
    return f'{name} connected, {storage}. Use {name}:MODEL.'


async def _connect(*, provider: str, name: str, console: Console) -> bool:
    if provider == 'openrouter':
        from pydantic_clai2.models import openrouter

        connection = await openrouter.prompt_connection(console=console)
        if connection is None:
            return False
        await to_thread.run_sync(partial(openrouter.save_connection, connection, account=name))
        return True
    from pydantic_clai2.models import vllm

    server = await vllm.prompt_connection()
    if server is None:
        return False
    await to_thread.run_sync(partial(vllm.save_connection, server, account=name))
    return True


async def _save_key(*, provider: str, name: str) -> bool:
    prompt: PromptSession[str] = PromptSession()
    token = await prompt_api_key(prompt=prompt, label=f'{provider} API key for {name}: ')
    if token is None:
        return False
    if not isinstance(token, KeyReference) and not token.strip():
        raise ValueError(f'An API key is required for {name}.')
    return await _save_token(
        provider=provider, name=name, token=token if isinstance(token, KeyReference) else SecretStr(token.strip())
    )


def _key_name(account: str) -> str:
    """The `/keys` name a key typed for `account` is saved under, such as `OPENAI_WORK` for `openai@work`."""
    return re.sub(r'\W', '_', account).upper()


async def _save_token(*, provider: str, name: str, token: SecretStr | KeyReference) -> bool:
    connection = KeyConnection(token=token)
    # Build it now, so a provider that also needs an endpoint or region fails here, not on the next turn.
    await to_thread.run_sync(partial(_checked_provider, provider=provider, name=name, token=connection.token))
    value = connection.model_dump(mode='json')
    if isinstance(connection.token, SecretStr):
        value['token'] = connection.token.get_secret_value()
    await to_thread.run_sync(
        partial(save_key_connection, account=name, token=connection.token, value=json.dumps(value))
    )
    return True


def model(name: str, *, base_url: str | None = None) -> Model:
    """Build a core provider's model with the profile's saved key, or its `base_url`, instead of the environment's."""
    ref = parse_model(name)
    if ref.profile is None:
        return infer_model(name, provider_factory=partial(build_provider, base_url=base_url))
    if keyed_provider(ref.provider) is None:
        raise _no_profiles(ref.provider)
    raw = load_codex_credentials(account=ref.account)
    if raw is None:
        raise UserError(f'{ref.account} is not connected. Run /login {ref.account}.')
    try:
        connection = KeyConnection.model_validate_json(raw)
    except ValidationError:
        raise UserError(f'Stored {ref.account} credentials are invalid. Run /login {ref.account}.') from None
    provider = build_provider(ref.provider, key=resolve_key(token=connection.token), base_url=base_url)
    return infer_model(f'{ref.provider}:{ref.name}', provider_factory=lambda _: provider)


def build_provider(provider: str, *, key: str | None = None, base_url: str | None = None) -> Provider[object]:
    """The provider core would build for `provider:` models, with `key` and `base_url` instead of the environment's.

    A `gateway/` route goes through core's gateway provider, as `infer_provider` does. Other settings,
    such as an Azure API version, still come from the environment.
    """
    if provider.startswith('gateway/'):
        from pydantic_ai.providers.gateway import gateway_provider

        return gateway_provider(provider.removeprefix('gateway/'), api_key=key, base_url=base_url)
    options = {'api_key': key, URL_ARGUMENTS.get(provider, 'base_url'): base_url}
    options = {name: value for name, value in options.items() if value is not None}
    provider_class = keyed_provider(provider, *options)
    if provider_class is None:
        raise _no_profiles(provider) if base_url is None else no_endpoint(provider)
    return provider_class(**options)


def no_endpoint(provider: str) -> UserError:
    """The error for a `base_url` saved for a provider, or a chain, that cannot take one."""
    return UserError(f'{provider} has no endpoint to override. Reset base_url in /model settings.')


def _no_profiles(provider: str) -> UserError:
    return UserError(
        f'{provider} has no profiles; only providers that sign in with an API key do. Use {provider}:MODEL.'
    )


def _checked_provider(*, provider: str, name: str, token: SecretStr | KeyReference) -> None:
    try:
        build_provider(provider, key=resolve_key(token=token))
    except (UserError, ValueError) as exc:
        raise UserError(f'{name} was not saved: {exc}') from None
