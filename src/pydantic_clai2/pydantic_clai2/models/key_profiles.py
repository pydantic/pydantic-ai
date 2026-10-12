"""Profiles for providers that sign in with an API key or a connection, not a subscription.

`/login openrouter@work` and `/login vllm@lab` save a connection the way `/model add` does, under the
profile's account. `/login openai@work` saves a key (or a `/keys` reference) for any Pydantic AI
provider that takes `api_key`; `openai@work:gpt-5` then runs with it. Models without a profile keep
their existing setup: `/model add` connections, or core's environment variables.
"""

import inspect
import json
from functools import partial
from typing import Annotated, Protocol, TypeGuard

from anyio import to_thread
from prompt_toolkit import PromptSession
from pydantic import BaseModel, Field, SecretStr, ValidationError

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider, infer_provider_class
from pydantic_clai2.config.api_keys import KeyReference, prompt_api_key, resolve_key, save_key_connection
from pydantic_clai2.config.credential_store import credentials_path, load_codex_credentials
from pydantic_clai2.models.profiles import account, parse_model

CONNECTIONS = ('openrouter', 'vllm')
"""Providers whose profile is a full connection, signed in the way `/model add` connects them."""


class KeyConnection(BaseModel):
    """A profile's key: a `/keys` reference, or a key typed at sign-in."""

    token: Annotated[SecretStr, Field(min_length=1)] | KeyReference
    """Required and never empty: an empty key would let the provider fall back to the environment's key."""


class KeyedProvider(Protocol):
    """A provider class that can be built from an API key alone."""

    def __call__(self, *, api_key: str) -> Provider[object]:
        """Build the provider."""
        ...


def _accepts_key(provider_class: type[Provider[object]]) -> TypeGuard[KeyedProvider]:
    return 'api_key' in inspect.signature(provider_class).parameters


def keyed_provider(provider: str) -> KeyedProvider | None:
    """The class Pydantic AI runs `provider:` models with, when it accepts `api_key`.

    A known provider whose SDK is missing raises core's `ImportError`, which names the extra to install.
    """
    try:
        provider_class = infer_provider_class(provider)
    except ValueError:
        return None
    return provider_class if _accepts_key(provider_class) else None


def supports(provider: str, *, profile: str | None) -> bool:
    """Whether `/login PROVIDER[@PROFILE]` is handled here; core providers need a profile."""
    return provider in CONNECTIONS or (profile is not None and keyed_provider(provider) is not None)


async def login(*, provider: str, profile: str | None) -> str:
    """Ask for the connection or key privately and save it under the profile's account."""
    name = account(provider, profile)
    if provider in CONNECTIONS:
        saved = await _connect(provider=provider, name=name)
    else:
        saved = await _save_key(provider=provider, name=name)
    if not saved:
        return 'Sign-in cancelled.'
    path = credentials_path(account=name)
    storage = f'saved in plaintext at {path}' if path.exists() else 'saved in the OS credential store'
    return f'{name} connected, {storage}. Use {name}:MODEL.'


async def _connect(*, provider: str, name: str) -> bool:
    if provider == 'openrouter':
        from pydantic_clai2.models import openrouter

        connection = await openrouter.prompt_connection()
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
    connection = KeyConnection(token=token if isinstance(token, KeyReference) else SecretStr(token.strip()))
    # Build it now, so a provider that also needs an endpoint or region fails here, not on the next turn.
    await to_thread.run_sync(partial(_checked_provider, provider=provider, name=name, token=connection.token))
    value = connection.model_dump(mode='json')
    if isinstance(connection.token, SecretStr):
        value['token'] = connection.token.get_secret_value()
    await to_thread.run_sync(
        partial(save_key_connection, account=name, token=connection.token, value=json.dumps(value))
    )
    return True


def model(name: str) -> Model:
    """Build a core provider's model with the profile's saved key instead of environment variables."""
    ref = parse_model(name)
    if keyed_provider(ref.provider) is None:
        raise _no_profiles(ref.provider)
    raw = load_codex_credentials(account=ref.account)
    if raw is None:
        raise UserError(f'{ref.account} is not connected. Run /login {ref.account}.')
    try:
        connection = KeyConnection.model_validate_json(raw)
    except ValidationError:
        raise UserError(f'Stored {ref.account} credentials are invalid. Run /login {ref.account}.') from None
    provider = build_provider(ref.provider, key=resolve_key(token=connection.token))
    return infer_model(f'{ref.provider}:{ref.name}', provider_factory=lambda _: provider)


def build_provider(provider: str, *, key: str) -> Provider[object]:
    """The provider core would build for `provider:` models, with `key` instead of the environment's key.

    A `gateway/` route goes through core's gateway provider, as `infer_provider` does, so it keeps the
    Gateway's endpoint. Other settings, such as an Azure endpoint, still come from the environment.
    """
    if provider.startswith('gateway/'):
        from pydantic_ai.providers.gateway import gateway_provider

        return gateway_provider(provider.removeprefix('gateway/'), api_key=key)
    provider_class = keyed_provider(provider)
    if provider_class is None:
        raise _no_profiles(provider)
    return provider_class(api_key=key)


def _no_profiles(provider: str) -> UserError:
    return UserError(
        f'{provider} has no profiles; only providers that sign in with an API key do. Use {provider}:MODEL.'
    )


def _checked_provider(*, provider: str, name: str, token: SecretStr | KeyReference) -> None:
    try:
        build_provider(provider, key=resolve_key(token=token))
    except (UserError, ValueError) as exc:
        raise UserError(f'{name} was not saved: {exc}') from None
