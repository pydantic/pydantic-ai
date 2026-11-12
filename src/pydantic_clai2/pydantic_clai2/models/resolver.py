"""Model names to models: provider integrations loaded on demand, accounts, profiles, and `chain:` fallbacks."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from anyio import to_thread
from rich.console import Console

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.profiles import ALL, DEFAULT, ModelRef, base_model, parse_model, with_profile
from pydantic_clai2.plugins import (
    ModelProvider,
    PluginLogin,
)
from pydantic_clai2.plugins.menus import Dialogs, SettingsMenu

if TYPE_CHECKING:
    from pydantic_ai.providers.openai_codex import OpenAICodexProvider
    from pydantic_clai2.auth import CodexAuth
    from pydantic_clai2.models.accounts import Account


def _parse(name: str) -> ModelRef:
    """Split a model name, reporting a malformed profile as a `UserError` that names the model."""
    try:
        return parse_model(name)
    except ValueError as exc:
        raise UserError(f'{name}: {exc}') from None


def _no_endpoint(name: str) -> UserError:
    from pydantic_clai2.models.key_profiles import no_endpoint

    return no_endpoint(name)


def _no_api_key(name: str) -> UserError:
    from pydantic_clai2.models.key_profiles import no_api_key

    return no_api_key(name)


def _check_built(provider: str, built: Mapping[str, str], *, plugin: bool) -> None:
    """Fail when `/model settings` saved a `base_url` or `api_key` that `provider`'s models are not built with."""
    if plugin and 'base_url' in built:
        raise _no_endpoint(provider)
    if 'api_key' in built:
        from pydantic_clai2.models.key_profiles import KEYED_ELSEWHERE

        if plugin or provider in KEYED_ELSEWHERE:
            raise _no_api_key(provider)


@dataclass(kw_only=True)
class ModelResolver:
    """Load provider integrations on demand, retaining Codex authentication per conversation."""

    console: Console
    plugins: Callable[[], Mapping[str, ModelProvider]] = lambda: {}
    """Model prefixes registered by loaded plugins, read per resolution so enabling one applies at once."""
    logins: Callable[[], Mapping[str, PluginLogin]] = lambda: {}
    """Sign-ins registered by loaded plugins, read per `/login` like `plugins`."""
    store: SettingsStore | None = None
    """Where a plugin sign-in saves its models."""
    pool_accounts: Callable[[], bool] = lambda: False
    """The `accounts.pool` setting, read per resolution so `/set` applies to the next run."""
    select_default: Callable[[str], bool] = lambda model: False
    """Select a model a sign-in prefers when the session has none chosen; whether it did."""
    codex_providers: 'dict[str, OpenAICodexProvider]' = field(default_factory=dict[str, 'OpenAICodexProvider'])
    """The native Codex provider per account, which the process's sessions share; sign-ins prompt on `console`."""
    _auth: 'CodexAuth | None' = None

    def codex_auth(self) -> 'CodexAuth':
        if self._auth is None:
            from pydantic_clai2.auth import CodexAuth

            self._auth = CodexAuth(self.console, providers=self.codex_providers)
        return self._auth

    async def login(self, args: list[str], dialogs: Dialogs | None = None) -> str:
        """`/login`; `dialogs` asks wherever a menu was opened, such as the web UI, instead of the terminal."""
        from pydantic_clai2.auth import login_command

        return await login_command(
            args,
            codex=self.codex_auth(),
            plugins=self.logins(),
            store=self.store,
            dialogs=dialogs,
            select_default=self.select_default,
        )

    def accounts_menu(self) -> SettingsMenu:
        """`/accounts` (and bare `/login`) as a declared menu, signing in through the dialogs it is shown with."""
        from pydantic_clai2.ui.menus.accounts_menu import accounts_settings_menu

        if self.store is None:  # pragma: no cover -- the shell always has a store.
            raise ValueError('Accounts need a settings database.')
        return accounts_settings_menu(
            self.store,
            sign_in=lambda target, dialogs: self.login([target], dialogs),
            plugins=self.logins,
            # Read when an account signs out, on the loop: the menu itself is built in a worker thread.
            forget=lambda account: self.codex_auth().forget(account),
        )

    async def resolve(self, name: str) -> Model | str:
        """Build `PROVIDER[@PROFILE]:NAME` or `chain:NAME`; a core model without a profile stays a string.

        A name that runs on several models, as `expand` lists them, is a `FallbackModel` that also falls
        back on a refusal, for running outside a session, whose runs use the `ModelFallback` capability.
        """
        from pydantic_ai.exceptions import ContentFilterError, ModelAPIError
        from pydantic_ai.models.fallback import FallbackModel

        first, *rest = [await self._one_account(model) for model in await self.expand(name)]
        return FallbackModel(first, *rest, fallback_on=(ModelAPIError, ContentFilterError)) if rest else first

    async def expand(self, name: str) -> list[str]:
        """The models `name` runs on in order, each on one account: a chain's, or every pooled account's.

        While `accounts.pool` is on, a name without a profile runs on every signed-in account, as `@*`
        does, once its provider has two or more; `@default` runs the default account alone.
        """
        from pydantic_clai2.models.chains import chain_name

        if (chain := chain_name(name)) is not None:
            if await self._saved(name, 'base_url') is not None:
                raise _no_endpoint(name)
            if await self._saved(name, 'api_key') is not None:
                raise _no_api_key(name)
            return [member for model in self._chain(chain) for member in await self.expand(model)]
        ref = _parse(name)
        if ref.profile == ALL and await self._saved(name, 'api_key') is not None:
            raise _no_api_key(name)  # Each pooled account keeps its own key.
        members = await self._accounts(ref)
        return [member.pinned(ref.name) for member in members] if members else [name]

    def _chain(self, chain: str) -> list[str]:
        """The models of a saved chain, in order."""
        models = self.store.chains().get(chain) if self.store is not None else None
        if not models:
            raise UserError(f'No chain named {chain}. Create one with /model chains.')
        return models

    async def _accounts(self, ref: ModelRef) -> 'list[Account]':
        """The accounts `@*`, or a pooled name without a profile, runs on; none for a name on one account."""
        if ref.profile == ALL:
            if not (members := await self._pool(ref.provider)):
                raise UserError(f'No {ref.provider} account is signed in. Add one with /accounts.')
            return members
        if ref.profile is None and ref.provider and self.pool_accounts():
            members = await self._pool(ref.provider)
            return members if len(members) > 1 else []
        return []

    async def _one_account(self, name: str) -> Model | str:
        """Build a model on the one account it names, the default one when it names none or `@default`."""
        ref = _parse(name)
        built = await self._built(name)
        endpoint = {'base_url': built['base_url']} if 'base_url' in built else {}
        if ref.profile == DEFAULT:
            name = base_model(name)
            ref = _parse(name)
        provider = self.plugins().get(ref.provider)
        _check_built(ref.provider, built, plugin=provider is not None)
        if ref.provider in ('openrouter', 'vllm', 'github-copilot'):
            from pydantic_clai2.models import github_copilot, openrouter, vllm

            build = {'openrouter': openrouter.model, 'vllm': vllm.model, 'github-copilot': github_copilot.model}
            return await to_thread.run_sync(partial(build[ref.provider], name, **endpoint), abandon_on_cancel=True)
        if ref.provider == 'openai-codex':
            return self.codex_auth().model(name, **endpoint)
        if provider is not None:
            if ref.profile is None:
                resolve = partial(provider.resolve, ref.name)
            elif provider.resolve_profile is not None:
                resolve = partial(provider.resolve_profile, ref.name, ref.profile)
            else:
                raise UserError(f'{ref.provider} does not support profiles. Use {ref.provider}:{ref.name}.')
            model = await to_thread.run_sync(resolve, abandon_on_cancel=True)
            if ref.provider == 'claude-code':
                from pydantic_clai2.models.retry_after import limit_retry_after

                limit_retry_after(model=model)
            return model
        if ref.profile is not None or built:
            from pydantic_clai2.models import key_profiles

            return await to_thread.run_sync(partial(key_profiles.model, name, **built), abandon_on_cancel=True)
        return name

    async def _built(self, name: str) -> dict[str, str]:
        """The `base_url` and `api_key` saved for building `name`, when set."""
        built = {'base_url': await self._base_url(name), 'api_key': await self._api_key(name)}
        return {setting: value for setting, value in built.items() if value is not None}

    async def _base_url(self, name: str) -> str | None:
        """The endpoint saved for `name`, or else for its pooled `PROVIDER@*:NAME` or plain `PROVIDER:NAME`."""
        plain = base_model(name)
        for candidate in dict.fromkeys((name, with_profile(plain, ALL), plain)):
            if (base_url := await self._saved(candidate, 'base_url')) is not None:
                return base_url
        return None

    async def _api_key(self, name: str) -> str | None:
        """The `/keys` entry saved for `name`, or for its plain `PROVIDER:NAME` on the default account it replaces."""
        if (api_key := await self._saved(name, 'api_key')) is not None or _parse(name).profile != DEFAULT:
            return api_key
        return await self._saved(base_model(name), 'api_key')

    async def _saved(self, name: str, setting: str) -> str | None:
        """A setting `/model settings` saved for `name` itself."""
        if (store := self.store) is None:
            return None
        value = await to_thread.run_sync(lambda: store.model_settings(name).get(setting), abandon_on_cancel=True)
        return value if isinstance(value, str) else None

    async def _pool(self, provider: str) -> 'list[Account]':
        """The provider's signed-in accounts, in `/accounts` order; none without a settings store."""
        from pydantic_clai2.models.accounts import pool

        store = self.store
        return await to_thread.run_sync(lambda: pool(store, provider), abandon_on_cancel=True) if store else []

    async def accounts(self, args: list[str]) -> str:
        """`/accounts`: list, add, rename, reorder, and sign out of accounts."""
        from pydantic_clai2.models.usage import usage_fetcher
        from pydantic_clai2.ui.menus.accounts_menu import open_accounts_menu

        if args:
            raise ValueError('Usage: /accounts (opens the menu)')
        if self.store is None:  # pragma: no cover -- the shell always has a store.
            raise ValueError('Accounts need a settings database.')
        auth = self.codex_auth()
        return await open_accounts_menu(
            self.store,
            login=self.login,
            plugins=self.logins,
            forget=auth.forget,
            usage=lambda item: usage_fetcher(item, codex=auth.account_provider, plugins=self.logins()),
        )
