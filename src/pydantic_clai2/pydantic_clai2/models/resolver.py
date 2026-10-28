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
from pydantic_clai2.models.profiles import ALL, DEFAULT, ModelRef, base_model, parse_model
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
            args, codex=self.codex_auth(), plugins=self.logins(), store=self.store, dialogs=dialogs
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

        While `accounts.pool` is on, a name without a profile runs on every signed-in account, as `@*`
        does, once its provider has two or more; `@default` runs the default account alone.
        """
        from pydantic_clai2.models.chains import chain_name

        if (chain := chain_name(name)) is not None:
            return await self._chain(chain)
        ref = _parse(name)
        if ref.profile == ALL:
            return await self._all_accounts(ref.provider, ref.name)
        if ref.profile == DEFAULT:
            return await self._one_account(base_model(name))
        if ref.profile is None and ref.provider and self.pool_accounts():
            members = await self._pool(ref.provider)
            if len(members) > 1:
                return await self._fall_back(members, ref.name)
        return await self._one_account(name)

    async def _one_account(self, name: str) -> Model | str:
        """Build a model on the one account it names, the default one when it names none."""
        ref = _parse(name)
        if ref.provider in ('openrouter', 'vllm', 'github-copilot'):
            from pydantic_clai2.models import github_copilot, openrouter, vllm

            build = {'openrouter': openrouter.model, 'vllm': vllm.model, 'github-copilot': github_copilot.model}
            return await to_thread.run_sync(build[ref.provider], name, abandon_on_cancel=True)
        if ref.provider == 'openai-codex':
            return self.codex_auth().model(name)
        if (provider := self.plugins().get(ref.provider)) is not None:
            if ref.profile is None:
                resolve = partial(provider.resolve, ref.name)
            elif provider.resolve_profile is not None:
                resolve = partial(provider.resolve_profile, ref.name, ref.profile)
            else:
                raise UserError(f'{ref.provider} does not support profiles. Use {ref.provider}:{ref.name}.')
            model = await to_thread.run_sync(resolve, abandon_on_cancel=True)
            if ref.provider == 'claude-code':
                from pydantic_clai2.models._retry_after import limit_retry_after

                limit_retry_after(model=model)
            return model
        if ref.profile is not None:
            from pydantic_clai2.models import key_profiles

            return await to_thread.run_sync(key_profiles.model, name, abandon_on_cancel=True)
        return name

    async def _chain(self, chain: str) -> Model:
        """Resolve every model of a saved chain and fall back through them in order."""
        from pydantic_ai.models.fallback import FallbackModel

        models = self.store.chains().get(chain) if self.store is not None else None
        if not models:
            raise UserError(f'No chain named {chain}. Create one with /model chains.')
        first, *rest = [await self.resolve(model) for model in models]
        return FallbackModel(first, *rest)

    async def _all_accounts(self, provider: str, name: str) -> Model | str:
        """`PROVIDER@*:NAME`: the model on every signed-in account, in `/accounts` order, falling back in turn."""
        members = await self._pool(provider)
        if not members:
            raise UserError(f'No {provider} account is signed in. Add one with /accounts.')
        return await self._fall_back(members, name)

    async def _pool(self, provider: str) -> 'list[Account]':
        """The provider's signed-in accounts, in `/accounts` order; none without a settings store."""
        from pydantic_clai2.models.accounts import pool

        store = self.store
        return await to_thread.run_sync(lambda: pool(store, provider), abandon_on_cancel=True) if store else []

    async def _fall_back(self, members: 'list[Account]', name: str) -> Model | str:
        """NAME on each account in turn, the next one taking over when a request fails."""
        from pydantic_ai.models.fallback import FallbackModel

        first, *rest = [await self._one_account(member.model(name)) for member in members]
        return FallbackModel(first, *rest) if rest else first

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
