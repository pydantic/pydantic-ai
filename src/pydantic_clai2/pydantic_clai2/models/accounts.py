"""Accounts: every signed-in account per provider, in the order `@*` tries them.

The settings database keeps only the order, a label, and which plugin sign-in owns an account. Logins
CLAI keeps in its credential store are found from their files, so ones made before accounts existed
appear too; a plugin keeps its own tokens, so its accounts are the ones signed in through `/login`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

from pydantic_clai2.config.api_keys import forget_connection
from pydantic_clai2.config.credential_store import has_credentials, profile_accounts
from pydantic_clai2.config.settings_store import SettingsStore, StoredAccount
from pydantic_clai2.models.profiles import DEFAULT, account, provider_of, with_profile
from pydantic_clai2.plugins import PluginLogin

SUBSCRIPTIONS = ('openai-codex', 'github-copilot')
"""Sign-ins CLAI ships, whose default account is a browser or device login."""

CONNECTIONS = ('openrouter', 'vllm')
"""Providers whose account is a saved connection; the default one is what `/model add` connects."""


@dataclass(frozen=True, kw_only=True)
class Account:
    """One account as `/accounts` shows it."""

    provider: str
    profile: str | None
    label: str | None
    plugin_login: str | None
    signed_in: bool

    @property
    def login(self) -> str:
        """What `/login` takes to sign it in again: `openai-codex@work`, or a plugin's `claude@work`."""
        return account(self.plugin_login or self.provider, self.profile)

    @property
    def name(self) -> str:
        """Its label, else its profile, else `default`."""
        return self.label or self.profile or 'default'

    def model(self, name: str) -> str:
        """`name` run on this account, such as `openai-codex@work:gpt-6-astra`."""
        return with_profile(f'{self.provider}:{name}', self.profile)

    def pinned(self, name: str) -> str:
        """`name` on this account alone, even while `accounts.pool` pools plain names: the default is `@default`."""
        return with_profile(f'{self.provider}:{name}', self.profile or DEFAULT)


def accounts(store: SettingsStore) -> list[Account]:
    """Every account, by provider in the saved order, after adding logins found in the credential store."""
    for provider, profile in _found():
        store.add_account(StoredAccount(provider=provider, profile=profile))
    return [
        Account(
            provider=row.provider,
            profile=row.profile,
            label=row.label,
            plugin_login=row.plugin_login,
            signed_in=row.plugin_login is not None or has_credentials(account=account(row.provider, row.profile)),
        )
        for row in store.accounts()
    ]


def _found() -> list[tuple[str, str | None]]:
    found: list[tuple[str, str | None]] = [
        (provider, None) for provider in (*SUBSCRIPTIONS, *CONNECTIONS) if has_credentials(account=provider)
    ]
    for name in profile_accounts():
        provider, _, profile = name.rpartition('@')
        found.append((provider, profile))
    return found


def pool(store: SettingsStore, provider: str) -> list[Account]:
    """The signed-in accounts `PROVIDER@*` tries, in order."""
    return [item for item in accounts(store) if item.provider == provider and item.signed_in]


def remember(store: SettingsStore, *, login: str, profile: str | None, plugin: PluginLogin | None = None) -> None:
    """Record a sign-in that succeeded, under the model prefix it unlocks."""
    if plugin is None:
        store.add_account(StoredAccount(provider=login, profile=profile))
        return
    provider = provider_of(plugin.models[0]) if plugin.models else plugin.name
    store.add_account(StoredAccount(provider=provider, profile=profile, plugin_login=plugin.name))


def sign_out(store: SettingsStore, item: Account) -> str:
    """Delete CLAI's copy of the credentials and forget the account; a plugin keeps its own tokens."""
    store.remove_account(provider=item.provider, profile=item.profile)
    if item.plugin_login is not None:
        return f'Removed {item.login} from the list. Its plugin keeps the sign-in; use its own logout to end it.'
    forget_connection(account=account(item.provider, item.profile))
    return f'Signed out of {item.login}.'


@dataclass(frozen=True, kw_only=True)
class LoginChoice:
    """A provider `/accounts` can add an account to."""

    login: str
    """The `/login` name."""
    provider: str
    """The model prefix the account runs."""
    kind: str
    """`subscription`, `connection`, `plugin`, or `api key`; an API-key provider's default is the environment."""
    profiles: bool
    """Whether it can have more than the default account."""

    @property
    def has_default(self) -> bool:
        """Whether the account without a profile is one CLAI signs in, rather than the environment's key."""
        return self.kind != 'api key'


def login_choices(plugins: Mapping[str, PluginLogin]) -> list[LoginChoice]:
    """Sign-ins first, then connections, then providers that take an API key."""
    choices = [LoginChoice(login=name, provider=name, kind='subscription', profiles=True) for name in SUBSCRIPTIONS]
    for name, plugin in sorted(plugins.items()):
        provider = provider_of(plugin.models[0]) if plugin.models else name
        choices.append(
            LoginChoice(login=name, provider=provider, kind='plugin', profiles=plugin.profile_handler is not None)
        )
    choices += [LoginChoice(login=name, provider=name, kind='connection', profiles=True) for name in CONNECTIONS]
    taken = {choice.provider for choice in choices}
    choices += [
        LoginChoice(login=name, provider=name, kind='api key', profiles=True)
        for name in _keyed_providers()
        if name not in taken
    ]
    return choices


@cache
def _keyed_providers() -> tuple[str, ...]:
    """Providers whose installed SDK takes an API key. Imports each provider once, so it is cached."""
    from pydantic_clai2.models.key_profiles import keyed_provider
    from pydantic_clai2.models.model_catalog import runnable_providers

    names: list[str] = []
    for name in sorted(runnable_providers()):
        try:
            if keyed_provider(name) is not None:
                names.append(name)
        except ImportError:
            continue  # Its SDK is not installed here.
    return tuple(names)
