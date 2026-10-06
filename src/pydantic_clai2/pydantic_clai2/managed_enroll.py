"""Managed enrolment: one sign-in sets up traces, Agent Control and the gateway for the project IT named."""

from typing import Any

from pydantic_clai2.builtin_plugins.logfire import REQUIRES, LogfireAccount, LogfireSettings
from pydantic_clai2.builtin_plugins.logfire_setup import Setup, SetupError, run_setup
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.managed import NOTICES, ManagedTarget

FACTORY = 'pydantic_clai2.builtin_plugins.logfire'


async def enroll(store: SettingsStore, target: ManagedTarget) -> None:
    """Show why, sign in, and save the observability settings for the managed project."""
    print(f'Your organization manages clai2 through Logfire ({target.organization}).')
    existing = next((plugin for plugin in store.plugins() if plugin.id == 'observability'), None)
    current: dict[str, Any] = dict(existing.settings) if existing is not None else {}
    config = LogfireSettings.model_validate(current) if current else LogfireSettings()
    try:
        chosen = await run_setup(
            Setup(announce=print),
            current=target.base_url,
            owned=config.token,
            owned_variables=config.api_key,
            team=config.team,
            destination=target.base_url,
            fixed_project=(target.organization, target.project),
        )
    except SetupError as error:
        raise SystemExit(str(error)) from None
    if chosen is None:
        raise SystemExit('Sign-in cancelled, so clai2 cannot start: your organization manages it.')
    email = chosen.account_email
    settings = config.model_copy(
        update={
            'token': chosen.token,
            'base_url': chosen.base_url,
            'account': LogfireAccount(email=email, token=chosen.token) if email else None,
            'send_to_logfire': 'if-token-present',
            'api_key': chosen.variables_key,
            'gateway': chosen.gateway,
            'project': target.project_label,
            'agent_control': True,
            'agent_control_name': target.agent,
            'team': chosen.team,
        }
    )
    store.save_plugin(
        PluginSettings(id='observability', factory=FACTORY, settings=settings.model_dump(mode='json', by_alias=True)),
        requires={key: frozenset(value) for key, value in REQUIRES.items()},
    )
    print(f'clai2 is set up for {target.project_label}.')
    NOTICES.append(f'Signed in: clai2 is managed through Logfire ({target.project_label}).')
