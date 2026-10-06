"""Logfire-managed setup that has to happen before the shell starts: the gateway, and managed enrolment.

Kept free of heavy imports: the CLI reads it before deciding what to load.
"""

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlsplit

from pydantic_clai2.config.api_keys import load_keys
from pydantic_clai2.config.settings_store import SettingsStore

MANAGED_ENV = 'CLAI2_MANAGED_URL'
MANAGED_FILES = {
    'linux': Path('/etc/clai2/managed.toml'),
    'darwin': Path('/Library/Application Support/clai2/managed.toml'),
}
"""Where IT can drop `url = "https://logfire-eu.pydantic.info/<org>/<project>"`; it wins over the env var."""


@dataclass(frozen=True)
class ManagedTarget:
    """The Logfire project an organization manages clai2 through, from the one URL IT ships."""

    base_url: str
    organization: str
    project: str
    agent: str = 'clai2'

    @property
    def project_label(self) -> str:
        return f'{self.organization}/{self.project}'


def managed_target() -> ManagedTarget | None:
    """The managed project, from the system file or `CLAI2_MANAGED_URL`; `None` when clai2 isn't managed."""
    url, agent = _managed_file()
    url = url or os.getenv(MANAGED_ENV)
    if not url:
        return None
    parts = urlsplit(url.strip())
    segments = [segment for segment in parts.path.split('/') if segment]
    if parts.scheme != 'https' or not parts.netloc or len(segments) < 2:
        raise SystemExit(f'{MANAGED_ENV} must look like https://logfire-eu.pydantic.info/<org>/<project>, not {url}')
    agent = agent or (parse_qs(parts.query).get('agent') or ['clai2'])[0]
    return ManagedTarget(base_url=f'https://{parts.netloc}', organization=segments[0], project=segments[1], agent=agent)


def _managed_file() -> tuple[str | None, str | None]:
    path = MANAGED_FILES.get(sys.platform)
    if path is None or not path.is_file():
        return None, None
    try:
        data = tomllib.loads(path.read_text(encoding='utf-8'))
    except (OSError, tomllib.TOMLDecodeError):
        return None, None
    url, agent = data.get('url'), data.get('agent')
    return (url if isinstance(url, str) else None), (agent if isinstance(agent, str) else None)


def enrolled(store: SettingsStore, target: ManagedTarget) -> bool:
    """Whether this config is set up for the managed project with a key Logfire still accepts."""
    declaration = next((plugin for plugin in store.plugins() if plugin.id == 'observability'), None)
    if declaration is None or not declaration.enabled:
        return False
    settings = declaration.settings
    if settings.get('project') != target.project_label or settings.get('base_url') != target.base_url:
        return False
    keys = load_keys()
    names = [
        reference.get('name')
        for reference in (settings.get('api_key'), settings.get('token'))
        if isinstance(reference, dict)
    ]
    if len(names) != 2 or not all(isinstance(name, str) and name in keys for name in names):
        return False
    return _key_accepted(target.base_url, keys[str(names[0])].get_secret_value())


def _key_accepted(base_url: str, key: str) -> bool:
    """One cheap request: an expired or revoked key gets 401/403. Offline counts as accepted."""
    import httpx

    try:
        response = httpx.get(f'{base_url}/v1/variables/', headers={'Authorization': f'bearer {key}'}, timeout=5)
    except httpx.HTTPError:
        return True
    return response.status_code not in (401, 403)


def ensure_enrolled(store: SettingsStore) -> ManagedTarget | None:
    """Before the first prompt: when managed and not (validly) set up, sign in, or exit. Ctrl-C exits."""
    target = managed_target()
    if target is None or enrolled(store, target):
        return target
    import asyncio

    from pydantic_clai2.managed_enroll import enroll

    try:
        asyncio.run(enroll(store, target))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    if not enrolled(store, target):
        raise SystemExit('Sign-in did not complete, so clai2 cannot start: your organization manages it.')
    _adopt_pushed_model(store, target)
    return target


def _adopt_pushed_model(store: SettingsStore, target: ManagedTarget) -> None:
    """On the first managed enrolment only: let the organization's model (from Agent Control) take over.

    Clears a model picked before enrolment, once (a marker beside the settings records it), so the model pushed
    in `agent__<agent>` is the default; a model picked afterwards sticks. The pushed model is also saved to the
    `/model` list so it can be picked again.
    """
    marker = store.path.parent / 'managed-model-adopted'
    if marker.exists():
        return
    model = _pushed_model(store, target)
    try:
        marker.write_text(target.project_label, encoding='utf-8')
    except OSError:
        pass
    if model is None:
        return
    if 'model' in store.overrides():
        store.reset('model')
    if model not in store.models():
        store.add_model(name=model)
    print(f"Your organization's default model is {model} (from Logfire). /model to change.")


def _pushed_model(store: SettingsStore, target: ManagedTarget) -> str | None:
    """The `model` the managed agent's config serves, read with the key enrolment saved."""
    import json

    import httpx
    from logfire.variables import VariablesConfig

    declaration = next((plugin for plugin in store.plugins() if plugin.id == 'observability'), None)
    reference = declaration.settings.get('api_key') if declaration is not None else None
    name = reference.get('name') if isinstance(reference, dict) else None
    key = load_keys().get(name) if isinstance(name, str) else None
    if key is None:
        return None
    try:
        response = httpx.get(
            f'{target.base_url}/v1/variables/',
            headers={'Authorization': f'bearer {key.get_secret_value()}'},
            timeout=10,
        )
        response.raise_for_status()
        resolved = VariablesConfig.model_validate(response.json()).resolve_serialized_value(
            f'agent__{target.agent}', None, {}
        )
        value: object = json.loads(resolved.value) if resolved.value else {}
    except (httpx.HTTPError, ValueError):
        return None
    model = cast(dict[str, object], value).get('model') if isinstance(value, dict) else None
    return model if isinstance(model, str) and model else None


def apply_gateway(store: SettingsStore) -> bool:
    """Hackathon: run `gateway/...` models with the key setup saved, if setup configured the gateway.

    Exports it as `PYDANTIC_AI_GATEWAY_API_KEY` for this process unless that is already set; the gateway provider
    then infers the region's gateway URL from the key itself. Returns whether the gateway is configured.
    """
    declaration = next((plugin for plugin in store.plugins() if plugin.id == 'observability'), None)
    if declaration is None or not declaration.enabled or declaration.settings.get('gateway') is not True:
        return False
    reference = declaration.settings.get('api_key')
    name = reference.get('name') if isinstance(reference, dict) else None
    key = load_keys().get(name) if isinstance(name, str) else None
    if key is None:
        return False
    os.environ.setdefault('PYDANTIC_AI_GATEWAY_API_KEY', key.get_secret_value())
    return True
