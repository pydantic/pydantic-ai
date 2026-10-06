"""Logfire-managed setup that has to happen before the shell starts: the gateway, and managed enrolment.

Kept free of heavy imports: the CLI reads it before deciding what to load.
"""

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
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
    return target


GATEWAY_DEFAULT_MODEL = 'gateway/anthropic:claude-sonnet-5-5'
"""The model when setup configured the gateway and the user has not picked one."""


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
