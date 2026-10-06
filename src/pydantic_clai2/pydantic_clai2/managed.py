"""Logfire-managed setup that has to happen before the shell starts: the gateway, and managed enrolment.

Kept free of heavy imports: the CLI reads it before deciding what to load.
"""

import os

from pydantic_clai2.config.api_keys import load_keys
from pydantic_clai2.config.settings_store import SettingsStore

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
