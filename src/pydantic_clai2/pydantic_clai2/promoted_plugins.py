"""Saved declarations from the retired harness catalog whose capability now has its own built-in plugin."""

from collections.abc import Sequence

from .config import PluginSettings
from .settings_store import SettingsStore

_PROMOTED = {'slack': 'pydantic_ai_harness.slack:Slack'}
"""The factory each promoted id had as a raw catalog row."""


def adopt_promoted(store: SettingsStore, builtin: Sequence[PluginSettings]) -> None:
    """Point a saved copy of a promoted catalog row at its built-in, keeping whether the user enabled it.

    Toggling a catalog row saved the whole declaration, and a saved declaration outranks the built-in, so
    without this the raw capability would keep loading instead. A declaration with the user's own settings is theirs.
    """
    shipped = {plugin.id: plugin for plugin in builtin}
    for saved in store.plugins():
        factory = _PROMOTED.get(saved.id)
        if factory is None or saved.id not in shipped:
            continue
        if saved == PluginSettings(id=saved.id, factory=factory, enabled=saved.enabled):
            store.save_plugin(shipped[saved.id].model_copy(update={'enabled': saved.enabled}))
