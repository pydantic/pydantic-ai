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
    for plugin in builtin:
        factory = _PROMOTED.get(plugin.id)
        if factory is None:
            continue

        def adopt(
            saved: PluginSettings, plugin: PluginSettings = plugin, factory: str = factory
        ) -> PluginSettings | None:
            if saved != PluginSettings(id=saved.id, factory=factory, enabled=saved.enabled):
                return None
            return plugin.model_copy(update={'enabled': saved.enabled})

        # One transaction: a declaration another CLAI saves meanwhile is never overwritten by a stale check.
        store.update_plugin(plugin.id, adopt)
