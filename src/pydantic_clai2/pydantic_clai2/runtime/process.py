"""What the live sessions of one CLAI process share."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Self

from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import DepsT
from pydantic_clai2.plugins.process_plugins import ProcessPlugins


@dataclass(kw_only=True)
class ClaiProcess(Generic[DepsT]):
    """The services a CLAI process holds once for every session it runs: the stores and the plugins they share.

    The terminal runs one session at a time. Each `SessionRuntime` is built over the process it belongs to.
    """

    store: SettingsStore
    conversations: SqliteConversationStore
    """Saved conversations, in `sessions.db` beside the settings database."""
    plugins: ProcessPlugins[DepsT]
    """Plugin declarations, changes, and process-scoped plugins; each session's `SessionPlugins` joins it."""

    @classmethod
    def open(cls, store: SettingsStore | None = None, *, builtin_plugins: Sequence[PluginSettings] = ()) -> Self:
        """The process over `store`, by default the user's settings database, with `builtin_plugins` shipped."""
        store = store or SettingsStore()
        return cls(
            store=store,
            conversations=SqliteConversationStore(database=store.path.with_name('sessions.db')),
            plugins=ProcessPlugins[DepsT](store=store, builtin=builtin_plugins),
        )
