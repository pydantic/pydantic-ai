"""What the live sessions of one CLAI process share."""

from dataclasses import dataclass
from typing import Self

from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2.config.settings_store import SettingsStore


@dataclass(kw_only=True)
class ClaiProcess:
    """The services a CLAI process holds once for every session it runs: the settings and conversation stores.

    The terminal runs one session at a time. Each `SessionRuntime` is built over the process it belongs to.
    """

    store: SettingsStore
    conversations: SqliteConversationStore
    """Saved conversations, in `sessions.db` beside the settings database."""

    @classmethod
    def open(cls, store: SettingsStore | None = None) -> Self:
        """The process over `store`, by default the user's settings database."""
        store = store or SettingsStore()
        return cls(store=store, conversations=SqliteConversationStore(database=store.path.with_name('sessions.db')))
