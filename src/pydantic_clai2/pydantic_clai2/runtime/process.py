"""What the live sessions of one CLAI process share."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime._session import step_store
from pydantic_clai2.runtime.session_naming import SessionNamer
from pydantic_clai2.runtime.tasks import TaskRouter

if TYPE_CHECKING:
    from pydantic_ai.providers.openai_codex import OpenAICodexProvider


@dataclass(kw_only=True)
class ClaiProcess:
    """The services a CLAI process holds once for every session it runs.

    The terminal runs one session at a time. Each `SessionRuntime` is built over the process it belongs to.
    """

    store: SettingsStore
    conversations: SqliteConversationStore
    """Saved conversations, in `sessions.db` beside the settings database."""
    tasks: TaskRouter
    """The one delegated-task owner; each session's `Tasks` is a view over it."""
    namer: SessionNamer
    """The one background naming worker; whoever runs the process runs `namer.run`."""
    codex_providers: 'dict[str, OpenAICodexProvider]' = field(default_factory=dict[str, 'OpenAICodexProvider'])
    """The native Codex provider per account, shared by every session's `ModelResolver`, so each refreshes once."""

    @classmethod
    def open(cls, store: SettingsStore | None = None) -> Self:
        """The process over `store`, by default the user's settings database."""
        store = store or SettingsStore()
        conversations = SqliteConversationStore(database=store.path.with_name('sessions.db'))
        return cls(
            store=store,
            conversations=conversations,
            tasks=TaskRouter(directory=Path(f'{conversations.database}.tasks'), step_store=step_store(conversations)),
            namer=SessionNamer(store=conversations),
        )
