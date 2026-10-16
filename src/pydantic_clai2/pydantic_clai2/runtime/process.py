"""What the live sessions of one CLAI process share."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from rich.console import Console

from pydantic_ai_harness.step_persistence import SqliteStepStore
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.resolver import ModelResolver
from pydantic_clai2.runtime.session_naming import SessionNamer
from pydantic_clai2.runtime.tasks import TaskRouter


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
    models: ModelResolver
    """The Codex sign-in every session's resolver shares, through `models.for_session()`."""

    @classmethod
    def open(cls, store: SettingsStore | None = None, *, console: Console | None = None) -> Self:
        """The process over `store`, by default the user's settings database.

        `console` is where a Codex sign-in prints.
        """
        store = store or SettingsStore()
        conversations = SqliteConversationStore(database=store.path.with_name('sessions.db'))
        return cls(
            store=store,
            conversations=conversations,
            tasks=TaskRouter(
                directory=Path(f'{conversations.database}.tasks'),
                step_store=SqliteStepStore(database=conversations.database, max_snapshots_per_run=8),
            ),
            namer=SessionNamer(store=conversations),
            models=ModelResolver(console=console or Console(), store=store),
        )
