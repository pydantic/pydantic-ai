"""What the live sessions of one CLAI process share."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Self

from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import DepsT
from pydantic_clai2.plugins.process_plugins import ProcessPlugins
from pydantic_clai2.runtime.live_runtimes import LiveRuntimes
from pydantic_clai2.runtime.session import step_store
from pydantic_clai2.runtime.session_naming import SessionNamer
from pydantic_clai2.runtime.tasks import TaskRouter

if TYPE_CHECKING:
    from pydantic_ai.providers.openai_codex import OpenAICodexProvider


@dataclass(kw_only=True)
class ClaiProcess(Generic[DepsT]):
    """The services a CLAI process holds once for every session it runs.

    Each `SessionRuntime` is built over the process it belongs to and joins its `live` sessions. The
    terminal runs one; `clai2 --web` opens more.
    """

    store: SettingsStore
    conversations: SqliteConversationStore
    """Saved conversations, in `sessions.db` beside the settings database."""
    tasks: TaskRouter
    """The one delegated-task owner; each session's `Tasks` is a view over it."""
    namer: SessionNamer
    """The one background naming worker; whoever runs the process runs `namer.run`."""
    plugins: ProcessPlugins[DepsT]
    """Plugin declarations, changes, and process-scoped plugins; each session's `SessionPlugins` joins it."""
    live: LiveRuntimes
    """The live sessions, by ID and by conversation, which process-scoped plugins reach as `host.live`."""
    codex_providers: 'dict[str, OpenAICodexProvider]' = field(default_factory=dict[str, 'OpenAICodexProvider'])
    """The native Codex provider per account, shared by every session's `ModelResolver`, so each refreshes once."""

    @classmethod
    def open(cls, store: SettingsStore | None = None, *, builtin_plugins: Sequence[PluginSettings] = ()) -> Self:
        """The process over `store`, by default the user's settings database, with `builtin_plugins` shipped."""
        store = store or SettingsStore()
        conversations = SqliteConversationStore(database=store.path.with_name('sessions.db'))
        plugins = ProcessPlugins[DepsT](store=store, builtin=builtin_plugins)
        plugins.live = live = LiveRuntimes(conversations=conversations)
        return cls(
            store=store,
            conversations=conversations,
            tasks=TaskRouter(directory=Path(f'{conversations.database}.tasks'), step_store=step_store(conversations)),
            namer=SessionNamer(store=conversations),
            plugins=plugins,
            live=live,
        )
