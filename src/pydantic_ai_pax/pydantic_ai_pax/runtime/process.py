"""What the live sessions of one Pax process share."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Self

from anyio import Lock, to_thread

from pydantic_ai_harness.step_persistence.conversations import SqliteSessionStore
from pydantic_ai_pax.config import PluginSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import DepsT
from pydantic_ai_pax.plugins.process_plugins import ProcessPlugins
from pydantic_ai_pax.runtime.live_runtimes import LiveRuntimes
from pydantic_ai_pax.runtime.session import step_store
from pydantic_ai_pax.runtime.session_naming import SessionNamer
from pydantic_ai_pax.runtime.tasks import TaskRouter
from pydantic_ai_pax.runtime.workspace_trust import WorkspaceTrust, saved_trust, trust_command, trust_scope

if TYPE_CHECKING:
    from pydantic_ai.providers.openai_codex import OpenAICodexProvider


@dataclass(kw_only=True)
class ClaiProcess(Generic[DepsT]):
    """The services a Pax process holds once for every session it runs.

    Each `SessionRuntime` is built over the process it belongs to and joins its `live` sessions. The
    terminal runs one; `pax --web` opens more.
    """

    store: SettingsStore
    conversations: SqliteSessionStore
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
    trusts: dict[Path, WorkspaceTrust] = field(default_factory=dict[Path, WorkspaceTrust])
    """Shared folder trust per `TrustScope.key`, accessed on the event loop so every live session in scope shares it."""
    _trust_lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)
    """Serializes async store reads and writes across `resolve_trust`, `decide_trust`, and `run_trust_command`."""

    def trust_for(self, directory: Path, *, initial: WorkspaceTrust | None = None) -> WorkspaceTrust:
        """The shared `WorkspaceTrust` for `directory`'s scope; `initial` seeds its key (for `/reload`)."""
        if initial is not None:
            self.trusts[initial.scope.key] = initial
            return initial
        scope = trust_scope(directory)
        if (existing := self.trusts.get(scope.key)) is not None:
            return existing
        trust = WorkspaceTrust.from_scope(self.store, scope)
        self.trusts[scope.key] = trust
        return trust

    async def resolve_trust(self, directory: Path) -> WorkspaceTrust:
        """The shared `WorkspaceTrust` for `directory`'s scope, offloading filesystem and store reads off the loop."""
        scope = await to_thread.run_sync(trust_scope, directory)
        if (existing := self.trusts.get(scope.key)) is not None:
            return existing
        async with self._trust_lock:
            if (existing := self.trusts.get(scope.key)) is not None:
                return existing
            trust = await to_thread.run_sync(WorkspaceTrust.from_scope, self.store, scope)
            return self.trusts.setdefault(scope.key, trust)

    async def decide_trust(self, trust: WorkspaceTrust, *, trusted: bool) -> None:
        """Record a decision on `trust` and refresh any live non-repository descendant scopes in this process."""
        async with self._trust_lock:
            await to_thread.run_sync(partial(trust.decide, self.store, trusted=trusted))
            await self._refresh_descendants(trust)

    async def run_trust_command(self, trust: WorkspaceTrust, directory: Path, args: list[str]) -> str:
        """Run `/trust` for `trust`, synchronizing changes and descendant refreshes with concurrent lookups."""
        if not args:
            return await to_thread.run_sync(trust_command, trust, self.store, directory, args)
        async with self._trust_lock:
            reply = await to_thread.run_sync(trust_command, trust, self.store, directory, args)
            await self._refresh_descendants(trust)
            return reply

    async def _refresh_descendants(self, changed: WorkspaceTrust) -> None:
        if changed.scope.repository or not changed.scope.remembered:
            return
        descendants = [
            other
            for key, other in self.trusts.items()
            if not other.scope.repository and key != changed.scope.key and key.is_relative_to(changed.scope.key)
        ]
        if not descendants:
            return
        updates = await to_thread.run_sync(
            lambda: [(other, saved_trust(self.store, other.scope)) for other in descendants]
        )
        for other, saved in updates:
            other.apply_saved(saved)

    @classmethod
    def open(
        cls,
        store: SettingsStore | None = None,
        *,
        builtin_plugins: Sequence[PluginSettings] = (),
        launch_plugins: Sequence[str] = (),
    ) -> Self:
        """The process over `store`, by default the user's settings database, with `builtin_plugins` shipped.

        `launch_plugins` are loaded at the start of every session, enabled or not, without saving anything.
        """
        store = store or SettingsStore()
        conversations = SqliteSessionStore(database=store.path.with_name('sessions.db'))
        plugins = ProcessPlugins[DepsT](store=store, builtin=builtin_plugins, launch=launch_plugins)
        plugins.live = live = LiveRuntimes(conversations=conversations)
        return cls(
            store=store,
            conversations=conversations,
            tasks=TaskRouter(directory=Path(f'{conversations.database}.tasks'), step_store=step_store(conversations)),
            namer=SessionNamer(store=conversations),
            plugins=plugins,
            live=live,
        )
