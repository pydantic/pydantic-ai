"""Save conversations so /resume can restore them, and name them in the background.

Shell integration for persisted conversations, the browser, auxiliary naming, and importing
Claude Code and Codex sessions.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Generic, TypeVar

from anyio.to_thread import run_sync
from rich.console import Console

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationSummary,
    SavedConversation,
    SqliteConversationStore,
    conversation_text,
    ensure_inactive,
)
from pydantic_ai_harness.step_persistence.recovery import inspect_recovery
from pydantic_ai_harness.subagents import DelegationTasks
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.plugins import Plugin
from pydantic_clai2.runtime._session import Session, fork_summary, interrupted_history
from pydantic_clai2.runtime.imported_sessions import (
    IMPORT_SOURCES,
    ImportCatalog,
    ImportSource,
    find_import,
    import_source,
    merge,
    save_import,
)
from pydantic_clai2.runtime.launch import launch_session_id
from pydantic_clai2.runtime.session_naming import NamingClient, NamingResult, SessionNamer, generate_name
from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.menus.session_browser import SessionBrowser
from pydantic_clai2.ui.rendering.usage_report import usage_command

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
ResultT = TypeVar('ResultT')


class Sessions(Generic[DepsT, OutputT]):
    """One session's services, over the naming worker its process shares; nothing is module-global."""

    def __init__(
        self,
        *,
        session: Session[DepsT, OutputT],
        store: SqliteConversationStore,
        context: CommandContext,
        launched: str | None = None,
        namer: SessionNamer | None = None,
        tasks: DelegationTasks | None = None,
    ) -> None:
        """Bind services to the active shell and its validated settings.

        `namer` is the process's naming worker, shared with its other sessions; without one, it gets its own.
        `tasks` is the process's task owner, which an imported session's sub-agents are saved to, open or not.
        """
        self.session = session
        self.store = store
        self.tasks = tasks
        self.context = context
        self.launched = launched
        """The conversation ID launch options picked, carried across `/reload`."""
        self.quiet: Callable[[], AbstractAsyncContextManager[None]] = nullcontext
        """Entered to tell plugins about a background rename; the shell holds it until no turn or command runs."""
        self.naming = NamingClient(generate=self.generate, enabled=lambda: self.context.settings.session_namer)
        """How this session names the conversations it queues on `namer`."""
        self.namer = namer or SessionNamer(store=store)
        self.join()
        self._newest_name: tuple[str, str] | None = None
        """The conversation and title `named` heard last, which wins over one still waiting for the terminal."""
        self.on_resume: Callable[[Sequence[ModelMessage]], Awaitable[None]] | None = None
        """Told the restored history after each resume, so the shell can show it."""
        self.relocate: Callable[[str], Awaitable[None]] | None = None
        """Moves the session into a directory, so `/resume ID` continues one where it was saved; `clai2 --web` only."""
        self.show_live: Callable[[str], Awaitable[str | None]] = _not_shown
        """Shows in the terminal another live session holding a conversation, with the notice; `None` when none does."""

    def join(self) -> None:
        """Hear the titles `namer` announces; again once a closed session is live again, as after a failed `/reload`."""
        if self.named not in self.namer.listeners:
            self.namer.listeners.append(self.named)

    def close(self) -> None:
        """Stop hearing the titles `namer` announces, so nothing the process shares keeps this session alive."""
        self.namer.listeners.remove(self.named)

    @property
    def follows_workspace(self) -> bool:
        """Whether `/resume ID` moves the session into its own directory rather than refusing another one."""
        return self.relocate is not None

    async def resume(self, conversation_id: str, *, allow_other_workspace: bool = False, unarchive: bool = True) -> str:
        """Restore a saved session, then show its history.

        With `relocate`, the session first moves into the directory it was saved in, unless the
        `/resume` browser confirmed continuing it here; one that no longer exists raises `ValueError`.
        `unarchive=False` leaves an archived session archived, for a restore that continues in a fork.
        One another live session of this CLAI holds raises `ValueError`: a conversation is live in one session.
        In the terminal, that session is shown instead, and this one stays live as it is.
        """
        if (shown := await self.show_live(conversation_id)) is not None:
            return shown
        # Held from before moving directory, so no other session resumes it meanwhile.
        with self.session.claim(conversation_id):
            if self.relocate is not None and not allow_other_workspace:
                workspace = (await self.store.get(conversation_id=conversation_id)).summary.workspace
                if workspace != self.session.workspace:
                    await self.relocate(workspace)
            notice = await self.session.resume(
                conversation_id, allow_other_workspace=allow_other_workspace, unarchive=unarchive
            )
        await self.show_resumed()
        return notice

    async def show_resumed(self) -> None:
        """Show the restored history; a startup restore calls this once plugin renderers have loaded."""
        if self.on_resume is not None:
            await self.on_resume(self.session.messages)

    async def named(self, conversation_id: str, title: str) -> None:
        """Retitle the current conversation after background naming, once the terminal is free."""
        if conversation_id != self.session.conversation_id:
            return
        before = self.session.title
        self._newest_name = (conversation_id, title)
        async with self.quiet():
            # A rename in `/resume` while this waited wins over the generated name, as does a newer name
            # announced meanwhile. A save in the meantime may already have adopted it from the store
            # without telling plugins.
            if self.session.title in (before, title) and self._newest_name == (conversation_id, title):
                await self.session.renamed(conversation_id=conversation_id, title=title)

    async def generate(self, prompt: str) -> NamingResult | None:
        """Resolve credentials on the owning loop, without loading any coding plugins."""
        name = self.context.settings.session_namer_model
        if name:
            model = self.session.resolve_model(name)
            if isinstance(model, Awaitable):
                model = await model
        else:
            model = await self.session.resolved_model()
        if model is None:
            return None
        return await generate_name(model=model, prompt=prompt)

    async def usage(self, *, console: Console) -> str:
        """Report auxiliary tokens separately from retained foreground-history cost."""
        report = usage_command(self.session.messages, console=console)
        if self.session.summary.revision:
            saved = await self.store.get(conversation_id=self.session.summary.id)
            report += f'\nBackground naming: {saved.summary.naming_tokens:,} tokens (outside retained-history cost).'
        return report

    async def start(
        self,
        *,
        resume: str | None,
        resume_from: ImportSource | None = None,
        session_id: str | None = None,
        fork: bool = False,
    ) -> str:
        """Apply the launch options and return the notice to show.

        `resume` restores a conversation (`''` opens the browser), first importing it from
        `resume_from` when given. With `fork`, a restored one is copied to `session_id`, or a random
        ID; otherwise `session_id` names the new conversation. Raises `ValueError` for options
        Claude Code would refuse, before changing anything.
        A named `resume` is not shown yet: the caller records and shows it once plugins have loaded.
        """
        session_id = launch_session_id(resume=resume, session_id=session_id, fork=fork)
        if resume:
            if resume_from is not None:
                resume = await self.import_session(resume_from, resume)
            # A fork continues in the copy, so an archived original stays archived.
            notice = await self.session.resume(resume, record=False, unarchive=not fork)
        elif resume is not None:
            notice = await self.command([resume_from] if resume_from else [], unarchive=not fork)
        else:
            notice = ''
        if fork and notice:
            # Keep the resume notice: it warns about an interrupted session, which the copy no longer records.
            notice = f'{notice}\n{await self.session._fork(session_id)}'  # pyright: ignore[reportPrivateUsage]
        elif session_id is not None:
            await self.session.clear(session_id)
        if notice or session_id is not None:
            self.launched = self.session.conversation_id
        return notice

    @property
    def chosen(self) -> bool:
        """Whether launch options picked the current conversation; see `SessionStart.conversation_chosen`."""
        return self.launched == self.session.conversation_id

    async def import_session(self, source: ImportSource, native_id: str) -> str:
        """Copy a Claude Code or Codex session into the store by its own ID, returning its CLAI ID.

        As with a saved session's ID, one from another directory is refused, before anything is saved.
        """
        imported = await run_sync(find_import, source, native_id)
        self.session.check_workspace(imported.summary.workspace)
        return await save_import(self.store, imported, delegations=self.tasks)

    async def listing(
        self, *, query: str = '', limit: int = 200, archived: bool | None = False
    ) -> list[ConversationSummary]:
        """Saved conversations from every directory, newest first, as `/resume` lists them, queueing untitled ones for naming."""
        entries = await self.store.listing(query=query, limit=limit, archived=archived)
        self.namer.backfill(entries, client=self.naming)
        return entries

    async def saved(self, conversation_id: str) -> SavedConversation:
        """A saved conversation and its history, without switching to it."""
        return await self.store.get(conversation_id=conversation_id)

    async def rename(self, conversation_id: str, title: str) -> None:
        """Retitle a saved conversation as a rename in `/resume` does, telling plugins when it is the current one."""
        title = ' '.join(title.split())
        if not title:
            raise ValueError('A session title cannot be empty.')
        await self._rename_saved((await self.store.get(conversation_id=conversation_id)).summary, title)
        await self.session.renamed(conversation_id=conversation_id, title=title)

    async def _rename_saved(self, source: ConversationSummary, title: str) -> None:
        """Save `title` as a manual title, keeping one a naming model would otherwise replace."""
        if not await self.store.name(
            source=source, title=title, subtitle=source.subtitle, tags=source.tags, manual=True
        ):
            raise ValueError('Session changed. Refresh and rename again.')

    async def delete(self, conversation_id: str) -> None:
        """Delete a saved conversation, not the current one, with its Monty state; one running elsewhere is refused."""
        if conversation_id == self.session.conversation_id:
            raise ValueError('The current session cannot be deleted. Switch to another one first.')
        source = (await self.store.get(conversation_id=conversation_id)).summary
        await self._delete(source)

    async def fork(self, conversation_id: str) -> ConversationSummary:
        """Save a copy of a conversation under a new ID, as `--fork-session` does, without switching to it.

        A copy of an interrupted conversation keeps its unanswered tool calls closed, so none is replayed.
        The copy gets its own copy of the session's Monty state, as `/fork` and `--fork-session` give theirs.
        """
        saved = await self.store.get(conversation_id=conversation_id)
        messages = saved.messages
        if saved.summary.outcome in ('running', 'failed', 'cancelled'):
            messages = interrupted_history(messages)
        copy = await self.store.save(summary=fork_summary(saved.summary), messages=messages)
        if self.session.repl_states is not None:
            self.session.repl_states.copy(source=saved.summary.id, target=copy.id)
        return copy

    async def archive(self, conversation_id: str) -> ConversationSummary:
        """Hide a saved conversation from listings, deleting nothing; not the current one, nor one running elsewhere."""
        if conversation_id == self.session.conversation_id:
            raise ValueError('The current session cannot be archived. Switch to another one first.')
        # Reserved throughout, so no other live session of this CLAI opens it meanwhile; one holding it refuses.
        with self.session.claim(conversation_id):
            source = (await self.store.get(conversation_id=conversation_id)).summary
            try:
                await run_sync(ensure_inactive, source)
            except ConversationConflict as busy:
                raise ValueError(f'{busy} A running session cannot be archived.') from busy
            return await self.store.set_archived(conversation_id=conversation_id, archived=True)

    async def unarchive(self, conversation_id: str) -> ConversationSummary:
        """List an archived conversation again."""
        return await self.store.set_archived(conversation_id=conversation_id, archived=False)

    async def command(self, args: list[str], *, unarchive: bool = True) -> str:
        """Shared command/startup resolver; loading history never executes pending tools.

        `claude` or `codex` first imports that agent's session, or browses only its sessions.
        `unarchive=False` leaves the chosen session archived, as `--fork-session` does.
        """
        source = import_source(args[0]) if args else None
        if source is not None:
            args = args[1:]
        if len(args) > 1:
            raise ValueError('Usage: /resume [claude|codex] [SESSION-ID]')
        if args:
            conversation_id = await self.import_session(source, args[0]) if source else args[0]
            return await self.resume(conversation_id, unarchive=unarchive)
        return await self.browse(source, unarchive=unarchive)

    async def browse(self, source: ImportSource | None, *, unarchive: bool = True) -> str:
        """Pick from CLAI's sessions and not-yet-imported ones, or from one agent's sessions only."""
        entries: list[ConversationSummary] = [] if source else await self.store.listing(archived=None)
        self.namer.backfill([entry for entry in entries if entry.archived_at is None], client=self.naming)
        imports = await run_sync(ImportCatalog, (source,) if source else IMPORT_SOURCES)
        loop = asyncio.get_running_loop()

        def apply(action: Coroutine[object, object, ResultT]) -> ResultT:
            return asyncio.run_coroutine_threadsafe(action, loop).result()

        def shown(
            saved: list[ConversationSummary], imported: list[ConversationSummary], *, limit: int, archived: bool
        ) -> list[ConversationSummary]:
            # An archived copy still replaces its Claude Code or Codex original, so neither is listed.
            entries = merge(saved, imported, limit=limit)
            return entries if archived else [entry for entry in entries if entry.archived_at is None]

        def listing(query: str, limit: int, archived: bool) -> list[ConversationSummary]:
            saved: list[ConversationSummary] = [] if source else apply(self.store.listing(query=query, limit=limit))
            return shown(saved, imports.listing(query, limit), limit=limit, archived=archived)

        retitled: dict[str, str] = {}

        async def archive(source: ConversationSummary, archived: bool) -> None:
            await (self.archive(source.id) if archived else self.unarchive(source.id))

        async def rename(source: ConversationSummary, title: str) -> None:
            await self._rename_saved(source, title)
            retitled[source.id] = title

        def browse() -> str:
            # Resolve Git identities on the menu worker, not the application loop.
            return SessionBrowser(
                entries=shown(entries, imports.listing(), limit=200, archived=False),
                workspace=self.session.workspace,
                active_id=self.session.summary.id,
                refresh=listing,
                preview=lambda session_id: apply(self._preview(session_id, imports=imports)),
                delete=lambda source: apply(self._delete(source)),
                rename=lambda source, title: apply(rename(source, title)),
                archive=lambda source, archived: apply(archive(source, archived)),
                # Saved sessions have a revision; one that is not yet imported has none.
                importing=lambda entry: not entry.revision,
            ).run()

        selected = await run_worker(browse)
        # Plugins hear of a rename once the browser has closed, so none of them draws over it.
        if (title := retitled.get(self.session.conversation_id)) is not None:
            await self.session.renamed(conversation_id=self.session.conversation_id, title=title)
        if not selected:
            return ''
        if (imported := imports.get(selected)) is not None:
            selected = await save_import(self.store, imported, delegations=self.tasks)
        # Another session can be shown while this one runs a turn: that turn goes on.
        if (notice := await self.show_live(selected)) is not None:
            return notice
        if self.session.running:
            # The browser opens mid-turn, but a running conversation cannot be swapped out.
            return f'A turn is running. Enter /resume {selected} to restore that session once it ends.'
        return await self.resume(selected, allow_other_workspace=True, unarchive=unarchive)

    async def _delete(self, source: ConversationSummary) -> None:
        """Delete a saved session, then the `run_code` REPL code mode kept for it; one another live session holds is refused."""
        with self.session.claim(source.id):
            await self.store.delete(source=source)
        if self.session.repl_states is not None:
            self.session.repl_states.delete(source.id)

    async def _preview(self, conversation_id: str, *, imports: ImportCatalog) -> str:
        try:
            saved = await self.store.get(conversation_id=conversation_id)
        except LookupError:
            imported = imports.get(conversation_id)
            if imported is None:
                raise
            saved = SavedConversation(summary=imported.summary, messages=await run_sync(imported.messages))
        meta = saved.summary
        effects = ''
        if self.session.step_store and meta.run_id and meta.outcome in ('running', 'failed', 'cancelled'):
            recovery = await inspect_recovery(store=self.session.step_store, run_id=meta.run_id)
            unknown = ', '.join(f'{e.tool_name} ({e.tool_call_id})' for e in recovery.unresolved) or 'none recorded'
            effects = (
                f'Unknown effects: {unknown}\n'
                f'Completed tools (results may not be checkpointed): {", ".join(recovery.completed_tools) or "none"}\n'
                f'Failed tools (partial effects possible): {", ".join(recovery.failed_tools) or "none"}\n'
            )
            if meta.outcome == 'running' and recovery.latest is not None:
                saved.messages = recovery.latest.messages
        return (
            f'{meta.title}\n{meta.id}\n{meta.workspace}\n'
            f'State: {meta.outcome}. Model: {meta.model or "agent default"}. '
            f'Naming usage: {meta.naming_tokens} tokens.\n\n'
            + effects
            + conversation_text(list(reversed(saved.messages)))
        )


async def _not_shown(conversation_id: str) -> str | None:
    return None


class PersistencePlugin(Plugin):
    """The normal step-capture capability over the shell's configured store."""

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        store = self.host.conversation.step_store
        return () if store is None else (StepPersistence(store=store, capture_frontier=True),)
