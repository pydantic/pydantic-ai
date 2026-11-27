"""Save conversations so /resume can restore them, and name them in the background.

Shell integration for persisted conversations, the browser, auxiliary naming, and importing
Claude Code and Codex sessions.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from functools import partial
from typing import Generic, TypeVar

from anyio.to_thread import run_sync
from rich.console import Console

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationOrder,
    ConversationSummary,
    SavedConversation,
    SqliteConversationStore,
    conversation_text,
    ensure_inactive,
)
from pydantic_ai_harness.step_persistence.recovery import inspect_recovery
from pydantic_ai_harness.sub_agents import DelegationTasks
from pydantic_ai_pax.cli.command_context import CommandContext
from pydantic_ai_pax.plugins import Plugin
from pydantic_ai_pax.runtime.imported_sessions import (
    IMPORT_SOURCES,
    ImportCatalog,
    ImportSource,
    find_import,
    import_source,
    merge,
    save_import,
    source_of,
)
from pydantic_ai_pax.runtime.launch import launch_session_id
from pydantic_ai_pax.runtime.session import Session, fork_summary, interrupted_history
from pydantic_ai_pax.runtime.session_naming import NamingClient, NamingResult, SessionNamer, generate_name
from pydantic_ai_pax.ui.menus.menu_worker import run_worker
from pydantic_ai_pax.ui.menus.session_browser import SessionBrowser
from pydantic_ai_pax.ui.rendering.usage_report import usage_command

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
        session.on_resumed = self.show_resumed
        self.store = store
        self.tasks = tasks
        self.context = context
        self.launched = launched
        """The conversation ID launch options picked, carried across `/reload`."""
        self.quiet: Callable[[], AbstractAsyncContextManager[None]] = nullcontext
        """Entered to tell plugins about a background rename; the shell holds it until no turn or command runs."""
        self.naming = NamingClient(
            generate=self.generate, enabled=lambda: self.context.settings.session_namer, may_see=self.naming_may_see
        )
        """How this session names the conversations it queues on `namer`."""
        self.namer = namer or SessionNamer(store=store)
        self.join()
        self._newest_name: tuple[str, str] | None = None
        """The conversation and title `named` heard last, which wins over one still waiting for the terminal."""
        self.on_resume: Callable[[Sequence[ModelMessage]], Awaitable[None]] | None = None
        """Told the restored history after each resume, so the shell can show it."""
        self.relocate: Callable[[str], Awaitable[None]] | None = None
        """Moves the session into a directory, so `/resume ID` continues one where it was saved; `pax --web` only."""
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
        One another live session of this Pax holds raises `ValueError`: a conversation is live in one session.
        In the terminal, that session is shown instead, and this one stays live as it is.
        """
        if (shown := await self.show_live(conversation_id)) is not None:
            return shown
        # One another process runs continues in its existing fork, if any, which is then the one to show or hold.
        continuation = await self.session.continuation(conversation_id)
        target = continuation.conversation_id
        if target != conversation_id and (shown := await self.show_live(target)) is not None:
            return shown
        # Held from before moving directory, so no other session resumes it meanwhile.
        with self.session.claim(target):
            if self.relocate is not None and not allow_other_workspace:
                workspace = (await self.store.get(conversation_id=target)).summary.workspace
                if workspace != self.session.workspace:
                    await self.relocate(workspace)
            notice = await self.session.resume(
                conversation_id,
                allow_other_workspace=allow_other_workspace,
                unarchive=unarchive,
                continuation=continuation,
            )
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

    def naming_may_see(self, summary: ConversationSummary) -> bool:
        """Whether a naming request may carry `summary`'s excerpts or title.

        When naming is on only by default, it uses only sessions held with the current model, so a model is never
        sent another conversation's text it was not already sent. Saving `sessions.naming true` or choosing a
        `sessions.naming_model` opts into naming every session, as it did while naming was opt-in.
        """
        settings = self.context.settings
        if 'session_namer' in settings.model_fields_set or settings.session_namer_model is not None:
            return True
        if summary.model != self.session.model:
            return False
        # No recorded model means Pax's default, unless the session came from another agent: an import, or a
        # fork of one, which leads its tags with the source and whose model is unknown.
        return summary.model is not None or not (summary.tags and import_source(summary.tags[0]))

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
        """Copy a Claude Code or Codex session into the store by its own ID, returning its Pax ID.

        As with a saved session's ID, one from another directory is refused, before anything is saved.
        """
        imported = await run_sync(find_import, source, native_id)
        self.session.check_workspace(imported.summary.workspace)
        return await save_import(self.store, imported, delegations=self.tasks)

    async def listing(
        self,
        *,
        query: str = '',
        limit: int = 200,
        archived: bool | None = False,
        order_by: ConversationOrder = 'updated_at',
    ) -> list[ConversationSummary]:
        """Saved conversations from every directory, newest first, as `/resume` lists them, queueing untitled ones for naming.

        `order_by='created_at'` pages by creation instead of activity, as the web sidebar lists them.
        """
        entries = await self.store.listing(query=query, limit=limit, archived=archived, order_by=order_by)
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
            await run_sync(partial(self.session.repl_states.copy, source=saved.summary.id, target=copy.id))
        return copy

    async def archive(self, conversation_id: str) -> ConversationSummary:
        """Hide a saved conversation from listings, deleting nothing; not the current one, nor one running elsewhere."""
        if conversation_id == self.session.conversation_id:
            raise ValueError('The current session cannot be archived. Switch to another one first.')
        # Reserved throughout, so no other live session of this Pax opens it meanwhile; one holding it refuses.
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
        """Pick from Pax's sessions and not-yet-imported ones, listing only one agent's sessions at first with `source`."""
        entries = await self.store.listing(archived=None)
        self.namer.backfill([entry for entry in entries if entry.archived_at is None], client=self.naming)
        entries = await self._with_own_copies(entries, query='', limit=200, source=source)
        imports = await run_sync(ImportCatalog, IMPORT_SOURCES)
        loop = asyncio.get_running_loop()

        def apply(action: Coroutine[object, object, ResultT]) -> ResultT:
            return asyncio.run_coroutine_threadsafe(action, loop).result()

        def on_disk(query: str, limit: int) -> list[ConversationSummary]:
            found = imports.listing(query, limit)
            if source is None:
                return found
            # The named agent's own newest are listed too, however many newer ones other sources have.
            ids = {summary.id for summary in found}
            return [
                *found,
                *(summary for summary in imports.listing(query, limit, sources=(source,)) if summary.id not in ids),
            ]

        def listing(query: str, limit: int, archived: bool) -> list[ConversationSummary]:
            page = apply(self.store.listing(query=query, limit=limit))
            saved = apply(self._with_own_copies(page, query=query, limit=limit, source=source))
            return self._browser_entries(saved, on_disk(query, limit), limit=limit, archived=archived, source=source)

        retitled: dict[str, str] = {}

        async def archive(source: ConversationSummary, archived: bool) -> None:
            await (self.archive(source.id) if archived else self.unarchive(source.id))

        async def rename(source: ConversationSummary, title: str) -> None:
            await self._rename_saved(source, title)
            retitled[source.id] = title

        def browse() -> str:
            # Resolve Git identities on the menu worker, not the application loop.
            browser = SessionBrowser(
                entries=self._browser_entries(entries, on_disk('', 200), limit=200, archived=False, source=source),
                workspace=self.session.workspace,
                active_id=self.session.summary.id,
                refresh=listing,
                preview=lambda session_id: apply(self._preview(session_id, imports=imports)),
                delete=lambda source: apply(self._delete(source)),
                rename=lambda source, title: apply(rename(source, title)),
                archive=lambda source, archived: apply(archive(source, archived)),
                # Saved sessions have a revision; one that is not yet imported has none.
                importing=lambda entry: not entry.revision,
                changed_since_import=imports.changed_since_import,
            )
            # Naming an agent lists its sessions first; `f` reaches the others.
            browser.select_source(source)
            return browser.run()

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

    async def _with_own_copies(
        self, saved: list[ConversationSummary], *, query: str, limit: int, source: ImportSource | None
    ) -> list[ConversationSummary]:
        """`saved`, the store's newest `limit`, with `source`'s own newest `limit` copies from the pages after it.

        Naming an agent lists its own newest originals past the overall cut, so their copies must be there
        too, or an original Pax already imported would be offered to import again in its copy's place.
        """
        if source is None:
            return saved
        entries, page, offset = list(saved), saved, len(saved)
        own = sum(source_of(entry) == source for entry in saved)
        while own < limit and len(page) == limit:
            page = await self.store.listing(query=query, limit=limit, offset=offset)
            offset += len(page)
            # Only the agent's own are kept from later pages; the rest stay past the cut.
            mine = [entry for entry in page if source_of(entry) == source]
            entries += mine
            own += len(mine)
        return entries

    @staticmethod
    def _browser_entries(
        saved: list[ConversationSummary],
        imported: list[ConversationSummary],
        *,
        limit: int,
        archived: bool,
        source: ImportSource | None = None,
    ) -> list[ConversationSummary]:
        """The newest `limit` of `saved` merged with `imported`, keeping archived ones only when `archived`.

        With `source`, that agent's own newest `limit` are kept too, so newer sessions elsewhere cannot crowd them out.
        """
        # An archived copy still replaces its Claude Code or Codex original, so neither is listed.
        entries: list[ConversationSummary] = []
        own = 0
        for index, entry in enumerate(merge(saved, imported, limit=len(saved) + len(imported))):
            mine = source is not None and source_of(entry) == source
            own += mine
            if index < limit or (mine and own <= limit):
                entries.append(entry)
        return entries if archived else [entry for entry in entries if entry.archived_at is None]

    async def _delete(self, source: ConversationSummary) -> None:
        """Delete a saved session, then the `run_code` REPL code mode kept for it; one another live session holds is refused."""
        with self.session.claim(source.id):
            await self.store.delete(source=source)
        if self.session.repl_states is not None:
            await run_sync(self.session.repl_states.delete, source.id)

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
