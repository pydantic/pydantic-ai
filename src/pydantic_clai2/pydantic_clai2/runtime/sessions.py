"""Save conversations so /resume can restore them, and name them in the background.

Shell integration for persisted conversations, the browser, auxiliary naming, and importing
Claude Code and Codex sessions.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Generic, TypeVar

from anyio.to_thread import run_sync
from rich.console import Console

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationSummary,
    SavedConversation,
    SqliteConversationStore,
    conversation_text,
)
from pydantic_ai_harness.step_persistence.recovery import inspect_recovery
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.plugins import Plugin
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.imported_sessions import (
    IMPORT_SOURCES,
    ImportCatalog,
    ImportSource,
    find_import,
    import_source,
    merge,
    save_import,
)
from pydantic_clai2.runtime.session_naming import NamingResult, SessionNamer, generate_name
from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.menus.session_browser import SessionBrowser
from pydantic_clai2.ui.rendering.usage_report import usage_command

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
ResultT = TypeVar('ResultT')


class Sessions(Generic[DepsT, OutputT]):
    """One application's services. No worker, registration, or selection is global."""

    def __init__(
        self, *, session: Session[DepsT, OutputT], store: SqliteConversationStore, context: CommandContext
    ) -> None:
        """Bind services to the active shell and its validated settings."""
        self.session = session
        self.store = store
        self.context = context
        self.namer = SessionNamer(
            store=store, generate=self.generate, enabled=lambda: self.context.settings.session_namer
        )
        self.on_resume: Callable[[Sequence[ModelMessage]], Awaitable[None]] | None = None
        """Told the restored history after each resume, so the shell can show it."""

    async def resume(self, conversation_id: str, *, allow_other_workspace: bool = False) -> str:
        """Restore a saved session, then show its history."""
        notice = await self.session.resume(conversation_id, allow_other_workspace=allow_other_workspace)
        if self.on_resume is not None:
            await self.on_resume(self.session.messages)
        return notice

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

    async def import_session(self, source: ImportSource, native_id: str) -> str:
        """Copy a Claude Code or Codex session into the store by its own ID, returning its CLAI ID.

        As with a saved session's ID, one from another directory is refused, before anything is saved.
        """
        imported = await run_sync(find_import, source, native_id)
        self.session.check_workspace(imported.summary.workspace)
        return await save_import(self.store, imported)

    async def command(self, args: list[str]) -> str:
        """Shared command/startup resolver; loading history never executes pending tools.

        `claude` or `codex` first imports that agent's session, or browses only its sessions.
        """
        source = import_source(args[0]) if args else None
        if source is not None:
            args = args[1:]
        if len(args) > 1:
            raise ValueError('Usage: /resume [claude|codex] [SESSION-ID]')
        if args:
            return await self.resume(await self.import_session(source, args[0]) if source else args[0])
        return await self.browse(source)

    async def browse(self, source: ImportSource | None) -> str:
        """Pick from CLAI's sessions and not-yet-imported ones, or from one agent's sessions only."""
        entries: list[ConversationSummary] = [] if source else await self.store.listing()
        self.namer.backfill(entries)
        imports = await run_sync(ImportCatalog, (source,) if source else IMPORT_SOURCES)
        loop = asyncio.get_running_loop()

        def apply(action: Coroutine[object, object, ResultT]) -> ResultT:
            return asyncio.run_coroutine_threadsafe(action, loop).result()

        def listing(query: str, limit: int) -> list[ConversationSummary]:
            saved: list[ConversationSummary] = [] if source else apply(self.store.listing(query=query, limit=limit))
            return merge(saved, imports.listing(query, limit), limit=limit)

        async def rename(source: ConversationSummary, title: str) -> None:
            if not await self.store.name(
                source=source, title=title, subtitle=source.subtitle, tags=source.tags, manual=True
            ):
                raise ValueError('Session changed. Refresh and rename again.')

        def browse() -> str:
            # Resolve Git identities on the menu worker, not the application loop.
            return SessionBrowser(
                entries=merge(entries, imports.listing(), limit=200),
                workspace=self.session.workspace,
                active_id=self.session.summary.id,
                refresh=listing,
                preview=lambda session_id: apply(self._preview(session_id, imports=imports)),
                delete=lambda source: apply(self.store.delete(source=source)),
                rename=lambda source, title: apply(rename(source, title)),
                # Saved sessions have a revision; one that is not yet imported has none.
                importing=lambda entry: not entry.revision,
            ).run()

        selected = await run_worker(browse)
        if not selected:
            return ''
        if (imported := imports.get(selected)) is not None:
            selected = await save_import(self.store, imported)
        if self.session.running:
            # The browser opens mid-turn, but a running conversation cannot be swapped out.
            return f'A turn is running. Enter /resume {selected} to restore that session once it ends.'
        return await self.resume(selected, allow_other_workspace=True)

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


class PersistencePlugin(Plugin):
    """The normal step-capture capability over the shell's configured store."""

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        store = self.host.conversation.step_store
        return () if store is None else (StepPersistence(store=store, capture_frontier=True),)
