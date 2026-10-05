"""Save conversations so /resume can restore them, and name them in the background.

Shell integration for persisted conversations, the browser, and auxiliary naming.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Generic, TypeVar

from rich.console import Console

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationSummary,
    SqliteConversationStore,
    conversation_text,
)
from pydantic_ai_harness.step_persistence.recovery import inspect_recovery
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.plugins import Plugin
from pydantic_clai2.runtime._session import Session
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
        self._launched: str | None = None
        self.quiet: Callable[[], AbstractAsyncContextManager[None]] = nullcontext
        """Entered to tell plugins about a background rename; the shell holds it until no turn or command runs."""
        self.namer = SessionNamer(
            store=store,
            generate=self.generate,
            enabled=lambda: self.context.settings.session_namer,
            on_named=self.named,
        )

    async def named(self, conversation_id: str, title: str) -> None:
        """Retitle the current conversation after background naming, once the terminal is free."""
        if conversation_id != self.session.conversation_id:
            return
        before = self.session.title
        async with self.quiet():
            # A rename in `/resume` while this waited wins over the generated name. A save in the
            # meantime may already have adopted it from the store without telling plugins.
            if self.session.title in (before, title):
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

    async def start(self, *, resume: str | None, session_id: str | None = None, fork: bool = False) -> str:
        """Apply the launch options and return the notice to show.

        `resume` restores a conversation (`''` opens the browser). With `fork`, a restored one is
        copied to `session_id`, or a random ID; otherwise `session_id` names the new conversation.
        """
        notice = await self.command([resume] if resume else []) if resume is not None else ''
        if fork and notice:
            # Keep the resume notice: it warns about an interrupted session, which the copy no longer records.
            notice = f'{notice}\n{await self.session.fork(session_id)}'
        elif session_id is not None:
            await self.session.clear(session_id)
        if notice or session_id is not None:
            self._launched = self.session.conversation_id
        return notice

    @property
    def chosen(self) -> bool:
        """Whether launch options picked the current conversation; see `SessionStart.conversation_chosen`."""
        return self._launched == self.session.conversation_id

    async def command(self, args: list[str]) -> str:
        """Shared command/startup resolver; loading history never executes pending tools."""
        if len(args) > 1:
            raise ValueError('Usage: /resume [SESSION-ID]')
        if args:
            return await self.session.resume(args[0])
        entries = await self.store.listing()
        self.namer.backfill(entries)
        loop = asyncio.get_running_loop()

        def apply(action: Coroutine[object, object, ResultT]) -> ResultT:
            return asyncio.run_coroutine_threadsafe(action, loop).result()

        async def preview(conversation_id: str) -> str:
            saved = await self.store.get(conversation_id=conversation_id)
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

        retitled: dict[str, str] = {}

        async def rename(source: ConversationSummary, title: str) -> None:
            if not await self.store.name(
                source=source, title=title, subtitle=source.subtitle, tags=source.tags, manual=True
            ):
                raise ValueError('Session changed. Refresh and rename again.')
            retitled[source.id] = title

        def browse() -> str:
            # Resolve Git identities on the menu worker, not the application loop.
            return SessionBrowser(
                entries=entries,
                workspace=self.session.workspace,
                active_id=self.session.summary.id,
                refresh=lambda query, limit: apply(self.store.listing(query=query, limit=limit)),
                preview=lambda session_id: apply(preview(session_id)),
                delete=lambda source: apply(self.store.delete(source=source)),
                rename=lambda source, title: apply(rename(source, title)),
            ).run()

        selected = await run_worker(browse)
        # Plugins hear of a rename once the browser has closed, so none of them draws over it.
        if (title := retitled.get(self.session.conversation_id)) is not None:
            await self.session.renamed(conversation_id=self.session.conversation_id, title=title)
        if not selected:
            return ''
        return await self.session.resume(selected, allow_other_workspace=True)


class PersistencePlugin(Plugin):
    """The normal step-capture capability over the shell's configured store."""

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        store = self.host.conversation.step_store
        return () if store is None else (StepPersistence(store=store, capture_frontier=True),)
