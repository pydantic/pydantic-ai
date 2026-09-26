"""Shell integration for persisted conversations, the browser, and auxiliary naming."""

import asyncio
from collections.abc import Awaitable, Coroutine
from typing import Generic, TypeVar

from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationSummary,
    SqliteConversationStore,
    conversation_text,
)
from pydantic_ai_harness.step_persistence.naming import NamingResult, SessionNamer, generate_name
from pydantic_ai_harness.step_persistence.recovery import inspect_recovery
from rich.console import Console

from ._session import Session
from .command_context import CommandContext
from .menu_worker import run_worker
from .plugins import PluginHost
from .session_browser import SessionBrowser
from .usage_report import usage_command

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

        async def rename(source: ConversationSummary, title: str) -> None:
            if not await self.store.name(
                source=source, title=title, subtitle=source.subtitle, tags=source.tags, manual=True
            ):
                raise ValueError('Session changed. Refresh and rename again.')

        browser = SessionBrowser(
            entries=entries,
            workspace=self.session.workspace,
            active_id=self.session.summary.id,
            refresh=lambda query, limit: apply(self.store.listing(query=query, limit=limit)),
            preview=lambda session_id: apply(preview(session_id)),
            delete=lambda source: apply(self.store.delete(source=source)),
            rename=lambda source, title: apply(rename(source, title)),
        )
        selected = await run_worker(browser.run)
        if not selected:
            return ''
        return await self.session.resume(selected, allow_other_workspace=True)


def activate(host: PluginHost[None]) -> None:
    """Declare the normal step-capture capability over the shell's configured store."""
    if host.conversation.step_store is not None:
        host.add(StepPersistence(store=host.conversation.step_store, capture_frontier=True))
