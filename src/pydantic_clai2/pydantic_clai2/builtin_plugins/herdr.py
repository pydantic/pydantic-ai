"""Opt-in herdr state, session, metadata, and conversation-title integration."""

import asyncio
import logging
import os
import sqlite3
import sys

import anyio
from pydantic import JsonValue

from pydantic_ai import AgentRunResult, RunContext, ToolDefinition
from pydantic_ai.capabilities import ValidatedToolArgs, WrapRunHandler, WrapToolExecuteHandler
from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness.ask_user import AskUserAnsweredEvent, AskUserRequestedEvent
from pydantic_ai_harness.compaction import ContextUsageEvent
from pydantic_clai2.builtin_plugins._herdr_client import HerdrClient
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart, TurnEnd, TurnStart
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.ui.rendering.usage_report import session_usage

_LOGGER = logging.getLogger(__name__)


class _Reporter:
    def __init__(self, *, host: PluginHost[None], client: HerdrClient) -> None:
        self.host = host
        self.client = client
        self.depth = 0
        self.waiting: set[str] = set()
        self.last_state: tuple[str, str] | None = None
        self.last_session: tuple[str, str] | None = None
        self.last_metadata: dict[str, JsonValue] | None = None
        self.last_title: str | None = None
        self.context: str | None = None
        self.watcher: asyncio.Task[None] | None = None

    def report(self, message: str = 'thinking') -> None:
        state = 'blocked' if self.waiting else 'working' if self.depth else 'idle'
        message = 'awaiting input' if self.waiting else message if self.depth else 'ready'
        current = (state, message)
        if current == self.last_state:
            return
        lane = 'state' if self.last_state is None or state != self.last_state[0] else 'activity'
        self.client.submit(lane, 'pane.report_agent', {'state': state, 'message': message})
        self.last_state = current

    async def title(self) -> str | None:
        conversation = self.host.conversation
        if not isinstance(conversation, Session) or conversation.conversations is None:
            return None
        reference = (conversation.summary.id, str(conversation.conversations.database.resolve()))
        if reference != self.last_session:
            self.context = None
            self.client.submit(
                'session',
                'pane.report_agent_session',
                {'agent_session_id': reference[0], 'agent_session_path': reference[1]},
            )
            self.last_session = reference
        if not conversation.summary.revision:
            return None
        try:
            saved = await conversation.conversations.get(conversation_id=reference[0])
            # A resume/clear may happen while the database read is in flight.
            if conversation.summary.id == reference[0]:
                return saved.summary.title
        except (OSError, sqlite3.Error, ValueError):
            _LOGGER.debug('herdr session metadata unavailable', exc_info=True)
        return None

    async def refresh(self) -> None:
        title = await self.title()
        total = session_usage(self.host.conversation.messages).total
        tokens: dict[str, JsonValue] = {'model': self.host.status.model, 'tokens': f'{total.total_tokens:,}'}
        if self.context is not None:
            tokens['context'] = self.context
        metadata: dict[str, JsonValue] = {
            'applies_to_source': 'herdr:clai2',
            'ttl_ms': 86_400_000,
            'tokens': tokens,
        }
        if title is not None:
            metadata['title'] = title
        else:
            metadata['clear_title'] = True
        if metadata != self.last_metadata:
            self.client.submit('metadata', 'pane.report_metadata', metadata)
            self.last_metadata = metadata
        if title != self.last_title:
            self.client.submit('title', 'tab.rename', {'label': title})
            self.last_title = title

    async def watch(self) -> None:
        while True:
            await asyncio.sleep(2)
            await self.refresh()

    async def stop(self, event: SessionEnd) -> None:
        with anyio.CancelScope(shield=True):
            try:
                if self.watcher is not None:
                    self.watcher.cancel()
                    try:
                        await self.watcher
                    except asyncio.CancelledError:
                        pass
            finally:
                await anyio.to_thread.run_sync(self.client.close)

    async def start(self, event: SessionStart) -> None:
        self.report()
        await self.refresh()
        self.watcher = asyncio.create_task(self.watch(), name='clai2-herdr-titles')

    async def prompt(self, event: TurnStart) -> None:
        self.context = None
        await self.refresh()

    async def run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        self.depth += 1
        self.report()
        try:
            return await handler()
        finally:
            self.depth -= 1
            if not self.depth:
                self.waiting.clear()
            self.report()

    async def tool(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        self.report(f'running {call.tool_name}')
        try:
            return await handler(args)
        finally:
            self.report()

    async def question(self, ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
        self.waiting.add(event.request.id)
        self.report()

    async def answered(self, ctx: RunContext[None], event: AskUserAnsweredEvent) -> None:
        self.waiting.discard(event.request_id)
        self.report()

    async def usage(self, ctx: RunContext[None], event: ContextUsageEvent) -> None:
        self.context = f'{event.fraction:.0%}' if event.resolved else None

    async def finished(self, event: TurnEnd) -> None:
        self.waiting.clear()
        self.report()
        await self.refresh()


def activate(host: PluginHost[None]) -> None:
    """Report only inside a herdr pane. No IO or tasks outside herdr or on Windows."""
    socket_path = os.environ.get('HERDR_SOCKET_PATH')
    pane_id = os.environ.get('HERDR_PANE_ID')
    if os.environ.get('HERDR_ENV') != '1' or not socket_path or not pane_id or sys.platform == 'win32':
        return
    reporter = _Reporter(
        host=host,
        client=HerdrClient(socket_path=socket_path, pane_id=pane_id, tab_id=os.environ.get('HERDR_TAB_ID')),
    )
    # Failed loads also call this handler, so register it before asynchronous work.
    host.on('session_end')(reporter.stop)
    host.on('session_start')(reporter.start)
    host.on('turn_start')(reporter.prompt)
    host.on('turn_end')(reporter.finished)
    host.on('run')(reporter.run)
    host.on('tool_execute')(reporter.tool)
    host.on(AskUserRequestedEvent)(reporter.question)
    host.on(AskUserAnsweredEvent)(reporter.answered)
    host.on(ContextUsageEvent)(reporter.usage)
