"""Opt-in herdr state, session, metadata, and conversation-title integration."""

import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio
from pydantic import JsonValue

from pydantic_ai import AgentRunResult, RunContext, ToolDefinition
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    ValidatedToolArgs,
    WrapRunHandler,
    WrapToolExecuteHandler,
    on_event,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness.ask_user import AskUserAnsweredEvent, AskUserRequestedEvent
from pydantic_ai_harness.compaction import ContextUsageEvent
from pydantic_clai2.builtin_plugins.herdr_client import SOURCE, HerdrClient
from pydantic_clai2.config.settings_store import config_dir
from pydantic_clai2.plugins import (
    ConversationChanged,
    NoSettings,
    Plugin,
    PluginHost,
    SessionEnd,
    SessionStart,
    TurnEnd,
    TurnStart,
)
from pydantic_clai2.runtime.session import Session
from pydantic_clai2.ui.rendering.usage_report import session_usage

_UNSAFE_ARGUMENT = re.compile(r"['\x00-\x1f\x7f-\x9f]")
"""Herdr rejects the whole report when a resume argument contains an apostrophe or a control character."""


def _resume_argv(conversation_id: str, settings_database: Path | None, *, saved: bool) -> list[JsonValue] | None:
    """The command herdr runs to reopen this conversation after a restart, if its settings path is known.

    Herdr keeps the last command it accepted, so an unsaved conversation needs one too: `--resume` cannot open it
    yet, and `--session-id` refuses an ID once it is saved.
    """
    if settings_database is None:
        return None
    argv = ['clai2', '--resume' if saved else '--session-id', conversation_id]
    if settings_database.resolve() != (config_dir() / 'config.db').resolve():
        argv += ['--database', str(settings_database.resolve())]
    if any(_UNSAFE_ARGUMENT.search(argument) for argument in argv):
        return None
    return list[JsonValue](argv)


class _Reporter:
    def __init__(self, *, host: PluginHost[None], client: HerdrClient) -> None:
        self.host = host
        self.client = client
        self.depth = 0
        self.waiting: set[str] = set()
        self.last_state: tuple[str, str] | None = None
        self.last_session: tuple[str, Path, bool] | None = None
        self.last_metadata: dict[str, JsonValue] | None = None
        self.last_title: str | None = None
        self.context: str | None = None

    def report(self, message: str = 'thinking') -> None:
        state = 'blocked' if self.waiting else 'working' if self.depth else 'idle'
        message = 'awaiting input' if self.waiting else message if self.depth else 'ready'
        current = (state, message)
        if current == self.last_state:
            return
        lane = 'state' if self.last_state is None or state != self.last_state[0] else 'activity'
        self.client.submit(lane, 'pane.report_agent', {'state': state, 'message': message})
        self.last_state = current

    def title(self) -> str | None:
        conversation = self.host.conversation
        if isinstance(conversation, Session) and conversation.conversations is not None:
            # Saving the first prompt makes the conversation resumable, so it is part of the reference.
            reference = (
                conversation.conversation_id,
                conversation.conversations.database.resolve(),
                bool(conversation.summary.revision),
            )
            if reference != self.last_session:
                conversation_id, database, saved = reference
                if self.last_session is None or self.last_session[:2] != reference[:2]:
                    self.context = None
                params: dict[str, JsonValue] = {
                    'agent_session_id': conversation_id,
                    'agent_session_path': str(database),
                }
                resume_argv = _resume_argv(conversation_id, self.host.settings_database, saved=saved)
                if resume_argv is not None:
                    params['resume_argv'] = resume_argv
                self.client.submit('session', 'pane.report_agent_session', params)
                self.last_session = reference
        return conversation.title

    def refresh(self) -> None:
        title = self.title()
        total = session_usage(self.host.conversation.messages).total
        tokens: dict[str, JsonValue] = {
            'model': self.host.status.model,
            'tokens': f'{total.total_tokens:,}',
            'context': self.context,
        }
        metadata: dict[str, JsonValue] = {
            'applies_to_source': SOURCE,
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

    async def stop(self, event: SessionEnd) -> None:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(self.client.close)

    async def start(self, event: SessionStart) -> None:
        self.report()
        self.refresh()

    async def prompt(self, event: TurnStart) -> None:
        self.context = None
        self.refresh()

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
        # Mid-turn, so the sidebar's context figure keeps up during a long run.
        self.refresh()

    async def finished(self, event: TurnEnd) -> None:
        self.waiting.clear()
        self.report()
        self.refresh()


@dataclass
class _Reporting(AbstractCapability[None]):
    """Report the run, its tool calls, open questions, and context usage as they happen."""

    reporter: _Reporter

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        return await self.reporter.run(ctx, handler=handler)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        return await self.reporter.tool(ctx, call=call, tool_def=tool_def, args=args, handler=handler)

    @on_event(AskUserRequestedEvent)
    async def _question(self, ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
        await self.reporter.question(ctx, event)

    @on_event(AskUserAnsweredEvent)
    async def _answered(self, ctx: RunContext[None], event: AskUserAnsweredEvent) -> None:
        await self.reporter.answered(ctx, event)

    @on_event(ContextUsageEvent)
    async def _usage(self, ctx: RunContext[None], event: ContextUsageEvent) -> None:
        await self.reporter.usage(ctx, event)


class HerdrPlugin(Plugin):
    """Report only inside a herdr pane. No IO or tasks outside herdr, on Windows, or without a terminal."""

    def __init__(self, host: PluginHost[None], settings: NoSettings) -> None:
        super().__init__(host, settings)
        socket_path = os.environ.get('HERDR_SOCKET_PATH')
        pane_id = os.environ.get('HERDR_PANE_ID')
        # The pane shows the session in the terminal; another session in this process, such as one the
        # browser opened, would report over it.
        self.reporter = (
            None
            if os.environ.get('HERDR_ENV') != '1'
            or not socket_path
            or not pane_id
            or sys.platform == 'win32'
            or not host.console.is_terminal
            else _Reporter(
                host=host,
                client=HerdrClient(socket_path=socket_path, pane_id=pane_id, tab_id=os.environ.get('HERDR_TAB_ID')),
            )
        )

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return () if self.reporter is None else (_Reporting(self.reporter),)

    async def on_session_start(self, event: SessionStart) -> None:
        if self.reporter is not None:
            await self.reporter.start(event)

    async def on_turn_start(self, event: TurnStart) -> None:
        if self.reporter is not None:
            await self.reporter.prompt(event)

    async def on_turn_end(self, event: TurnEnd) -> None:
        if self.reporter is not None:
            await self.reporter.finished(event)

    async def on_conversation_changed(self, event: ConversationChanged) -> None:
        if self.reporter is not None:
            self.reporter.refresh()

    async def on_session_end(self, event: SessionEnd) -> None:
        # Failed loads also call this, so it must tolerate a session that never started.
        if self.reporter is not None:
            await self.reporter.stop(event)
