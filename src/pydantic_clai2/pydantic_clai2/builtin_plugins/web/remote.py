"""The session as the browser drives it: what it sees on connect, what each action does, and the sidebar's sessions."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

from anyio import get_cancelled_exc_class, to_thread

from pydantic_ai import AgentRunResult, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.messages import UserContent, UserPromptPart
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserResponse, check_response
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationSummary,
    ensure_inactive,
)
from pydantic_ai_harness.subagents import DelegationTasks
from pydantic_clai2.builtin_plugins.web.live import BrowserAnswerer, Hub
from pydantic_clai2.builtin_plugins.web.protocol import (
    CompletionInfo,
    EventTranslator,
    Item,
    ItemLog,
    Notice,
    Project,
    SessionEntry,
    SessionInfo,
    SessionsChanged,
    SessionState,
    Snapshot,
    StatusChanged,
    StatusInfo,
    TitleChanged,
    Transcript,
    TurnEnded,
    TurnStarted,
    Usage,
    history_items,
    user_text,
)
from pydantic_clai2.plugins import CommandEnd, ConversationChanged, PluginHost, SessionCatalog, TurnOutcome
from pydantic_clai2.runtime.project_identity import ProjectIdentity, project_identity

STATUS_INTERVAL = 0.5
"""Seconds between status updates sent while a turn streams; turn and command ends always send one."""


class Remote:
    """One shell session, as every connected browser shares it.

    Everything a browser does goes through the prompt, as typing there would, so the terminal and
    the browser always drive the same session. What happens comes back through the plugin's moments
    and the run's events.
    """

    def __init__(self, *, host: PluginHost[None], hub: Hub, answerer: BrowserAnswerer) -> None:
        self.host = host
        self.hub = hub
        self.answerer = answerer
        self.running = False
        self._translator = EventTranslator()
        """One for the whole load, so item IDs stay unique across turns in a browser that keeps them all."""
        self._turn = ItemLog()
        self._status_sent = 0.0
        self._identities: dict[str, ProjectIdentity] = {}
        self._shown = host.conversation.conversation_id
        """The conversation browsers were last sent, so a retitle sends only the title."""

    # What the browser sees.

    def snapshot(self) -> Snapshot:
        conversation = self.host.conversation
        items = history_items(conversation.messages)
        if self.running:
            items += self._turn.items.values()
        return Snapshot(
            session=SessionInfo(
                id=conversation.conversation_id, title=conversation.title, workspace=self.host.status.workspace
            ),
            items=items,
            running=self.running,
            status=self.status(),
            questions=self.answerer.questions,
        )

    def status(self) -> StatusInfo:
        status = self.host.status
        return StatusInfo(
            model=status.model,
            activity=status.activity if self.running else 'ready',
            context_tokens=status.context_tokens,
            context_window=status.context_window,
            cost=None if status.cost is None else str(status.cost),
        )

    def publish_status(self) -> None:
        self._status_sent = time.monotonic()
        self.hub.publish(StatusChanged(status=self.status()))

    # What happens in the session, from the plugin's moments and capability.

    def foreground(self, ctx: RunContext[None]) -> bool:
        """A run of the conversation the shell holds, rather than a background `/fork` or a delegated task."""
        return ctx.conversation_id == self.host.conversation.conversation_id and DelegationTasks.child_id() is None

    def observe(self, event: AgentStreamEvent) -> None:
        for message in self._translator.translate(event):
            self._turn.apply(message)
            self.hub.publish(message)
        if time.monotonic() - self._status_sent >= STATUS_INTERVAL:
            self.publish_status()

    def run_started(self, prompt: str | Sequence[UserContent] | None) -> None:
        """A run of the held conversation started; a run woken by a finished task has no prompt to show."""
        self.running = True
        self._turn.clear()
        item = None
        if prompt is not None:
            item = Item(id=f'prompt{time.monotonic_ns()}', kind='user', text=user_text(UserPromptPart(prompt)))
        self.hub.publish(TurnStarted(item=item))
        self.publish_status()

    def run_ended(
        self, outcome: TurnOutcome, *, error: BaseException | None = None, usage: RunUsage | None = None
    ) -> None:
        self.running = False
        self._turn.clear()
        self.hub.publish(
            TurnEnded(
                outcome=outcome,
                error=None if error is None else str(error) or type(error).__name__,
                usage=None
                if usage is None
                else Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, requests=usage.requests),
            )
        )
        self.publish_status()
        self.hub.publish(SessionsChanged())

    def conversation_changed(self, event: ConversationChanged) -> None:
        if event.conversation_id == self._shown:
            self.hub.publish(TitleChanged(title=event.title))
        else:
            self._shown = event.conversation_id
            self.hub.publish(self.snapshot())
        self.hub.publish(SessionsChanged())

    def command_ended(self, event: CommandEnd) -> None:
        if event.error is not None:
            self.hub.publish(Notice(text=str(event.error), level='error', command=event.text))
        elif event.output:
            self.hub.publish(Notice(text=event.output, command=event.text))
        self.publish_status()

    # What the browser does.

    def send(self, text: str) -> None:
        """Enter a prompt or `/command` as if typed at the prompt."""
        text = text.strip()
        if not text:
            raise ValueError('Type a prompt or a /command.')
        if text.startswith('!'):
            raise ValueError('Shell lines (!) run only in the terminal.')
        self.host.send_input(text)

    async def complete(self, text: str) -> list[CompletionInfo]:
        return [
            CompletionInfo(
                text=completion.text,
                start=-completion.start_position,
                display=completion.display or completion.text,
                meta=completion.display_meta,
            )
            for completion in await self.host.complete(text)
        ]

    async def models(self) -> list[str]:
        """The saved models and chains `/model` offers, as `PROVIDER:NAME`."""
        return [completion.text for completion in await self.host.complete('/model ') if ':' in completion.text]

    def answer(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        """Answer an open question, or decline it with `None`; `ValueError` when the answers do not fit it."""
        request = self.answerer.request(request_id)
        response = AskUserResponse(cancelled=True) if answers is None else AskUserResponse(answers=tuple(answers))
        check_response(request, response)
        self.answerer.answer(request_id, response)

    # The sidebar's sessions.

    @property
    def catalog(self) -> SessionCatalog:
        if self.host.sessions is None:
            raise LookupError('This CLAI does not save sessions.')
        return self.host.sessions

    async def projects(self, query: str = '') -> list[Project]:
        """Saved sessions grouped by repository, the current one's project first."""
        entries = await self.catalog.listing(query=query)
        workspaces = {entry.workspace for entry in entries} | {self.host.status.workspace}
        unknown = [workspace for workspace in workspaces if workspace and workspace not in self._identities]
        if unknown:
            found = await to_thread.run_sync(lambda: [project_identity(workspace) for workspace in unknown])
            self._identities.update(zip(unknown, found))
        here = self._identity(self.host.status.workspace).key
        projects: dict[str, Project] = {}
        for entry in entries:
            identity = self._identity(entry.workspace)
            project = projects.setdefault(
                identity.key,
                Project(key=identity.key, name=identity.name, current=identity.key == here, sessions=[]),
            )
            project.sessions.append(self._entry(entry))
        return sorted(projects.values(), key=lambda project: not project.current)

    def _identity(self, workspace: str) -> ProjectIdentity:
        return self._identities.get(workspace) or ProjectIdentity(key=workspace, name=workspace, root=workspace)

    def _entry(self, summary: ConversationSummary) -> SessionEntry:
        current = summary.id == self.host.conversation.conversation_id
        state: SessionState = summary.outcome
        if current and self.running:
            state = 'running'
        elif summary.outcome == 'running':
            state = 'busy' if _busy(summary) else 'cancelled'
        return SessionEntry(
            id=summary.id,
            title=summary.title,
            workspace=summary.workspace,
            updated_at=summary.updated_at,
            state=state,
            messages=summary.message_count,
            model=summary.model,
            current=current,
        )

    async def transcript(self, conversation_id: str) -> Transcript:
        saved = await self.catalog.saved(conversation_id)
        return Transcript(
            id=saved.summary.id,
            title=saved.summary.title,
            workspace=saved.summary.workspace,
            items=history_items(saved.messages),
            openable=saved.summary.workspace == self.host.status.workspace,
        )

    def open(self, conversation_id: str) -> None:
        self.host.send_input(f'/resume {conversation_id}')

    def new(self) -> None:
        self.host.send_input('/new')

    async def rename(self, conversation_id: str, title: str) -> None:
        await self.catalog.rename(conversation_id, title)
        self.hub.publish(SessionsChanged())

    async def fork(self, conversation_id: str) -> str:
        """Copy a session and switch to the copy."""
        copy = await self.catalog.fork(conversation_id)
        self.hub.publish(SessionsChanged())
        if copy.workspace == self.host.status.workspace:
            self.open(copy.id)
        return copy.id

    async def delete(self, conversation_id: str) -> None:
        await self.catalog.delete(conversation_id)
        self.hub.publish(SessionsChanged())


def _busy(summary: ConversationSummary) -> bool:
    """Whether another live CLAI is running this saved session now, rather than having died mid-turn."""
    try:
        ensure_inactive(summary)
    except ConversationConflict:
        return True
    return False


@dataclass
class Observer(AbstractCapability[None]):
    """Hands the held conversation's runs and their events to the browser.

    Turns follow these runs rather than the turn moments, which a background `/fork` fires too.
    """

    remote: Remote

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        if not self.remote.foreground(ctx):
            return await handler()
        self.remote.run_started(ctx.prompt)
        try:
            result = await handler()
        except get_cancelled_exc_class():
            self.remote.run_ended('cancelled')
            raise
        except Exception as error:
            self.remote.run_ended('failed', error=error)
            raise
        self.remote.run_ended('completed', usage=result.usage)
        return result

    async def on_event(self, ctx: RunContext[None], *, event: AgentStreamEvent) -> None:
        if self.remote.foreground(ctx):
            self.remote.observe(event)
