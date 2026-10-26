"""The orchestrator: an agent of its own in the web UI, which runs workspaces for the user.

It has its own conversation and only the tools in `workspace_tools` and `session_tools`, plus
`ask_user_question`, answered in its own chat. The session's agent gets none of them. The sessions it
starts in workspaces are live sessions of this CLAI, listed in the sidebar like any other. Its conversation
is one JSON file in CLAI's state directory, so it survives a reload and a restart, and never shows
among the saved sessions.
"""

import asyncio
import functools
import hashlib
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from anyio import get_cancelled_exc_class, to_thread
from pydantic import ValidationError

from pydantic_ai import Agent, AgentStreamEvent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, repair_messages
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ask_user import AskUser, AskUserAnswer
from pydantic_clai2.builtin_plugins.web.live import BrowserAnswerer
from pydantic_clai2.builtin_plugins.web.protocol import (
    EventTranslator,
    Item,
    ItemAdded,
    ItemLog,
    OrchestratorEvent,
    OrchestratorMessage,
    OrchestratorState,
    ServerMessage,
    TurnEnded,
    TurnStarted,
    history_items,
    usage_info,
)
from pydantic_clai2.builtin_plugins.workspaces import configured
from pydantic_clai2.config.settings_store import state_dir
from pydantic_clai2.plugins import TurnOutcome

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.web.service import WebService
    from pydantic_clai2.builtin_plugins.web.workspace_tools import WorkspaceTools

HISTORY = 'orchestrator.json'
"""The conversation's file in CLAI's state directory."""

INSTRUCTIONS = """\
You are CLAI's orchestrator. You manage workspaces: checkouts of a repository on their own branch, each \
with a CLAI session of its own that you give tasks to. Create them, give their agents tasks, watch them, steer \
them, and report back to the user what they did. `workspace_list` shows the workspaces and the kinds you \
can make; `workspace_create` makes one (it does not start anything); `session_start` starts its session with \
a prompt; `session_wait` waits for the turn to end or for a question; `session_read` shows the latest \
items; `session_send` steers it with another prompt; `session_answer` answers its questions; \
`session_interrupt` stops a turn. Ask the user with `ask_user_question` when a choice is theirs. Never \
destroy a workspace unasked: before `workspace_destroy`, ask with `ask_user_question`, offering the \
options `Destroy ID` and `Keep ID`, and destroy it only if the user picks `Destroy ID`."""


@dataclass(frozen=True, kw_only=True)
class Saved:
    """The conversation as its file held it, and that file's revision, which a later save checks."""

    messages: list[ModelMessage]
    revision: str | None
    """A digest of the file's contents; `None` when there was none."""


class ChatChanged(Exception):
    """Another CLAI changed the conversation's file since this one read it."""


def _revision(path: Path) -> tuple[bytes | None, str | None]:
    try:
        data = path.read_bytes()
    except OSError:
        return None, None
    return data, hashlib.sha256(data).hexdigest()


def read_saved(path: Path) -> Saved:
    """The conversation in `path`; empty when there is none or it cannot be read."""
    data, revision = _revision(path)
    try:
        messages = [] if data is None else ModelMessagesTypeAdapter.validate_json(data)
    except ValidationError:
        messages = []
    return Saved(messages=messages, revision=revision)


def write_saved(path: Path, messages: list[ModelMessage], *, revision: str | None) -> Saved:
    """Replace `path` with `messages` through a new file only you can read; `ChatChanged` if it is not at `revision`."""
    if _revision(path)[1] != revision:
        raise ChatChanged
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ModelMessagesTypeAdapter.dump_json(messages)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f'.{HISTORY}.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'wb') as file:
            file.write(data)
        os.replace(name, path)
    except BaseException:
        os.unlink(name)
        raise
    return Saved(messages=messages, revision=hashlib.sha256(data).hexdigest())


class Orchestrator:
    """The orchestrator chat every connected browser shares: one conversation, one turn at a time."""

    def __init__(self, *, service: 'WebService') -> None:
        self.service = service
        self.host = service.host
        self._publish: Callable[[ServerMessage], None] = service.hub.publish
        self.answerer = BrowserAnswerer(publish=lambda message, _session: self._send(message), terminal=None)
        """Asks in the orchestrator chat alone, so only the user there answers, never the terminal."""
        self._translator = EventTranslator()
        self._turn = ItemLog()
        self._saved = Saved(messages=[], revision=None)
        """The conversation as last read or saved; the running turn started from it."""
        self._unsaved: list[ModelMessage] | None = None
        """A turn's conversation its file could not take, kept here until a later save succeeds."""
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        """The running turn was told to stop, or is saving: it is not cancelled again, so it always ends."""
        self._tools: WorkspaceTools | None = None
        self._agent: Agent[None, str] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def path(self) -> Path:
        return state_dir() / HISTORY

    async def load(self) -> Saved:
        """The conversation, read from its file again between turns, as another CLAI may have saved it since."""
        if self.running:
            return self._saved
        if self._unsaved is not None:
            return Saved(messages=self._unsaved, revision=self._saved.revision)
        self._saved = await to_thread.run_sync(read_saved, self.path)
        return self._saved

    def enabled(self) -> bool:
        """Whether the `workspaces` plugin is on, which the orchestrator's tools work over."""
        return configured(self.host.plugins) is not None

    async def state(self) -> OrchestratorState:
        saved = await self.load()
        return self._state(saved.messages, await self.host.default_model_name())

    def _state(self, messages: Sequence[ModelMessage], model: str | None) -> OrchestratorState:
        items = history_items(messages)
        if self.running:
            items += self._turn.items.values()
        return OrchestratorState(
            enabled=self.enabled(),
            running=self.running,
            model=model,
            items=items,
            questions=self.answerer.questions,
        )

    def _send(self, message: OrchestratorEvent) -> None:
        self._publish(OrchestratorMessage(message=message))

    def _observe(self, event: AgentStreamEvent) -> None:
        for message in self._translator.translate(event):
            self._turn.apply(message)
            self._send(message)

    def agent(self) -> Agent[None, str]:
        """The orchestrator's agent, made on first use: the workspace and session tools and `ask_user_question`."""
        if self._agent is None:
            from pydantic_clai2.builtin_plugins.web import session_tools
            from pydantic_clai2.builtin_plugins.web.workspace_tools import WorkspaceTools

            self._tools = WorkspaceTools(self.service)
            sessions = session_tools.SessionTools(self.host, publish=self._publish)
            self._agent = Agent(
                name='orchestrator',
                deps_type=type(None),
                instructions=[INSTRUCTIONS, session_tools.INSTRUCTIONS],
                capabilities=[
                    Capability[None](tools=[*self._tools.tools(), *sessions.tools()]),
                    AskUser[None](answerer=self.answerer),
                ],
            )
        return self._agent

    # What the browser does.

    async def send(self, text: str) -> None:
        """Start a turn with `text`; one runs at a time, and only while the `workspaces` plugin is on."""
        text = text.strip()
        if not text:
            raise ValueError('Type a message for the orchestrator.')
        if not self.enabled():
            raise RuntimeError('Turn the `workspaces` plugin on first.')
        saved = await self.load()
        if self.running:  # Checked after the read, as another request may have started a turn meanwhile.
            raise RuntimeError('The orchestrator is still working. Wait for it, or stop it.')
        self._task = asyncio.get_running_loop().create_task(self._run(text, saved), name='clai2-orchestrator')

    async def _run(self, prompt: str, saved: Saved) -> None:
        item = Item(id=f'orchestrator{time.monotonic_ns()}', kind='user', text=prompt)
        self._stopping = False
        self._turn.clear()
        self._turn.apply(ItemAdded(item=item))
        self._send(TurnStarted(item=item))
        # The history this turn starts from, which another CLAI may have changed since the page loaded.
        self._send(self._state(saved.messages, await self.host.default_model_name()))
        messages = saved.messages
        outcome: TurnOutcome = 'failed'
        error: str | None = None
        usage: RunUsage | None = None
        try:
            # The saved default, as a new session starts with, never one session's own choice.
            model = await self.host.default_model()
            if model is None:
                raise RuntimeError('Choose a model with /model first.')
            async with self.agent().iter(prompt, message_history=messages, model=model) as run:
                try:
                    async for node in run:
                        if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                            async with node.stream(run.ctx) as events:
                                async for event in events:
                                    self._observe(event)
                finally:
                    # What ran so far is kept, its unfinished tool calls closed, even when the turn is stopped.
                    messages = repair_messages(run.all_messages())
            usage = run.usage
            outcome = 'completed'
        except get_cancelled_exc_class():
            outcome = 'cancelled'
            raise
        except Exception as exc:  # noqa: BLE001 -- the turn fails in the chat, as a session turn does.
            error = str(exc) or type(exc).__name__
        finally:
            self._stopping = True
            self._turn.clear()
            try:
                if messages is not saved.messages:  # Nothing ran, such as without a model: nothing to save.
                    self._saved = await to_thread.run_sync(
                        functools.partial(write_saved, self.path, messages, revision=saved.revision)
                    )
                    self._unsaved = None
            except ChatChanged:
                # Not written over: what the other CLAI saved is the chat now.
                self._unsaved = None
                outcome, error = (
                    'failed',
                    'Another CLAI changed the orchestrator chat during this turn, so it was not kept.',
                )
            except OSError as exc:
                self._unsaved = messages
                outcome, error = 'failed', f'The orchestrator chat could not be saved: {exc}'
            self._send(TurnEnded(outcome=outcome, error=error, usage=None if usage is None else usage_info(usage)))

    def interrupt(self) -> bool:
        """Stop the running turn; `False` when none runs."""
        if self._task is None or self._task.done():
            return False
        if not self._stopping:
            self._stopping = True
            self._task.cancel()
        return True

    def answer(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        """Answer the orchestrator's open question, or decline it with `None`."""
        self.answerer.reply(request_id, answers)

    async def clear(self) -> None:
        """Start a new orchestrator chat, forgetting this one; its workspaces are kept."""
        if self.running:
            raise RuntimeError('Stop the orchestrator before starting a new chat.')
        await to_thread.run_sync(functools.partial(self.path.unlink, missing_ok=True))
        self._unsaved = None
        self._send(await self.state())

    async def close(self) -> None:
        """Stop a running turn and stop following the workspaces' sessions."""
        task = self._task
        if task is not None and self.interrupt():
            await asyncio.gather(task, return_exceptions=True)
        if self._tools is not None:
            await self._tools.close()
