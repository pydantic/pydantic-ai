"""The orchestrator: an agent of its own in the web UI, which runs workspaces for the user.

It has its own conversation and only the tools in `workspace_tools`, plus `ask_user_question`, answered
in its own chat. The session's agent gets none of them. Its conversation is one JSON file in CLAI's state
directory, so it survives a reload and a restart, and never shows among the saved sessions.
"""

import asyncio
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from anyio import get_cancelled_exc_class
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
    Usage,
    history_items,
)
from pydantic_clai2.config.settings_store import state_dir
from pydantic_clai2.plugins import PluginHost, TurnOutcome

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.web.workspace_tools import WorkspaceTools

HISTORY = 'orchestrator.json'
"""The conversation's file in CLAI's state directory."""

INSTRUCTIONS = """\
You are CLAI's orchestrator. You manage workspaces: checkouts of a repository on their own branch, each \
with its own CLAI session that you give tasks to. Create them, give their agents tasks, watch them, steer \
them, and report back to the user what they did. `workspace_list` shows the workspaces and the kinds you \
can make; `workspace_create` makes one (it does not start anything); `session_start` starts its CLAI with \
a prompt; `session_wait` waits for the turn to end or for a question; `session_read` shows the latest \
items; `session_send` steers it with another prompt; `session_answer` answers its questions; \
`session_interrupt` stops a turn. Ask the user with `ask_user_question` when a choice is theirs. Never \
destroy a workspace unasked: before `workspace_destroy`, ask with `ask_user_question`, offering the \
options `Destroy ID` and `Keep ID`, and destroy it only if the user picks `Destroy ID`."""


class Orchestrator:
    """The orchestrator chat every connected browser shares: one conversation, one turn at a time."""

    def __init__(self, *, host: PluginHost[None], publish: Callable[[ServerMessage], None]) -> None:
        self.host = host
        self._publish = publish
        self.answerer = BrowserAnswerer(publish=self._send, terminal=None)
        """Asks in the orchestrator chat alone, so only the user there answers, never the terminal."""
        self._translator = EventTranslator()
        self._turn = ItemLog()
        self._messages: list[ModelMessage] | None = None
        self._task: asyncio.Task[None] | None = None
        self._tools: WorkspaceTools | None = None
        self._agent: Agent[None, str] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def path(self) -> Path:
        return state_dir() / HISTORY

    @property
    def messages(self) -> list[ModelMessage]:
        """The conversation, read from its file on first use; empty when there is none or it cannot be read."""
        if self._messages is None:
            try:
                self._messages = ModelMessagesTypeAdapter.validate_json(self.path.read_bytes())
            except (OSError, ValidationError):
                self._messages = []
        return self._messages

    def _save(self, messages: list[ModelMessage]) -> None:
        """Keep `messages`, replacing the file through a new one only you can read."""
        self._messages = messages
        folder = self.path.parent
        folder.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=folder, prefix=f'.{HISTORY}.', suffix='.tmp')
        with os.fdopen(descriptor, 'wb') as file:
            file.write(ModelMessagesTypeAdapter.dump_json(messages))
        os.replace(name, self.path)

    def enabled(self) -> bool:
        """Whether the `workspaces` plugin is on, which the orchestrator's tools work over."""
        # Imported here: the `workspaces` plugin imports this package for the token header.
        from pydantic_clai2.builtin_plugins.workspaces import configured

        return configured(self.host.plugins) is not None

    def state(self) -> OrchestratorState:
        items = history_items(self.messages)
        if self.running:
            items += self._turn.items.values()
        return OrchestratorState(
            enabled=self.enabled(),
            running=self.running,
            items=items,
            questions=self.answerer.questions,
            model=self.host.status.model,
        )

    def _send(self, message: OrchestratorEvent) -> None:
        self._publish(OrchestratorMessage(message=message))

    def _observe(self, event: AgentStreamEvent) -> None:
        for message in self._translator.translate(event):
            self._turn.apply(message)
            self._send(message)

    def agent(self) -> Agent[None, str]:
        """The orchestrator's agent, made on first use: the workspace tools and `ask_user_question`, nothing else."""
        if self._agent is None:
            from pydantic_clai2.builtin_plugins.web.workspace_tools import WorkspaceTools

            self._tools = WorkspaceTools(self.host)
            self._agent = Agent(
                name='orchestrator',
                deps_type=type(None),
                instructions=INSTRUCTIONS,
                capabilities=[Capability[None](tools=self._tools.tools()), AskUser[None](answerer=self.answerer)],
            )
        return self._agent

    # What the browser does.

    def send(self, text: str) -> None:
        """Start a turn with `text`; one runs at a time, and only while the `workspaces` plugin is on."""
        text = text.strip()
        if not text:
            raise ValueError('Type a message for the orchestrator.')
        if self.running:
            raise RuntimeError('The orchestrator is still working. Wait for it, or stop it.')
        if not self.enabled():
            raise RuntimeError('Turn the workspaces plugin on first.')
        self._task = asyncio.get_running_loop().create_task(self._run(text), name='clai2-orchestrator')

    async def _run(self, prompt: str) -> None:
        item = Item(id=f'orchestrator{time.monotonic_ns()}', kind='user', text=prompt)
        self._turn.clear()
        self._turn.apply(ItemAdded(item=item))
        self._send(TurnStarted(item=item))
        outcome: TurnOutcome = 'failed'
        error: str | None = None
        usage: RunUsage | None = None
        messages = self.messages
        try:
            model = await self.host.conversation.resolved_model()
            if model is None:
                raise RuntimeError('Choose a model for this session first.')
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
            self._turn.clear()
            used = None
            if usage is not None:
                used = Usage(
                    input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, requests=usage.requests
                )
            self._send(TurnEnded(outcome=outcome, error=error, usage=used))
            self._save(messages)

    def interrupt(self) -> bool:
        """Stop the running turn; `False` when none runs."""
        if self._task is None or self._task.done():
            return False
        self._task.cancel()
        return True

    def answer(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        """Answer the orchestrator's open question, or decline it with `None`."""
        self.answerer.reply(request_id, answers)

    def clear(self) -> None:
        """Start a new orchestrator chat, forgetting this one; its workspaces are kept."""
        if self.running:
            raise RuntimeError('Stop the orchestrator before starting a new chat.')
        self.path.unlink(missing_ok=True)
        self._messages = []
        self._send(self.state())

    async def close(self) -> None:
        """Stop a running turn and disconnect from the workspaces; their CLAIs keep running."""
        task = self._task
        if task is not None and self.interrupt():
            await asyncio.gather(task, return_exceptions=True)
        if self._tools is not None:
            await self._tools.close()
