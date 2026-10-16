"""The orchestrator: an agent of its own in the web UI, which runs workspaces for the user.

It has its own conversation and only the tools in `workspace_tools`, plus `ask_user_question`, answered
in its own chat. The session's agent gets none of them. Its conversation is one JSON file in CLAI's state
directory, so it survives a reload and a restart, and never shows among the saved sessions; the model
chosen for it is another file beside it. Its `/commands` are in `chat_commands`.
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
from pydantic import BaseModel, ValidationError

from pydantic_ai import Agent, AgentStreamEvent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, repair_messages
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ask_user import AskUser, AskUserAnswer
from pydantic_clai2.builtin_plugins.web.chat_commands import (
    FOLLOW,
    OWN,
    orchestrator_help,
    parse,
    picks_session_model,
    result_notice,
    saved_models,
    shared,
)
from pydantic_clai2.builtin_plugins.web.live import BrowserAnswerer
from pydantic_clai2.builtin_plugins.web.protocol import (
    EventTranslator,
    Item,
    ItemAdded,
    ItemLog,
    Notice,
    OrchestratorEvent,
    OrchestratorMessage,
    OrchestratorModel,
    OrchestratorState,
    ServerMessage,
    TurnEnded,
    TurnStarted,
    history_items,
    usage_info,
)
from pydantic_clai2.commands import expand_bare_command, is_command_input
from pydantic_clai2.config.settings_store import state_dir
from pydantic_clai2.models.model_catalog import check_installed
from pydantic_clai2.plugins import CommandEnd, PluginHost, TurnOutcome
from pydantic_clai2.ui.menus.model_picker import MODEL_SUBCOMMANDS

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.web.workspace_tools import WorkspaceTools

HISTORY = 'orchestrator.json'
"""The conversation's file in CLAI's state directory."""
SETTINGS = 'orchestrator-settings.json'
"""What the user chose for the orchestrator, beside its conversation, so every CLAI and a new chat keep it."""

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
    data = ModelMessagesTypeAdapter.dump_json(messages)
    _replace(path, data)
    return Saved(messages=messages, revision=hashlib.sha256(data).hexdigest())


def _replace(path: Path, data: bytes) -> None:
    """Replace `path` with `data` through a new file only you can read, so no reader sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'wb') as file:
            file.write(data)
        os.replace(name, path)
    except BaseException:
        os.unlink(name)
        raise


class OrchestratorSettings(BaseModel):
    """What the user chose for the orchestrator chat; a choice not made follows the session."""

    model: str | None = None
    """The model its turns run on; `None` uses the session's."""


def read_settings(path: Path) -> OrchestratorSettings:
    """The choices in `path`; none when there is no file or it cannot be read."""
    try:
        return OrchestratorSettings.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return OrchestratorSettings()


class Orchestrator:
    """The orchestrator chat every connected browser shares: one conversation, one turn at a time."""

    def __init__(self, *, host: PluginHost[None], publish: Callable[[ServerMessage], None]) -> None:
        self.host = host
        self._publish = publish
        self.answerer = BrowserAnswerer(publish=self._send, terminal=None)
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
        self._model: str | None = None
        """The model chosen for its turns, as last read or chosen; `None` follows the session's."""
        self._entered: list[str] = []
        """CLAI commands this chat entered at the prompt whose end it has not seen yet, to show their results.

        Matched by text, all `CommandEnd` carries: one cancelled while it runs ends without one, so the same
        text entered next from the session's composer answers here instead.
        """

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def path(self) -> Path:
        return state_dir() / HISTORY

    @property
    def settings_path(self) -> Path:
        return state_dir() / SETTINGS

    async def load(self) -> Saved:
        """The conversation and its model, read again between turns, as another CLAI may have saved them since."""
        if self.running:
            return self._saved
        self._model = (await to_thread.run_sync(read_settings, self.settings_path)).model
        if self._unsaved is not None:
            return Saved(messages=self._unsaved, revision=self._saved.revision)
        self._saved = await to_thread.run_sync(read_saved, self.path)
        return self._saved

    def enabled(self) -> bool:
        """Whether the `workspaces` plugin is on, which the orchestrator's tools work over."""
        # Imported here: the `workspaces` plugin imports this package for the token header.
        from pydantic_clai2.builtin_plugins.workspaces import configured

        return configured(self.host.plugins) is not None

    async def state(self) -> OrchestratorState:
        return self._state((await self.load()).messages)

    def _state(self, messages: Sequence[ModelMessage]) -> OrchestratorState:
        items = history_items(messages)
        if self.running:
            items += self._turn.items.values()
        return OrchestratorState(
            enabled=self.enabled(),
            running=self.running,
            items=items,
            questions=self.answerer.questions,
            model=self._model,
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

    async def send(self, text: str) -> None:
        """Run a `/command`, or start a turn with `text`: one at a time, and only while `workspaces` is on."""
        text = expand_bare_command(text.strip())
        if not text:
            raise ValueError('Type a message for the orchestrator.')
        if is_command_input(text):
            await self._command(text)
            return
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
        self._send(self._state(saved.messages))
        messages = saved.messages
        outcome: TurnOutcome = 'failed'
        error: str | None = None
        usage: RunUsage | None = None
        try:
            conversation = self.host.conversation
            model = await (
                conversation.resolved_model() if self._model is None else conversation.build_model(self._model)
            )
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

    async def _command(self, text: str) -> None:
        """Run one of the chat's own commands, or enter one of CLAI's shared ones at the prompt.

        The prompt runs CLAI's as it runs the session's, and `command_ended` shows the result here.
        """
        name, args = parse(text)
        if shared(self.host, name) and not picks_session_model(name, args):
            self._entered.append(text)
            try:
                self.host.send_input(text)
            except RuntimeError:
                self._entered.remove(text)
                raise
            return
        try:
            output = await self._run_own(name, args)
        except (ValueError, RuntimeError) as error:
            self._send(Notice(text=str(error), level='error', command=text))
        else:
            if output:
                self._send(Notice(text=output, command=text))

    async def _run_own(self, name: str, args: list[str]) -> str:
        if name == 'model':
            return await self._model_command(args)
        if name in OWN and args:
            raise ValueError(f'Usage: /{name}')
        if name in ('new', 'clear'):
            await self.clear()
            return ''
        if name == 'help':
            return await orchestrator_help(self.host)
        if name == 'stop':
            return '' if self.interrupt() else 'The orchestrator is not working.'
        if picks_session_model(name, args):
            raise ValueError(
                f"/{name} model picks the session's model, in a session; /model NAME picks the orchestrator's."
            )
        if self.host.command(name) is not None:
            raise ValueError(f'/{name} works in a session, not in the orchestrator chat. See /help.')
        raise ValueError(f'Unknown command /{name}. Use /help.')

    async def _model_command(self, args: list[str]) -> str:
        """`/model`: show the orchestrator's model and the saved ones, pick one, or follow the session's again."""
        if args and args[0] in MODEL_SUBCOMMANDS:
            raise ValueError(f'/model {args[0]} works in a session, not in the orchestrator chat.')
        if len(args) > 1:
            raise ValueError('Usage: /model [NAME|session]')
        if args and args[0] in FOLLOW:
            await self.select_model(None)
            return f"The orchestrator follows the session's model, {self.host.status.model}, from its next turn."
        if args:
            await self.select_model(args[0])
            return f'The orchestrator runs on {args[0]} from its next turn.'
        await self.load()
        current = self._model or f"{self.host.status.model}, the session's"
        saved = ', '.join(await saved_models(self.host)) or 'none'
        return (
            f'The orchestrator runs on {current}.\nSaved models: {saved}.\n'
            "/model NAME picks one for it alone; /model session follows the session's again."
        )

    async def select_model(self, name: str | None) -> None:
        """Run the next turns on `name`, checked as `/model NAME` checks it, or on the session's model with `None`.

        The session's own model stays as it is.
        """
        if name is not None:
            check_installed(name)
        data = OrchestratorSettings(model=name).model_dump_json().encode()
        try:
            await to_thread.run_sync(_replace, self.settings_path, data)
        except OSError as exc:
            raise RuntimeError(f"The orchestrator's model could not be saved: {exc}") from exc
        self._model = name
        self._send(OrchestratorModel(model=name))

    def command_ended(self, event: CommandEnd) -> bool:
        """Show the result of a CLAI command this chat entered; `False` for any other, which the session shows."""
        if event.text not in self._entered:
            return False
        self._entered.remove(event.text)
        if (notice := result_notice(event)) is not None:
            self._send(notice)
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
        """Stop a running turn and disconnect from the workspaces; their CLAIs keep running."""
        task = self._task
        if task is not None and self.interrupt():
            await asyncio.gather(task, return_exceptions=True)
        if self._tools is not None:
            await self._tools.close()
