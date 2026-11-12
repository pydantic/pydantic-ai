"""The web UI's orchestrator and its tools, running a session in each workspace: a live session of the same CLAI.

The parent CLAI serves its live sessions as `clai2 --web` does, so a workspace's session is built in its
checkout, runs the workspace model below, and is followed in-process as a page watching it would.
"""

import asyncio
import io
import os
import shutil
import subprocess
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import TypeAdapter
from rich.console import Console

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_clai2.builtin_plugins import web as web_module
from pydantic_clai2.builtin_plugins.web import WebPlugin, live as live_module, workspace_tools as tools_module
from pydantic_clai2.builtin_plugins.web.app import HEADER, create_app
from pydantic_clai2.builtin_plugins.web.client import SessionView, WaitOutcome
from pydantic_clai2.builtin_plugins.web.local_client import LocalClient
from pydantic_clai2.builtin_plugins.web.orchestrator import Orchestrator
from pydantic_clai2.builtin_plugins.web.protocol import (
    DialogOpened,
    Item,
    ItemAdded,
    LiveChanged,
    Notice,
    OrchestratorEvent,
    OrchestratorMessage,
    QuestionAsked,
    ServerMessage,
    SessionsChanged,
    Snapshot,
    TurnEnded,
)
from pydantic_clai2.builtin_plugins.web.service import WebService
from pydantic_clai2.builtin_plugins.web.workspace_tools import SessionAnswer, WorkspaceTools, summarize
from pydantic_clai2.builtin_plugins.web.workspaces import checkouts, open_workspace, required_workspaces
from pydantic_clai2.builtin_plugins.workspaces import Workspaces, WorktreeKind
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import ConversationChanged, LiveSession
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_clai2.runtime.session_workspace import SessionWorkspace

from .test_web import HOSTS, TOKEN, static_dir

pytestmark = pytest.mark.anyio

QUESTION = {
    'questions': [{'question': 'Which colour?', 'header': 'Colour', 'options': [{'label': 'Red'}, {'label': 'Blue'}]}]
}
WEB = PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', settings={'open_browser': False})
WORKSPACES = PluginSettings(id='workspaces', factory='pydantic_clai2.builtin_plugins.workspaces')


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(directory), *args], check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    repo = tmp_path / 'project'
    repo.mkdir()
    git(repo, 'init', '-b', 'main')
    git(repo, 'config', 'user.name', 'CLAI Test')
    git(repo, 'config', 'user.email', 'clai@example.com')
    (repo / 'app.py').write_text('print("hi")\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'First')
    return repo


def prompt_of(messages: Sequence[ModelMessage]) -> str:
    return next(
        part.content
        for message in reversed(messages)
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


def child_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The workspace's model: answers, asks a question, blocks in a tool, or fails, as the prompt says."""
    last = messages[-1]
    if isinstance(last, ModelRequest) and not any(isinstance(part, UserPromptPart) for part in last.parts):
        return ModelResponse(parts=[TextPart('Done after the tool.')])
    prompt = prompt_of(messages)
    if 'ask' in prompt:
        return ModelResponse(parts=[ToolCallPart('ask_user_question', QUESTION)])
    if 'block' in prompt:
        return ModelResponse(parts=[ToolCallPart('block', {})])
    if 'fail' in prompt:
        raise RuntimeError('the model broke')
    return ModelResponse(parts=[TextPart(f'Hello from the workspace: {prompt}. ' + 'x' * 5000)])


async def streamed(response: ModelResponse) -> AsyncIterator[str | DeltaToolCalls]:
    """`response`, streamed, as CLAI and the orchestrator request it."""
    for part in response.parts:
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}


def child_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    return streamed(child_model(messages, info))


@dataclass
class Parent:
    shell: SessionRuntime[None, str]
    web: WebPlugin
    tools: WorkspaceTools
    blocked: anyio.Event
    """Set once the workspace model's `block` tool runs."""
    built: list[Path]
    """The folders the sessions CLAI opened beside the launch session were built in."""

    @property
    def service(self) -> WebService:
        return self.web.service

    @property
    def orchestrator(self) -> Orchestrator:
        return self.service.orchestrator

    def sessions_in(self, folder: Path) -> list[LiveSession]:
        return [session for session in self.service.live.listing() if session.workspace == folder]


@asynccontextmanager
async def parent(
    tmp_path: Path, *, workspaces: bool = True, model: FunctionModel | None = None
) -> AsyncGenerator[Parent]:
    """`clai2 --web` with `web`, and `workspaces` unless told otherwise, whose other sessions run the workspace model.

    `model` is the launch session's agent's own, which the orchestrator uses too while no default model is saved.
    """
    blocked = anyio.Event()
    blocking = Capability[None]()

    @blocking.tool_plain
    async def block() -> str:
        """Wait forever."""
        blocked.set()
        await anyio.sleep_forever()
        return 'never'  # pragma: no cover

    child = Agent(
        FunctionModel(child_model, stream_function=child_stream), deps_type=type(None), capabilities=[blocking]
    )
    home = tmp_path / 'home'
    home.mkdir()
    shell = create_shell(
        Agent(model or FunctionModel(child_model), deps_type=type(None)),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), width=100),
        settings=None,
        store=SettingsStore(tmp_path / 'parent.db'),
        builtin_plugins=[WEB, WORKSPACES] if workspaces else [WEB],
        project=ProjectSettings(),
        headless=True,
        workspace=home,
    )
    built: list[Path] = []

    def build(directory: Path | None) -> SessionRuntime[None, str]:
        assert directory is not None, 'every workspace session starts in its checkout'
        built.append(directory)
        runtime = create_shell(
            child,
            deps=None,
            plugins=(),
            usage_limits=None,
            console=Console(file=io.StringIO(), width=100),
            settings=None,
            store=None,
            builtin_plugins=(),
            project=ProjectSettings(),
            headless=True,
            workspace=directory,
            process=shell.process,
            signals=False,
        )
        # As `clai2 --web` does: `/workspace DIRECTORY` and `/resume` move it.
        SessionWorkspace(
            session=runtime.session,
            context=runtime.context,
            loader=runtime.loader,
            status=runtime.status,
            settings=lambda project: runtime.context.settings,
        ).attach(sessions=runtime.sessions, commands=runtime.commands)
        return runtime

    shell.process.live.build = build
    await shell.loader.load_all()
    [web] = [entry.loaded.plugin for entry in shell.loader.entries() if entry.name == 'web' and entry.loaded]
    assert isinstance(web, WebPlugin)
    serving = anyio.Event()

    async def serve() -> None:
        # As `clai2 --web` does: the other sessions close as serving ends, then the launch session's plugins.
        async with shell.process.live.serving():
            serving.set()
            reason = await shell.run()
        await shell.close(reason)

    async with anyio.create_task_group() as group:
        group.start_soon(serve)
        await serving.wait()
        tools = WorkspaceTools(web.service)
        try:
            yield Parent(shell=shell, web=web, tools=tools, blocked=blocked, built=built)
        finally:
            await tools.close()
            shell.send_input('/exit')


async def created(tools: WorkspaceTools, repository: Path) -> str:
    """A relative `repo` is in the launch session's directory, not the process's."""
    made = await tools.workspace_create(kind='local-worktree', repo=f'../{repository.name}', ref='main')
    return made.split()[2]


async def test_a_workspace_session_from_start_to_answer(tmp_path: Path, repository: Path) -> None:
    """Make a workspace, start a session there with a prompt, wait, read, steer, answer its question, and interrupt."""
    async with parent(tmp_path) as made:
        tools = made.tools
        assert await tools.workspace_list() == 'No workspaces yet.\nKinds: local-worktree.'
        workspace = await created(tools, repository)
        assert workspace.startswith('project-')
        path = (await tools.workspaces().get(workspace)).path
        assert await tools.workspace_list() == (
            f'{workspace}  local-worktree  clai-{workspace}  {path}\nKinds: local-worktree.'
        )

        started = await tools.session_start(workspace, 'say hi')
        conversation = (await tools.workspaces().get(workspace)).conversation
        assert started == (
            f'Workspace {workspace} runs session {conversation}, listed in the sidebar under its repository; '
            'the prompt is sent. Wait with `session_wait`.'
        )
        assert made.built == [path] and [session.id for session in made.sessions_in(path)] != []
        waited = await tools.session_wait(workspace, timeout=10)
        assert waited.startswith(f'Workspace {workspace} is idle; last turn: completed. Its final answer:\n')
        assert 'Hello from the workspace: say hi.' in waited and waited.endswith('… (1,034 more characters)')

        # The sidebar lists it under its repository, on the workspace's branch, as any other session.
        [project] = [project for project in await made.service.projects() if project.name == 'project']
        [checkout] = project.worktrees
        assert (checkout.root, checkout.label) == (str(path), f'clai-{workspace}')
        assert [session.id for session in checkout.sessions] == [conversation]

        read = await tools.session_read(workspace, last=1)
        assert read.startswith(
            f"Workspace {workspace}: idle; session 'say hi'.\nLast turn: completed.\nLast 1 of 2 items:\n"
        )
        assert (await tools.session_read(workspace, last=0)).endswith('Last 0 of 2 items:')

        assert await tools.session_send(workspace, 'ask me') == 'Sent. Wait with `session_wait`.'
        asking = await tools.session_wait(workspace, timeout=10)
        [request] = (await tools.view(workspace)).questions
        assert asking == (
            f'Workspace {workspace} asks:\nQuestion request {request}:\n  - Colour: Which colour? Options: Red, Blue\n'
            'Answer with `session_answer`.'
        )
        assert f'Question request {request}' in await tools.session_read(workspace)
        assert await tools.session_answer(workspace, 'nope') == f"Workspace {workspace} has no open question 'nope'."
        answered = await tools.session_answer(workspace, request, [SessionAnswer(header='Colour', selected=['Blue'])])
        assert answered == 'Answered. Wait with `session_wait`.'
        assert 'Done after the tool.' in await tools.session_wait(workspace, timeout=10)

        await tools.session_send(workspace, 'ask me again')
        await tools.session_wait(workspace, timeout=10)
        [request] = (await tools.view(workspace)).questions
        await tools.session_answer(workspace, request, [SessionAnswer(header='Colour', custom='Teal')])
        assert 'Done after the tool.' in await tools.session_wait(workspace, timeout=10)

        await tools.session_send(workspace, 'block')
        await made.blocked.wait()
        assert 'is still running' in await tools.session_wait(workspace, timeout=0.05)
        assert 'a turn is running' in await tools.session_read(workspace)
        assert await tools.session_interrupt(workspace) == 'Interrupted.'
        assert 'last turn: cancelled' in await tools.session_wait(workspace, timeout=10)

        await tools.session_send(workspace, 'fail')
        await tools.session_wait(workspace, timeout=10)
        assert 'Last turn: failed (' in await tools.session_read(workspace)
        assert made.built == [path], 'one session the whole time'


def destroy_question(workspace: str) -> dict[str, object]:
    return {
        'questions': [
            {
                'question': f'Destroy workspace {workspace}?',
                'header': 'Destroy',
                'options': [{'label': f'Destroy {workspace}'}, {'label': f'Keep {workspace}'}],
            }
        ]
    }


def returned(messages: Sequence[ModelMessage]) -> list[str]:
    """What the workspace tools returned, in order; `ask_user_question` answers left out."""
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name != 'ask_user_question'
    ]


def orchestrating(repository: Path) -> FunctionModel:
    """The orchestrator's model: make a workspace, run a prompt there, read it, and destroy it once the user agrees."""

    def destroy(workspace: str) -> ToolCallPart:
        return ToolCallPart('workspace_destroy', {'workspace': workspace})

    def ask(workspace: str) -> ToolCallPart:
        return ToolCallPart('ask_user_question', destroy_question(workspace))

    steps: list[Callable[[str], ToolCallPart | TextPart]] = [
        lambda _: ToolCallPart('workspace_create', {'kind': 'local-worktree', 'repo': str(repository), 'ref': 'main'}),
        lambda workspace: ToolCallPart('session_start', {'workspace': workspace, 'prompt': 'say hi'}),
        lambda workspace: ToolCallPart('session_wait', {'workspace': workspace, 'timeout': 10}),
        lambda workspace: ToolCallPart('session_read', {'workspace': workspace, 'last': 1}),
        *[destroy, ask] * 3,
        destroy,
        lambda _: TextPart('It said hi; the workspace is gone.'),
    ]

    def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        made = next((text for text in returned(messages) if text.startswith('Created workspace ')), '')
        workspace = made.split()[2] if made else ''
        step = steps[sum(isinstance(message, ModelResponse) for message in messages)]
        return streamed(ModelResponse(parts=[step(workspace)]))

    return FunctionModel(stream_function=stream)


WIRE = TypeAdapter[ServerMessage](ServerMessage)


async def converse(
    orchestrator: Orchestrator, queue: asyncio.Queue[str], picks: list[int | None]
) -> list[OrchestratorEvent]:
    """The orchestrator chat's messages until its turn ends, answering each question with the next of `picks`.

    A pick is an option's index, or `None` to decline, as the user would in the page.
    """
    seen: list[OrchestratorEvent] = []
    with anyio.fail_after(30):
        while True:
            message = WIRE.validate_json(await queue.get())
            # The sessions' own messages, and the sidebar's, are skipped.
            event = message.message if isinstance(message, OrchestratorMessage) else None
            seen += [] if event is None else [event]
            if isinstance(event, QuestionAsked):
                pick = picks.pop(0)
                [question] = event.questions.questions
                label = None if pick is None else question.options[pick].label
                answers = None if label is None else [AskUserAnswer(header=question.header, selected=(label,))]
                orchestrator.answer(event.questions.id, answers)
            elif isinstance(event, TurnEnded):
                return seen


async def test_the_orchestrator_runs_a_workspace_and_asks_before_destroying(tmp_path: Path, repository: Path) -> None:
    """The orchestrator's own agent makes a workspace, prompts a session there, reads it, and destroys it once agreed.

    `workspace_destroy` refuses until the user picked `Destroy ID` in the orchestrator chat: declining and
    keeping it are no confirmation. The model never answers for the user. Destroying closes the session first.
    """
    async with parent(tmp_path, model=orchestrating(repository)) as made:
        orchestrator = made.orchestrator
        with made.service.hub.subscribe() as client:
            await orchestrator.send('Say hi in a new workspace, then clean up.')
            seen = await converse(orchestrator, client.queue, [1, None, 0])
        end = seen[-1]
        assert isinstance(end, TurnEnded) and end.outcome == 'completed'
        tools = [event.item.name for event in seen if isinstance(event, ItemAdded) and event.item.kind == 'tool']
        assert tools[:4] == ['workspace_create', 'session_start', 'session_wait', 'session_read']
        results = returned((await orchestrator.load()).messages)
        workspace = results[0].split()[2]
        path = Path(results[0].split()[4])
        assert 'Hello from the workspace: say hi.' in results[2]
        assert results[4].startswith(
            f'Not destroyed. Ask the user first with `ask_user_question`, offering `Destroy {workspace}`'
        )
        assert results[5].startswith('Not destroyed.'), 'keeping it is no confirmation'
        assert results[6].startswith('Not destroyed.'), 'declining is no confirmation'
        assert results[7] == f'Destroyed workspace {workspace}; its checkout at {path} was removed, its branch kept.'
        assert made.built == [path] and not path.exists()
        assert made.sessions_in(path) == [], 'its session closed before the checkout went'
        assert [session.workspace for session in made.service.live.listing()] == [tmp_path / 'home']
        assert await made.tools.workspace_list() == 'No workspaces yet.\nKinds: local-worktree.'

        # The conversation is saved, so another load of the web UI shows it.
        again = Orchestrator(service=made.service)
        kinds = [item.kind for item in (await again.state()).items]
        assert (
            kinds[0] == 'user'
            and kinds[-1] == 'assistant'
            and len((await again.load()).messages) == len((await orchestrator.load()).messages)
        )


def test_a_later_keep_takes_back_an_earlier_destroy() -> None:
    """Only the user's latest pick for a workspace counts, so a `Keep ID` after a `Destroy ID` refuses again."""

    def answer(label: str) -> ModelRequest:
        content = {'Destroy workspace w?': [label]}
        return ModelRequest(parts=[ToolReturnPart(tool_name='ask_user_question', content=content, tool_call_id=label)])

    assert tools_module.confirmed([answer('Destroy w')], 'w')
    assert not tools_module.confirmed([answer('Destroy w'), answer('Keep w')], 'w')
    assert tools_module.confirmed([answer('Keep w'), answer('Destroy w')], 'w')
    assert not tools_module.confirmed([answer('Destroy other')], 'w')


class Silent(LocalClient):
    """A client whose messages end at once, before any snapshot."""

    @asynccontextmanager
    async def messages(self) -> AsyncGenerator[AsyncIterator[ServerMessage]]:
        async def nothing() -> AsyncIterator[ServerMessage]:
            for message in list[ServerMessage]():
                yield message  # pragma: no cover

        yield nothing()


async def test_problems_come_back_as_errors(tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with parent(tmp_path) as made:
        tools = made.tools
        assert (await tools.workspace_list()).startswith('No workspaces')
        workspace = await created(tools, repository)
        path = (await tools.workspaces().get(workspace)).path
        reading = tools_module.reported(tools.session_read)
        no_session = f'Error: Workspace {workspace} has no session yet. Start one with `session_start`.'
        assert await reading(workspace) == no_session
        await tools.workspaces().attach(workspace, 'never-saved')
        assert await reading(workspace) == no_session, 'a session never saved is no session'
        assert await tools_module.reported(tools.workspace_create)(kind='docker') == (
            "Error: No workspace kind 'docker'. Kinds: local-worktree."
        )

        # A session that closes ends the following; the next use opens it again, as it was saved.
        await tools.session_start(workspace, 'block')
        await made.blocked.wait()
        entered = anyio.Event()
        wait = SessionView.wait

        async def entering(view: SessionView, *, timeout: float) -> WaitOutcome:
            entered.set()
            return await wait(view, timeout=timeout)

        stopped: list[str] = []
        with monkeypatch.context() as patched:
            patched.setattr(SessionView, 'wait', entering)
            async with anyio.create_task_group() as group:

                async def waiting() -> None:
                    stopped.append(await tools.session_wait(workspace, timeout=10))

                group.start_soon(waiting)
                await entered.wait()
                [session] = made.sessions_in(path)
                await made.service.live.close(session.id)
        assert stopped == [f'The session of workspace {workspace} closed.']
        # A turn cut off by closing may not have saved its end, so it can open again in a fork (`'block (fork)'`).
        reread = await tools.session_read(workspace)
        assert reread.startswith(f"Workspace {workspace}: idle; session 'block"), reread
        assert made.built == [path, path], 'opened again in its checkout'
        await tools.disconnect(workspace)
        waited = await tools.session_wait(workspace, timeout=10)
        assert waited.startswith(f'Workspace {workspace} is idle; last turn: no turn ran.')

        # Falling too far behind ends the following too; the next use follows it again from a snapshot.
        view = await tools.view(workspace)
        with monkeypatch.context() as patched:
            patched.setattr(live_module, 'QUEUE_LIMIT', 3)
            for _ in range(3):
                made.service.hub.publish(SessionsChanged())
        with anyio.fail_after(10):
            await tools.links[workspace].task  # Following ends.
        assert view.closed and await tools.view(workspace) is not view

        # A session whose messages end before its snapshot cannot be followed.
        await tools.disconnect(workspace)
        monkeypatch.setattr(tools_module, 'LocalClient', Silent)
        assert await reading(workspace) == f'Error: Could not follow the session of workspace {workspace}.'

        await tools.close()
        assert tools.links == {}


async def test_the_pages_dialogs_are_not_a_sessions(tmp_path: Path, repository: Path) -> None:
    """A settings dialog the user opened in the page never shows to the orchestrator, which cannot answer it."""
    async with parent(tmp_path) as made:
        tools, service = made.tools, made.service
        workspace = await created(tools, repository)
        await tools.session_start(workspace, 'hi')
        await tools.session_wait(workspace, timeout=10)
        view = await tools.view(workspace)
        with service.hub.subscribe() as page:
            async with anyio.create_task_group() as group:
                group.start_soon(service.run_menu_action, 'workspaces', 'add')
                await published(page.queue, DialogOpened)
                [dialog] = service.dialogs.dialogs
                # Followed again, its snapshot leaves the open dialog out too.
                await tools.disconnect(workspace)
                view = await tools.view(workspace)
                assert view.dialogs == {}
                assert (await tools.session_wait(workspace, timeout=1)).startswith(f'Workspace {workspace} is idle;')
                with pytest.raises(LookupError, match='dialogs belong to the page'):
                    await view.client.answer_dialog(dialog.id, 'docs')
                service.answer_dialog(dialog.id, None)
        await tools.session_send(workspace, 'hi again')
        assert 'hi again' in await tools.session_wait(workspace, timeout=10)
        assert view.dialogs == {}

        # A bare command that opens a menu in the page is no input for a session.
        with pytest.raises(ValueError, match='/set opens a menu in the page'):
            await LocalClient(service, made.sessions_in((await tools.workspaces().get(workspace)).path)[0].id).send(
                '/set'
            )


async def test_without_the_workspaces_plugin(tmp_path: Path) -> None:
    async with parent(tmp_path, workspaces=False) as made:
        listing = tools_module.reported(made.tools.workspace_list)
        assert await listing() == 'Error: The `workspaces` plugin is off. Ask the user to turn it on.'
        assert await checkouts(made.web.host.plugins) == []
    assert await checkouts(None) == []
    with pytest.raises(RuntimeError, match='plugin is off'):
        required_workspaces(None)


async def test_a_live_change_queued_before_a_switch_does_not_end_the_stream(tmp_path: Path) -> None:
    """A `LiveChanged` queued before `/new` names the old conversation, but the session is still live."""
    async with parent(tmp_path, workspaces=False) as made:
        service = made.service
        channel = service.default()
        before = channel.id
        async with LocalClient(service, channel.session.id).messages() as received:
            assert isinstance(await anext(received), Snapshot)
            with service.hub.subscribe() as page:
                service.watch(page, [before])
                await published(page.queue, Snapshot)
                service.hub.publish(service.live_message())
                channel.send('/new')
                # Switched before the client reads what was queued for it earlier.
                switched = await published(page.queue, Snapshot)
            assert isinstance(switched, Snapshot) and switched.session.id != before
            message = await anext(received, None)
        assert isinstance(message, LiveChanged), 'the stream ended while the session was still live'
        assert [entry.id for entry in message.sessions] == [before]


async def test_commands_are_not_forwarded(tmp_path: Path, repository: Path) -> None:
    """A `/command` would run in the workspace's session with no one to confirm it, as `/workspaces destroy` would."""
    async with parent(tmp_path) as made:
        workspace = await created(made.tools, repository)
        refusal = 'Error: Send prompts only. A `/command`, such as `/workspaces destroy`'
        starting = tools_module.reported(made.tools.session_start)
        assert (await starting(workspace, ' /workspaces destroy other')).startswith(refusal)
        assert (await made.tools.workspaces().get(workspace)).conversation is None, 'nothing started'
        await made.tools.session_start(workspace, 'hi')
        sending = tools_module.reported(made.tools.session_send)
        assert (await sending(workspace, '/plugins add evil evil:Plugin')).startswith(refusal)
        await made.tools.session_wait(workspace, timeout=10)


async def test_a_bug_in_a_tool_fails_the_turn() -> None:
    """A `KeyError` is a bug, not a problem for the model: it propagates rather than coming back as an error line."""

    async def broken() -> str:
        raise KeyError('oops')

    with pytest.raises(KeyError):
        await tools_module.reported(broken)()


def test_items_in_short() -> None:
    tool = Item(id='t', kind='tool', name='run_code', args='{"code": "print(1)"}', text='1', state='done')
    assert summarize(tool) == '- tool `run_code` (done): {"code": "print(1)"} -> 1'
    assert summarize(Item(id='a', kind='assistant', text=' short ')) == '- assistant: short'


async def test_the_page_opens_a_workspaces_session(tmp_path: Path, repository: Path) -> None:
    """The sidebar lists a workspace's checkout under its repository, and a tool card's link opens its session here."""
    async with parent(tmp_path) as made:
        service = made.service
        workspace = await created(made.tools, repository)
        path = (await made.tools.workspaces().get(workspace)).path
        gone = await created(made.tools, repository)
        shutil.rmtree((await made.tools.workspaces().get(gone)).path)
        # Before any session runs there, for New session to offer; a search lists sessions only, and a checkout
        # removed meanwhile is not offered.
        [project] = [project for project in await service.projects() if project.name == 'project']
        assert [(checkout.root, checkout.label, checkout.sessions) for checkout in project.worktrees] == [
            (str(path), f'clai-{workspace}', [])
        ]
        assert [project.name for project in await service.projects('hi')] == []

        app = create_app(service=service, token=TOKEN, hosts=HOSTS, static=static_dir(tmp_path))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url='http://testserver', headers={HEADER: TOKEN}
        ) as page:
            opened = (await page.post(f'/api/workspaces/{workspace}/open')).json()['id']
            assert (await made.tools.workspaces().get(workspace)).conversation == opened
            assert service.default().id == opened, 'the page shows it'
            assert [session.conversation.conversation_id for session in made.sessions_in(path)] == [opened]
            assert (await page.post(f'/api/workspaces/{workspace}/open')).json()['id'] == opened, 'the same one'
            assert (await page.post('/api/workspaces/nope-1/open')).status_code == 404
        assert made.built == [path]


async def published(queue: asyncio.Queue[str], kind: type[ServerMessage]) -> ServerMessage:
    """The next message of `kind` the hub sent, the others skipped."""
    with anyio.fail_after(10):
        while not isinstance(message := WIRE.validate_json(await queue.get()), kind):
            pass
    return message


def applied(view: SessionView, wanted: Callable[[ServerMessage], bool], monkeypatch: pytest.MonkeyPatch) -> anyio.Event:
    """An event set once `view` has applied a message that is `wanted`."""
    seen = anyio.Event()
    apply = view.apply

    def applying(message: ServerMessage) -> None:
        apply(message)
        if wanted(message):
            seen.set()

    monkeypatch.setattr(view, 'apply', applying)
    return seen


async def test_the_sidebar_and_one_session_per_workspace(tmp_path: Path, repository: Path) -> None:
    """Creating and destroying refresh the sidebar, and tool calls the model makes at once share one session."""
    async with parent(tmp_path) as made:
        tools, service = made.tools, made.service
        with service.hub.subscribe() as client:
            workspace = await created(tools, repository)
            await published(client.queue, SessionsChanged)
        path = (await tools.workspaces().get(workspace)).path
        # The page's Open and two tool calls at once start one session, followed once.
        shown, first, second = await asyncio.gather(
            open_workspace(service, workspace), tools.view(workspace, start=True), tools.view(workspace, start=True)
        )
        assert first is second and made.built == [path]
        assert (await tools.workspaces().get(workspace)).conversation == shown

        # A session that another process holds goes on in a copy here, which is the workspace's from then on.
        await tools.disconnect(workspace)
        [session] = made.sessions_in(path)
        await service.live.close(session.id)
        copy = await service.live.new(workspace=path)
        opened = service.live.open

        async def copying(conversation_id: str) -> LiveSession:
            return (
                copy
                if first.session is not None and conversation_id == first.session.id
                else await opened(conversation_id)
            )

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(service.live, 'open', copying)
            await tools.view(workspace)
        assert (await tools.workspaces().get(workspace)).conversation == copy.conversation.conversation_id

        with service.hub.subscribe() as client:
            confirmed = ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ask_user_question',
                        content={f'Destroy workspace {workspace}?': [f'Destroy {workspace}']},
                        tool_call_id='asked',
                    )
                ]
            )
            ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), messages=[confirmed])
            assert (await tools.workspace_destroy(ctx, workspace)).startswith(f'Destroyed workspace {workspace};')
            await published(client.queue, SessionsChanged)
        assert workspace not in service.workspace_locks


async def test_destroying_a_kept_checkout_leaves_its_sessions(tmp_path: Path, repository: Path) -> None:
    """A kind that keeps the checkout closes none of the sessions working there: they go on in the folder."""
    async with parent(tmp_path) as made:
        kept = Workspaces(
            kinds={'kept': WorktreeKind(repo=str(repository), teardown='keep')}, directory=tmp_path / 'kept'
        )
        workspace = await kept.create(kind='kept')
        session = await made.service.live.new(workspace=workspace.path)
        assert (await kept.destroy(workspace.id, live=made.service.live)).teardown == 'keep'
        assert workspace.path.exists() and await kept.listing() == []
        assert made.sessions_in(workspace.path) == [session], 'still live'


async def test_falling_behind_mid_wait_is_no_close(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wait whose following falls too far behind says the session is still open, so the model waits again."""
    async with parent(tmp_path) as made:
        tools = made.tools
        workspace = await created(tools, repository)
        await tools.session_start(workspace, 'block')
        await made.blocked.wait()
        entered = anyio.Event()
        wait = SessionView.wait

        async def entering(view: SessionView, *, timeout: float) -> WaitOutcome:
            entered.set()
            return await wait(view, timeout=timeout)

        waited: list[str] = []
        monkeypatch.setattr(SessionView, 'wait', entering)
        async with anyio.create_task_group() as group:

            async def waiting() -> None:
                waited.append(await tools.session_wait(workspace, timeout=10))

            group.start_soon(waiting)
            await entered.wait()
            with monkeypatch.context() as patched:
                patched.setattr(live_module, 'QUEUE_LIMIT', 3)
                for _ in range(3):
                    made.service.hub.publish(SessionsChanged())
        assert waited == [
            f'Lost track of the session of workspace {workspace}, which is still open. Wait again to follow it.'
        ]
        assert await tools.session_interrupt(workspace) == 'Interrupted.', 'followed again, mid-turn'


async def test_the_workspace_session_is_the_users_too(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user can type in the workspace's session too.

    A destroy there is refused. A `/new` there is the workspace's from then on, even after it closed (or a fork
    of it, when the close beat its turn's last save). Moved to another folder with `/workspace`, it is not, even
    before the orchestrator's view hears of the move, and the workspace gets a session in its checkout again.
    The page follows a switch even when recording it for the workspace fails.
    """
    async with parent(tmp_path) as made:
        tools, service = made.tools, made.service
        workspace = await created(tools, repository)
        await tools.session_start(workspace, 'say hi')
        await tools.session_wait(workspace, timeout=10)
        view = await tools.view(workspace)
        path = (await tools.workspaces().get(workspace)).path
        [session] = made.sessions_in(path)
        channel = service.channel(session)
        before = channel.id

        refused = applied(view, lambda message: isinstance(message, Notice), monkeypatch)
        channel.send(f'/workspaces destroy {workspace}')
        with anyio.fail_after(10):
            await refused.wait()
        assert 'Run this from a session outside it.' in view.notices[-1].text
        assert path.exists(), 'nothing was destroyed'

        switched = applied(
            view, lambda message: isinstance(message, Snapshot) and message.session.id != before, monkeypatch
        )
        channel.send('/new')
        with anyio.fail_after(10):
            await switched.wait()
        assert view.session is not None and view.session.id != before
        renewed = view.session.id
        assert (await tools.workspaces().get(workspace)).conversation == renewed, 'recorded as the page saw it'
        assert await tools.view(workspace) is view, 'still followed'
        assert await open_workspace(service, workspace) == renewed, 'the page opens it too'
        await tools.session_send(workspace, 'again')
        assert 'Hello from the workspace: again.' in await tools.session_wait(workspace, timeout=10)
        # Closed, it opens again in the conversation the user switched to, or in a fork of it: closed right after
        # its turn, before that turn's end was saved, it reads as still running elsewhere.
        await tools.disconnect(workspace)
        await service.live.close(session.id)
        view = await tools.view(workspace)
        reopened = view.session
        assert reopened is not None and reopened.title is not None and reopened.title.startswith('again')
        assert (await tools.workspaces().get(workspace)).conversation == reopened.id and made.built == [path, path]
        renewed = reopened.id

        [session] = made.sessions_in(path)
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        # The view never hears of the move, as when the orchestrator asks before its snapshot arrives:
        # the live session, not the view, tells that it left.
        moved = anyio.Event()
        apply = view.apply

        def unaware(message: ServerMessage) -> None:
            if isinstance(message, Snapshot) and message.session.workspace == str(elsewhere.resolve()):
                moved.set()
            else:
                apply(message)

        monkeypatch.setattr(view, 'apply', unaware)
        service.channel(session).send(f'/workspace {elsewhere}')
        with anyio.fail_after(10):
            await moved.wait()
        assert view.session is not None and view.session.workspace == str(path)
        assert (await tools.workspaces().get(workspace)).conversation == renewed, "not the workspace's any more"
        again = await tools.view(workspace)
        assert again is not view and again.session is not None and again.session.workspace == str(path)
        assert made.built == [path, path, path], 'a session in the checkout again'

        # The page hears of a switch even when recording it for the workspace fails.
        async def failing(service: WebService, event: ConversationChanged) -> None:
            raise OSError('the registry is read-only')

        monkeypatch.setattr(web_module, 'conversation_switched', failing)
        current = again.session.id
        switched = applied(
            again, lambda message: isinstance(message, Snapshot) and message.session.id != current, monkeypatch
        )
        [session] = made.sessions_in(path)
        service.channel(session).send('/new')
        with anyio.fail_after(10):
            await switched.wait()
        assert (await tools.workspaces().get(workspace)).conversation == current
