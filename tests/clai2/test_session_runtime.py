"""`SessionRuntime`: live sessions built over the `ClaiProcess` whose stores and services they share."""

import io
import signal
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.subagents import DelegationEndEvent, DelegationStartEvent, DelegationTaskEvent
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import TaskProgress
from pydantic_clai2.runtime.process import ClaiProcess
from pydantic_clai2.runtime.session import resolve_model_name
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_clai2.runtime.tasks import TaskRouter, Tasks
from pydantic_clai2.ui.prompt.interrupts import Interrupts
from tests.clai2.test_tasks import task


def runtime_over(
    process: ClaiProcess, workspace: Path, agent: Agent[None, str] | None = None
) -> tuple[SessionRuntime[None, str], io.StringIO]:
    workspace.mkdir(exist_ok=True)
    output = io.StringIO()
    runtime = create_shell(
        agent or Agent(TestModel(), deps_type=type(None)),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=output, force_terminal=False, width=120),
        settings=Settings(model='test') if agent is None else None,
        store=None,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        workspace=workspace,
        process=process,
        signals=False,
    )
    return runtime, output


async def test_a_runtime_without_a_terminal_serves_input_over_the_process_stores(tmp_path: Path) -> None:
    """With no terminal, `send_input` feeds the read loop; the turn is saved in the process's conversation store."""
    process = ClaiProcess.open(SettingsStore(tmp_path / 'config.db'))
    runtime, output = runtime_over(process, tmp_path / 'project')
    assert (runtime.process, runtime.context.store, runtime.sessions.store) == (
        process,
        process.store,
        process.conversations,
    )

    runtime.send_input('hello')
    runtime.send_input('/exit')

    assert await runtime.run() == 'exit'
    assert '> hello\n\nsuccess (no tool calls)' in output.getvalue()
    saved = await process.conversations.get(conversation_id=runtime.session.summary.id)
    assert saved.summary.workspace == str((tmp_path / 'project').resolve())


async def test_runtimes_share_the_process_services(tmp_path: Path) -> None:
    """One task owner, one naming worker and one Codex sign-in, whichever session uses them."""
    process = ClaiProcess.open(SettingsStore(tmp_path / 'config.db'))
    a, _ = runtime_over(process, tmp_path / 'a')
    b, _ = runtime_over(process, tmp_path / 'b')
    assert a.tasks is not b.tasks
    assert a.tasks.owner is b.tasks.owner is process.tasks.owner
    assert process.tasks.views == [a.tasks, b.tasks]
    assert a.sessions.namer is b.sessions.namer is process.namer
    assert a.sessions.naming is not b.sessions.naming
    assert process.tasks.owner.directory == tmp_path / 'sessions.db.tasks'
    # Every account's Codex provider, and so its token refresh, is built once for all sessions.
    assert await resolve_model_name(a.session.resolve_model, 'openai-codex:gpt-5') is not None
    provider = process.codex_providers['openai-codex']
    assert await resolve_model_name(b.session.resolve_model, 'openai-codex:gpt-5') is not None
    assert list(process.codex_providers) == ['openai-codex']
    assert process.codex_providers['openai-codex'] is provider
    assert not (a.interrupts.signals or b.interrupts.signals)
    alone, _ = runtime_over(ClaiProcess.open(SettingsStore(tmp_path / 'other.db')), tmp_path / 'alone')
    assert alone.tasks.owner is not process.tasks.owner


async def test_closing_one_runtime_leaves_the_others_tasks_running(tmp_path: Path) -> None:
    """The owner stays open while any runtime runs: A's exit cancels A's tasks, whose plugins close, not B's."""
    process = ClaiProcess.open(SettingsStore(tmp_path / 'config.db'))
    started, release = anyio.Event(), anyio.Event()

    async def blocked(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        started.set()
        await release.wait()
        yield 'B finished.'

    a, a_output = runtime_over(process, tmp_path / 'a')
    b, b_output = runtime_over(process, tmp_path / 'b', Agent(FunctionModel(stream_function=blocked)))
    background = task(conversation_id=b.session.summary.id)
    background.background = True
    # A's own: one of its conversation, and one it started in a conversation it has since left.
    own, earlier = task(task_id='d' * 32, conversation_id=a.session.summary.id), task(task_id='e' * 32)
    # B's, though in the conversation A holds now: it runs with B's plugins.
    foreign = task(task_id='f' * 32, conversation_id=a.session.summary.id)
    reasons: dict[str, str] = {}

    async def run(name: str, runtime: SessionRuntime[None, str]) -> None:
        reasons[name] = await runtime.run()

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(run, 'b', b)
            b.send_input('wait here')
            await started.wait()
            process.tasks.owner.records[background.id] = background
            assert b.background_running() and not a.background_running()
            assert b.plugins_busy('/plugins list') and not a.plugins_busy('/plugins list')
            process.tasks.owner.records.update({own.id: own, earlier.id: earlier, foreign.id: foreign})
            process.tasks.started.update({earlier.id: a.tasks, foreign.id: b.tasks})
            assert a.background_running()
            assert b.tasks.runs(foreign) and not a.tasks.runs(foreign)
            a.send_input('hello')
            a.send_input('/exit')
            await run('a', a)
            assert '> hello\n\nsuccess (no tool calls)' in a_output.getvalue()
            # B's turn is still streaming, and its task is neither cancelled nor finished.
            assert [(record.status, record.outcome) for record in (background, foreign)] == [('running', None)] * 2
            assert [(record.outcome, record.user_stopped) for record in (own, earlier)] == [('cancelled', False)] * 2
            release.set()
            b.send_input('/exit')
    assert reasons == {'a': 'exit', 'b': 'exit'}
    assert 'B finished.' in b_output.getvalue()
    # The last runtime out closes the owner, which cancels what still runs.
    assert (background.status, background.outcome) == ('finished', 'cancelled')


async def test_task_events_reach_only_the_session_holding_the_conversation() -> None:
    """Wake-ups, plugin events and rows go to the conversation's session; every task's progress is kept."""
    router = TaskRouter(directory=None)
    held = {'a': 'a', 'b': 'b'}
    woken: list[str] = []
    fired: dict[str, list[TaskProgress]] = {'a': [], 'b': []}
    outputs = {'a': io.StringIO(), 'b': io.StringIO()}

    def view(name: str) -> Tasks:
        async def fire(event: TaskProgress) -> None:
            fired[name].append(event)

        tasks = Tasks(
            console=Console(file=outputs[name], width=120),
            conversation_id=lambda: held[name],
            router=router,
            fire=fire,
        )
        tasks.wake = lambda: woken.append(name)
        return tasks

    a, b = view('a'), view('b')
    assert router.view('b') is b and router.view('missing') is None
    record = task(conversation_id='b')
    record.background = True
    router.owner.records[record.id] = record
    assert b.running() and not a.running()
    record.status, record.outcome = 'finished', 'ok'
    end = DelegationEndEvent(
        agent_name='worker', outcome='failed', output='', truncated=False, usage=None, duration_seconds=1.0
    )
    await router.observe(DelegationTaskEvent(task=record, event=end))
    assert woken == ['b']
    assert [event.task.id for event in fired['b']] == [record.id] and not fired['a']
    assert 'failed' in outputs['b'].getvalue() and not outputs['a'].getvalue()
    # A finished task is no session's any more.
    assert not router.started

    # A session that moves to another conversation keeps the tasks it started counted as its own.
    record.status, record.outcome = 'running', None
    await router.observe(DelegationTaskEvent(task=record))
    assert router.started == {record.id: b}
    held['b'] = 'b2'
    assert b.running() and not b.records() and not a.running()
    # A session that takes the conversation over does not take over B's task, until it resumes the task.
    held['a'] = 'b'
    assert a.records() == (record,) and not a.running()
    start = DelegationStartEvent(agent_name='worker', task='inspect', truncated=False, inherits_tools=True, model=None)
    await router.observe(DelegationTaskEvent(task=record, event=start))
    assert router.started == {record.id: a}
    assert a.running() and not b.running()
    # A child runs with its parent's plugins, whichever session holds the conversation when it starts.
    child = task(task_id='c' * 32, parent_id=record.id, conversation_id='b')
    router.started[record.id] = b
    await router.observe(DelegationTaskEvent(task=child, event=start))
    assert router.started[child.id] is b and not a.runs(child)
    router.started[record.id] = a
    # With no session holding its conversation, a task's events reach none, and its progress is still kept.
    held['a'] = 'a'
    router.progress.clear()
    await a.observe(DelegationTaskEvent(task=record))
    assert woken == ['b'] and record.id in router.progress


async def test_the_owner_opens_once_for_overlapping_users(tmp_path: Path) -> None:
    router = TaskRouter(directory=tmp_path / 'tasks')
    first, second = (
        Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: name, router=router)
        for name in ('root', 'other')
    )
    async with router.opened(view=first) as owner:
        record = task()
        owner.records[record.id] = record
        async with router.opened(view=second) as again:
            assert again is owner
        # The session that left first ran none of it.
        assert record.status == 'running'
    assert (record.status, record.outcome) == ('finished', 'cancelled')
    assert (tmp_path / 'tasks' / f'{record.id}.json').exists()
    async with router.opened(view=first) as reopened:
        assert reopened.records[record.id].outcome == 'cancelled'


async def test_only_the_signalling_session_installs_the_ctrl_c_handler() -> None:
    """Overlapping runs, ending in either order, leave the terminal session's handler and then the original."""
    terminal, background = Interrupts(signals=True), Interrupts(signals=False)
    original = signal.getsignal(signal.SIGINT)
    seen: dict[str, object] = {}
    results: dict[str, bool] = {}
    started = {'terminal': anyio.Event(), 'background': anyio.Event()}
    release = {'terminal': anyio.Event(), 'background': anyio.Event()}
    done = {'terminal': anyio.Event(), 'background': anyio.Event()}

    async def run(interrupts: Interrupts, name: str) -> None:
        async def operation() -> None:
            started[name].set()
            await release[name].wait()
            seen[name] = signal.getsignal(signal.SIGINT)

        results[name] = await interrupts.run(operation())
        done[name].set()

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            # The background run starts first and ends while the terminal's still runs.
            group.start_soon(run, background, 'background')
            await started['background'].wait()
            assert signal.getsignal(signal.SIGINT) == original
            group.start_soon(run, terminal, 'terminal')
            await started['terminal'].wait()
            installed = signal.getsignal(signal.SIGINT)
            assert installed != original
            release['background'].set()
            await done['background'].wait()
            assert signal.getsignal(signal.SIGINT) == installed
            release['terminal'].set()
            await done['terminal'].wait()
            assert seen == {'background': installed, 'terminal': installed}
    assert results == {'background': True, 'terminal': True}
    assert signal.getsignal(signal.SIGINT) == original

    # A session without signals is still cancelled from a key or the browser.
    started['background'] = anyio.Event()
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(run, background, 'background')
            await started['background'].wait()
            assert signal.getsignal(signal.SIGINT) == original
            assert background.cancel(exit_on_repeat=False)
    assert results['background'] is False
    assert signal.getsignal(signal.SIGINT) == original


def test_a_runtime_takes_its_store_from_the_process(tmp_path: Path) -> None:
    """`store` and `process` name the settings twice, so passing both is refused rather than one dropped."""
    store = SettingsStore(tmp_path / 'config.db')
    with pytest.raises(TypeError, match='`store` or `process`, not both'):
        create_shell(
            Agent(TestModel(), deps_type=type(None)),
            deps=None,
            plugins=(),
            usage_limits=None,
            console=Console(file=io.StringIO()),
            settings=None,
            store=store,
            builtin_plugins=(),
            project=ProjectSettings(),
            headless=True,
            process=ClaiProcess.open(store),
        )
