"""`SessionRuntime`: live sessions built over the `ClaiProcess` whose stores and services they share."""

import asyncio
import io
import signal
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.subagents import DelegationEndEvent, DelegationTaskEvent
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import TaskEvent
from pydantic_clai2.runtime.process import ClaiProcess
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


def test_runtimes_share_the_process_services(tmp_path: Path) -> None:
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
    consoles = Console(file=io.StringIO()), Console(file=io.StringIO())
    first, second = (process.models.for_session(console=console) for console in consoles)
    assert first.store is second.store is process.store
    # Each session's sign-in prompts on its own console; every account's provider, and its refresh, is shared.
    assert (first.codex_auth().console, second.codex_auth().console) == consoles
    assert first.codex_auth() is first.codex_auth()
    provider = first.codex_auth().account_provider('openai-codex')
    assert second.codex_auth().account_provider('openai-codex') is provider
    assert process.models.codex_auth().provider is provider
    assert not (a.interrupts.signals or b.interrupts.signals)
    alone, _ = runtime_over(ClaiProcess.open(SettingsStore(tmp_path / 'other.db')), tmp_path / 'alone')
    assert alone.tasks.owner is not process.tasks.owner


async def test_closing_one_runtime_leaves_the_others_tasks_running(tmp_path: Path) -> None:
    """The owner stays open while any runtime runs, so A's exit does not cancel B's background task."""
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
            a.send_input('hello')
            a.send_input('/exit')
            await run('a', a)
            assert '> hello\n\nsuccess (no tool calls)' in a_output.getvalue()
            # B's turn is still streaming, and its task is neither cancelled nor finished.
            assert (background.status, background.outcome) == ('running', None)
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
    fired: dict[str, list[TaskEvent]] = {'a': [], 'b': []}
    outputs = {'a': io.StringIO(), 'b': io.StringIO()}

    def view(name: str) -> Tasks:
        async def fire(event: TaskEvent) -> None:
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
    assert router.started == {record.id: b}

    # A session that moves to another conversation keeps the tasks it started counted as its own.
    record.status, record.outcome = 'running', None
    held['b'] = 'b2'
    assert b.running() and not b.records() and not a.running()
    router.detach(b)
    assert router.views == [a] and not router.started
    await a.observe(DelegationTaskEvent(task=record))
    assert woken == ['b'] and record.id in router.progress


async def test_the_owner_opens_once_for_overlapping_users(tmp_path: Path) -> None:
    router = TaskRouter(directory=tmp_path / 'tasks')
    async with router.opened() as owner:
        record = task()
        owner.records[record.id] = record
        async with router.opened() as again:
            assert again is owner
        assert record.status == 'running'
    assert (record.status, record.outcome) == ('finished', 'cancelled')
    assert (tmp_path / 'tasks' / f'{record.id}.json').exists()
    async with router.opened() as reopened:
        assert reopened.records[record.id].outcome == 'cancelled'


async def test_only_the_signalling_session_installs_the_ctrl_c_handler() -> None:
    """Overlapping runs, ending in either order, leave the terminal session's handler and then the original."""
    terminal, background = Interrupts(signals=True), Interrupts(signals=False)
    original = signal.getsignal(signal.SIGINT)
    seen: dict[str, object] = {}
    gates = {name: asyncio.Event() for name in ('terminal', 'background', 'background_started', 'terminal_started')}

    async def run(interrupts: Interrupts, name: str) -> bool:
        async def operation() -> None:
            gates[f'{name}_started'].set()
            await gates[name].wait()
            seen[name] = signal.getsignal(signal.SIGINT)

        return await interrupts.run(operation())

    # The background run starts first and ends while the terminal's still runs.
    first = asyncio.create_task(run(background, 'background'))
    await gates['background_started'].wait()
    assert signal.getsignal(signal.SIGINT) == original
    second = asyncio.create_task(run(terminal, 'terminal'))
    await gates['terminal_started'].wait()
    installed = signal.getsignal(signal.SIGINT)
    assert installed != original
    gates['background'].set()
    assert await first
    assert signal.getsignal(signal.SIGINT) == installed
    gates['terminal'].set()
    assert await second
    assert seen == {'background': installed, 'terminal': installed}
    assert signal.getsignal(signal.SIGINT) == original

    # A session without signals is still cancelled from a key or the browser.
    for gate in gates.values():
        gate.clear()
    cancelled = asyncio.create_task(run(background, 'background'))
    await gates['background_started'].wait()
    assert background.cancel(exit_on_repeat=False)
    assert not await cancelled
    assert signal.getsignal(signal.SIGINT) == original
