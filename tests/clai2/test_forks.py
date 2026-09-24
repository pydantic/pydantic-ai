"""`/fork` and `/forks`: parsing, history snapshots, cancellation, and status."""

import asyncio
import io
import signal
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from rich.console import Console
from rich.text import Text

from pydantic_clai2 import chat
from pydantic_clai2._app import create_shell
from pydantic_clai2._session import Session
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.forks import USAGE, Forks, parse_fork_args
from pydantic_clai2.image_input import ImageInput
from pydantic_clai2.interrupts import Interrupts
from pydantic_clai2.live_prompt import LivePrompt
from pydantic_clai2.plugins import HostEvent, TurnEnd, TurnStart
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.spinners import BUILTIN_SPINNERS, DEFAULT_SPINNER

PromptT = TypeVar('PromptT')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def last_prompt(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    raise AssertionError('no prompt')  # pragma: no cover


class Model:
    """Answers each prompt, records what the model saw, and can block or fail on request."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def respond(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        prompt = last_prompt(messages)
        self.seen[prompt] = len(messages)
        if prompt == 'block':
            self.started.set()
            await self.release.wait()
        if prompt == 'explode':
            raise RuntimeError('provider down\nsecond line')
        if prompt == 'cancel me':
            signal.raise_signal(signal.SIGINT)
            await asyncio.Event().wait()
        yield f'**answer** to {prompt}'


def shell_for(tmp_path: Path, model: Model, output: io.StringIO):
    return create_shell(
        Agent(FunctionModel(stream_function=model.respond)),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=output, width=200),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('fix the bug', (None, 'fix the bug')),
        ('@openai:gpt-5  fix it ', ('openai:gpt-5', 'fix it')),
        ('@test', ('test', '')),
        ('@ fix it', (None, 'fix it')),
        ('@test\tfix it', ('test', 'fix it')),
        ('@test\nfix it\nand this', ('test', 'fix it\nand this')),
    ],
)
def test_parse_fork_args(text: str, expected: tuple[str | None, str]) -> None:
    assert parse_fork_args(text) == expected


def test_raw_commands_keep_quotes() -> None:
    commands = Commands()
    commands.register(Command(name='echo', description='echo', handler=lambda args: repr(args), raw=True))
    commands.register(Command(name='words', description='words', handler=lambda args: repr(args)))
    assert commands.execute('/echo  don\'t "split" me ') == repr(['don\'t "split" me '])
    assert commands.execute('/echo') == '[]'
    assert commands.execute('/words "a b" c') == repr(['a b', 'c'])
    assert commands.execute('/') == '/echo: echo\n/words: words'


async def test_fork_copies_history_and_reports(tmp_path: Path) -> None:
    model, output = Model(), io.StringIO()
    shell = shell_for(tmp_path, model, output)
    await shell.session.prompt('first')
    before = shell.session.messages

    started = await shell.commands.execute_async("/fork what's next")
    assert started.startswith('fork #1 (agent default) started in the background.')
    (record,) = shell.forks.records
    await record.task

    assert model.seen["what's next"] == len(before) + 1
    assert shell.session.messages == before
    assert record.status == 'done'
    assert record.session_id is not None
    assert record.session_id != shell.session.summary.id
    text = output.getvalue()
    assert 'FORK #1 RESPONSE' in text
    assert "\x1b[1manswer\x1b[22m to what's next" in text  # Rendered as Markdown, not raw asterisks.
    assert 'FORK #1 RESPONSE  agent default' in text
    assert 'fork #1 finished in' in text
    assert f'Continue it with /resume {record.session_id}' in text


async def test_fork_without_history_starts_fresh_with_model_override(tmp_path: Path) -> None:
    model, output = Model(), io.StringIO()
    shell = shell_for(tmp_path, model, output)
    await shell.commands.execute_async('/fork @test hello')
    (record,) = shell.forks.records
    await record.task
    assert model.seen == {}  # `@test` replaced the agent's FunctionModel.
    assert record.model == 'test'
    assert 'success (no tool calls)' in output.getvalue()


async def test_snapshot_failure_forks_fresh(tmp_path: Path) -> None:
    model, output = Model(), io.StringIO()
    console = Console(file=output, width=200)

    def broken() -> list[ModelMessage]:
        raise RuntimeError('no history')

    forks = Forks(
        console=console,
        history=broken,
        spawn=lambda _, history: Session(
            Agent(FunctionModel(stream_function=model.respond)), deps=None, message_history=history
        ),
    )
    await forks.fork_command(['hello'])
    (record,) = forks.records
    await record.task
    assert "couldn't copy the current conversation" in output.getvalue()
    assert model.seen == {'hello': 1}
    assert record.session_id is None
    assert 'Continue it with' not in output.getvalue()


async def test_cancel_status_and_failures(tmp_path: Path) -> None:
    model, output = Model(), io.StringIO()
    shell = shell_for(tmp_path, model, output)
    execute = shell.commands.execute_async

    assert await execute('/forks') == 'No forks yet. Start one with /fork [@model] PROMPT.'
    assert await execute('/fork') == USAGE
    with pytest.raises(ValueError, match='Fork what'):
        await execute('/fork @test')

    await execute('/fork block')
    await execute('/fork explode')
    await model.started.wait()
    first, second = shell.forks.records
    await asyncio.gather(second.task)
    assert second.status == 'failed'
    assert 'fork #2 failed after' in output.getvalue()
    assert 'provider down' in output.getvalue()
    assert 'second line' not in output.getvalue()

    assert await execute('/forks') == '1 running, 1 failed'
    assert 'block' in output.getvalue()
    with pytest.raises(ValueError, match='Usage: /forks'):
        await execute('/forks now')

    with pytest.raises(ValueError, match="'x' is not a fork id"):
        await execute('/fork cancel x')
    with pytest.raises(ValueError, match='No fork #9'):
        await execute('/fork cancel 9')
    assert await execute('/fork cancel 1') == 'Cancelling fork #1...'
    await asyncio.gather(first.task, return_exceptions=True)
    assert first.status == 'cancelled'
    assert 'fork #1 cancelled after' in output.getvalue()
    assert await execute('/fork cancel 1') == 'fork #1 already cancelled.'
    assert await execute('/forks') == '1 failed, 1 cancelled'


async def test_output_waits_while_busy(tmp_path: Path) -> None:
    model, output = Model(), io.StringIO()
    shell = shell_for(tmp_path, model, output)
    forks = shell.forks
    async with forks.busy():
        async with forks.busy():
            await forks.fork_command(['hello'])
            (record,) = forks.records
            while record.status == 'running':
                await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert 'FORK #1 RESPONSE' not in output.getvalue()
        assert forks.cancel_running() == 0
    await record.task
    assert 'FORK #1 RESPONSE' in output.getvalue()


async def test_notices_wait_while_busy(tmp_path: Path) -> None:
    model, output = Model(), io.StringIO()
    shell = shell_for(tmp_path, model, output)
    forks = shell.forks
    async with forks.busy():
        await forks.fork_command(['block'])
        await forks.fork_command(['explode'])
        await model.started.wait()
        first, second = forks.records
        await asyncio.gather(second.task)
        assert forks.cancel('1') == 'Cancelling fork #1...'
        await asyncio.gather(first.task, return_exceptions=True)
        assert (first.status, second.status) == ('cancelled', 'failed')
        assert 'fork #' not in output.getvalue()
    text = output.getvalue()
    assert 'fork #2 failed after' in text
    assert 'fork #1 cancelled after' in text


async def test_announcements_own_the_terminal(tmp_path: Path) -> None:
    model, output = Model(), io.StringIO()
    shell = shell_for(tmp_path, model, output)
    forks = shell.forks
    async with forks.busy():
        await forks.fork_command(['one'])
        await forks.fork_command(['two'])
        first, second = forks.records
        while 'running' in (first.status, second.status):
            await asyncio.sleep(0)
    # Idle wakes both forks, but a command claims the terminal before either takes the lock.
    async with forks.busy():
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert 'FORK #' not in output.getvalue()
    await asyncio.gather(first.task, second.task)
    text = output.getvalue()
    # Announcements run one at a time: each banner is followed by its own finished line.
    blocks = sorted((text.index(f'FORK #{n} RESPONSE'), text.index(f'fork #{n} finished')) for n in (1, 2))
    assert blocks[0][1] < blocks[1][0]


async def test_structured_output_prints_as_is() -> None:
    output = io.StringIO()
    forks = Forks(
        console=Console(file=output, width=200),
        history=lambda: [],
        spawn=lambda _, history: Session(Agent(TestModel(), output_type=list[int]), deps=None),
    )
    await forks.fork_command(['numbers'])
    (record,) = forks.records
    await record.task
    text = output.getvalue()
    assert 'FORK #1 RESPONSE' in text
    assert '\n[0]\n' in text


async def test_first_forks_on_a_new_database_all_run(tmp_path: Path) -> None:
    for attempt in range(25):
        model, output = Model(), io.StringIO()
        shell = shell_for(tmp_path / str(attempt), model, output)
        for prompt in ('one', 'two', 'three'):
            await shell.commands.execute_async(f'/fork {prompt}')
        await asyncio.gather(*(record.task for record in shell.forks.records))
        assert [record.status for record in shell.forks.records] == ['done', 'done', 'done'], output.getvalue()


async def test_forks_fire_turn_hooks() -> None:
    model, output = Model(), io.StringIO()
    events: list[HostEvent] = []

    async def fire(event: HostEvent) -> None:
        events.append(event)
        if isinstance(event, TurnStart) and event.text == 'forbidden':
            event.cancel('policy says no')
        elif isinstance(event, TurnStart) and event.text == 'draft':
            event.text = 'rewritten'
        elif isinstance(event, TurnStart) and event.text == 'broken hook':
            raise RuntimeError('hook broke')

    forks = Forks(
        console=Console(file=output, width=200),
        history=lambda: [],
        spawn=lambda _, history: Session(Agent(FunctionModel(stream_function=model.respond)), deps=None),
        fire=fire,
    )
    with pytest.raises(ValueError, match='Fork cancelled by a plugin: policy says no'):
        await forks.fork_command(['forbidden'])
    with pytest.raises(RuntimeError, match='hook broke'):
        await forks.fork_command(['broken hook'])
    assert forks.records == ()
    await forks.fork_command(['draft'])
    await forks.fork_command(['explode'])
    await forks.fork_command(['block'])
    await model.started.wait()
    done, failed, blocked = forks.records
    assert done.prompt == 'rewritten'
    await asyncio.gather(done.task, failed.task)
    forks.cancel('3')
    await asyncio.gather(blocked.task, return_exceptions=True)
    assert 'rewritten' in model.seen
    ends = {event.text: event for event in events if isinstance(event, TurnEnd)}
    assert ends['rewritten'].outcome == 'completed'
    assert ends['rewritten'].result is not None
    assert ends['explode'].outcome == 'failed'
    assert isinstance(ends['explode'].error, RuntimeError)
    assert ends['block'].outcome == 'cancelled'
    # A refused fork still closes its turn, as a refused foreground turn does.
    assert ends['forbidden'].outcome == 'cancelled'
    assert ends['broken hook'].outcome == 'failed'
    assert isinstance(ends['broken hook'].error, RuntimeError)


async def test_live_rows_follow_each_fork(tmp_path: Path) -> None:
    model, output = Model(), io.StringIO()
    forks = shell_for(tmp_path, model, output).forks
    assert forks.rows('*') == ()
    async with forks.busy():
        await forks.fork_command(['block'])
        await forks.fork_command(['hello'])
        await model.started.wait()
        running, finished = forks.records
        while finished.status == 'running':
            await asyncio.sleep(0)
        first, second = (Text.from_ansi(row).plain for row in forks.rows('*'))
        assert first.startswith(' FORK #1  agent default  * 00:0')
        assert first.endswith('starting')
        assert second.startswith(' FORK #2  agent default  \u2713 00:0')
        assert second.endswith('done, prints after this turn')
    await asyncio.sleep(0)
    assert finished.announced
    assert [Text.from_ansi(row).plain[:9] for row in forks.rows('*')] == [' FORK #1 ']
    for activity in ('thinking', 'tool: grep', 'running: grep', 'responding', 'working'):
        running.progress.activity = activity
        assert Text.from_ansi(forks.rows('*')[0]).plain.endswith(activity)
    forks.cancel('1')
    await asyncio.gather(running.task, return_exceptions=True)
    assert forks.rows('*') == ()


async def test_stream_events_drive_the_activity(tmp_path: Path) -> None:
    model, output = Model(), io.StringIO()
    forks = shell_for(tmp_path, model, output).forks
    await forks.fork_command(['hello'])
    (record,) = forks.records
    await record.task
    assert record.progress.activity == 'responding'


@pytest.mark.parametrize(
    ('height', 'panel_rows'),
    [
        (12, ['row 0', 'row 1', '+3 more']),
        (8, ['row 0', '+4 more']),
        (7, ['+5 more']),
        (6, []),
    ],
)
def test_editor_paints_fork_rows_above_the_rule(height: int, panel_rows: list[str]) -> None:
    rows = [f'row {index}' for index in range(5)]
    seen: list[str] = []

    def panel(glyph: str) -> list[str]:
        seen.append(glyph)
        return rows

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        live = LivePrompt(
            console=Console(file=io.StringIO(), force_terminal=True, width=40, height=height),
            commands=Commands(),
            history=InMemoryHistory(),
            images=ImageInput(),
            interrupts=Interrupts(),
            toolbar=lambda: [('', 'ready')],
            clock=lambda: 0.0,
            panel=panel,
        )
        plain = [Text.from_ansi(row).plain for row in live.frame()]
    assert seen == [BUILTIN_SPINNERS[DEFAULT_SPINNER].frames[0]]
    # The surface keeps `height - 2` rows, so fork rows give way before the title, draft, rule, and footer.
    assert len(plain) <= height - 2
    assert plain[: len(panel_rows)] == panel_rows
    assert plain[len(panel_rows)].startswith('\u2500')


async def test_a_fork_that_cannot_start_closes_its_turn() -> None:
    events: list[HostEvent] = []

    async def fire(event: HostEvent) -> None:
        events.append(event)

    def spawn(model: str | None, history: Sequence[ModelMessage]) -> Session[None, str]:
        raise ValueError(f'Unknown model {model}')

    forks = Forks(console=Console(file=io.StringIO()), history=lambda: [], spawn=spawn, fire=fire)
    with pytest.raises(ValueError, match='Unknown model nope'):
        await forks.fork_command(['@nope hello'])
    assert forks.records == ()
    start, end = events
    assert isinstance(start, TurnStart)
    assert isinstance(end, TurnEnd)
    assert (end.text, end.outcome, str(end.error)) == ('hello', 'failed', 'Unknown model nope')


def test_completion(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.set('model', 'test')
    shell = create_shell(
        Agent('test'),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    forks = shell.forks
    assert list(forks.complete([''])) == ['cancel', *(f'@{name}' for name in store.models())]
    assert list(forks.complete(['@test', ''])) == []


class Script:
    def __init__(self, steps: list[str | Callable[[], Awaitable[str]]]) -> None:
        self.steps = steps

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        steps = self.steps

        class Prompt(Generic[PromptT]):
            def __init__(self, **kwargs: object) -> None:
                pass

            async def prompt_async(self, label: str, **kwargs: object) -> str:
                step = steps.pop(0)
                return step if isinstance(step, str) else await step()

        monkeypatch.setattr('pydantic_clai2._app.PromptSession', Prompt)


async def test_cancelled_turn_takes_forks_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, output = Model(), io.StringIO()

    async def after_fork_started() -> str:
        await model.started.wait()
        return 'cancel me'

    Script(['/fork block', after_fork_started, '/exit']).install(monkeypatch)
    await chat(
        Agent(FunctionModel(stream_function=model.respond)),
        deps=None,
        console=Console(file=output, width=200),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    text = output.getvalue()
    assert 'Turn cancelled' in text
    assert 'Cancelled 1 running fork(s) with the turn.' in text
    assert 'fork #1 cancelled after' in text


async def test_shell_passthrough_holds_fork_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, output = Model(), io.StringIO()

    async def after_fork_started() -> str:
        await model.started.wait()
        return '!make test'

    async def run_shell_command(command: str, *, console: Console, interrupts: object) -> None:
        # The fork finishes while the command owns the terminal; its output must wait.
        model.release.set()
        for _ in range(20):
            await asyncio.sleep(0.01)
        assert 'FORK #1' not in output.getvalue()
        console.print(f'ran {command}')

    monkeypatch.setattr('pydantic_clai2._app.run_shell_command', run_shell_command)

    async def after_command() -> str:
        # Idle again: the held banner prints before the next prompt returns.
        for _ in range(20):
            if 'FORK #1' in output.getvalue():
                break
            await asyncio.sleep(0.01)
        return '/exit'

    Script(['/fork block', after_fork_started, after_command]).install(monkeypatch)
    await chat(
        Agent(FunctionModel(stream_function=model.respond)),
        deps=None,
        console=Console(file=output, width=200),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    text = output.getvalue()
    assert text.index('ran make test') < text.index('FORK #1 RESPONSE')


async def test_exit_cancels_running_forks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, output = Model(), io.StringIO()

    async def after_fork_started() -> str:
        await model.started.wait()
        return '/exit'

    Script(['/fork block', after_fork_started]).install(monkeypatch)
    await chat(
        Agent(FunctionModel(stream_function=model.respond)),
        deps=None,
        console=Console(file=output, width=200),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    assert 'fork #1 cancelled after' in output.getvalue()
