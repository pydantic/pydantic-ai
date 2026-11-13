"""The live `/tasks` view: the sidebar, the selected task's transcript, and steering it."""

import asyncio
import io
import time
from itertools import count
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.sub_agents import DelegationOutcome, DelegationTask
from pydantic_clai2.runtime.tasks import Tasks
from pydantic_clai2.ui.menus.task_view import Action, TaskView, perform, task_rows, transcript_blocks
from tests.clai2.live_script import frame, press

SIZE = (110, 26)


def tasks_ui() -> Tasks:
    return Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)


_STARTS = count(1)
"""Start times in the order the tasks are made, whatever the clock's resolution."""


def task(
    letter: str = 'a', *, agent: str = 'worker', parent: str | None = None, outcome: DelegationOutcome | None = None
) -> DelegationTask:
    record = DelegationTask(
        id=letter * 32,
        parent_id=parent and parent * 32,
        agent_name=agent,
        prompt=f'inspect {letter}',
        conversation_id='root',
        started_at=float(next(_STARTS)),
    )
    if outcome is not None:
        record.status, record.outcome, record.finished_at = 'finished', outcome, time.time()
    return record


def marked(screen: str) -> str:
    """The sidebar line of the selected task."""
    return next(line for line in screen.splitlines() if '❯' in line)


class Driver:
    """A `TaskView` over real `Tasks`, with the actions it asks for recorded instead of run."""

    def __init__(self, *records: DelegationTask, size: tuple[int, int] = SIZE, error: str | None = None) -> None:
        self.ui = tasks_ui()
        self.ui.owner.records = {record.id: record for record in records}
        self.acts: list[tuple[Action, str, str]] = []
        self.error = error
        self.view = TaskView(snapshot=lambda: task_rows(self.ui), act=self.act, size=lambda: size)
        self.view.app.running = True

    def act(self, action: Action, task_id: str, text: str) -> str:
        self.acts.append((action, task_id[0], text))
        if self.error is not None:
            raise ValueError(self.error)
        return f'{action} done'

    def press(self, *keys: str) -> str:
        """Press `keys`, typing the single characters among them, and return the next frame."""
        return press(self.view.app, *keys)

    def type(self, text: str) -> str:
        return self.press(*text)

    def screen(self, *, refresh: bool = False) -> str:
        """Draw a frame, first letting enough time pass for a new snapshot when `refresh`."""
        for _ in range(3 * refresh):
            frame(self.view.app, 0.1)
        return frame(self.view.app)


def test_sidebar_shows_the_tree_and_the_transcript_follows_the_selection() -> None:
    parent, child, failed = task('a', agent='Explore'), task('b', parent='a'), task('c', agent='Plan', outcome='failed')
    failed.output, failed.resumable = 'the model refused', False
    driver = Driver(parent, child, failed)
    assert '❯   ⠋ worker bbbbbbbb' in driver.screen(), 'opens on the newest running task, indented under its parent'
    screen = driver.press('up')
    assert '❯ ⠋ Explore aaaaaaaa' in screen and '✗ Plan cccccccc' in screen
    assert 'Explore · aaaaaaaa · starting' in screen
    assert 'inspect a' in screen and '› Type to steer Explore, then Enter.' in screen
    assert 'Ctrl+B background - Ctrl+X stop - Esc close - ↑↓ tasks - PgUp/PgDn scroll' in screen

    screen = driver.press('down', 'down', 'down')
    assert '❯ ✗ Plan cccccccc' in screen, 'the selection stops at the last task'
    assert 'Plan · cccccccc · failed' in screen
    assert 'failed: the model refused' in screen and 'inspect a' not in screen
    assert 'Plan cannot be resumed, so it takes no messages.' in screen
    assert '/tasks  Esc close - ↑↓ tasks' in screen, 'a one-shot task that finished answers to no control key'
    assert '❯ ⠋ Explore aaaaaaaa' in driver.press('up', 'up', 'up')


def test_selection_stays_on_its_task_as_the_tasks_change() -> None:
    first, second = task('a'), task('b')
    driver = Driver(first, second)
    driver.ui.owner.records = {record.id: record for record in (task('0'), first, second)}
    second.status, second.outcome, second.finished_at = 'finished', 'ok', time.time()
    screen = driver.screen(refresh=True)
    assert '✓ worker bbbbbbbb' in marked(screen)
    assert screen.count('bbbbbbbb is now ok.') == 2, 'over the input and in the activity log'

    driver.ui.owner.records = {first.id: first}
    assert 'worker aaaaaaaa' in marked(driver.screen(refresh=True)), 'a task that left gives its selection away'
    driver.ui.owner.records = {}
    screen = driver.screen(refresh=True)
    assert 'No tasks yet: ask the agent to delegate work.' in screen and '/tasks  Esc close - ↑↓ tasks' in screen
    assert driver.press('down', 'enter', 'ctrl-x') and driver.acts == []


def test_transcript_streams_the_selected_task() -> None:
    other, record = task('b'), task('a')
    record.messages = [
        ModelRequest(parts=[UserPromptPart('inspect a')]),
        ModelResponse(
            parts=[TextPart('Reading the **registry**.'), ToolCallPart('grep', {'pattern': '**/*.py'}, 'call')]
        ),
        ModelRequest(
            parts=[ToolReturnPart('grep', 'tasks.py: def steer', 'call'), UserPromptPart('focus on the tests')]
        ),
    ]
    driver = Driver(record, other)
    screen = driver.screen()
    assert 'Reading the registry.' in screen and 'grep {"pattern":"**/*.py"}' in screen, (
        'a tool call is code, not Markdown'
    )
    assert screen.index('inspect a') < screen.index('Reading') < screen.index('focus on the tests')
    assert 'def steer' not in screen, 'a tool result stays out of the transcript'

    streaming = driver.view.transcript.widget
    driver.ui.partial[record.id] = 'The tests cover'
    assert 'The tests cover' in driver.screen(refresh=True)
    driver.ui.partial[record.id] = 'The tests cover steering.'
    assert 'The tests cover steering.' in driver.screen(refresh=True)
    del driver.ui.partial[record.id]
    assert 'The tests cover steering.' in driver.screen(refresh=True), (
        'a part that ended stays until its response lands'
    )
    record.messages.append(
        ModelResponse(parts=[TextPart('The tests cover steering.'), ToolCallPart('pytest', {}, 'run')])
    )
    screen = driver.screen(refresh=True)
    assert screen.count('The tests cover steering.') == 1 and 'pytest {}' in screen
    assert driver.view.transcript.widget is streaming, 'text that only grew is appended, keeping the scroll position'

    assert 'inspect b' in driver.press('up')
    assert driver.view.transcript.widget is not streaming


def test_transcript_keeps_its_newest_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_clai2.ui.menus.task_view._TAIL', 40)
    record = task()
    record.messages = [ModelResponse(parts=[TextPart(f'answer {number} ' + 'x' * 20)]) for number in range(6)]
    driver = Driver(record)
    screen = driver.screen()
    assert 'earlier messages are hidden' in screen and 'answer 5' in screen
    assert 'answer 0' not in screen and 'inspect a' not in screen

    record.messages[:] = [ModelResponse(parts=[TextPart('compacted')])]
    screen = driver.screen(refresh=True)
    assert 'compacted' in screen and 'answer 5' not in screen, 'a history rewritten shorter is shown from its start'


async def test_a_settled_task_shows_the_history_and_usage_its_owner_saved(tmp_path: Path) -> None:
    """A settled task's history is in its owner's saved file only: the view and its usage read it from there once."""
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=tmp_path)
    record = task(outcome='ok')
    record.messages = [
        ModelRequest(parts=[UserPromptPart('inspect a')]),
        ModelResponse(parts=[TextPart('Found the **registry**.')], usage=RequestUsage(input_tokens=7)),
    ]
    reads: list[str] = []

    def history(task_id: str) -> list[ModelMessage]:
        reads.append(task_id)
        return ui.history(task_id)

    def act(action: Action, task_id: str, text: str) -> str:
        raise AssertionError('no action is taken')  # pragma: no cover

    async with ui.router.opened(view=ui):
        assert await ui.owner.restore(record)
        assert record.messages == []
        # Saved by another owner, so its usage is read from the file, in a worker thread as async code does.
        (info,) = await anyio.to_thread.run_sync(ui.listing)
        assert (info.usage.requests, info.usage.input_tokens) == (1, 7)

        def show() -> None:
            view = TaskView(snapshot=lambda: task_rows(ui), act=act, history=history, size=lambda: SIZE)
            view.app.running = True
            assert 'Found the registry.' in frame(view.app)
            for _ in range(3):
                frame(view.app, 0.1)
            assert 'Found the registry.' in frame(view.app)

        # The view runs on its own thread, as `open_tasks` runs it.
        await anyio.to_thread.run_sync(show)
        assert reads == [record.id], 'read once while it stays selected'


def test_transcript_blocks_escape_terminal_control_and_show_the_directory() -> None:
    record = task()
    record.working_directory = '/policy'
    record.messages = [ModelResponse(parts=[TextPart('safe\x1b[2Jtext')])]
    assert transcript_blocks(record) == ['*Directory: /policy*', '> inspect a', 'safe\\x1b[2Jtext']


def test_enter_steers_a_running_task_and_prompts_a_finished_one() -> None:
    running, finished = task('a'), task('b', outcome='ok')
    driver = Driver(running, finished)
    assert 'worker aaaaaaaa' in marked(driver.screen()), 'opens on a running task, not a newer finished one'
    assert '› focus on tx▌' in driver.type('focus on tx')
    assert '› focus on the tests▌' in driver.press('backspace', '\t', *'he tests'), 'a tab is not text'
    pasted = driver.press(*' first', 'enter', 'enter', *'then')
    assert '› focus on the tests first then▌' in pasted and driver.acts == [], 'a pasted line break is a space'
    driver.press(*['backspace'] * len(' first then'))
    screen = driver.press('enter')
    assert driver.acts == [('steer', 'a', 'focus on the tests')]
    assert 'Steer · steer done' in screen and '› Type to steer worker, then Enter.' in screen

    screen = driver.press('down')
    assert 'Prompt ━' in screen and '› Type to prompt worker again, then Enter.' in screen
    assert '/tasks  Ctrl+R resume - Esc close' in screen
    driver.type('and the docs?')
    driver.press('enter')
    assert driver.acts[-1] == ('prompt', 'b', 'and the docs?')
    assert driver.press('enter') and len(driver.acts) == 2, 'an empty message is not sent'


def test_a_message_that_could_not_be_sent_stays_in_the_input() -> None:
    driver = Driver(task(), error='aaaaaaaa finished before it could read that.')
    driver.type('focus')
    screen = driver.press('enter')
    assert '› focus▌' in screen
    assert 'Steer · aaaaaaaa finished before it could read that.' in screen

    one_shot = task(outcome='failed')
    one_shot.resumable = False
    finished = Driver(one_shot)
    finished.type('again')
    screen = finished.press('enter')
    assert '› again▌' in screen and finished.acts == []
    assert 'Finished · worker cannot be resumed, so it takes no messages.' in screen


def test_control_keys_act_on_the_selected_task() -> None:
    running, finished = task('a'), task('b', outcome='ok')
    driver = Driver(running, finished)
    driver.press('ctrl-b', 'ctrl-x')
    assert driver.acts == [('background', 'a', ''), ('stop', 'a', '')]
    assert 'Resume needs a finished task.' in driver.press('ctrl-r')
    assert driver.view.app.running

    assert 'Stop needs a running task.' in driver.press('down', 'ctrl-x')
    assert 'Background needs a running task.' in driver.press('ctrl-b')
    driver.press('ctrl-r')
    assert driver.acts[2:] == [('resume', 'b', '')]
    assert not driver.view.app.running, 'resuming hands the terminal back to the agent'


def test_a_failed_resume_keeps_the_view_open() -> None:
    driver = Driver(task(outcome='ok'), error='This CLAI has no prompt to ask the agent to resume a task at.')
    assert 'This CLAI has no prompt' in driver.press('ctrl-r')
    assert driver.view.app.running


def test_escape_clears_the_message_then_closes() -> None:
    driver = Driver(task())
    driver.type('never mind')
    assert 'Type to steer worker, then Enter.' in driver.press('escape')
    assert driver.view.app.running
    driver.press('escape')
    assert not driver.view.app.running


def test_page_keys_scroll_the_transcript() -> None:
    record = task()
    record.messages = [ModelResponse(parts=[TextPart(f'line {number}')]) for number in range(60)]
    driver = Driver(record)
    assert 'line 59' in driver.screen()
    screen = driver.press('page-up')
    assert 'line 59' not in screen
    assert 'line 59' in driver.press('page-down', 'page-down')


def test_the_view_closes_when_its_owner_stops_it(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = Driver(task())
    monkeypatch.setattr('pydantic_clai2.ui.menus.task_view.worker_stopping', lambda: True)
    driver.screen()
    assert not driver.view.app.running


@pytest.mark.parametrize(
    ('size', 'shown', 'hidden'),
    [
        ((110, 26), ['Tasks', 'Activity', 'Steer'], []),
        ((110, 16), ['Tasks', 'Steer'], ['Activity']),
        ((60, 26), ['Steer', 'worker · aaaaaaaa'], ['Tasks', 'Activity']),
    ],
)
def test_a_small_terminal_gives_its_space_to_the_transcript(
    size: tuple[int, int], shown: list[str], hidden: list[str]
) -> None:
    record = task()
    record.finished_at = None
    record.started_at = time.time() - 3725
    screen = Driver(record, size=size).screen()
    assert all(title in screen for title in shown) and not any(title in screen for title in hidden)
    assert ('1h02m' in screen) == ('Tasks' in shown), 'a running task shows how long it has run'
    assert ('PgUp/PgDn scroll' in screen) == (size[0] == 110), 'the footer keeps the hints that fit'


async def test_perform_steers_while_running_then_offers_a_prompt() -> None:
    gate = asyncio.Event()
    prompted: list[str] = []
    runs: list[asyncio.Task[object] | None] = []

    async def child(record: DelegationTask) -> str:
        await gate.wait()
        return 'done'

    async def prompt_task(record: DelegationTask, text: str) -> None:
        prompted.append(text)
        runs.append(asyncio.current_task())

    ui = Tasks(
        console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None, prompt_task=prompt_task
    )
    async with ui.router.opened(view=ui):
        await ui.owner.delegate(
            agent_name='worker',
            prompt='go',
            conversation_id='root',
            model=None,
            background=True,
            resume=None,
            run=child,
        )
        (record,) = ui.owner.records.values()
        assert await perform(ui, 'steer', record.id, 'focus on the tests') == f'Steered {record.id[:8]}.'
        assert await perform(ui, 'background', record.id) == f'Backgrounded {record.id[:8]}.'
        gate.set()
        await ui.owner.wait_children(None)
        with pytest.raises(ValueError, match='finished before it could read that'):
            await perform(ui, 'steer', record.id, 'too late')
        assert await perform(ui, 'prompt', record.id, 'and the docs?') == f'Prompted {record.id[:8]}.'
        await asyncio.gather(*filter(None, runs))
        assert prompted == ['and the docs?']
        with pytest.raises(LookupError):
            await perform(ui, 'stop', 'unknown')


async def test_perform_stops_and_resumes() -> None:
    sent: list[str] = []
    started = asyncio.Event()

    async def child(record: DelegationTask) -> str:
        started.set()
        await asyncio.Event().wait()
        return 'unreachable'  # pragma: no cover

    ui = Tasks(
        console=Console(file=io.StringIO()),
        conversation_id=lambda: 'root',
        directory=None,
        send_input=sent.append,
    )
    async with ui.router.opened(view=ui):
        await ui.owner.delegate(
            agent_name='worker',
            prompt='go',
            conversation_id='root',
            model=None,
            background=True,
            resume=None,
            run=child,
        )
        (record,) = ui.owner.records.values()
        await started.wait()
        assert await perform(ui, 'stop', record.id) == f'Stopped {record.id[:8]}.'
        assert record.user_stopped and record.status == 'finished'
        assert await perform(ui, 'resume', record.id) == f'Resuming {record.id[:8]}.'
        assert len(sent) == 1 and record.id in sent[0] and not record.user_stopped
