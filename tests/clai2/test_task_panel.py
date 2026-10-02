"""The task panel's theme roles and available action hints."""

import io
import time
from typing import Literal

import pytest
from rich.console import Console

from pydantic_ai_harness.subagents import DelegationTaskEvent
from pydantic_clai2.runtime.tasks import Tasks
from pydantic_clai2.ui.rendering import theme
from tests.clai2.test_tasks import task


@pytest.mark.parametrize('palette', ['default', theme.names()[1]])
@pytest.mark.parametrize('background', [False, True])
def test_selected_theme_and_available_background_action(palette: str, background: bool) -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    record = task()
    record.background, record.backgroundable = background, True
    ui.owner.records[record.id] = record
    with theme.use(lambda: palette):
        rows = ui.rows('*')
        accent, muted = theme.sgr(theme.ACCENT), theme.sgr(theme.MUTED)
        mode = 'background' if background else 'foreground'
        assert rows[0].startswith(accent + '* ' + theme.sgr(theme.INFO) + 'worker ')
        assert f'{muted}[{record.id[:8]}] 0s · {mode} · {accent}starting' in rows[0]
        assert theme.sgr(theme.ACCENT) + '/tasks' in rows[-1]
        assert ('Ctrl+B' in rows[-1]) is (not background)


@pytest.mark.parametrize(('outcome', 'role'), [('failed', theme.ERROR), ('cancelled', theme.WARNING)])
def test_unsuccessful_outcomes_use_their_own_colour(outcome: Literal['failed', 'cancelled'], role: str) -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    record = task()
    record.status, record.outcome, record.finished_at = 'finished', outcome, time.time()
    ui.owner.records[record.id] = record
    with theme.use(lambda: theme.names()[1]):
        assert theme.sgr(role) + outcome in ui.rows('*')[0]


def test_start_row_is_styled_like_a_tool_call() -> None:
    from pydantic_ai_harness.subagents import DelegationStartEvent
    from pydantic_clai2.runtime.tasks import task_row

    event = DelegationStartEvent(
        agent_name='self', task='count\nthe files', truncated=False, inherits_tools=True, model=None
    )
    event.task_id = 'abcdef0123'
    row = task_row(event)
    assert row is not None
    assert row.plain == '● general-purpose [abcdef01] count the files'
    assert row.no_wrap and row.overflow == 'ellipsis'
    assert [(row.plain[span.start : span.end], span.style) for span in row.spans] == [
        ('general-purpose', theme.color(theme.ACCENT)),
        (' [abcdef01] count the files', theme.color(theme.MUTED)),
    ]


@pytest.mark.parametrize('activity', ['running: run_code', 'tool: run_code'])
def test_panel_shows_the_tool_name_without_a_prefix(activity: str) -> None:
    from pydantic_clai2.ui.rendering.status import Status

    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    record = task()
    ui.owner.records[record.id] = record
    ui.progress[record.id] = Status(activity=activity)
    row = ui.rows('*')[0]
    assert f'{theme.sgr(theme.ACCENT)}run_code' in row
    assert 'running:' not in row and 'tool:' not in row


def test_non_backgroundable_foreground_task_has_no_background_hint() -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    record = task()
    record.backgroundable = False
    ui.owner.records[record.id] = record
    assert '/tasks' in ui.rows('*')[-1]
    assert 'Ctrl+B' not in ui.rows('*')[-1]


async def test_completed_task_hint_has_no_background_action() -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    record = task()
    record.status, record.outcome, record.background = 'finished', 'ok', True
    record.finished_at = time.time()
    ui.owner.records[record.id] = record
    await ui.observe(DelegationTaskEvent(task=record))
    assert '/tasks' in ui.rows('*')[-1]
    assert 'Ctrl+B' not in ui.rows('*')[-1]
