"""The task panel's theme roles and available action hints."""

import io
import sys
import time
from datetime import UTC, datetime
from typing import Literal

import pytest
from rich.console import Console

from pydantic_ai_harness.sub_agents import DelegationTaskEvent
from pydantic_ai_pax.runtime.tasks import Tasks
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.duration import format_duration
from tests.pax.test_tasks import task


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
    from pydantic_ai_harness.sub_agents import DelegationStartedEvent
    from pydantic_ai_pax.runtime.tasks import task_row

    event = DelegationStartedEvent(
        agent_name='self', task='count\nthe files', truncated=False, inherits_tools=True, model=None
    )
    event.task_id = 'abcdef0123'
    row = task_row(event)
    assert row is not None
    assert row.plain == '● general-purpose [abcdef01] count the files'
    with_model = task_row(event, model_name='child-model')
    assert with_model is not None
    assert with_model.plain == '● general-purpose [abcdef01] · child-model count the files'
    assert with_model.spans[-1].style == theme.color(theme.MUTED)
    hidden = task_row(event, model_name='child-model', show_model=False)
    assert hidden is not None and hidden.plain == row.plain
    assert row.no_wrap and row.overflow == 'ellipsis'
    assert [(row.plain[span.start : span.end], span.style) for span in row.spans] == [
        ('general-purpose', theme.color(theme.ACCENT)),
        (' [abcdef01] count the files', theme.color(theme.MUTED)),
    ]


@pytest.mark.parametrize('activity', ['running: run_code', 'tool: run_code'])
def test_panel_shows_the_tool_name_without_a_prefix(activity: str) -> None:
    from pydantic_ai_pax.ui.rendering.status import Status

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


@pytest.mark.parametrize(
    ('seconds', 'expected'),
    [
        (-5, '0s'),
        (0, '0s'),
        (42, '42s'),
        (59, '59s'),
        (60, '1m'),
        (99, '1m 39s'),
        (100, '1m 40s'),
        (192, '3m 12s'),
        (3599, '59m 59s'),
        (3600, '1h'),
        (7448, '2h 4m 8s'),
        (86399, '23h 59m 59s'),
        (86400, '1d'),
        (100000, '1d 3h 46m'),
        (100201, '1d 3h 50m'),
        (2591999, '29d 23h 59m'),
    ],
)
def test_humane_duration_boundaries(seconds: float, expected: str) -> None:
    assert format_duration(seconds, started_at=0) == expected
    assert len(expected.split()) <= 3
    if 'd' in expected:
        assert 's' not in expected


@pytest.mark.parametrize('days', [30, 31, 365])
def test_old_duration_shows_the_actual_local_start_date(days: int) -> None:
    start = datetime(2026, 1, 2, 12, tzinfo=UTC).timestamp()
    expected = datetime.fromtimestamp(start).strftime('%b %d')
    assert format_duration(days * 86400, started_at=start) == expected


@pytest.mark.skipif(sys.platform == 'win32', reason='time.tzset is unavailable on Windows')
@pytest.mark.parametrize(
    ('timezone', 'start_hour', 'local_day'),
    [('UTC0', 12, 2), ('UTC-14', 12, 3), ('UTC12', 0, 1)],
)
def test_old_duration_uses_the_local_date_across_day_boundaries(
    monkeypatch: pytest.MonkeyPatch, timezone: str, start_hour: int, local_day: int
) -> None:
    start = datetime(2026, 1, 2, start_hour, tzinfo=UTC).timestamp()
    expected = datetime(2026, 1, local_day).strftime('%b %d')
    try:
        with monkeypatch.context() as environment:
            environment.setenv('TZ', timezone)
            time.tzset()
            assert format_duration(30 * 86400, started_at=start) == expected
    finally:
        time.tzset()


def test_completed_duration_freezes_but_panel_expiry_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2026, 1, 2, 12, tzinfo=UTC).timestamp()
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    record = task()
    record.started_at = start
    record.finished_at = start + 192
    record.status, record.outcome = 'finished', 'failed'
    ui.owner.records[record.id] = record
    monkeypatch.setattr('pydantic_ai_pax.runtime.tasks.time.time', lambda: start + 200)
    assert '3m 12s' in ui.rows('*')[0]
    monkeypatch.setattr('pydantic_ai_pax.runtime.tasks.time.time', lambda: start + 220)
    assert '3m 12s' in ui.rows('*')[0]
    monkeypatch.setattr('pydantic_ai_pax.runtime.tasks.time.time', lambda: start + 223)
    assert ui.rows('*') == ()
