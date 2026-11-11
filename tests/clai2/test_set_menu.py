"""The `/set` menu, driven headless with scripted key presses."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.plugins.menus import FieldRow
from pydantic_clai2.ui.menus import field_menu
from pydantic_clai2.ui.menus.field_menu import (
    SAVE_AND_CLOSE,
    FieldMenu,
    is_save_and_close,
    run_flow,
    run_flow_async,
    save_and_close_item,
)
from pydantic_clai2.ui.menus.set_menu import SettingsSource, open_settings_menu
from tests.clai2.menu_script import Script, make_context, pick, typed


def test_rows_details_and_validation(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    menu = FieldMenu(SettingsSource(context))
    keys = [row.key for row in menu.rows]
    assert keys[:3] == ['model', 'run.request_limit', 'display.thinking']
    assert 'run.instructions' not in keys, '/system_prompt edits them, keeping their line breaks'
    items = menu.items()
    assert items[0].label.startswith('model') and 'openai-codex:gpt-6-astra' in items[0].label
    thinking = next(row for row in menu.rows if row.key == 'display.thinking')
    model = menu.rows[0]
    assert thinking.choices == ('true', 'false')
    assert len(model.choices) > 8
    assert 'choices  true, false' in menu.details(items[2])
    assert 'options; Enter opens a searchable list' in menu.details(items[0])
    assert 'current  true (default)' in menu.details(items[2])
    assert 'choices' not in menu.details(items[1])
    assert menu.details(MenuItem('stray', value='nope')) == ''
    assert menu.initial(model) == '', 'a value at its default opens empty'
    context.settings = context.settings.model_copy(update={'model': None})
    assert 'current  (not set)' in menu.details(items[0])
    assert menu.initial(model) == '', 'a value that is not set opens empty'
    limit = menu.rows[1]
    source = SettingsSource(context)
    assert source.problem(limit, '50') is None
    assert source.problem(limit, '-1') is not None
    assert 'JSON' in (source.problem(limit, 'not json') or '')
    assert menu.build(99) is not None
    assert menu.build_choices(thinking) is not None
    assert menu.build_choices(model) is not None
    assert menu.build_editor(limit) is not None
    assert menu.reset_marker(object(), items[1]).item is not None


def test_flow_edits_resets_and_reports(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    menu = FieldMenu(SettingsSource(context))
    script = Script(
        lists=[
            pick('display.thinking'),
            pick('run.request_limit'),
            pick('run.request_limit'),
            menu.reset_marker(object(), MenuItem('run.request_limit', value='run.request_limit')),
            menu.reset_marker(object(), MenuItem('ghost', value='ghost')),
            pick('model'),
            pick('model'),
            pick('display.splash'),
            pick('display.splash'),
            pick('unknown-key'),
        ],
        choices=[
            pick('false'),
            pick('Type a value...'),
            pick('Type a value...'),
            pick('Keep current'),
            MenuResult(cancelled=True),
        ],
        texts=[typed('50'), typed('bad'), typed('test'), typed('')],
    )
    messages = run_flow(menu, script.runners)
    assert messages == [
        'Saved display.thinking. Applied.',
        'Saved run.request_limit. Applied.',
        'run.request_limit: Invalid JSON: expected value at line 1 column 1',
        'Reset run.request_limit. Applied.',
        'Saved model. Applied.',
        'Reset model. Applied.',
    ]
    assert not context.settings.thinking
    assert context.settings.request_limit == 10000
    assert context.settings.model == 'openai-codex:gpt-6-astra'
    assert context.store.overrides() == {'display.thinking': False}
    assert applied == ['display.thinking', 'run.request_limit', 'run.request_limit', 'model', 'model']
    assert script.opened.count('choice') == 5


@pytest.mark.parametrize('searchable', [True, False])
@pytest.mark.parametrize('choices', [('true', 'false'), ('false', 'true')])
def test_space_toggles_scrolled_setting_and_save_close_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    searchable: bool,
    choices: tuple[str, str],
) -> None:
    context, applied = make_context(tmp_path)
    source = SettingsSource(context)
    rows: list[FieldRow] = [
        replace(row, choices=choices) if row.key == 'run.code_mode' else row for row in source.rows()
    ]
    monkeypatch.setattr(source, 'rows', lambda: rows)
    fields = FieldMenu(source, searchable=searchable)
    index = next(i for i, row in enumerate(fields.rows) if row.key == 'run.code_mode')

    def keys() -> Iterator[str]:
        yield from ['down'] * index
        for expected in (False, True, False):
            assert context.settings.code_mode is expected
            assert running.highlighted is not None and running.highlighted.value == 'run.code_mode'
            # Check the painted viewport too: a stable cursor alone does not prove stable scrolling.
            frame = capsys.readouterr().out.split('\x1b[H')[-1]
            assert '> run.code_mode' in frame
            assert '  model ' not in frame, 'the scrolled viewport must not jump to the top'
            assert f'current  {str(expected).lower()}' in frame
            yield ' '
        assert context.settings.code_mode is True
        yield 'end'
        yield ' '
        yield 'enter'

    pressed = keys()
    monkeypatch.setattr(field_menu, 'menu_key', lambda: next(pressed))
    monkeypatch.setattr('termflow.tui.menu.terminal_size', lambda: (100, 10))
    running = fields.build()
    result = running.run()
    assert result.item is not None and is_save_and_close(result.item)
    assert context.store.load().code_mode is True
    assert applied == ['run.code_mode'] * 3
    assert fields.messages == ['Saved run.code_mode. Applied.'] * 3
    assert 'true' in next(item.label for item in fields.items() if item.value == 'run.code_mode')


@pytest.mark.parametrize('searchable', [True, False])
@pytest.mark.parametrize('key', ['model', 'display.theme', 'display.spinner', 'display.tool_calls', 'updates.channel'])
def test_space_cycles_declared_choices_and_wraps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, searchable: bool, key: str
) -> None:
    context, applied = make_context(tmp_path)
    source = SettingsSource(context, keys=[key])
    fields = FieldMenu(source, searchable=searchable)
    row = fields.rows[0]
    assert len(row.choices) > 1
    assert source.apply(row, row.choices[-1]) == f'Saved {key}. Applied.'
    applied.clear()

    def keys() -> Iterator[str]:
        yield ' '
        assert source.current(row) == row.choices[0]
        assert running.highlighted is not None and running.highlighted.value == key
        yield ' '
        assert source.current(row) == row.choices[1]
        assert running.highlighted is not None and running.highlighted.value == key
        yield 'end'
        yield 'enter'

    pressed = keys()
    monkeypatch.setattr(field_menu, 'menu_key', lambda: next(pressed))
    running = fields.build()
    result = running.run()
    assert result.item is not None and is_save_and_close(result.item)
    reloaded, _ = make_context(tmp_path)
    assert SettingsSource(reloaded, keys=[key]).current(row) == row.choices[1]
    assert applied == [key, key]
    assert fields.messages == [f'Saved {key}. Applied.'] * 2


@pytest.mark.parametrize('current', [None, 'custom:model-not-in-the-catalog'])
def test_space_starts_unlisted_values_at_the_first_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current: str | None
) -> None:
    context, applied = make_context(tmp_path)
    update: dict[str, str | None] = {'model': current}
    context.settings = context.settings.model_copy(update=update)
    source = SettingsSource(context, keys=['model'])
    fields = FieldMenu(source)
    row = fields.rows[0]
    assert source.current(row) not in row.choices
    pressed = iter([' ', 'end', 'enter'])
    monkeypatch.setattr(field_menu, 'menu_key', lambda: next(pressed))
    assert run_flow(fields) == ['Saved model. Applied.']
    assert context.store.load().model == row.choices[0]
    assert applied == ['model']


def test_space_search_text_and_filtered_toggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, applied = make_context(tmp_path)
    fields = FieldMenu(SettingsSource(context))
    # The trailing Space belongs to the query. Enter ends search; Space then toggles the match.
    pressed = iter(['/', *'run.code_mode', 'R', 'backspace', ' ', 'enter', ' ', 'escape', 'end', 'enter'])
    monkeypatch.setattr(field_menu, 'menu_key', lambda: next(pressed))
    assert run_flow(fields) == ['Saved run.code_mode. Applied.']
    assert applied == ['run.code_mode']
    assert context.store.load().code_mode is True


@pytest.mark.parametrize('searchable', [True, False])
def test_space_on_choice_free_field_keeps_enter_editing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, searchable: bool
) -> None:
    context, applied = make_context(tmp_path)
    fields = FieldMenu(SettingsSource(context), searchable=searchable)
    pressed = iter(['down', ' ', 'enter', 'end', 'enter'])
    monkeypatch.setattr(field_menu, 'menu_key', lambda: next(pressed))
    script = Script(lists=[], choices=[], texts=[typed('42')])
    runners = field_menu.Runners(run_text=script.run_text)
    assert run_flow(fields, runners) == ['Saved run.request_limit. Applied.']
    assert context.store.load().request_limit == 42
    assert applied == ['run.request_limit']
    assert script.opened == ['text']


@pytest.mark.parametrize('searchable', [True, False])
@pytest.mark.parametrize(
    ('key', 'cycling'), [('display.thinking', True), ('display.tool_calls', True), ('run.request_limit', False)]
)
def test_space_hint_requires_multiple_choices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    searchable: bool,
    key: str,
    cycling: bool,
) -> None:
    context, _ = make_context(tmp_path)
    fields = FieldMenu(SettingsSource(context, keys=[key]), searchable=searchable)
    monkeypatch.setattr(field_menu, 'menu_key', lambda: 'escape')
    fields.build().run()
    assert ('Space cycle' in capsys.readouterr().out) is cycling


def test_choice_free_search_keeps_type_to_filter_and_literal_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    context, applied = make_context(tmp_path)
    fields = FieldMenu(SettingsSource(context, keys=['run.request_limit', 'display.shell_lines']))
    pressed = iter([*'shell', 'enter', 'R', *'shell ', 'escape'])
    monkeypatch.setattr(field_menu, 'menu_key', lambda: next(pressed))
    result = fields.build().run()
    assert result.item is not None and result.item.value == 'display.shell_lines'
    assert fields.build(1).run() == fields.reset_marker(object(), result.item)
    capsys.readouterr()
    assert fields.build().run().cancelled
    output = capsys.readouterr().out
    assert '(no matches)' in output, 'Space is query text, not a hotkey, in a choice-free list'
    assert 'Space cycle' not in output and '/ search' not in output
    assert 'type to filter' in output
    assert applied == []
    assert context.store.overrides() == {}


@pytest.mark.parametrize('asynchronous', [True, False])
async def test_space_does_not_bypass_boolean_row_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], asynchronous: bool
) -> None:
    context, applied = make_context(tmp_path)
    fields = FieldMenu(SettingsSource(context, keys=['run.code_mode']))
    pressed = iter([' ', 'enter', 'end', 'enter'])
    monkeypatch.setattr(field_menu, 'menu_key', lambda: next(pressed))
    calls: list[str] = []

    def action() -> list[str]:
        calls.append('action')
        return ['Action ran.']

    async def async_action() -> list[str]:
        return action()

    if asynchronous:
        messages = await run_flow_async(fields, submenus={'run.code_mode': async_action})
    else:
        messages = run_flow(fields, submenus={'run.code_mode': action})
    assert messages == ['Action ran.']
    assert calls == ['action']
    assert applied == []
    assert context.store.overrides() == {}
    assert 'Space cycle' not in capsys.readouterr().out


def test_flow_stops_on_cancel(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    script = Script(lists=[MenuResult(cancelled=True)], choices=[], texts=[])
    assert run_flow(FieldMenu(SettingsSource(context)), script.runners) == []
    script = Script(
        lists=[pick('run.request_limit'), MenuResult(item=None)], choices=[], texts=[TextInputResult(cancelled=True)]
    )
    assert run_flow(FieldMenu(SettingsSource(context)), script.runners) == []


def test_save_and_close_is_the_last_row_and_leaves_with_edits_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, _ = make_context(tmp_path)
    menu = FieldMenu(SettingsSource(context))
    first, *_, last = menu.items()
    assert last.label == SAVE_AND_CLOSE and is_save_and_close(last) and not is_save_and_close(first)
    assert menu.details(last) == 'Leave this menu. Each change was saved as you made it.'
    pressed = iter(['end', 'R', 'enter', 'R'])
    monkeypatch.setattr(field_menu, 'menu_key', lambda: next(pressed))
    left = menu.build().run()
    assert left.item is not None and is_save_and_close(left.item), 'R on Save & close resets nothing'
    assert menu.build().run() == menu.reset_marker(object(), first)
    script = Script(
        lists=[pick('display.thinking'), MenuResult(item=save_and_close_item())], choices=[pick('false')], texts=[]
    )
    assert run_flow(menu, script.runners) == ['Saved display.thinking. Applied.']
    assert context.store.overrides() == {'display.thinking': False}
    assert script.opened == ['list', 'choice', 'list']


async def test_async_flow_matches_the_sync_one_without_submenus(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    script = Script(
        lists=[pick('run.request_limit'), MenuResult(item=None)], choices=[], texts=[TextInputResult(cancelled=True)]
    )
    assert await run_flow_async(FieldMenu(SettingsSource(context)), script.runners) == []
    assert script.opened == ['list', 'text', 'list']


async def test_open_menu_runs_in_a_thread(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    assert await open_settings_menu(context, run=lambda menu: []) == 'No changes.'
    assert await open_settings_menu(context, run=lambda menu: [menu.apply(menu.rows[1], '7')]) == (
        'Saved run.request_limit. Applied.'
    )
    assert context.settings.request_limit == 7
    assert await open_settings_menu(context, run=lambda menu: [menu.apply(menu.rows[1], '  ')]) == (
        'Reset run.request_limit. Applied.'
    )


def test_project_overrides_are_marked_on_the_row(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    context.project = ProjectSettings(path=tmp_path / '.clai' / 'settings.json', overrides={'display.thinking': False})
    menu = FieldMenu(SettingsSource(context))
    items = {item.value: item for item in menu.items()}
    assert items['display.thinking'].description.endswith('project')
    assert items['display.thinking'].description.startswith('\x1b[')
    assert items['model'].description == ''
    assert 'origin   project' in menu.details(items['display.thinking'])
    assert 'origin' not in menu.details(items['model'])
    assert context.reset_setting('display.thinking').endswith('sets it again at next start.')


class _Stored:
    """A one-row source whose saved value the test sets."""

    title = 'Stored'

    def __init__(self, value: str) -> None:
        self.value = value

    def rows(self) -> list[FieldRow]:
        return [FieldRow(key='root', description='', default='')]

    def current(self, row: FieldRow) -> str:
        return self.value

    def problem(self, row: FieldRow, text: str) -> str | None:
        raise NotImplementedError

    def apply(self, row: FieldRow, raw: str) -> str:
        raise NotImplementedError

    def reset(self, row: FieldRow) -> str:
        raise NotImplementedError


@pytest.mark.parametrize('searchable', [True, False])
def test_space_does_not_save_a_single_choice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], searchable: bool
) -> None:
    source = _Stored('one')
    rows: list[FieldRow] = [FieldRow(key='root', description='', default='', choices=('one',))]
    monkeypatch.setattr(source, 'rows', lambda: rows)
    fields = FieldMenu(source, searchable=searchable)
    pressed = iter([' ', 'escape'])
    monkeypatch.setattr(field_menu, 'menu_key', lambda: next(pressed))
    assert fields.build().run().cancelled
    assert fields.messages == []
    assert source.value == 'one'
    assert 'Space cycle' not in capsys.readouterr().out


@pytest.mark.parametrize(('value', 'initial'), [('src', 'src'), ('', ''), ('evil\x1b[2J', '')])
def test_editor_opens_holding_a_printable_value(value: str, initial: str) -> None:
    menu = FieldMenu(_Stored(value))
    editor = menu.build_editor(menu.rows[0])
    assert (editor.text, editor._placeholder) == (initial, 'empty resets to (empty)')  # pyright: ignore[reportPrivateUsage]


def test_repo_context_settings_edit_paths_and_json_lists(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    source = SettingsSource(context)
    menu = FieldMenu(source)
    rows = {row.key: row for row in menu.rows}
    roots = rows['repo_context.walk_up_roots']
    targets = rows['repo_context.autoload_symlink_roots']
    assert roots.default == '["~"]'
    assert targets.default == '[]'
    assert source.current(roots) == '["~"]'
    assert source.current(targets) == '[]'
    assert source.problem(roots, '~') is not None
    assert source.problem(roots, '["relative"]') is not None
    assert source.problem(targets, '/not-json') is not None
    assert source.problem(targets, '["relative"]') is not None
    raw_roots = '["~/my projects", "/offline mount"]'
    assert source.apply(roots, raw_roots) == 'Saved repo_context.walk_up_roots. Applied.'
    raw = '["~/shared tooling", "/offline instructions"]'
    assert source.apply(targets, raw) == 'Saved repo_context.autoload_symlink_roots. Applied.'
    assert source.current(roots) == raw_roots
    assert source.current(targets) == raw
    assert menu.initial(roots) == raw_roots, 'lists must roundtrip as JSON, not Python repr'
    assert menu.initial(targets) == raw, 'lists must roundtrip as JSON, not Python repr'
    assert context.store.load().repo_context_walk_up_roots == ['~/my projects', '/offline mount']
    assert context.store.load().repo_context_autoload_symlink_roots == ['~/shared tooling', '/offline instructions']
    source.reset(roots)
    source.reset(targets)
    assert context.settings.repo_context_walk_up_roots == ['~']
    assert context.settings.repo_context_autoload_symlink_roots == []
    assert applied == [roots.key, targets.key, roots.key, targets.key]
