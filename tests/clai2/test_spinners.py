"""The `/spinner` catalogue, command, picker, customization surfaces, and painters."""

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from menu_script import Script, make_context, pick
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import chat, theme
from pydantic_clai2.command_context import CommandContext
from pydantic_clai2.commands import Command, Commands, set_completions
from pydantic_clai2.config import Settings
from pydantic_clai2.field_menu import Runners
from pydantic_clai2.image_input import ImageInput
from pydantic_clai2.interrupts import Interrupts
from pydantic_clai2.live_prompt import LivePrompt
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.spinner_picker import SpinnerPicker, spinner_command, spinner_completions
from pydantic_clai2.spinners import (
    BUILTIN_SPINNERS,
    DEFAULT_SPINNER,
    STARTER_FILE,
    Spinner,
    Spinners,
    make_spinner,
    user_spinners_path,
)

PromptT = TypeVar('PromptT')
CODE_PUPPY_BUILTINS = (
    'puppy',
    'bone',
    'zoomies',
    'paws',
    'dots',
    'dotsWide',
    'dots8Bit',
    'dotsCircle',
    'sand',
    'growVertical',
    'growHorizontal',
    'noise',
    'binary',
    'chevrons',
    'bouncingBar',
    'bouncingBall',
    'pong',
    'fistBump',
    'aesthetic',
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def catalogue(tmp_path: Path, *, selected: str = DEFAULT_SPINNER, registered: tuple[Spinner, ...] = ()) -> Spinners:
    return Spinners(selected=lambda: selected, registered=lambda: registered, path=tmp_path / 'spinners.json')


def write(spinners: Spinners, data: object) -> None:
    text = data if isinstance(data, str) else json.dumps(data)
    spinners.path.write_text(text)


class TestCatalogue:
    def test_every_code_puppy_builtin_plus_working(self) -> None:
        assert set(BUILTIN_SPINNERS) == {'working', *CODE_PUPPY_BUILTINS}
        assert len(BUILTIN_SPINNERS) == 20
        assert DEFAULT_SPINNER == 'working' and Settings().spinner == 'working'

    def test_working_matches_the_prompt_title(self) -> None:
        working = BUILTIN_SPINNERS['working']
        assert ''.join(working.frames) == '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' and working.interval == 0.1
        assert working.frame(0) == '⠋' and working.frame(0.25) == '⠹' and working.frame(1.0) == '⠋'

    @pytest.mark.parametrize('name', sorted(BUILTIN_SPINNERS))
    def test_frames_share_one_cell_width(self, name: str) -> None:
        spinner = BUILTIN_SPINNERS[name]
        assert len({cell_len(frame) for frame in spinner.frames}) == 1
        assert spinner.description and spinner.source == 'builtin'

    def test_make_spinner_normalizes(self) -> None:
        spinner = make_spinner(' wide ', ['\U0001f436', 'ab', 'x' * 50], interval=5)
        assert spinner.name == 'wide' and spinner.interval == 1.0
        assert spinner.frames[0] == '\U0001f436' + ' ' * 38 and len(spinner.frames[2]) == 40
        assert make_spinner('fast', ['a'], interval=0.001).interval == 0.02

    @pytest.mark.parametrize(
        ('name', 'frames', 'error'),
        [
            ('', ['a'], 'no spaces'),
            ('two words', ['a'], 'no spaces'),
            ('empty', [], 'non-empty frame'),
            ('blank', ['a', ''], 'non-empty frame'),
            ('escape', ['\x1b[31m'], 'control character'),
        ],
    )
    def test_make_spinner_rejects(self, name: str, frames: list[str], error: str) -> None:
        with pytest.raises(ValueError, match=error):
            make_spinner(name, frames)

    @pytest.mark.parametrize('interval', [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_interval_is_rejected_before_it_reaches_a_painter(self, interval: float) -> None:
        host = PluginHost[None](name='p', console=Console(file=io.StringIO()), settings={})
        with pytest.raises(ValueError, match='finite number of seconds'):
            host.spinner('broken', ['a'], interval=interval)
        assert host.spinners == []

    def test_layers_and_lookup(self, tmp_path: Path) -> None:
        plugin = make_spinner('dots', ['o', 'O'], source='plugin')
        spinners = catalogue(tmp_path, selected='DOTSWIDE', registered=(plugin, make_spinner('extra', ['e'])))
        assert list(spinners.catalogue())[:4] == ['aesthetic', 'binary', 'bone', 'bouncingBall']
        assert spinners.catalogue()['dots'] is plugin and 'extra' in spinners.catalogue()
        assert spinners.active().name == 'dotsWide'
        assert spinners.find('nope') is None and spinners.problems() == ()
        assert catalogue(tmp_path, selected='gone').active() is BUILTIN_SPINNERS['working']

    def test_user_file_wins_and_reloads_on_change(self, tmp_path: Path) -> None:
        plugin = make_spinner('shared', ['p'], source='plugin')
        spinners = catalogue(tmp_path, selected='mine', registered=(plugin,))
        write(
            spinners,
            {
                'mine ': {'frames': ['a', 'bb'], 'description': 'custom'},
                'zoomies': {'interval': 0.5},
                'shared': {'description': 'retold'},
            },
        )
        found = spinners.catalogue()
        assert spinners.active() == Spinner(
            name='mine', frames=('a ', 'bb'), interval=0.2, description='custom', source='user'
        )
        assert found['zoomies'].frames == BUILTIN_SPINNERS['zoomies'].frames and found['zoomies'].interval == 0.5
        assert found['shared'].frames == ('p',) and found['shared'].description == 'retold'
        write(spinners, {'mine': {'frames': ['z'], 'interval': 0.3}})
        assert spinners.active().frames == ('z',) and spinners.active().interval == 0.3
        mtime = spinners.path.stat().st_mtime_ns
        write(spinners, {'mine': {'frames': ['y'], 'interval': 0.3}})
        os.utime(spinners.path, ns=(mtime, mtime))
        # Same size and timestamp: only the contents tell the edit apart.
        assert spinners.active().frames == ('y',)
        spinners.path.unlink()
        assert spinners.active().name == 'working'

    @pytest.mark.parametrize(
        ('data', 'problem'),
        [
            ('[1]', 'must be a JSON object'),
            ('{not json', 'must be a JSON object'),
            ({'ghost': {'interval': 0.1}}, 'skipped \'ghost\': needs "frames"'),
            ({'bad': {'frames': 'abc'}}, "skipped 'bad': Input should be a valid list"),
            ({'bad': {'frames': ['a'], 'colour': 'red'}}, "skipped 'bad': Extra inputs"),
            ({'bad': {'frames': ['a'], 'interval': True}}, "skipped 'bad': Input should be a valid number"),
            ({'bad': {'frames': ['\x07']}}, "skipped 'bad': Spinner 'bad' has a control character"),
        ],
    )
    def test_bad_entries_are_reported_and_skipped(self, tmp_path: Path, data: object, problem: str) -> None:
        spinners = catalogue(tmp_path, selected='bad')
        write(spinners, data)
        assert spinners.active().name == 'working'
        assert any(problem in line for line in spinners.problems()), spinners.problems()

    def test_unreadable_file_is_reported(self, tmp_path: Path) -> None:
        spinners = catalogue(tmp_path)
        spinners.path.mkdir()
        assert 'could not be read' in spinners.problems()[0]
        assert len(spinners.catalogue()) == len(BUILTIN_SPINNERS)

    def test_padded_keys_match_their_trimmed_name(self, tmp_path: Path) -> None:
        spinners = catalogue(tmp_path, selected='mine')
        write(spinners, {' zoomies': {'interval': 0.5}, 'mine ': {'frames': ['a'], 'interval': 0.3}})
        assert spinners.catalogue()['zoomies'].interval == 0.5
        spinners.save_interval('mine', 0.7)
        assert json.loads(spinners.path.read_text())['mine '] == {'frames': ['a'], 'interval': 0.7}
        assert spinners.active().interval == 0.7 and spinners.problems() == ()

    def test_save_interval_keeps_other_keys(self, tmp_path: Path) -> None:
        spinners = catalogue(tmp_path)
        spinners.save_interval('puppy', 0.333)
        write(spinners, {**json.loads(spinners.path.read_text()), 'mine': {'frames': ['a'], 'description': 'd'}})
        spinners.save_interval('mine', 3)
        assert json.loads(spinners.path.read_text()) == {
            'puppy': {'interval': 0.33},
            'mine': {'frames': ['a'], 'description': 'd', 'interval': 1.0},
        }

    def test_concurrent_saves_both_land(self, tmp_path: Path) -> None:
        first, second = catalogue(tmp_path), catalogue(tmp_path)
        jobs = [(first, 'puppy', 0.3), (second, 'bone', 0.4)] * 20
        with ThreadPoolExecutor(8) as pool:
            for future in [pool.submit(spinners.save_interval, name, seconds) for spinners, name, seconds in jobs]:
                future.result()
        assert json.loads(first.path.read_text()) == {'puppy': {'interval': 0.3}, 'bone': {'interval': 0.4}}
        assert sorted(path.name for path in tmp_path.iterdir()) == ['.spinners.json.lock', 'spinners.json']

    def test_init_writes_once(self, tmp_path: Path) -> None:
        spinners = catalogue(tmp_path)
        assert spinners.init() and spinners.path.read_text() == STARTER_FILE
        assert set(spinners.catalogue()) >= {'sniffer', 'zoomies'}
        spinners.path.write_text('{}')
        assert not spinners.init() and spinners.path.read_text() == '{}'

    def test_default_path_is_next_to_settings(self) -> None:
        assert user_spinners_path() == SettingsStore().path.with_name('spinners.json')
        assert Spinners(selected=lambda: 'puppy').path == user_spinners_path()


def spinners_for(context: CommandContext, tmp_path: Path) -> Spinners:
    return Spinners(selected=lambda: context.settings.spinner, path=tmp_path / 'spinners.json')


class TestCommand:
    async def test_by_name_persists_and_applies(self, tmp_path: Path) -> None:
        context, applied = make_context(tmp_path)
        spinners = spinners_for(context, tmp_path)
        message = await spinner_command(context, spinners, ['PUPPY'])
        assert message == 'Spinner set to puppy (8 frames at 0.06s).'
        assert applied == ['display.spinner'] and context.store.overrides() == {'display.spinner': 'puppy'}
        assert spinners.active().name == 'puppy' and not spinners.path.exists()

    async def test_seconds_save_to_the_user_file(self, tmp_path: Path) -> None:
        context, _ = make_context(tmp_path)
        spinners = spinners_for(context, tmp_path)
        message = await spinner_command(context, spinners, ['zoomies', '0.5'])
        assert message.startswith('Spinner set to zoomies (6 frames at 0.50s). Speed saved to ')
        assert json.loads(spinners.path.read_text()) == {'zoomies': {'interval': 0.5}}

    async def test_problems_follow_the_confirmation(self, tmp_path: Path) -> None:
        context, _ = make_context(tmp_path)
        spinners = spinners_for(context, tmp_path)
        write(spinners, {'ghost': {}})
        message = await spinner_command(context, spinners, ['dots'])
        assert message.splitlines()[1].startswith("spinners.json: skipped 'ghost'")

    @pytest.mark.parametrize(
        ('args', 'error'),
        [
            (['nope'], "Unknown spinner 'nope'. Choose from: aesthetic, binary"),
            (['dots', 'fast'], "'fast' is not a number of seconds"),
            (['dots', '0'], "'0' is not a number of seconds"),
            (['dots', 'nan'], "'nan' is not a number of seconds"),
            (['dots', 'inf'], "'inf' is not a number of seconds"),
            (['dots', '1', 'x'], 'Usage: /spinner'),
        ],
    )
    async def test_rejects(self, tmp_path: Path, args: list[str], error: str) -> None:
        context, applied = make_context(tmp_path)
        with pytest.raises(ValueError, match=error):
            await spinner_command(context, spinners_for(context, tmp_path), args)
        assert applied == [] and context.store.overrides() == {}

    async def test_speed_is_not_saved_over_a_broken_file(self, tmp_path: Path) -> None:
        context, applied = make_context(tmp_path)
        spinners = spinners_for(context, tmp_path)
        write(spinners, '[]')
        with pytest.raises(ValueError, match='is not a JSON object; speed not saved'):
            await spinner_command(context, spinners, ['dots', '0.3'])
        assert spinners.path.read_text() == '[]' and applied == []

    async def test_init(self, tmp_path: Path) -> None:
        context, applied = make_context(tmp_path)
        spinners = spinners_for(context, tmp_path)
        assert (await spinner_command(context, spinners, ['init'])).startswith(f'Wrote {spinners.path}.')
        assert 'already exists' in await spinner_command(context, spinners, ['init'])
        assert applied == []

    @pytest.mark.parametrize('result', [MenuResult(cancelled=True), MenuResult(), pick(3)])
    async def test_cancelled_picker_changes_nothing(self, tmp_path: Path, result: MenuResult) -> None:
        context, applied = make_context(tmp_path)
        script = Script(lists=[result], choices=[], texts=[])
        assert await spinner_command(context, spinners_for(context, tmp_path), [], runners=script.runners) == ''
        assert applied == [] and script.opened == ['list']

    async def test_picker_applies_and_saves_a_changed_speed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        context, _ = make_context(tmp_path)
        spinners = spinners_for(context, tmp_path)
        monkeypatch.setattr('sys.stdout', io.StringIO())
        keys = iter([*'bone', '+', '+', '=', 'right', '-', 'left', '+', 'enter'])
        monkeypatch.setattr('pydantic_clai2.spinner_picker.menu_key', lambda: next(keys))
        message = await spinner_command(context, spinners, [], runners=Runners(run_list=lambda menu: menu.run()))
        assert message.startswith('Spinner set to bone (8 frames at 0.14s). Speed saved')
        assert context.settings.spinner == 'bone'
        assert await spinner_command(context, spinners, [], runners=Script([pick('dots')], [], []).runners)
        assert context.settings.spinner == 'dots'
        assert json.loads(spinners.path.read_text()) == {'bone': {'interval': 0.14}}

    def test_completions(self, tmp_path: Path) -> None:
        spinners = catalogue(tmp_path)
        assert tuple(spinner_completions(spinners, ['']))[:2] == ('init', 'aesthetic')
        assert tuple(spinner_completions(spinners, ['dots', ''])) == ()
        assert tuple(set_completions(['display.spinner', ''])) == tuple(BUILTIN_SPINNERS)


class TestPicker:
    def test_preview_animates_and_shows_speed(self, tmp_path: Path) -> None:
        now = [0.0]
        picker = SpinnerPicker(catalogue(tmp_path, selected='dots'), clock=lambda: now[0])
        item = MenuItem('working', value='working')
        first = Text.from_ansi(picker.preview(item)).plain
        now[0] = 0.1
        second = Text.from_ansi(picker.preview(item)).plain
        assert '─ Working ⠋ ─' in first and '─ Working ⠙ ─' in second
        assert '10 frames at 0.10s per frame' in first and 'working (builtin)' in first
        assert picker.build().highlighted == MenuItem('dots (current)', value='dots')

    def test_menu_repaints_while_idle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        output = io.StringIO()
        monkeypatch.setattr('sys.stdout', output)
        monkeypatch.setenv('COLUMNS', '100')
        monkeypatch.setenv('LINES', '30')
        keys = iter(['', '', 'escape'])
        monkeypatch.setattr('pydantic_clai2.spinner_picker.menu_key', lambda: next(keys))
        ticks = iter([0.0, 0.1, 0.2, 0.3])
        picker = SpinnerPicker(catalogue(tmp_path), clock=lambda: next(ticks))
        assert picker.build().run().cancelled
        painted = Text.from_ansi(output.getvalue()).plain
        assert all(f'Working {glyph} ─' in painted for glyph in '⠋⠙⠹')


PLUGIN = """
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost) -> None:
    host.spinner('wave', ['~ ', ' ~'], interval=0.05, description='from a plugin')
"""


class TestRegistration:
    def test_host_registers_a_normalized_spinner(self) -> None:
        host = PluginHost[None](name='p', console=Console(file=io.StringIO()), settings={})
        spinner = host.spinner('wave', ['~', '~~'], interval=0.001, description='d')
        assert host.spinners == [spinner]
        assert spinner == Spinner(name='wave', frames=('~ ', '~~'), interval=0.02, description='d', source='plugin')
        with pytest.raises(ValueError, match='non-empty frame'):
            host.spinner('nothing', [])

    async def test_shell_offers_plugin_spinners_and_persists_the_choice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        values = ['/spinner wave', '/spinner fistbump 0.3', '/exit']

        class Prompt(Generic[PromptT]):
            def __init__(self, **kwargs: object) -> None:
                pass

            async def prompt_async(self, label: str, **kwargs: object) -> str:
                return values.pop(0)

        monkeypatch.setattr('pydantic_clai2._app.PromptSession', Prompt)
        store = SettingsStore(tmp_path / 'config.db')
        store.plugins_dir.mkdir(parents=True, exist_ok=True)
        (store.plugins_dir / 'wave.py').write_text(PLUGIN)
        output = io.StringIO()
        await chat(
            Agent('test'),
            deps=None,
            settings=Settings(model='test'),
            console=Console(file=output, width=200),
            store=store,
        )
        assert 'Spinner set to wave (2 frames at 0.05s).' in output.getvalue()
        assert 'Spinner set to fistBump (7 frames at 0.30s).' in output.getvalue()
        assert store.overrides()['display.spinner'] == 'fistBump'
        assert json.loads(user_spinners_path().read_text()) == {'fistBump': {'interval': 0.3}}


class Busy(Interrupts):
    @property
    def active(self) -> bool:
        return True


@pytest.mark.parametrize('width', [80, 10])
async def test_prompt_title_follows_the_selection(width: int) -> None:
    selected = ['working']
    commands = Commands()
    commands.register(Command(name='help', description='Help', handler=lambda args: 'help'))
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        live = LivePrompt(
            console=Console(file=io.StringIO(), force_terminal=True, width=width, height=24),
            commands=commands,
            history=InMemoryHistory(),
            images=ImageInput(),
            interrupts=Busy(),
            toolbar=lambda: [('', 'ready')],
            clock=lambda: 0.0,
            spinner=lambda: BUILTIN_SPINNERS[selected[0]],
        )
        accent = theme.sgr(theme.ACCENT)
        selected[0] = 'binary'
        title = live.frame()[0]
        plain = Text.from_ansi(title).plain
        if width == 80:
            frame = BUILTIN_SPINNERS['binary'].frames[0]
            # An empty queue has nothing to steer, so the title carries no queue hints.
            assert plain.rstrip('─') == f' Working {frame} '
            assert f'{accent}{frame}' in title
            live.submit('follow up')
            follow_up, plain = (Text.from_ansi(row).plain for row in live.frame()[:2])
            assert follow_up == 'Follow-up: follow up'
            assert plain.startswith(f' Working {frame} | Enter: queue | Alt+Enter: steer queued ')
        else:
            # Too narrow to show the whole frame: nothing is highlighted rather than half a frame.
            assert plain.startswith(' Working') and accent not in title
        async with live.opened():
            pass
