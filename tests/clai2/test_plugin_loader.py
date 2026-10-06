"""Loading, unloading, reloading, and dispatching between turns."""

import asyncio
import io
import sys
from pathlib import Path
from types import ModuleType

import anyio
import pytest
from pydantic import BaseModel
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import Plugin, SessionEnd, SessionStart, TurnEnd, TurnStart
from pydantic_clai2.plugins.loader import TURN_NOTICE, PluginError, PluginLoader, PluginSettingsError

RECORDER = """
from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import Plugin, SessionEnd, SessionStart, TurnEnd, TurnStart

LOADS = globals().get('LOADS', 0) + 1


class Recorder(Plugin):
    def get_commands(self):
        return [Command(name='{command}', description='From plugin', handler=lambda _: 'ok')]

    async def on_session_start(self, event: SessionStart) -> None:
        self.host.console.print('{name} started')

    async def on_session_end(self, event: SessionEnd) -> None:
        host = self.host
        host.console.print('{name} stopped ' + event.reason)
        {end_body}

    async def on_turn_start(self, event: TurnStart) -> None:
        {start_body}

    async def on_turn_end(self, event: TurnEnd) -> None:
        {turn_end_body}
"""


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        builtin: tuple[PluginSettings, ...] = (),
        project: tuple[PluginSettings, ...] = (),
    ) -> None:
        self.store = SettingsStore(tmp_path / 'config.db')
        self.store.plugins_dir.mkdir(exist_ok=True)
        self.output = io.StringIO()
        self.commands = Commands()
        self.commands.register(Command(name='help', description='Built in', handler=lambda _: ''))
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200),
            commands=self.commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=builtin,
            project=project,
        )

    def write(
        self,
        name: str,
        *,
        command: str | None = None,
        end_body: str = 'pass',
        start_body: str = 'pass',
        turn_end_body: str = 'pass',
    ) -> Path:
        path = self.store.plugins_dir / f'{name}.py'
        path.write_text(
            RECORDER.format(
                name=name,
                command=command or name,
                end_body=end_body,
                start_body=start_body,
                turn_end_body=turn_end_body,
            )
        )
        return path

    @property
    def text(self) -> str:
        return self.output.getvalue()


SEGMENT = """
from pydantic_clai2.plugins import Plugin


class Segment(Plugin):
    def get_status_segments(self):
        return [lambda: 'in {name}']
"""


async def test_status_segments_follow_load_and_unload(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    (harness.store.plugins_dir / 'segment.py').write_text(SEGMENT.format(name='segment'))
    assert harness.loader.status_segments() == []
    await harness.loader.load_all()
    assert [segment() for segment in harness.loader.status_segments()] == ['in segment']
    await harness.loader.disable('segment')
    assert harness.loader.status_segments() == []


@pytest.mark.parametrize(('first', 'shown'), [('zulu', 'zulu'), ('logfire', 'observability')])
async def test_registration_order_follows_the_shipped_declarations(tmp_path: Path, first: str, shown: str) -> None:
    """Names order the menu and the list; the built-in declarations order what a turn sees first."""

    def shipped(name: str) -> PluginSettings:
        path = tmp_path / f'shipped_{name}.py'
        path.write_text(
            RECORDER.format(name=name, command=name, end_body='pass', start_body='pass', turn_end_body='pass')
        )
        return PluginSettings(id=name, factory=name, path=str(path))

    harness = Harness(tmp_path, builtin=(shipped(first), shipped('alpha')))
    harness.write('mike')
    await harness.loader.load_all()
    assert [entry.name for entry in harness.loader.entries()] == ['alpha', 'mike', shown]
    assert harness.text.index(f'{first} started') < harness.text.index('alpha started')
    assert harness.text.index('alpha started') < harness.text.index('mike started')


async def test_startup_creates_the_plugins_folder(tmp_path: Path) -> None:
    """So installing a drop-in is one copy, without first finding and creating the folder."""
    harness = Harness(tmp_path)
    harness.store.plugins_dir.rmdir()
    await harness.loader.load_all()
    assert harness.store.plugins_dir.is_dir()
    await harness.loader.load_all()  # already there: nothing to do
    assert harness.text == ''


async def test_startup_leaves_the_plugins_folder_alone_when_plugins_are_off(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.store.plugins_dir.rmdir()
    harness.loader.enabled = False
    await harness.loader.load_all()
    assert not harness.store.plugins_dir.exists()


async def test_startup_reports_a_plugins_folder_it_cannot_create_and_carries_on(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.store.plugins_dir.rmdir()
    harness.store.plugins_dir.write_text('a file where the folder belongs')
    await harness.loader.load_all()
    assert 'Cannot create the plugins folder:' in harness.text
    assert harness.loader.entries() == []


async def test_folder_discovery_and_load_order(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('beta')
    harness.write('alpha')
    (harness.store.plugins_dir / '_private.py').write_text('raise AssertionError')
    (harness.store.plugins_dir / 'not-a-name.py').write_text('raise AssertionError')
    (harness.store.plugins_dir / 'pkg').mkdir()
    (harness.store.plugins_dir / 'pkg' / '__init__.py').write_text(
        'from pydantic_clai2.plugins import Plugin\nclass Empty(Plugin):\n    pass\n'
    )
    (harness.store.plugins_dir / 'empty').mkdir()
    await harness.loader.load_all()
    assert [entry.name for entry in harness.loader.entries()] == ['alpha', 'beta', 'pkg']
    assert all(entry.loaded is not None for entry in harness.loader.entries())
    assert harness.text.index('alpha started') < harness.text.index('beta started')
    assert {command.name for command in harness.commands} == {'help', 'alpha', 'beta'}
    await harness.loader.close('eof')
    assert harness.text.index('beta stopped eof') < harness.text.index('alpha stopped eof')
    assert {command.name for command in harness.commands} == {'help'}
    assert harness.loader.capabilities() == []


async def test_add_saves_settings_only_once_activation_accepts_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / 'site' / 'clai_strict'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text(
        'from pydantic import BaseModel, ConfigDict\n'
        'from pydantic_clai2.plugins import Plugin, SessionStart\n'
        'class Settings(BaseModel):\n'
        "    model_config = ConfigDict(extra='forbid')\n"
        '    fail_on_start: bool = False\n'
        'class Strict(Plugin[Settings]):\n'
        '    async def on_session_start(self, event: SessionStart) -> None:\n'
        '        if self.settings.fail_on_start:\n'
        "            Settings.model_validate({'unexpected': 1})\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path / 'site'))  # pyright: ignore[reportUnknownMemberType]
    harness = Harness(tmp_path)
    with pytest.raises(PluginSettingsError):
        await harness.loader.command(['add', 'strict', 'clai_strict', '{"secret": "s3cr3t"}'])
    assert harness.store.plugins() == []
    assert b's3cr3t' not in (tmp_path / 'config.db').read_bytes()
    with pytest.raises(PluginError) as raised:
        await harness.loader.command(['add', 'strict', 'clai_strict', '{"fail_on_start": true}'])
    assert not isinstance(raised.value, PluginSettingsError), 'a ValidationError after activation is not about settings'
    assert [plugin.settings for plugin in harness.store.plugins()] == [{'fail_on_start': True}]


async def test_add_keeps_settings_the_plugin_saves_while_activating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / 'site' / 'clai_migrating'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text(
        'from pydantic import BaseModel\n'
        'from pydantic_clai2.plugins import Plugin\n'
        'class Settings(BaseModel):\n'
        '    version: int = 1\n'
        'class Migrating(Plugin[Settings]):\n'
        '    def __init__(self, host, settings):\n'
        '        super().__init__(host, settings)\n'
        '        host.save_settings(settings.model_copy(update={"version": 2}))\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path / 'site'))  # pyright: ignore[reportUnknownMemberType]
    harness = Harness(tmp_path)
    assert await harness.loader.command(['add', 'migrating', 'clai_migrating', '{"version": 1}']) == (
        'Added and loaded migrating.'
    )
    assert [plugin.settings for plugin in harness.store.plugins()] == [{'version': 2}]


async def test_declared_plugin_and_capability_classes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / 'site' / 'clai_extras'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'caps.py').write_text(
        'from pydantic_ai.capabilities import Capability\n'
        'from pydantic_clai2.plugins import Plugin\n'
        'class Greeter(Capability[None]):\n'
        "    def __init__(self, *, greeting: str = 'hi') -> None:\n"
        '        super().__init__(instructions=greeting)\n'
        'class Installer(Plugin):\n'
        '    def get_capabilities(self):\n'
        "        return [Greeter(greeting='yo')]\n"
        'class Other(Plugin):\n'
        '    pass\n'
        'NOT_A_PLUGIN = 3\n'
    )
    (package / 'single.py').write_text(
        'from clai_extras.caps import Installer\n'
        'from pydantic_clai2.plugins import Plugin\n'
        'class _Base(Plugin):\n'
        '    pass\n'
        'class Single(_Base):\n'
        '    def get_capabilities(self):\n'
        '        return Installer.get_capabilities(self)\n'
    )
    (package / 'legacy.py').write_text('def activate(host):\n    pass\n')
    monkeypatch.syspath_prepend(str(tmp_path / 'site'))  # pyright: ignore[reportUnknownMemberType]
    harness = Harness(tmp_path)
    assert (
        await harness.loader.command(['add', 'greeter', 'clai_extras.caps:Greeter', '{"greeting": "hello"}'])
        == 'Added and loaded greeter.'
    )
    assert (
        await harness.loader.command(['add', 'installer', 'clai_extras.caps:Installer'])
        == 'Added and loaded installer.'
    )
    assert await harness.loader.command(['add', 'single', 'clai_extras.single']) == 'Added and loaded single.'
    assert len(harness.loader.capabilities()) == 3
    with pytest.raises(PluginError, match='neither a `Plugin` nor a capability class'):
        await harness.loader.command(['add', 'number', 'clai_extras.caps:NOT_A_PLUGIN'])
    with pytest.raises(PluginError, match='neither a `Plugin` nor a capability class'):
        await harness.loader.command(['add', 'wrong', 'pathlib:Path'])
    with pytest.raises(PluginError, match='neither a `Plugin` nor a capability class'):
        await harness.loader.command(['add', 'empty', 'clai_extras'])
    with pytest.raises(PluginError, match=r'defines several plugins \(Installer, Other\); name one as'):
        await harness.loader.command(['add', 'several', 'clai_extras.caps'])
    for name, factory in [('legacy', 'clai_extras.legacy'), ('legacy_attr', 'clai_extras.legacy:activate')]:
        with pytest.raises(PluginError) as legacy:
            await harness.loader.command(['add', name, factory])
        assert str(legacy.value) == (
            f'Plugin {name!r}: TypeError: {factory} is an `activate(host)` function; plugins are now `Plugin`'
            ' subclasses. See "What a plugin declares" in PLUGINS.md.'
        )
    plugins_md = Path(__file__).parents[2] / 'src' / 'pydantic_clai2' / 'PLUGINS.md'
    assert '\n## What a plugin declares\n' in plugins_md.read_text()
    with pytest.raises(PluginError, match='ModuleNotFoundError'):
        await harness.loader.command(['add', 'missing', 'clai_extras.nope'])
    listing = await harness.loader.command(['list'])
    assert 'greeter: clai_extras.caps:Greeter (enabled, loaded)' in listing
    assert 'missing: clai_extras.nope (enabled, failed: ModuleNotFoundError' in listing
    await harness.loader.command(['reload', 'greeter'])
    assert len(harness.loader.capabilities()) == 3
    assert await harness.loader.command(['remove', 'greeter']) == 'Removed greeter.'
    assert len(harness.loader.capabilities()) == 2


async def test_load_all_skips_missing_plugin_modules_quietly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / 'site' / 'clai_broken'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'plugin.py').write_text('import clai_missing_dependency\n')
    monkeypatch.syspath_prepend(str(tmp_path / 'site'))  # pyright: ignore[reportUnknownMemberType]
    harness = Harness(tmp_path)
    # Saved by a CLAI version that shipped a `retired` built-in; this one does not.
    harness.store.save_plugin(PluginSettings(id='retired', factory='pydantic_clai2.retired'))
    harness.store.save_plugin(PluginSettings(id='gone', factory='clai_gone.plugin:activate'))
    harness.store.save_plugin(PluginSettings(id='broken', factory='clai_broken.plugin'))
    await harness.loader.load_all()
    states = {entry.name: entry.state for entry in harness.loader.entries()}
    assert states['retired'] == "enabled, failed: ModuleNotFoundError: No module named 'pydantic_clai2.retired'"
    assert states['gone'] == "enabled, failed: ModuleNotFoundError: No module named 'clai_gone'"
    assert harness.text == ("Plugin 'broken': ModuleNotFoundError: No module named 'clai_missing_dependency'\n")
    with pytest.raises(PluginError, match=r"No module named 'pydantic_clai2\.retired'"):
        await harness.loader.command(['enable', 'retired'])


async def test_failed_load_leaves_nothing_registered(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('clash', command='help')
    harness.write('explode', end_body='pass', start_body='pass')
    (harness.store.plugins_dir / 'explode.py').write_text(
        'from pydantic_clai2.commands import Command\n'
        'from pydantic_clai2.plugins import Plugin, SessionStart\n'
        'class Explode(Plugin):\n'
        '    def get_commands(self):\n'
        "        return [Command(name='boom', description='x', handler=lambda _: '')]\n"
        '    async def on_session_start(self, event: SessionStart) -> None:\n'
        "        raise RuntimeError('start failed')\n"
    )
    (harness.store.plugins_dir / 'syntax.py').write_text('def activate(host):\n    return (\n')
    await harness.loader.load_all()
    assert {command.name for command in harness.commands} == {'help'}
    states = {entry.name: entry.state for entry in harness.loader.entries()}
    assert states['clash'].startswith('enabled, failed: ValueError')
    assert states['explode'] == 'enabled, failed: RuntimeError: start failed'
    assert states['syntax'].startswith('enabled, failed: SyntaxError')
    assert "Plugin 'clash'" in harness.text and "Plugin 'syntax'" in harness.text
    assert harness.loader.capabilities() == []
    with pytest.raises(PluginError, match='duplicate command: help'):
        await harness.loader.load('clash')


async def test_saved_settings_persist_and_a_failed_start_after_saving_can_load_again(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    (harness.store.plugins_dir / 'tuned.py').write_text(
        'from pydantic import BaseModel\n'
        'from pydantic_clai2.plugins import Plugin, SessionStart\n'
        'class Tuned(BaseModel):\n'
        '    level: int = 1\n'
        'class TunedPlugin(Plugin[Tuned]):\n'
        '    async def on_session_start(self, event: SessionStart) -> None:\n'
        '        level = self.settings.level\n'
        '        self.host.save_settings(Tuned(level=level + 1))\n'
        '        if level == 1:\n'
        "            raise RuntimeError('first start fails')\n"
    )
    await harness.loader.load_all()
    [entry] = harness.loader.entries()
    assert entry.state == 'enabled, failed: RuntimeError: first start fails'
    assert harness.store.plugins()[0].settings == {'level': 2}
    await harness.loader.load('tuned')
    assert harness.loader.entries()[0].state == 'enabled, loaded'
    assert harness.store.plugins()[0].settings == {'level': 3}


async def test_enable_disable_reload_persist_and_refresh_module(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('counter', end_body='pass')
    await harness.loader.load_all()
    await harness.loader.load('counter')
    assert await harness.loader.command(['disable', 'counter']) == 'Disabled counter.'
    assert 'counter stopped exit' in harness.text
    assert harness.loader.capabilities() == []
    assert [plugin.enabled for plugin in harness.store.plugins()] == [False]
    assert 'counter' not in {command.name for command in harness.commands}
    fresh = PluginLoader[None](
        store=harness.store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=harness.store.load()),
    )
    await fresh.load_all()
    assert fresh.entries()[0].state == 'disabled'
    assert await harness.loader.command(['enable', 'counter']) == 'Enabled counter.'
    assert harness.store.plugins()[0].enabled
    assert await harness.loader.command(['reload', 'counter']) == 'Reloaded counter.'
    assert harness.text.count('counter started') == 3
    message = await harness.loader.command(['remove', 'counter'])
    assert message.startswith('Disabled counter. Delete ')
    assert not harness.store.plugins()[0].enabled
    assert harness.loader.entries()[0].state == 'disabled'


async def test_plugins_unloaded_during_a_turn_end_after_it(tmp_path: Path) -> None:
    """The running agent still holds what they declared, so `session_end` waits for the turn."""
    harness = Harness(tmp_path)
    harness.write('alpha', end_body="host.console.print('alpha cleaned')")
    harness.write('beta', end_body="raise RuntimeError('beta cleanup')")
    (harness.store.plugins_dir / 'gamma.py').write_text(
        'from pydantic_ai.capabilities import Hooks\nfrom pydantic_clai2.plugins import Plugin\n\n\n'
        'class Gamma(Plugin):\n    def get_capabilities(self):\n        return (Hooks(),)\n'
    )
    await harness.loader.load_all()
    async with harness.loader.turn():
        assert await harness.loader.command(['disable', 'alpha']) == 'Disabled alpha.'
        assert await harness.loader.command(['disable', 'gamma']) == f'Disabled gamma.\n{TURN_NOTICE}'
        assert await harness.loader.command(['reload', 'beta']) == 'Reloaded beta.'
        assert 'alpha' not in harness.commands
        assert [plugin.enabled for plugin in harness.store.plugins()] == [False, False]
        assert harness.loader.capabilities() == []
        assert harness.text.count('beta started') == 2
        assert 'stopped' not in harness.text
    assert harness.text.index('alpha stopped exit') < harness.text.index('alpha cleaned')
    assert harness.text.index('alpha cleaned') < harness.text.index('beta stopped exit')
    assert "Plugin 'beta': RuntimeError: beta cleanup" in harness.text
    await harness.loader.disable('beta')
    assert harness.text.count('beta stopped exit') == 2


async def test_a_cancelled_turn_still_ends_the_plugins_it_unloaded(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write(
        'alpha', end_body="await __import__('anyio').lowlevel.checkpoint()\n        host.console.print('cleaned')"
    )
    await harness.loader.load_all()
    with anyio.CancelScope() as scope:
        async with harness.loader.turn():
            await harness.loader.disable('alpha')
            scope.cancel()
            await anyio.lowlevel.checkpoint()
    assert scope.cancelled_caught
    assert 'alpha stopped exit\ncleaned' in harness.text


async def test_save_settings_keeps_a_declaration_saved_after_load(tmp_path: Path) -> None:
    """Another CLAI process may replace the declaration while this one has the plugin loaded."""

    class Chosen(BaseModel):
        level: int

    harness = Harness(tmp_path)
    path = harness.write('counter')
    await harness.loader.load('counter')
    newer = PluginSettings(id='counter', factory='counter:Recorder', path=str(path), settings={'level': 1})
    harness.store.save_plugin(newer)
    loaded = harness.loader.entries()[0].loaded
    assert loaded is not None
    loaded.host.save_settings(Chosen(level=2))
    assert harness.store.plugins() == [newer.model_copy(update={'settings': {'level': 2}})]


async def test_fire_reports_observers_and_fails_closed_on_turn_start(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.write('rude', turn_end_body="raise RuntimeError('late')")
    harness.write('polite', start_body="event.text += '!'")
    await harness.loader.load_all()
    start = TurnStart(text='hi')
    await harness.loader.fire(start)
    assert start.text == 'hi!'
    await harness.loader.fire(TurnEnd(text='hi', outcome='completed'))
    assert "Plugin 'rude': RuntimeError: late" in harness.text
    harness.write('rude', start_body="raise RuntimeError('no')", end_body="raise RuntimeError('bye')")
    await harness.loader.reload('rude')
    with pytest.raises(PluginError, match="Plugin 'rude': RuntimeError: no"):
        await harness.loader.fire(TurnStart(text='again'))
    await harness.loader.unload('rude')
    assert "Plugin 'rude': RuntimeError: bye" in harness.text
    await harness.loader.unload('rude')


async def test_command_usage_and_unknown_names(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    assert await harness.loader.command([]) == 'No plugins.'
    with pytest.raises(ValueError, match='Usage'):
        await harness.loader.command(['enable'])
    with pytest.raises(ValueError, match='Unknown plugins action'):
        await harness.loader.command(['frobnicate', 'x'])
    with pytest.raises(ValueError, match='Unknown plugin: ghost'):
        await harness.loader.command(['enable', 'ghost'])
    with pytest.raises(ValueError, match='Usage'):
        await harness.loader.command(['add', 'only-name'])


async def test_loaded_entry_survives_file_removal_until_unloaded(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    path = harness.write('ephemeral')
    await harness.loader.load_all()
    path.unlink()
    entry = harness.loader.entries()[0]
    assert entry.name == 'ephemeral' and entry.loaded is not None
    await harness.loader.unload('ephemeral')
    assert harness.loader.entries() == []
    harness.store.save_plugin(PluginSettings(id='ephemeral', factory='ephemeral'))
    assert harness.loader.entries()[0].source == 'ephemeral'
    assert harness.loader.plugins_dir == harness.store.plugins_dir


BUILTIN = (
    PluginSettings(id='hello', factory='pydantic_ai.capabilities:Capability', settings={'instructions': 'Say hello.'}),
)


async def test_builtin_plugins_are_on_by_default_and_overridable(tmp_path: Path) -> None:
    harness = Harness(tmp_path, builtin=BUILTIN)
    await harness.loader.load_all()
    entry = harness.loader.entries()[0]
    assert entry.name == 'hello' and entry.builtin
    assert entry.source == 'pydantic_ai.capabilities:Capability (built-in)'
    assert entry.state == 'enabled, loaded'
    assert len(harness.loader.capabilities()) == 1
    assert harness.store.plugins() == []

    assert await harness.loader.command(['disable', 'hello']) == 'Disabled hello.'
    assert harness.loader.capabilities() == []
    fresh = Harness(tmp_path, builtin=BUILTIN)
    await fresh.loader.load_all()
    assert fresh.loader.entries()[0].state == 'disabled'
    assert fresh.loader.entries()[0].builtin

    message = await fresh.loader.command(['remove', 'hello'])
    assert message == 'hello is built in; restored its defaults. Use /plugins disable hello to turn it off.'
    assert fresh.store.plugins() == []
    assert fresh.loader.entries()[0].state == 'enabled, loaded'


async def test_store_declaration_replaces_a_builtin(tmp_path: Path) -> None:
    harness = Harness(tmp_path, builtin=BUILTIN)
    harness.write('hello')
    harness.store.save_plugin(
        PluginSettings(id='hello', factory='hello', path=str(harness.store.plugins_dir / 'hello.py'))
    )
    await harness.loader.load_all()
    entry = harness.loader.entries()[0]
    assert not entry.builtin and entry.path is not None
    assert 'hello' in {command.name for command in harness.commands}
    assert (await harness.loader.command(['remove', 'hello'])).startswith('Disabled hello.')


async def test_add_replaces_a_builtin_and_remove_restores_it(tmp_path: Path) -> None:
    harness = Harness(tmp_path, builtin=BUILTIN)
    await harness.loader.load_all()
    message = await harness.loader.command(
        ['add', 'hello', 'pydantic_ai.capabilities:Capability', '{"instructions": "Say goodbye."}']
    )
    assert message == 'Replaced built-in hello.'
    entry = harness.loader.entries()[0]
    assert not entry.builtin and entry.state == 'enabled, loaded'
    assert harness.store.plugins()[0].settings == {'instructions': 'Say goodbye.'}
    assert len(harness.loader.capabilities()) == 1

    message = await harness.loader.command(['remove', 'hello'])
    assert message == 'hello is built in; restored its defaults. Use /plugins disable hello to turn it off.'
    entry = harness.loader.entries()[0]
    assert entry.builtin and entry.state == 'enabled, loaded'
    assert harness.store.plugins() == []
    replace = ['add', 'hello', 'pydantic_ai.capabilities:Capability']
    assert await harness.loader.command(replace) == 'Replaced built-in hello.'
    with pytest.raises(ValueError, match='already exists'):
        await harness.loader.command(replace)


PROJECT = (
    PluginSettings(
        id='hello', factory='pydantic_ai.capabilities:Capability', enabled=False, settings={'instructions': 'Project.'}
    ),
)
"""As `load_project_settings` hands them over: declared by the repository, off until the user approves."""


async def test_project_declarations_sit_above_builtins_and_below_the_store(tmp_path: Path) -> None:
    harness = Harness(tmp_path, builtin=BUILTIN, project=PROJECT)
    await harness.loader.load_all()
    hello = harness.loader.entries()[0]
    assert hello.project and not hello.builtin and hello.declaration.settings == {'instructions': 'Project.'}
    assert hello.source == 'pydantic_ai.capabilities:Capability (project)'
    assert hello.state == 'disabled', 'repository code does not run until the user approves it'
    assert harness.loader.capabilities() == [] and harness.store.plugins() == []

    assert await harness.loader.command(['enable', 'hello']) == 'Enabled hello.'
    assert len(harness.loader.capabilities()) == 1
    fresh = Harness(tmp_path, builtin=BUILTIN, project=PROJECT)
    await fresh.loader.load_all()
    entry = fresh.loader.entries()[0]
    assert entry.project and entry.state == 'enabled, loaded', 'approval is remembered in the user store'

    message = await fresh.loader.command(['remove', 'hello'])
    assert (
        message == 'hello is declared by the project; restored its defaults. Use /plugins disable hello to turn it off.'
    )
    assert fresh.loader.entries()[0].state == 'disabled' and fresh.store.plugins() == []

    replace = ['add', 'hello', 'pydantic_ai.capabilities:Capability', '{"instructions": "Mine."}']
    assert await fresh.loader.command(replace) == 'Replaced project hello.'
    assert not fresh.loader.entries()[0].project and fresh.store.plugins()[0].settings == {'instructions': 'Mine.'}
    with pytest.raises(ValueError, match='already exists'):
        await fresh.loader.command(replace)


async def test_remove_restores_a_project_plugin_that_names_a_file(tmp_path: Path) -> None:
    path = tmp_path / 'repo' / 'filed.py'
    path.parent.mkdir()
    path.write_text('from pydantic_clai2.plugins import Plugin\nclass Filed(Plugin):\n    pass\n')
    project = (PluginSettings(id='filed', factory='filed', path=str(path), enabled=False),)
    harness = Harness(tmp_path, project=project)
    await harness.loader.load_all()
    assert harness.loader.entries()[0].project and harness.loader.entries()[0].path == path
    assert await harness.loader.command(['enable', 'filed']) == 'Enabled filed.'

    message = await harness.loader.command(['remove', 'filed'])
    assert message.startswith('filed is declared by the project; restored its defaults.')
    assert harness.store.plugins() == [] and harness.loader.entries()[0].state == 'disabled'


@pytest.mark.parametrize(
    'plugin_id',
    [
        'ask_user',
        'repo_context',
        'compaction',
        'persistence',
        'observability',
        'notifications',
        'github',
        'slack',
        'posthog',
        'grain',
        'linear',
    ],
)
def test_saved_builtin_factory_paths_upgrade_without_losing_toggles(tmp_path: Path, plugin_id: str) -> None:
    """Previously saved built-ins follow their new import paths; user replacements do not."""
    builtin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == plugin_id)
    old_factory = {
        'persistence': 'pydantic_clai2.sessions',
        'ask_user': 'pydantic_clai2.ask_user_menu:activate',
    }.get(plugin_id, builtin.factory.replace('pydantic_clai2.builtin_plugins.', 'pydantic_clai2.'))
    harness = Harness(tmp_path, builtin=(builtin,))
    harness.store.save_plugin(builtin.model_copy(update={'factory': old_factory, 'enabled': False}))
    entry = harness.loader.entries()[0]
    assert entry.builtin and entry.declaration.factory == builtin.factory and not entry.declaration.enabled

    harness.store.save_plugin(builtin.model_copy(update={'factory': old_factory, 'settings': {'custom': True}}))
    entry = harness.loader.entries()[0]
    assert not entry.builtin and entry.declaration.factory == builtin.factory
    assert entry.declaration.settings == {'custom': True}

    # An independently named plugin can also have used the old import string.
    harness.store.save_plugin(PluginSettings(id='plain', factory=old_factory))
    assert (
        next(item for item in harness.loader.entries() if item.name == 'plain').declaration.factory == builtin.factory
    )


def test_saved_activate_factory_upgrades_to_the_plugin_module(tmp_path: Path) -> None:
    """`ask_user` was declared as `module:activate` before plugins were `Plugin` classes."""
    builtin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'ask_user')
    harness = Harness(tmp_path, builtin=(builtin,))
    old_factory = f'{builtin.factory}:activate'
    harness.store.save_plugin(builtin.model_copy(update={'factory': old_factory, 'enabled': False}))
    entry = harness.loader.entries()[0]
    assert entry.builtin and entry.declaration.factory == builtin.factory and not entry.declaration.enabled


async def test_repo_context_builtin_loads_the_workspace_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'AGENTS.md').write_text('Answer in haiku.\n')
    monkeypatch.chdir(workspace)
    repo_context = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'repo_context')
    harness = Harness(tmp_path, builtin=(repo_context,))
    await harness.loader.load_all()
    entry = harness.loader.entries()[0]
    assert entry.builtin and entry.state == 'enabled, loaded'
    model = TestModel(call_tools=[])
    await Agent(model, deps_type=type(None), capabilities=harness.loader.capabilities()).run(
        'hi', workspace=LocalWorkspaceBackend(working_dir=workspace)
    )
    assert model.last_model_request_parameters is not None
    parts = model.last_model_request_parameters.instruction_parts or []
    assert sum('Answer in haiku.' in part.content for part in parts) == 1
    assert model.last_model_request_parameters.function_tools == []

    assert await harness.loader.command(['disable', 'repo_context']) == 'Disabled repo_context.'
    assert harness.loader.capabilities() == []

    knobs = [
        'add',
        'repo_context',
        'pydantic_clai2.builtin_plugins.repo_context',
        '{"inventory_tool": true, "walk_up": true}',
    ]
    assert await harness.loader.command(knobs) == 'Replaced built-in repo_context.'
    model = TestModel(call_tools=[])
    await Agent(model, deps_type=type(None), capabilities=harness.loader.capabilities()).run(
        'hi', workspace=LocalWorkspaceBackend(working_dir=workspace)
    )
    assert model.last_model_request_parameters is not None
    assert [tool.name for tool in model.last_model_request_parameters.function_tools] == ['inventory_agent_context']
    assert (await harness.loader.command(['remove', 'repo_context'])).startswith('repo_context is built in')
    with pytest.raises(PluginError, match='extra_forbidden'):
        await harness.loader.command(
            ['add', 'repo_context', 'pydantic_clai2.builtin_plugins.repo_context', '{"filenames": []}']
        )


@pytest.mark.parametrize('error', ['RuntimeError', 'CancelledError'])
async def test_failed_start_reports_cleanup_error_without_masking_start_failure(tmp_path: Path, error: str) -> None:
    harness = Harness(tmp_path)
    (harness.store.plugins_dir / 'broken_start.py').write_text(
        'from asyncio import CancelledError\n'
        'from pydantic_clai2.commands import Command\n'
        'from pydantic_clai2.plugins import Plugin\n'
        'class BrokenStart(Plugin):\n'
        '    def get_commands(self):\n'
        '        return [Command(name="temporary", description="temporary", handler=lambda _: "ok")]\n'
        '    async def on_session_start(self, event):\n'
        '        raise RuntimeError("start failed")\n'
        '    async def on_session_end(self, event):\n'
        '        self.host.console.print("cleanup reason " + event.reason)\n'
        f'        raise {error}("cleanup failed")\n'
    )
    with pytest.raises(PluginError, match='start failed'):
        await harness.loader.load('broken_start')
    assert 'cleanup reason error' in harness.text
    assert 'cleanup failed' in harness.text
    assert 'temporary' not in {command.name for command in harness.commands}
    assert harness.loader.entries()[0].loaded is None
    assert harness.loader.entries()[0].error == 'RuntimeError: start failed'


async def test_external_cancellation_during_failed_start_cleanup_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = anyio.Event()
    cancelled = anyio.Event()
    cleaned = anyio.Event()
    task: asyncio.Task[None] | None = None
    module = ModuleType('cancel_during_cleanup')

    class CancelDuringCleanup(Plugin):
        async def on_session_start(self, event: SessionStart) -> None:
            raise RuntimeError('startup failed')

        async def on_session_end(self, event: SessionEnd) -> None:
            try:
                entered.set()
                await anyio.sleep_forever()
            finally:
                cleaned.set()

    CancelDuringCleanup.__module__ = module.__name__
    module.__dict__['CancelDuringCleanup'] = CancelDuringCleanup
    monkeypatch.setitem(sys.modules, module.__name__, module)
    harness = Harness(tmp_path, builtin=(PluginSettings(id='cancel', factory=module.__name__),))

    async def load() -> None:
        nonlocal task
        task = asyncio.current_task()
        try:
            await harness.loader.load('cancel')
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(load)
            await entered.wait()
            assert task is not None
            task.cancel()
            await cancelled.wait()
    assert cleaned.is_set()
    assert harness.loader.entries()[0].loaded is None
    assert not harness.loader.capabilities()


@pytest.mark.parametrize('cancel_scope', [False, True])
async def test_failed_load_cleanup_is_bounded_and_continues_after_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_scope: bool
) -> None:
    entered = anyio.Event()
    stopped = anyio.Event()
    failed = anyio.Event()
    cancelled = anyio.Event()
    module = ModuleType('stalled_cleanup')

    class StalledCleanup(Plugin):
        async def on_session_start(self, event: SessionStart) -> None:
            raise RuntimeError('startup failed')

        async def on_session_end(self, event: SessionEnd) -> None:
            try:
                entered.set()
                await anyio.sleep_forever()
            finally:
                stopped.set()

    StalledCleanup.__module__ = module.__name__
    module.__dict__['StalledCleanup'] = StalledCleanup
    monkeypatch.setitem(sys.modules, module.__name__, module)
    harness = Harness(tmp_path, builtin=(PluginSettings(id='stalled', factory=module.__name__),))

    async def load() -> None:
        try:
            await harness.loader.load('stalled')
        except PluginError as exc:
            assert not cancel_scope
            assert 'startup failed' in str(exc)
            failed.set()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with anyio.fail_after(15):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(load)
            await entered.wait()
            if cancel_scope:
                tasks.cancel_scope.cancel()
            else:
                await failed.wait()
    assert cancelled.is_set() == cancel_scope
    assert stopped.is_set()
    assert 'TimeoutError' in harness.text
    assert harness.loader.entries()[0].loaded is None
