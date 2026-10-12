"""Feature tags on saved plugin settings: other builds' settings are ignored, never misapplied."""

import io
import sqlite3
import sys
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, Field, JsonValue
from rich.console import Console
from termflow.tui import MenuItem

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, Capability, Hooks
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.subagents import SubAgents
from pydantic_clai2._app import _Shell, create_shell  # pyright: ignore[reportPrivateUsage]
from pydantic_clai2.commands import Commands, plugins_command
from pydantic_clai2.config import PluginSettings, features
from pydantic_clai2.config.features import check_feature_name
from pydantic_clai2.config.plugin_requirements import (
    UNREADABLE,
    apply_requirements,
    declared_requirements,
    ignored_notice,
    merged_requirements,
    stored_requirements,
    withheld,
)
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionStart, TurnStart
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.capability_guard import CapabilitySetupError
from pydantic_clai2.ui.menus.plugin_menu import PluginMenu

CODER = 'pydantic_ai_harness.coder:Coder'

FANCY = """
from pydantic import BaseModel
from pydantic_clai2.plugins import Plugin, PluginHost

SEEN = []


class Settings(BaseModel):
    mode: str = 'plain'
    color: str = 'blue'


class Fancy(Plugin[Settings]):
    @classmethod
    def from_host(cls, host):
        return cls(host, host.settings(Settings, requires={'mode': ['fancy-mode']}))

    def __init__(self, host, settings):
        super().__init__(host, settings)
        SEEN.append(settings.model_dump())
        host.console.print(f'mode={settings.mode} color={settings.color}')

    async def configure(self) -> str:
        self.host.save_settings(self.settings.model_copy(update={'color': 'green'}))
        return 'saved'
"""


def tags(store: SettingsStore, plugin_id: str) -> JsonValue | None:
    return store.plugin_requirements(plugin_id)


def save_raw(store: SettingsStore, plugin_id: str, declaration: str, requirements: str | None) -> None:
    """Rows written as another build would, not through today's models."""
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('INSERT OR REPLACE INTO plugins VALUES (?, ?)', (plugin_id, declaration))
        if requirements is not None:
            connection.execute('INSERT OR REPLACE INTO plugin_requirements VALUES (?, ?)', (plugin_id, requirements))


class Harness:
    def __init__(self, tmp_path: Path, *, builtin: tuple[PluginSettings, ...] = ()) -> None:
        self.store = SettingsStore(tmp_path / 'config.db')
        self.store.plugins_dir.mkdir(exist_ok=True)
        (self.store.plugins_dir / 'fancy.py').write_text(FANCY)
        self.output = io.StringIO()
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=builtin,
        )

    @property
    def text(self) -> str:
        return self.output.getvalue()

    def fancy(self, settings: str, requirements: str | None) -> None:
        path = self.store.plugins_dir / 'fancy.py'
        declaration = f'{{"id": "fancy", "factory": "fancy", "path": "{path}", "settings": {settings}}}'
        save_raw(self.store, 'fancy', declaration, requirements)


def test_feature_names_are_hyphenated_lowercase_words() -> None:
    assert check_feature_name('stock-bound-delegation') == 'stock-bound-delegation'
    for bad in ('Stock', 'two  words', 'trailing-', '-leading', 'under_score', '', UNREADABLE):
        with pytest.raises(ValueError, match='lowercase words joined by hyphens'):
            check_feature_name(bad)


def test_declarations_are_validated() -> None:
    assert declared_requirements({'sub_agents': ['stock-bound-delegation']}) == {
        'sub_agents': frozenset({'stock-bound-delegation'})
    }
    with pytest.raises(ValueError, match='lists no required features'):
        declared_requirements({'sub_agents': []})
    with pytest.raises(ValueError, match='lowercase words'):
        declared_requirements({'sub_agents': ['Bad Name']})


def test_stored_rows_are_read_safely() -> None:
    settings: dict[str, JsonValue] = {'a': 1, 'b': 2, 'c': 3}
    assert stored_requirements(None, settings) == {}
    assert stored_requirements({'a': ['x'], 'b': 'x', 'gone': ['y'], 'c': []}, settings) == {
        'a': frozenset({'x'}),
        'b': frozenset({UNREADABLE}),
        'c': frozenset(),
    }
    assert stored_requirements('not json', {'a': 1}) == {'a': frozenset({UNREADABLE})}


def test_unsupported_settings_fall_back_to_defaults_only() -> None:
    settings: dict[str, JsonValue] = {'shipped': 1, 'plugin': 2, 'kept': 3}
    requirements = {
        'shipped': frozenset({'missing'}),
        'plugin': frozenset({'missing', 'present'}),
        'kept': frozenset({'present'}),
        'absent': frozenset({'missing'}),
    }
    applied = apply_requirements(settings, requirements, defaults={'shipped': 0}, supported=frozenset({'present'}))
    assert applied.settings == {'shipped': 0, 'kept': 3}
    assert applied.ignored == {'shipped': frozenset({'missing'}), 'plugin': frozenset({'missing'})}
    assert withheld(settings, {'plugin': frozenset({'unknown-feature'})}) == {'plugin': 2}
    assert ignored_notice('coder', {'sub_agents': frozenset({'stock-bound-delegation'})}) == (
        'coder: ignored saved sub_agents (needs stock-bound-delegation); using defaults.'
    )


def test_tags_survive_writers_that_do_not_change_the_value() -> None:
    old: dict[str, JsonValue] = {'same': True, 'changed': True, 'odd': 1, 'untagged': 0}
    new: dict[str, JsonValue] = {'same': True, 'changed': False, 'odd': 1, 'untagged': 0, 'added': 5}
    row = merged_requirements(
        old_settings=old,
        old_row={'same': ['old-feature'], 'changed': ['old-feature'], 'odd': {'future': 'format'}},
        new_settings=new,
        declared={'same': frozenset({'new-feature'}), 'added': frozenset({'new-feature'})},
    )
    assert row == {
        'same': ['new-feature', 'old-feature'],
        'odd': {'future': 'format'},
        'added': ['new-feature'],
    }
    # A stored `null` entry stays unreadable rather than counting as untagged.
    nulled = merged_requirements(old_settings=old, old_row={'same': None}, new_settings=new, declared={})
    assert nulled == {'same': [UNREADABLE]}
    unreadable = merged_requirements(old_settings=old, old_row='garbage', new_settings=new, declared={})
    assert unreadable == {'same': [UNREADABLE], 'odd': [UNREADABLE], 'untagged': [UNREADABLE]}
    assert merged_requirements(old_settings={}, old_row=None, new_settings=new, declared={}) == {}


BAD_TAGS = """
from pydantic import BaseModel
from pydantic_clai2.plugins import Plugin


class Settings(BaseModel):
    mode: str = 'plain'


class BadTags(Plugin[Settings]):
    @classmethod
    def from_host(cls, host):
        return cls(host, host.settings(Settings, requires={'mode': ['Not A Name']}))
"""


async def test_rejected_declaration_leaves_stored_data_intact(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    (harness.store.plugins_dir / 'fancy.py').write_text(BAD_TAGS)
    harness.fancy('{"mode": "fancy"}', '{"mode": ["fancy-mode"]}')
    snapshot = harness.store.path.read_bytes()
    await harness.loader.load_all()
    assert "Plugin 'fancy': ValueError: Feature name 'Not A Name'" in harness.text
    assert harness.store.path.read_bytes() == snapshot


def test_store_keeps_tags_beside_declarations(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    plugin = PluginSettings(id='p', factory='p', settings={'mode': 'fancy', 'color': 'red'})
    store.save_plugin(plugin, requires={'mode': frozenset({'fancy-mode'})})
    assert tags(store, 'p') == {'mode': ['fancy-mode']}
    # A build that knows nothing of the feature saves an unchanged value: the tag stays.
    store.save_plugin(plugin.model_copy(update={'enabled': False}))
    assert tags(store, 'p') == {'mode': ['fancy-mode']}
    # The declaration itself never carries tags, so older builds still validate it.
    with closing(sqlite3.connect(store.path)) as connection:
        assert PluginSettings.model_validate_json(connection.execute('SELECT declaration FROM plugins').fetchone()[0])
    store.save_plugin(plugin.model_copy(update={'settings': {'mode': 'plain'}}))
    assert tags(store, 'p') is None
    store.save_plugin(plugin, requires={'mode': frozenset({'fancy-mode'})})
    store.delete_plugin('p')
    assert tags(store, 'p') is None and store.plugins() == []


def test_replacing_the_factory_drops_the_old_plugins_tags(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(
        PluginSettings(id='p', factory='old', settings={'mode': 'fancy'}), requires={'mode': frozenset({'old-only'})}
    )
    store.save_plugin(PluginSettings(id='p', factory='new', settings={'mode': 'fancy'}))
    assert tags(store, 'p') is None


def test_null_requirements_row_is_unreadable(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    save_raw(store, 'p', '{"id": "p", "factory": "p", "settings": {"mode": "fancy"}}', 'null')
    assert stored_requirements(store.plugin_requirements('p'), {'mode': 'fancy'}) == {'mode': frozenset({UNREADABLE})}


def test_renamed_plugin_tags_follow_its_stored_id(tmp_path: Path) -> None:
    """`observability` is stored as `logfire` for older builds; its tags live under the same ID."""
    store = SettingsStore(tmp_path / 'config.db')
    plugin = PluginSettings(id='observability', factory='p', settings={'mode': 'fancy'})
    store.save_plugin(plugin, requires={'mode': frozenset({'fancy-mode'})})
    assert tags(store, 'observability') == tags(store, 'logfire') == {'mode': ['fancy-mode']}
    with closing(sqlite3.connect(store.path)) as connection:
        assert [row[0] for row in connection.execute('SELECT id FROM plugin_requirements')] == ['logfire']
    store.save_plugin(plugin.model_copy(update={'enabled': False}))
    assert tags(store, 'observability') == {'mode': ['fancy-mode']}
    store.delete_plugin('observability')
    assert tags(store, 'logfire') is None


def test_unreadable_and_stale_rows(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    save_raw(store, 'p', '{"id": "p", "factory": "p", "settings": {"mode": "fancy"}}', 'not json')
    assert tags(store, 'p') == 'not json'
    store.save_plugin(PluginSettings(id='p', factory='p', settings={'mode': 'fancy'}))
    assert tags(store, 'p') == {'mode': [UNREADABLE]}
    # Tags left by a build that deleted the declaration without knowing the table do not carry over.
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('DELETE FROM plugins')
    store.save_plugin(PluginSettings(id='p', factory='p', settings={'mode': 'fancy'}))
    assert tags(store, 'p') is None
    # A declaration row another build wrote unreadably is replaced like any other.
    save_raw(store, 'q', 'not json', None)
    store.save_plugin(PluginSettings(id='q', factory='q', settings={'a': 1}), requires={'a': frozenset({'x'})})
    assert tags(store, 'q') == {'a': ['x']}


async def test_unsupported_setting_is_dropped_with_one_notice(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.fancy('{"mode": "fancy", "color": "red"}', '{"mode": ["fancy-mode"]}')
    await harness.loader.load_all()
    assert 'mode=plain color=red' in harness.text
    assert harness.text.count('fancy: ignored saved mode (needs fancy-mode); using defaults.') == 1
    assert '(needs fancy-mode)' in await harness.loader.command(['reload', 'fancy'])
    menu = PluginMenu(harness.loader, apply=lambda action: None)
    assert 'fancy: ignored saved mode' in menu.details(MenuItem('fancy', value='fancy'))
    # Nothing was rewritten by reading.
    assert harness.store.plugins()[0].settings == {'mode': 'fancy', 'color': 'red'}


async def test_supported_setting_is_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, 'SUPPORTED_FEATURES', frozenset({'fancy-mode'}))
    harness = Harness(tmp_path)
    harness.fancy('{"mode": "fancy", "color": "red"}', '{"mode": ["fancy-mode"]}')
    await harness.loader.load_all()
    assert 'mode=fancy color=red' in harness.text
    assert 'ignored' not in harness.text
    assert await harness.loader.command(['reload', 'fancy']) == 'Reloaded fancy.'


async def test_unknown_or_unreadable_feature_drops_the_setting(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.fancy('{"mode": "fancy", "color": "red"}', '{"mode": ["feature-from-the-future"], "color": 7}')
    await harness.loader.load_all()
    assert 'mode=plain color=blue' in harness.text
    assert f'ignored saved color, mode (needs {UNREADABLE}, feature-from-the-future)' in harness.text


async def test_untagged_settings_and_staged_adds_load_as_saved(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.fancy('{"mode": "fancy"}', '{"color": ["fancy-mode"]}')
    await harness.loader.load_all()
    assert 'mode=fancy' in harness.text and 'ignored' not in harness.text


async def test_saving_writes_tags_and_keeps_ignored_values(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.fancy('{"mode": "fancy", "color": "red"}', '{"mode": ["fancy-mode"]}')
    await harness.loader.load_all()
    assert await harness.loader.configure('fancy') == 'saved'
    # This build could not judge `mode`, so it wrote the saved value back, still tagged.
    assert harness.store.plugins()[0].settings == {'mode': 'fancy', 'color': 'green'}
    assert tags(harness.store, 'fancy') == {'mode': ['fancy-mode']}


async def test_plugin_declared_tags_are_attached_on_enable_and_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(features, 'SUPPORTED_FEATURES', frozenset({'fancy-mode'}))
    harness = Harness(tmp_path)
    harness.fancy('{"mode": "fancy"}', None)
    await harness.loader.command(['disable', 'fancy'])
    assert tags(harness.store, 'fancy') is None  # Not loaded, so it has not said what it needs yet.
    await harness.loader.enable('fancy')
    assert tags(harness.store, 'fancy') == {'mode': ['fancy-mode']}
    assert harness.store.plugins()[0].settings == {'mode': 'fancy'}
    await harness.loader.command(['disable', 'fancy'])
    # Enabling from the command also opens its settings menu, which saves too.
    assert await harness.loader.command(['enable', 'fancy']) == 'Enabled fancy.\nsaved'
    assert tags(harness.store, 'fancy') == {'mode': ['fancy-mode']}
    await harness.loader.command(['disable', 'fancy'])
    await harness.loader.command(['enable', 'fancy'])
    await harness.loader.command(['remove', 'fancy'])
    assert tags(harness.store, 'fancy') == {'mode': ['fancy-mode']}


async def test_add_attaches_the_plugins_own_tags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(tmp_path)
    (tmp_path / 'fancy_module.py').write_text(FANCY)
    monkeypatch.setattr(sys, 'path', [str(tmp_path), *sys.path])
    added = await harness.loader.command(['add', 'mine', 'fancy_module', '{"mode": "fancy"}'])
    assert added == 'Added and loaded mine.\nsaved'
    assert tags(harness.store, 'mine') == {'mode': ['fancy-mode']}
    await harness.loader.close('exit')


def test_host_rejects_bad_requirement_declarations() -> None:
    class Model(BaseModel):
        mode: str = Field(default='plain', alias='Mode')

    host = PluginHost[None](name='p', console=Console(file=io.StringIO()), settings={})
    assert host.settings(Model, requires={'Mode': ['fancy-mode']}).mode == 'plain'
    assert host.requirements == {'Mode': frozenset({'fancy-mode'})}
    with pytest.raises(ValueError, match='has no setting colour'):
        host.settings(Model, requires={'colour': ['fancy-mode']})
    with pytest.raises(ValueError, match='lowercase words'):
        host.settings(Model, requires={'mode': ['Fancy']})
    assert host.requirements == {'Mode': frozenset({'fancy-mode'})}


def test_offline_add_uses_the_capability_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(features.CAPABILITY_REQUIREMENTS, CODER, {'sub_agents': frozenset({'stock-bound-delegation'})})
    store = SettingsStore(tmp_path / 'config.db')
    plugins_command(store, ['add', 'mine', CODER, '{"sub_agents": true}'])
    assert tags(store, 'mine') == {'sub_agents': ['stock-bound-delegation']}


GATE = """
from pydantic_ai.capabilities import Capability, Hooks
from pydantic_ai.exceptions import UserError
from pydantic_clai2.plugins import Plugin


async def refuse(ctx, *, handler):
    raise UserError('blocked by policy')


class Gate(Plugin):
    def get_capabilities(self):
        return [Capability(instructions='Be brief.'), Hooks(run=refuse)]
"""


WRAPPED_GATE = """
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability, Hooks, WrapperCapability
from pydantic_ai.exceptions import UserError


async def refuse(ctx, *, handler):
    raise UserError('blocked by policy')


@dataclass
class Gated(WrapperCapability[None]):
    wrapped: AbstractCapability[None] = field(default_factory=lambda: Hooks(run=refuse))
"""


async def run_unguarded(tmp_path: Path, guarded: list[AgentCapability[None]]) -> None:
    reported: list[CapabilitySetupError] = []
    conversation = Session(Agent(TestModel()), deps=None, plugins=guarded, workspace=tmp_path)
    conversation.on_setup_error = reported.append
    with pytest.raises(UserError, match='blocked by policy'):
        await conversation.prompt('hello')
    assert reported == []


async def test_plugin_hooks_and_contributions_stay_unguarded(tmp_path: Path) -> None:
    """Fail-soft covers only capabilities built from saved settings: a gate that raises still stops every run."""
    harness = Harness(tmp_path)
    (harness.store.plugins_dir / 'fancy.py').unlink()
    (harness.store.plugins_dir / 'gate.py').write_text(GATE)
    await harness.loader.load_all()
    guarded = harness.loader.run_capabilities()
    assert [type(capability) for capability in guarded] == [Capability, Hooks]
    await run_unguarded(tmp_path, guarded)


async def test_settings_built_capability_with_nested_hooks_stays_unguarded(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    (harness.store.plugins_dir / 'fancy.py').unlink()
    path = harness.store.plugins_dir / 'wrapped_gate.py'
    path.write_text(WRAPPED_GATE)
    save_raw(
        harness.store,
        'wrapped_gate',
        f'{{"id": "wrapped_gate", "factory": "wrapped_gate:Gated", "path": "{path}"}}',
        None,
    )
    await harness.loader.load_all()
    guarded = harness.loader.run_capabilities()
    assert [type(capability).__name__ for capability in guarded] == ['Gated']
    await run_unguarded(tmp_path, guarded)


def coder_shell(tmp_path: Path, output: io.StringIO, *, sub_agents_tags: str | None) -> _Shell[None, str]:
    store = SettingsStore(tmp_path / 'settings.db')
    store.plugins_dir.mkdir(exist_ok=True)
    path = store.plugins_dir / 'bare_coder.py'
    path.write_text('from pydantic_ai_harness.coder import Coder\n')
    factory = 'bare_coder:Coder'
    save_raw(
        store,
        'coder',
        f'{{"id": "coder", "factory": "{factory}", "path": "{path}", "settings": '
        '{"unrestricted_filesystem": true, "repo_context": false, "sub_agents": true}}',
        sub_agents_tags,
    )
    return create_shell(
        Agent(TestModel(call_tools=[])),
        deps=None,
        plugins=(),
        usage_limits=None,
        settings=None,
        project=ProjectSettings(),
        console=Console(file=output, width=300),
        store=store,
        builtin_plugins=[PluginSettings(id='coder', factory=factory, path=str(path), settings={'sub_agents': False})],
    )


def delegates(capabilities: Sequence[AgentCapability[None]]) -> bool:
    leaves: list[AbstractCapability[None]] = []
    for capability in capabilities:
        assert isinstance(capability, AbstractCapability)
        cast(AbstractCapability[None], capability).apply(leaves.append)
    return any(isinstance(leaf, SubAgents) for leaf in leaves)


def test_forks_report_setup_errors_like_the_foreground(tmp_path: Path) -> None:
    shell = coder_shell(tmp_path, io.StringIO(), sub_agents_tags=None)
    assert shell.fork_session(None, []).on_setup_error == shell.capability_failed


async def test_coder_delegation_saved_by_another_build_is_ignored(tmp_path: Path) -> None:
    """The incident: a branch saved `sub_agents: true`, which this build cannot run."""
    output = io.StringIO()
    shell = coder_shell(tmp_path, output, sub_agents_tags='{"sub_agents": ["stock-bound-delegation"]}')
    await shell.loader.load_all()
    try:
        assert 'coder: ignored saved sub_agents (needs stock-bound-delegation); using defaults.' in output.getvalue()
        assert not delegates(shell.loader.capabilities())
        ended = await shell.run_turn(TurnStart(text='hello'), headless=True)
        assert ended.outcome == 'completed', ended.error
    finally:
        await shell.loader.close('exit')


async def test_untagged_setup_error_costs_one_turn_then_the_capability(tmp_path: Path) -> None:
    """Fail-soft: the same setting saved untagged makes `Coder` refuse the run. That turn fails closed;
    later turns run without `Coder` until it is reloaded."""
    output = io.StringIO()
    shell = coder_shell(tmp_path, output, sub_agents_tags=None)
    await shell.loader.load_all()
    try:
        assert delegates(shell.loader.capabilities())
        # Stock-agent binding compares capability identity; reading the snapshot must not rebuild guards.
        guarded = shell.loader.run_capabilities()
        assert guarded[0] is shell.loader.run_capabilities()[0]
        first = await shell.run_turn(TurnStart(text='hello'), headless=True)
        assert first.outcome == 'failed'
        assert isinstance(first.error, CapabilitySetupError)
        assert "Plugin 'coder': UserError: `SubAgents(include_self=True)`" in str(first.error)
        second = await shell.run_turn(TurnStart(text='again'), headless=True)
        assert second.outcome == 'completed', second.error
        text = output.getvalue()
        assert text.count('Leaving the failing coder capability out of later turns') == 1
        assert 'run /plugins reload coder' in text
        assert shell.loader.capabilities() == []
        await shell.loader.command(['reload', 'coder'])
        assert delegates(shell.loader.capabilities())
    finally:
        await shell.loader.close('exit')
