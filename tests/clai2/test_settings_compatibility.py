"""Settings survive upgrades, branch switches, and rejected operations."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from pydantic_clai2.commands import config_command, plugins_command
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.model_settings import model_settings_from_json
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource


@pytest.mark.parametrize(('version', 'has_model_settings'), [(0, False), (1, False), (1, True)])
def test_upgrade_legacy_database_preserves_data(tmp_path: Path, version: int, has_model_settings: bool) -> None:
    path = tmp_path / 'config.db'
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(f'PRAGMA user_version = {version}')
        connection.execute('CREATE TABLE settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)')
        connection.execute('CREATE TABLE plugins (id TEXT PRIMARY KEY, declaration TEXT NOT NULL)')
        connection.executemany(
            'INSERT INTO settings VALUES (?, ?)', [('model', '"test"'), ('display.thinking', 'false')]
        )
        connection.execute(
            'INSERT INTO plugins VALUES (?, ?)',
            ('notify', '{"id":"notify","factory":"notify","enabled":false,"settings":{"sound":false}}'),
        )
        if has_model_settings:
            connection.execute('CREATE TABLE model_settings (model TEXT PRIMARY KEY, settings_json TEXT NOT NULL)')
            connection.execute('INSERT INTO model_settings VALUES (?, ?)', ('test', '{"max_tokens":64}'))

    store = SettingsStore(path)
    assert store.load() == Settings(model='test', thinking=False)
    # Databases from before speculative execution keep it off.
    assert store.load().speculative_code_mode is False
    # Databases from before `/spinner` keep the braille they always showed.
    assert store.load().spinner == 'working'
    assert store.overrides() == {'model': 'test', 'display.thinking': False}
    assert store.plugins() == [PluginSettings(id='notify', factory='notify', enabled=False, settings={'sound': False})]
    assert store.models() == []
    assert store.model_settings('test') == ({'max_tokens': 64} if has_model_settings else {})
    store.add_model(name='test')
    store.save_model_settings('test', {'max_tokens': 100})
    with closing(sqlite3.connect(path)) as connection:
        snapshot = list(connection.iterdump())
        assert connection.execute('PRAGMA user_version').fetchone() == (1,)

    reopened = SettingsStore(path)
    assert reopened.load() == store.load()
    assert reopened.plugins() == store.plugins()
    assert reopened.models() == ['test']
    assert reopened.model_settings('test') == {'max_tokens': 100}
    with closing(sqlite3.connect(path)) as connection:
        assert list(connection.iterdump()) == snapshot
        assert connection.execute('PRAGMA user_version').fetchone() == (1,)


@pytest.mark.parametrize(
    ('key', 'value_json'),
    [('future.setting', '{"enabled":true}'), ('future.setting', 'unrecognized encoding')],
)
def test_unknown_saved_settings_survive_edits(tmp_path: Path, key: str, value_json: str) -> None:
    path = tmp_path / 'config.db'
    store = SettingsStore(path)
    store.set('model', 'test')
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('INSERT INTO settings VALUES (?, ?)', (key, value_json))

    store = SettingsStore(path)
    assert store.load() == Settings(model='test')
    assert store.overrides() == {'model': 'test'}
    assert Settings.model_validate_json(config_command(store, ['show'])) == Settings(model='test')
    config_command(store, ['set', 'display.thinking', 'false'])
    assert not SettingsStore(path).load().thinking
    config_command(store, ['reset', 'display.thinking'])
    assert SettingsStore(path).load() == Settings(model='test')
    snapshot = path.read_bytes()
    with pytest.raises(ValueError, match='Unknown settings:'):
        store.set(key, 'replacement')
    with pytest.raises(ValueError, match='Unknown setting:'):
        store.reset(key)
    with pytest.raises(ValidationError):
        store.set('model', '')
    with pytest.raises(ValidationError):
        store.set('run.request_limit', -1)
    assert path.read_bytes() == snapshot
    with closing(sqlite3.connect(path)) as connection:
        assert dict(connection.execute('SELECT key, value_json FROM settings')) == {
            'model': '"test"',
            key: value_json,
        }


@pytest.mark.parametrize(
    ('key', 'value_json'),
    [
        ('run.request_limit', '-1'),
        ('run.request_limit', '"10"'),
        ('run.request_limit', 'invalid json'),
        ('display.theme', '"light"'),
        ('display.spinner', '""'),
        ('display.spinner', '3'),
        ('run.speculative_code_mode', '"yes"'),
    ],
)
def test_invalid_known_settings_fail_without_data_loss(tmp_path: Path, key: str, value_json: str) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.set('model', 'test')
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('INSERT INTO settings VALUES (?, ?)', (key, value_json))
    snapshot = store.path.read_bytes()
    with pytest.raises(ValidationError):
        store.load()
    assert store.path.read_bytes() == snapshot


def test_incompatible_schema_is_not_modified(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.set('model', 'test')
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('PRAGMA user_version = 2')
        connection.execute('CREATE TABLE future_data (value TEXT NOT NULL)')
        connection.execute('INSERT INTO future_data VALUES (?)', ('keep me',))
    snapshot = store.path.read_bytes()
    with pytest.raises(ValueError, match='Unsupported settings schema version: 2'):
        SettingsStore(store.path)
    assert store.path.read_bytes() == snapshot


def test_historical_model_preferences_survive_new_editor(tmp_path: Path) -> None:
    path = tmp_path / 'config.db'
    store = SettingsStore(path)
    # Literal persisted JSON, not generated from today's schema or defaults.
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            'INSERT INTO model_settings VALUES (?, ?)',
            (
                'openai:gpt-4o',
                '{"temperature":0.5,"custom_params":{"chat_template_kwargs.reasoning_level":30},'
                '"future_option":{"enabled":true}}',
            ),
        )
    original = store.model_settings('openai:gpt-4o')
    assert model_settings_from_json(original).to_model_settings() == {
        'temperature': 0.5,
        'extra_body': {'chat_template_kwargs': {'reasoning_level': 30}},
    }
    assert SettingsStore(path).model_settings('openai:gpt-4o') == original
    source = ModelSettingsSource(store, 'openai:gpt-4o')
    row = FieldMenu(source).row_for('temperature')
    assert row is not None
    assert source.apply(row, '0.8').startswith('Saved')
    assert SettingsStore(path).model_settings('openai:gpt-4o') == {**original, 'temperature': 0.8}
    source.reset(row)
    expected = dict(original)
    expected.pop('temperature')
    assert SettingsStore(path).model_settings('openai:gpt-4o') == expected


@pytest.mark.parametrize('tier', ['auto', 'default', 'flex', 'priority'])
def test_codex_speed_labels_preserve_existing_preferences(tmp_path: Path, tier: str) -> None:
    path = tmp_path / 'config.db'
    store = SettingsStore(path)
    name = 'openai-codex:gpt-6-astra'
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            'INSERT INTO model_settings VALUES (?, ?)',
            (name, '{"service_tier":"' + tier + '","openai_reasoning_effort":"high","future_option":42}'),
        )
    original = store.model_settings(name)
    source = ModelSettingsSource(SettingsStore(path), name)
    menu = FieldMenu(source)
    row = menu.row_for('service_tier')
    assert row is not None and source.current(row) == tier
    assert SettingsStore(path).model_settings(name) == original
    assert source.apply(row, 'priority').startswith('Saved')
    assert SettingsStore(path).model_settings(name) == {**original, 'service_tier': 'priority'}
    settings = model_settings_from_json(SettingsStore(path).model_settings(name), model=name).to_model_settings()
    assert settings is not None and settings.get('service_tier') == 'priority'
    assert settings.get('openai_reasoning_effort') == 'high'
    snapshot = path.read_bytes()
    assert not source.apply(row, 'fast').startswith('Saved')
    assert path.read_bytes() == snapshot
    source.reset(row)
    assert SettingsStore(path).model_settings(name) == {'openai_reasoning_effort': 'high', 'future_option': 42}
    assert source.current(row) == 'default'


def test_saved_coder_declarations_keep_delegation_off(tmp_path: Path) -> None:
    """A `coder` saved before `sub_agents` existed keeps its previous delegation opt-out."""
    store = SettingsStore(tmp_path / 'settings.db')
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.executemany(
            'INSERT INTO plugins VALUES (?, ?)',
            [
                (
                    'coder',
                    '{"id": "coder", "factory": "pydantic_ai_harness.coder:Coder", "enabled": false, '
                    '"settings": {"unrestricted_filesystem": true, "repo_context": false}}',
                ),
                (
                    'mine',
                    '{"id": "mine", "factory": "pydantic_ai_harness.coder:Coder", "settings": {"sub_agents": true}}',
                ),
                ('other', '{"id": "other", "factory": "my_package.other"}'),
            ],
        )
    assert {plugin.id: plugin.settings for plugin in store.plugins()} == {
        'coder': {'unrestricted_filesystem': True, 'repo_context': False, 'sub_agents': False},
        'mine': {'sub_agents': True},
        'other': {},
    }


def test_saved_spinner_from_a_removed_plugin_is_kept(tmp_path: Path) -> None:
    """Plugin and user spinners are unknown when settings load, so any saved name survives."""
    path = tmp_path / 'config.db'
    SettingsStore(path).set('display.spinner', 'wave')
    store = SettingsStore(path)
    assert store.load().spinner == 'wave'
    config_command(store, ['set', 'display.thinking', 'false'])
    assert SettingsStore(path).overrides() == {'display.spinner': 'wave', 'display.thinking': False}


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('factory', ['pydantic_clai2.logfire', 'pydantic_clai2.builtin_plugins.logfire'])
def test_logfire_preferences_follow_the_observability_name(tmp_path: Path, enabled: bool, factory: str) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    legacy = (
        f'{{"id":"logfire","factory":"{factory}","enabled":{str(enabled).lower()},'
        '"settings":{"token":{"name":"TEAM_LOGFIRE"},"base_url":"https://logfire-eu.pydantic.dev",'
        '"include_content":false,"ui_events":true}}'
    )
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('INSERT INTO plugins VALUES (?, ?)', ('logfire', legacy))

    expected = PluginSettings.model_validate_json(legacy).model_copy(update={'id': 'observability'})
    assert store.plugins() == [expected]
    assert SettingsStore(store.path).plugins() == [expected]
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute('SELECT id, declaration FROM plugins').fetchall() == [('logfire', legacy)]

    updated = expected.model_copy(
        update={'enabled': not enabled, 'settings': {**expected.settings, 'ui_events': False}}
    )
    store.save_plugin(updated)
    assert SettingsStore(store.path).plugins() == [updated]
    # Older builds still see the same plugin and its latest settings, not a second tracing plugin.
    with closing(sqlite3.connect(store.path)) as connection:
        rows = connection.execute('SELECT id, declaration FROM plugins').fetchall()
    assert len(rows) == 1 and rows[0][0] == 'logfire'
    assert PluginSettings.model_validate_json(rows[0][1]) == updated.model_copy(update={'id': 'logfire'})

    store.delete_plugin('observability')
    assert SettingsStore(store.path).plugins() == []


def test_new_observability_preferences_remain_readable_by_older_builds(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    plugin = PluginSettings(id='observability', factory='my_custom_tracing', enabled=False)
    store.save_plugin(plugin)
    store.save_plugin(PluginSettings(id='mcp', factory='pydantic_clai2.mcp'))
    assert [saved.id for saved in store.plugins()] == ['mcp', 'observability']
    with closing(sqlite3.connect(store.path)) as connection:
        stored = connection.execute('SELECT declaration FROM plugins WHERE id = ?', ('logfire',)).fetchone()
    assert stored is not None
    assert PluginSettings.model_validate_json(stored[0]) == plugin.model_copy(update={'id': 'logfire'})
    store.delete_plugin('observability')
    assert [saved.id for saved in store.plugins()] == ['mcp']


@pytest.mark.parametrize('with_legacy', [False, True])
@pytest.mark.parametrize('action', ['disable', 'remove'])
def test_existing_observability_rows_are_coalesced(tmp_path: Path, with_legacy: bool, action: str) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    current = PluginSettings(id='observability', factory='custom_tracing', settings={'include_content': False})
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('INSERT INTO plugins VALUES (?, ?)', (current.id, current.model_dump_json()))
        if with_legacy:
            legacy = PluginSettings(id='logfire', factory='pydantic_clai2.builtin_plugins.logfire', enabled=False)
            connection.execute('INSERT INTO plugins VALUES (?, ?)', (legacy.id, legacy.model_dump_json()))
    assert store.plugins() == [current]
    plugins_command(store, [action, 'observability'])
    assert store.plugins() == ([current.model_copy(update={'enabled': False})] if action == 'disable' else [])
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute('SELECT id FROM plugins').fetchall() == (
            [('logfire',)] if action == 'disable' else []
        )


def test_plugin_alias_coalescing_rolls_back_on_failure(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    plugin = PluginSettings(id='observability', factory='custom_tracing')
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('INSERT INTO plugins VALUES (?, ?)', (plugin.id, plugin.model_dump_json()))
        connection.execute("CREATE TRIGGER refuse_insert BEFORE INSERT ON plugins BEGIN SELECT RAISE(FAIL, 'no'); END")
        original = list(connection.iterdump())
    with pytest.raises(sqlite3.IntegrityError):
        store.save_plugin(plugin.model_copy(update={'enabled': False}))
    with closing(sqlite3.connect(store.path)) as connection:
        assert list(connection.iterdump()) == original


def test_legacy_logfire_commands_edit_the_renamed_plugin(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    plugins_command(store, ['add', 'logfire', 'pydantic_clai2.builtin_plugins.logfire'])
    plugins_command(store, ['disable', 'logfire'])
    assert store.plugins()[0].id == 'observability' and not store.plugins()[0].enabled
    plugins_command(store, ['enable', 'logfire'])
    assert store.plugins()[0].enabled
    plugins_command(store, ['remove', 'logfire'])
    assert store.plugins() == []
