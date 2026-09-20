"""Settings survive upgrades, branch switches, and rejected operations."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from pydantic_clai2.commands import config_command
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.model_menu import ModelSettingsSource
from pydantic_clai2.model_settings import model_settings_from_json
from pydantic_clai2.settings_store import SettingsStore


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
    [('display.theme', '"light"'), ('future.setting', '{"enabled":true}'), ('future.setting', 'unrecognized encoding')],
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


@pytest.mark.parametrize('value_json', ['-1', '"10"', 'invalid json'])
def test_invalid_known_settings_fail_without_data_loss(tmp_path: Path, value_json: str) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.set('model', 'test')
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute('INSERT INTO settings VALUES (?, ?)', ('run.request_limit', value_json))
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
