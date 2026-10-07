"""Settings survive upgrades, branch switches, and rejected operations."""

import io
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.console import Console

from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent
from pydantic_clai2 import StreamRenderer
from pydantic_clai2.builtin_plugins.logfire import LogfireAccount, LogfireSettings, LogfireSource
from pydantic_clai2.commands import config_command, plugins_command
from pydantic_clai2.config import PluginSettings, Settings, features
from pydantic_clai2.config.api_keys import KeyReference
from pydantic_clai2.config.plugin_requirements import apply_requirements, stored_requirements
from pydantic_clai2.config.settings_store import SettingsStore, StoredAccount
from pydantic_clai2.models.model_settings import model_settings_from_json
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource
from tests.clai2.test_logfire import Recorder, observability_loader, recorder as recorder


@pytest.mark.parametrize('user_tag', [None, 'git-email', 'false'])
async def test_logfire_user_tag_settings_survive_older_builds(
    tmp_path: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch, user_tag: str | None
) -> None:
    loader, store = observability_loader(tmp_path)
    # Saved by a build before user tags: set up with a token, but no sign-in email was recorded.
    previous = PluginSettings(
        id='observability',
        factory='pydantic_clai2.builtin_plugins.logfire',
        settings={
            'send_to_logfire': False,
            'include_content': False,
            'service_name': 'shared-project',
            'token': {'name': 'LOGFIRE_TOKEN_TEAM'},
        },
    )
    store.save_plugin(previous)
    try:
        await loader.load_all()
        loaded = loader.entries()[0].loaded
        assert loaded is not None
        settings = loaded.plugin.host.settings(LogfireSettings)
        assert (settings.user_tag, settings.account) == ('logfire-account', None)  # No identity to tag with.
        assert settings.httpx is False
        assert store.plugins() == [previous]  # Loading an old declaration does not rewrite it.
        source = LogfireSource(loaded.plugin.host)
        rows = {row.key: row for row in source.rows()}
        if user_tag is None:
            source.apply(rows['service_name'], 'changed-project')  # Even an unrelated edit saves the defaults.
        else:
            source.apply(rows['user_tag'], user_tag)
        [saved] = store.plugins()
        assert (
            saved.settings['user_tag'] == {None: 'logfire-account', 'git-email': 'git-email', 'false': False}[user_tag]
        )
        assert saved.settings['account'] is None
        requirements = store.plugin_requirements('observability')
        assert requirements == {
            'user_tag': ['logfire-user-tag'],
            'account': ['logfire-user-tag'],
            'httpx': ['logfire-httpx'],
        }
        old_view = apply_requirements(
            saved.settings, stored_requirements(requirements, saved.settings), defaults={}, supported=frozenset()
        )
        assert old_view.settings == {
            key: value for key, value in saved.settings.items() if key not in ('user_tag', 'account', 'httpx')
        }
        monkeypatch.setattr(features, 'SUPPORTED_FEATURES', frozenset[str]())
        await loader.reload('observability')
        reloaded = loader.entries()[0].loaded
        assert reloaded is not None
        old_settings = reloaded.plugin.host.settings(LogfireSettings)
        assert old_settings.user_tag == 'logfire-account'
        assert not old_settings.include_content
        assert store.plugins() == [saved]  # An older build can read without discarding the newer preference.
    finally:
        await loader.close('exit')
    assert recorder.exporters and all(exporter.closed for exporter in recorder.exporters)


async def test_httpx_opt_in_is_ignored_by_older_builds(tmp_path: Path, recorder: Recorder) -> None:
    loader, store = observability_loader(tmp_path)
    try:
        await loader.load_all()
        host = _observability_host(loader)
        source = LogfireSource(host)
        row = next(row for row in source.rows() if row.key == 'httpx')
        source.apply(row, 'true')
        [saved] = store.plugins()
        assert saved.settings['httpx'] is True
        requirements = store.plugin_requirements('observability')
        assert requirements == {
            'httpx': ['logfire-httpx'],
            'user_tag': ['logfire-user-tag'],
            'account': ['logfire-user-tag'],
        }
        old_view = apply_requirements(
            saved.settings,
            stored_requirements(requirements, saved.settings),
            defaults={'httpx': False},
            supported=frozenset({'logfire-user-tag'}),
        )
        assert old_view.settings['httpx'] is False
        assert store.plugins() == [saved]
    finally:
        await loader.close('exit')


@pytest.mark.parametrize('ui_events', [None, False, True])
async def test_logfire_ui_events_default_preserves_saved_overrides(
    tmp_path: Path, recorder: Recorder, ui_events: bool | None
) -> None:
    loader, store = observability_loader(tmp_path)
    # Historical declarations either omitted UI events (then default off), or saved an explicit choice.
    previous = PluginSettings(
        id='observability',
        factory='pydantic_clai2.builtin_plugins.logfire',
        settings={'send_to_logfire': False, 'include_content': False, 'service_name': 'shared-project'},
    )
    if ui_events is not None:
        previous.settings['ui_events'] = ui_events
    store.save_plugin(previous)
    expected = ui_events is not False
    try:
        await loader.load_all()
        host = _observability_host(loader)
        assert host.settings(LogfireSettings).ui_events is expected
        assert store.plugins() == [previous]  # No migration or write on load.
        source = LogfireSource(host)
        rows = {row.key: row for row in source.rows()}
        source.apply(rows['service_name'], 'changed-project')
        [saved] = store.plugins()
        assert saved.settings['ui_events'] is expected
        assert saved.settings['include_content'] is False
        assert saved.settings['send_to_logfire'] is False
        assert SettingsStore(store.path).plugins() == [saved]
        await loader.reload('observability')
        assert _observability_host(loader).settings(LogfireSettings).ui_events is expected
        assert store.plugins() == [saved]
    finally:
        await loader.close('exit')
    assert recorder.exporters and all(exporter.closed for exporter in recorder.exporters)


def _observability_host(loader: PluginLoader[None]) -> PluginHost[None]:
    loaded = loader.entries()[0].loaded
    assert loaded is not None
    return loaded.plugin.host


@pytest.mark.parametrize('older_token', [None, KeyReference(name='LOGFIRE_TOKEN_OTHER')])
async def test_an_older_build_changing_the_token_retires_the_sign_in_email(
    tmp_path: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch, older_token: KeyReference | None
) -> None:
    loader, store = observability_loader(tmp_path)
    team = KeyReference(name='LOGFIRE_TOKEN_TEAM')
    supported = features.SUPPORTED_FEATURES
    store.save_plugin(
        PluginSettings(
            id='observability', factory='pydantic_clai2.builtin_plugins.logfire', settings={'send_to_logfire': False}
        )
    )
    try:
        await loader.load_all()
        # This build's project setup saves the token with the account that signed in, then reloads.
        host = _observability_host(loader)
        account = LogfireAccount(email='mike@example.com', token=team)
        host.save_settings(host.settings(LogfireSettings).model_copy(update={'token': team, 'account': account}))
        await loader.reload('observability')
        # An older build, which ignores `account`, sets up another project or resets the project row.
        monkeypatch.setattr(features, 'SUPPORTED_FEATURES', frozenset[str]())
        await loader.reload('observability')
        host = _observability_host(loader)
        assert host.settings(LogfireSettings).account is None
        host.save_settings(host.settings(LogfireSettings).model_copy(update={'token': older_token}))
        [saved] = store.plugins()
        assert saved.settings['account'] == account.model_dump(mode='json')  # Written back, as unknown settings are.
        monkeypatch.setattr(features, 'SUPPORTED_FEATURES', supported)
        await loader.reload('observability')
    finally:
        await loader.close('exit')
    tags = [(span.attributes or {})['logfire.tags'] for span in recorder.spans() if span.name == 'CLAI session']
    assert tags == [(), ('mike@example.com',), (), ()]


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
    # Databases from before `/update` follow stable releases.
    assert store.load().update_channel == 'stable'
    # Databases from before grouped tool calls keep one line per call.
    assert store.load().tool_calls == 'detailed'
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


@pytest.mark.parametrize('value_json', ['false', 'true'])
async def test_historical_tool_output_preference_is_preserved(tmp_path: Path, value_json: str) -> None:
    path = tmp_path / 'config.db'
    store = SettingsStore(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('INSERT INTO settings VALUES (?, ?)', ('display.tool_output', value_json))

    # File diffs no longer depend on this setting; its saved shell/grep preference stays intact.
    assert store.load().tool_output is (value_json == 'true')
    reopened = SettingsStore(path)
    assert reopened.overrides() == {'display.tool_output': value_json == 'true'}
    output = io.StringIO()
    renderer = StreamRenderer(
        Console(file=output), stop_loading=lambda: None, show_tool_output=reopened.load().tool_output
    )
    for event in (
        FileChangeRequestEvent(
            path='file.txt', root_dir='/tmp', operation='write', diff='visible diff', truncated=False
        ),
        FileWrittenEvent(path='file.txt', root_dir='/tmp', content_hash='hash'),
        CommandStartedEvent(command='echo preview', pid=1),
        CommandOutputEvent(text='shell preview\n'),
        CommandFinishedEvent(pid=1, output_path='/tmp/output', status_path='/tmp/status', exit_code=0, truncated=False),
        FunctionToolCallEvent(part=ToolCallPart('grep', {'pattern': 'preview'}, tool_call_id='grep')),
        FunctionToolResultEvent(part=ToolReturnPart('grep', 'grep preview\n', tool_call_id='grep')),
    ):
        await renderer.on_stream_event(event)
    await renderer.finish()
    assert 'visible diff' in output.getvalue()
    assert ('shell preview' in output.getvalue()) == (value_json == 'true')
    assert ('grep preview' in output.getvalue()) == (value_json == 'true')
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute(
            'SELECT value_json FROM settings WHERE key = ?', ('display.tool_output',)
        ).fetchone() == (value_json,)
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
        ('accounts.pool', '"off"'),
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


def test_database_from_before_account_pooling_pools_and_keeps_its_settings(tmp_path: Path) -> None:
    path = tmp_path / 'config.db'
    SettingsStore(path)
    # Literal rows an earlier build wrote, with a saved profile model and no `accounts.pool`.
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executemany(
            'INSERT INTO settings VALUES (?, ?)',
            [('model', '"claude-code@work:opus"'), ('display.thinking', 'false'), ('future.setting', '1')],
        )
    store = SettingsStore(path)
    settings = store.load()
    # Turning pooling on for existing databases is the intended default; named profiles stay pinned.
    assert settings.pool_accounts is True
    assert settings.model == 'claude-code@work:opus' and settings.thinking is False
    config_command(store, ['set', 'accounts.pool', 'false'])
    assert SettingsStore(path).load().pool_accounts is False
    config_command(store, ['reset', 'accounts.pool'])
    assert SettingsStore(path).load() == settings
    with closing(sqlite3.connect(path)) as connection:
        assert dict(connection.execute('SELECT key, value_json FROM settings')) == {
            'model': '"claude-code@work:opus"',
            'display.thinking': 'false',
            'future.setting': '1',
        }


def test_update_channel_main_keeps_its_former_name_on_disk(tmp_path: Path) -> None:
    """`main` was called `bleeding`: older builds' rows read as `main`, and `main` is saved as `bleeding` for them."""
    path = tmp_path / 'config.db'
    store = SettingsStore(path)
    # Literal rows an earlier build wrote.
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executemany(
            'INSERT INTO settings VALUES (?, ?)',
            [('updates.channel', '"bleeding"'), ('display.thinking', 'false'), ('future.setting', '1')],
        )
    settings = SettingsStore(path).load()
    assert (settings.update_channel, settings.thinking) == ('main', False)
    assert config_command(store, ['get', 'updates.channel']) == '"main"'
    config_command(store, ['set', 'updates.channel', 'stable'])
    assert SettingsStore(path).load().update_channel == 'stable'
    config_command(store, ['set', 'updates.channel', 'main'])
    assert SettingsStore(path).load() == settings
    snapshot = path.read_bytes()
    with pytest.raises(ValidationError):
        store.set('updates.channel', 'nightly')
    assert path.read_bytes() == snapshot
    with closing(sqlite3.connect(path)) as connection:
        assert dict(connection.execute('SELECT key, value_json FROM settings')) == {
            'updates.channel': '"bleeding"',
            'display.thinking': 'false',
            'future.setting': '1',
        }


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


def test_database_without_requirement_tags_loads_unchanged(tmp_path: Path) -> None:
    """A database from before requirement tags, written as those builds wrote it, keeps every setting."""
    path = tmp_path / 'config.db'
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('PRAGMA user_version = 1')
        connection.execute('CREATE TABLE settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)')
        connection.execute('CREATE TABLE plugins (id TEXT PRIMARY KEY, declaration TEXT NOT NULL)')
        connection.execute(
            'INSERT INTO plugins VALUES (?, ?)',
            (
                'coder',
                '{"id": "coder", "factory": "pydantic_ai_harness.coder:Coder", "enabled": true, '
                '"settings": {"unrestricted_filesystem": true, "repo_context": false, "sub_agents": true}}',
            ),
        )
    store = SettingsStore(path)
    assert store.plugins() == [
        PluginSettings(
            id='coder',
            factory='pydantic_ai_harness.coder:Coder',
            settings={'unrestricted_filesystem': True, 'repo_context': False, 'sub_agents': True},
        )
    ]
    assert store.plugin_requirements('coder') is None
    with closing(sqlite3.connect(path)) as connection:
        snapshot = list(connection.iterdump())
        assert connection.execute('PRAGMA user_version').fetchone() == (1,)
    SettingsStore(path)
    SettingsStore(path)
    with closing(sqlite3.connect(path)) as connection:
        assert list(connection.iterdump()) == snapshot
        # Older builds refuse any other version, so the requirements table must not bump it.
        assert connection.execute('PRAGMA user_version').fetchone() == (1,)
        # Tags sit in their own table; older builds never read it.
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert 'plugin_requirements' in tables


def test_database_without_chains_keeps_models_and_gains_chains(tmp_path: Path) -> None:
    """A database from before fallback chains and auth profiles keeps its models and settings unchanged."""
    path = tmp_path / 'config.db'
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('PRAGMA user_version = 1')
        connection.execute('CREATE TABLE settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)')
        connection.execute('CREATE TABLE models (name TEXT PRIMARY KEY)')
        connection.execute('CREATE TABLE model_settings (model TEXT PRIMARY KEY, settings_json TEXT NOT NULL)')
        connection.execute('INSERT INTO settings VALUES (?, ?)', ('model', '"openai-codex:gpt-6-astra"'))
        connection.execute('INSERT INTO models VALUES (?)', ('openai-codex:gpt-6-astra',))
        connection.execute(
            'INSERT INTO model_settings VALUES (?, ?)', ('openai-codex:gpt-6-astra', '{"service_tier":"priority"}')
        )
    store = SettingsStore(path)
    assert store.load().model == 'openai-codex:gpt-6-astra'
    assert store.models() == ['openai-codex:gpt-6-astra']
    assert store.model_settings('openai-codex:gpt-6-astra') == {'service_tier': 'priority'}
    assert store.chains() == {}
    with closing(sqlite3.connect(path)) as connection:
        snapshot = list(connection.iterdump())
    SettingsStore(path)
    with closing(sqlite3.connect(path)) as connection:
        assert list(connection.iterdump()) == snapshot
        # Older builds refuse any other version; chains live in a table they never read.
        assert connection.execute('PRAGMA user_version').fetchone() == (1,)
    store.save_chain(name='pool', models=['openai-codex:gpt-6-astra', 'openai-codex@work:gpt-6-astra'])
    with closing(sqlite3.connect(path)) as connection, connection:
        # A row another build wrote in a shape this one cannot read is skipped, not rewritten.
        connection.execute('INSERT INTO model_chains VALUES (?, ?)', ('future', '{"models": []}'))
    reopened = SettingsStore(path)
    assert reopened.chains() == {'pool': ['openai-codex:gpt-6-astra', 'openai-codex@work:gpt-6-astra']}
    assert reopened.models() == ['chain:pool', 'openai-codex:gpt-6-astra']
    assert reopened.model_settings('openai-codex:gpt-6-astra') == {'service_tier': 'priority'}
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT models_json FROM model_chains WHERE name = 'future'").fetchone() == (
            '{"models": []}',
        )


def test_database_without_accounts_keeps_its_data_and_gains_accounts(tmp_path: Path) -> None:
    """A database from before `/accounts` opens unchanged; accounts go in a table older builds never read."""
    path = tmp_path / 'config.db'
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('PRAGMA user_version = 1')
        connection.execute('CREATE TABLE settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)')
        connection.execute('CREATE TABLE models (name TEXT PRIMARY KEY)')
        connection.execute('CREATE TABLE model_chains (name TEXT PRIMARY KEY, models_json TEXT NOT NULL)')
        connection.execute('INSERT INTO settings VALUES (?, ?)', ('model', '"openai-codex@work:gpt-6-astra"'))
        connection.execute('INSERT INTO models VALUES (?)', ('openai-codex@work:gpt-6-astra',))
        connection.execute('INSERT INTO model_chains VALUES (?, ?)', ('pool', '["openai:gpt-5", "openai@work:gpt-5"]'))
    store = SettingsStore(path)
    assert store.accounts() == []
    with closing(sqlite3.connect(path)) as connection:
        snapshot = list(connection.iterdump())
    SettingsStore(path)
    with closing(sqlite3.connect(path)) as connection:
        assert list(connection.iterdump()) == snapshot
        assert connection.execute('PRAGMA user_version').fetchone() == (1,)
    store.add_account(StoredAccount(provider='openai-codex', profile='work', label='Work'))
    reopened = SettingsStore(path)
    assert reopened.load().model == 'openai-codex@work:gpt-6-astra'
    assert reopened.models() == ['openai-codex@work:gpt-6-astra']
    assert reopened.chains() == {'pool': ['openai:gpt-5', 'openai@work:gpt-5']}
    assert reopened.accounts() == [StoredAccount(provider='openai-codex', profile='work', label='Work')]


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
