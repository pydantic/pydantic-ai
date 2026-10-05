"""SQLite preferences with short transactions and explicit schema ownership."""

import os
import sqlite3
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from pydantic_clai2.config import SETTING_FIELDS, PluginSettings, Settings, resolve_settings
from pydantic_clai2.config.plugin_requirements import Requirements, merged_requirements

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_CHAIN: TypeAdapter[list[str]] = TypeAdapter(list[str])
# Keep the original IDs on disk so older builds share the same preferences, without loading a second plugin.
_PLUGIN_NAMES = {'logfire': 'observability'}
_STORED_PLUGIN_NAMES = {name: stored for stored, name in _PLUGIN_NAMES.items()}


def canonical_plugin_id(plugin_id: str) -> str:
    """Use one runtime identity for old and current plugin names."""
    return _PLUGIN_NAMES.get(plugin_id, plugin_id)


def canonical_plugin_declarations(plugins: Iterable[PluginSettings]) -> dict[str, PluginSettings]:
    """Keep declaration order, preferring an explicit current name over its old alias."""
    declarations = {plugin.id: plugin for plugin in plugins}
    renamed: dict[str, PluginSettings] = {}
    for plugin in declarations.values():
        current = canonical_plugin_id(plugin.id)
        if current != plugin.id and current in declarations:
            continue
        renamed[current] = plugin if current == plugin.id else plugin.model_copy(update={'id': current})
    return renamed


def _stored_plugin_id(plugin_id: str) -> str:
    """The ID a plugin's rows are stored under, so older builds that know the old name still find them."""
    current_id = canonical_plugin_id(plugin_id)
    return _STORED_PLUGIN_NAMES.get(current_id, current_id)


def config_dir() -> Path:
    """The user's CLAI folder, honouring `XDG_CONFIG_HOME`."""
    return Path(os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'pydantic-clai2'


class SettingsStore:
    """Persist overrides, never credentials or conversation messages."""

    def __init__(self, path: Path | None = None) -> None:
        """Open or initialize a settings database at an explicit or user path."""
        self.path = path or config_dir() / 'config.db'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f'Unsupported settings schema version: {version}')
            connection.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS plugins (id TEXT PRIMARY KEY, declaration TEXT NOT NULL)')
            connection.execute(
                'CREATE TABLE IF NOT EXISTS model_settings (model TEXT PRIMARY KEY, settings_json TEXT NOT NULL)'
            )
            connection.execute('CREATE TABLE IF NOT EXISTS models (name TEXT PRIMARY KEY)')
            # Requirement tags sit beside `plugins`, never inside a declaration: older builds validate
            # declarations strictly and never look at this table. It needs no schema version bump,
            # which those builds would refuse.
            connection.execute(
                'CREATE TABLE IF NOT EXISTS plugin_requirements (id TEXT PRIMARY KEY, requirements_json TEXT NOT NULL)'
            )
            # Fallback chains are their own table for the same reason: builds without chains never read it.
            connection.execute(
                'CREATE TABLE IF NOT EXISTS model_chains (name TEXT PRIMARY KEY, models_json TEXT NOT NULL)'
            )
            connection.execute('PRAGMA user_version = 1')

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def overrides(self) -> dict[str, JsonValue]:
        """Read recognized preferences without modifying unknown stored entries."""
        with self._connect() as connection:
            return {
                key: _JSON.validate_json(value)
                for key, value in connection.execute('SELECT key, value_json FROM settings')
                if key in SETTING_FIELDS
            }

    def load(self) -> Settings:
        """Resolve persisted overrides against built-in defaults."""
        return resolve_settings(self.overrides())

    def set(self, key: str, value: JsonValue) -> None:
        """Validate before committing a single override."""
        resolve_settings({key: value})
        with self._connect() as connection:
            connection.execute(
                'INSERT INTO settings VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json',
                (key, _JSON.dump_json(value).decode()),
            )

            if key == 'model' and isinstance(value, str):
                connection.execute('INSERT OR IGNORE INTO models VALUES (?)', (value,))

    def models(self) -> list[str]:
        """Models explicitly saved for reuse, in name order."""
        with self._connect() as connection:
            return [row[0] for row in connection.execute('SELECT name FROM models ORDER BY name')]

    def add_model(self, *, name: str) -> None:
        """Remember a model without changing the active preference."""
        with self._connect() as connection:
            connection.execute('INSERT OR IGNORE INTO models VALUES (?)', (name,))

    def remove_model(self, *, name: str) -> bool:
        """Forget a model and its overrides; return `False` if it is the saved default."""
        with self._connect() as connection:
            # Keep the default check and both deletes atomic across CLAI sessions.
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute("SELECT value_json FROM settings WHERE key = 'model'").fetchone()
            if row is not None and _JSON.validate_json(row[0]) == name:
                return False
            connection.execute('DELETE FROM models WHERE name = ?', (name,))
            connection.execute('DELETE FROM model_settings WHERE model = ?', (name,))
        return True

    def chains(self) -> dict[str, list[str]]:
        """Saved fallback chains by name, each its models in order; an unreadable row is skipped, not changed."""
        chains: dict[str, list[str]] = {}
        with self._connect() as connection:
            for name, models in connection.execute('SELECT name, models_json FROM model_chains ORDER BY name'):
                try:
                    chains[name] = _CHAIN.validate_json(models)
                except ValidationError:
                    continue
        return chains

    def save_chain(self, *, name: str, models: list[str]) -> None:
        """Save a chain and offer it as `chain:NAME` in the model list."""
        with self._connect() as connection:
            connection.execute(
                'INSERT INTO model_chains VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET models_json = excluded.models_json',
                (name, _CHAIN.dump_json(models).decode()),
            )
            connection.execute('INSERT OR IGNORE INTO models VALUES (?)', (f'chain:{name}',))

    def delete_chain(self, *, name: str) -> bool:
        """Forget a chain, its model-list entry, and its settings; `False` if it is the saved default."""
        if not self.remove_model(name=f'chain:{name}'):
            return False
        with self._connect() as connection:
            connection.execute('DELETE FROM model_chains WHERE name = ?', (name,))
        return True

    def reset(self, key: str) -> None:
        """Remove a setting override, restoring its default."""
        if key not in SETTING_FIELDS:
            raise ValueError(f'Unknown setting: {key}')
        with self._connect() as connection:
            connection.execute('DELETE FROM settings WHERE key = ?', (key,))

    def plugins(self) -> list[PluginSettings]:
        """Return declarations in stable identifier order without importing code."""
        with self._connect() as connection:
            plugins = [
                PluginSettings.model_validate_json(row[0])
                for row in connection.execute('SELECT declaration FROM plugins ORDER BY id')
            ]
        return sorted(canonical_plugin_declarations(plugins).values(), key=lambda plugin: plugin.id)

    def plugin_requirements(self, plugin_id: str) -> JsonValue | None:
        """The stored requirement tags for a plugin's settings, as saved; `None` when it has none.

        Text that is not JSON comes back as a string, which `stored_requirements` treats as unreadable.
        """
        with self._connect() as connection:
            return self._requirements_row(connection, _stored_plugin_id(plugin_id))

    def save_plugin(self, plugin: PluginSettings, *, requires: Requirements | None = None) -> None:
        """Persist an explicitly trusted plugin declaration with the writer's requirement tags.

        `requires` is what the plugin declares for its settings. Stored tags on values left unchanged
        are kept, so saving never strips a tag another build attached. Both rows change in one transaction.
        """
        current_id = canonical_plugin_id(plugin.id)
        plugin = plugin.model_copy(update={'id': _stored_plugin_id(current_id)})
        with self._connect() as connection:
            old = connection.execute('SELECT declaration FROM plugins WHERE id = ?', (plugin.id,)).fetchone()
            saved = _saved_declaration(old[0]) if old is not None else {}
            # Tags without a declaration were left by a build that deleted it without knowing this table,
            # and tags on another factory's declaration describe that plugin, not this one.
            same_plugin = old is not None and saved.get('factory') == plugin.factory
            old_row = self._requirements_row(connection, plugin.id) if same_plugin else None
            settings = saved.get('settings')
            row = merged_requirements(
                old_settings=settings if same_plugin and isinstance(settings, dict) else {},
                old_row=old_row,
                new_settings=plugin.settings,
                declared=requires or {},
            )
            if current_id != plugin.id:
                connection.execute('DELETE FROM plugins WHERE id = ?', (current_id,))
                connection.execute('DELETE FROM plugin_requirements WHERE id = ?', (current_id,))
            connection.execute(
                'INSERT INTO plugins VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET declaration = excluded.declaration',
                (plugin.id, plugin.model_dump_json()),
            )
            if row:
                connection.execute(
                    'INSERT INTO plugin_requirements VALUES (?, ?) '
                    'ON CONFLICT(id) DO UPDATE SET requirements_json = excluded.requirements_json',
                    (plugin.id, _JSON_OBJECT.dump_json(row).decode()),
                )
            else:
                connection.execute('DELETE FROM plugin_requirements WHERE id = ?', (plugin.id,))

    @staticmethod
    def _requirements_row(connection: sqlite3.Connection, plugin_id: str) -> JsonValue | None:
        row = connection.execute(
            'SELECT requirements_json FROM plugin_requirements WHERE id = ?', (plugin_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            value = _JSON.validate_json(row[0])
        except ValidationError:
            return row[0]
        # A stored JSON `null` is not "no tags": hand back the text, which readers treat as unreadable.
        return row[0] if value is None else value

    def delete_plugin(self, plugin_id: str) -> None:
        """Forget a declaration and its requirement tags; a plugin file in the plugins folder is not deleted."""
        current_id = canonical_plugin_id(plugin_id)
        stored_id = _stored_plugin_id(current_id)
        with self._connect() as connection:
            connection.execute('DELETE FROM plugins WHERE id IN (?, ?)', (current_id, stored_id))
            connection.execute('DELETE FROM plugin_requirements WHERE id IN (?, ?)', (current_id, stored_id))

    def model_settings(self, model: str) -> dict[str, JsonValue]:
        """Saved overrides for one model; empty when none."""
        with self._connect() as connection:
            row = connection.execute('SELECT settings_json FROM model_settings WHERE model = ?', (model,)).fetchone()
        return _JSON_OBJECT.validate_json(row[0]) if row else {}

    def save_model_settings(self, model: str, settings: dict[str, JsonValue]) -> None:
        """Replace one model's overrides; an empty dict removes the row."""
        with self._connect() as connection:
            if not settings:
                connection.execute('DELETE FROM model_settings WHERE model = ?', (model,))
                return
            connection.execute(
                'INSERT INTO model_settings VALUES (?, ?) '
                'ON CONFLICT(model) DO UPDATE SET settings_json = excluded.settings_json',
                (model, _JSON_OBJECT.dump_json(settings).decode()),
            )

    @property
    def plugins_dir(self) -> Path:
        """Folder scanned for drop-in plugins, next to the settings database."""
        return self.path.parent / 'plugins'


def _saved_declaration(declaration: str) -> dict[str, JsonValue]:
    """A stored declaration as plain JSON, read leniently: the row may come from another build."""
    try:
        return _JSON_OBJECT.validate_json(declaration)
    except ValidationError:
        return {}
