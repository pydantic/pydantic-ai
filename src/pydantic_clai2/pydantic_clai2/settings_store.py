"""SQLite preferences with short transactions and explicit schema ownership."""

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from .config import SETTING_FIELDS, PluginSettings, Settings, resolve_settings

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


class SettingsStore:
    """Persist overrides, never credentials or conversation messages."""

    def __init__(self, path: Path | None = None) -> None:
        """Open or initialize a settings database at an explicit or user path."""
        self.path = (
            path or Path(os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'pydantic-clai2/config.db'
        )
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
        """Read explicit preferences, validating the serialized values."""
        with self._connect() as connection:
            return {
                key: _JSON.validate_json(value)
                for key, value in connection.execute('SELECT key, value_json FROM settings')
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

    def reset(self, key: str) -> None:
        """Remove a setting override, restoring its default."""
        if key not in SETTING_FIELDS:
            raise ValueError(f'Unknown setting: {key}')
        with self._connect() as connection:
            connection.execute('DELETE FROM settings WHERE key = ?', (key,))

    def plugins(self) -> list[PluginSettings]:
        """Return declarations in stable identifier order without importing code."""
        with self._connect() as connection:
            return [
                PluginSettings.model_validate_json(row[0])
                for row in connection.execute('SELECT declaration FROM plugins ORDER BY id')
            ]

    def save_plugin(self, plugin: PluginSettings) -> None:
        """Persist an explicitly trusted plugin declaration."""
        with self._connect() as connection:
            connection.execute(
                'INSERT INTO plugins VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET declaration = excluded.declaration',
                (plugin.id, plugin.model_dump_json()),
            )

    def delete_plugin(self, plugin_id: str) -> None:
        """Forget a declaration; a plugin file in the plugins folder is not deleted."""
        with self._connect() as connection:
            connection.execute('DELETE FROM plugins WHERE id = ?', (plugin_id,))

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
