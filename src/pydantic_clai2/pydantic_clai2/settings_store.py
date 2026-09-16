"""SQLite preferences with short transactions and explicit schema ownership."""

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from .config import SETTING_FIELDS, PluginSettings, Settings, resolve_settings

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


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
