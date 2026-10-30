import sys
from pathlib import Path

from pydantic_clai2.config.settings_store import SettingsStore

workspace = Path(sys.argv[1])
database = workspace / 'settings.db'
SettingsStore(database).set('run.code_mode', False)
print(database)
