import sys
from pathlib import Path

# Runs against the published engine, `pydantic-ai-pax`.
from pydantic_ai_pax.config.settings_store import SettingsStore

workspace = Path(sys.argv[1])
database = workspace / 'settings.db'
SettingsStore(database).set('run.code_mode', False)
print(database)
