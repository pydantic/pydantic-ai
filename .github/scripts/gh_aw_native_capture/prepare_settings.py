import sys
from pathlib import Path

# Runs against the published engine, `pydantic-clai2`; in this checkout the module is only a run-time alias.
from pydantic_clai2.config.settings_store import SettingsStore  # pyright: ignore[reportMissingImports, reportUnknownVariableType]

workspace = Path(sys.argv[1])
database = workspace / 'settings.db'
SettingsStore(database).set('run.code_mode', False)  # pyright: ignore[reportUnknownMemberType]
print(database)
