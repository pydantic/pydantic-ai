import os
import sys
from pathlib import Path

from pydantic_clai2.config.settings_store import SettingsStore

workspace = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ['GITHUB_WORKSPACE'])
database = workspace / 'settings.db'
SettingsStore(database).set('run.code_mode', False)
print(database)
