"""Import-light entry point: animate before loading the agent/UI dependencies."""

import json
import os
import sqlite3
import sys
from pathlib import Path

from .splash import Splash


def main() -> None:
    """Cover heavyweight startup imports with the CLAI splash."""
    os.environ['PYDANTIC_AI_NO_BANNER'] = '1'
    enabled = len(sys.argv) == 1 and not os.getenv('CLAI_NO_SPLASH')
    database = Path(os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'pydantic-clai2/config.db'
    if enabled and database.exists():
        try:
            connection = sqlite3.connect(f'{database.as_uri()}?mode=ro', uri=True)
            try:
                row = connection.execute("SELECT value_json FROM settings WHERE key = 'display.splash'").fetchone()
                enabled = row is None or json.loads(row[0]) is True
            finally:
                connection.close()
        except (sqlite3.Error, ValueError):
            enabled = False
    splash = Splash(enabled=bool(enabled))
    splash.start()
    try:
        from ._cli import run

        run(splash=splash)
    finally:
        splash.stop()


if __name__ == '__main__':
    main()
