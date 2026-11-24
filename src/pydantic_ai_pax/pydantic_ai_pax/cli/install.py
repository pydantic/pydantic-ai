"""Which package installed the running Pax, and so which command starts it."""

import sys
from pathlib import Path

DISTRIBUTION = 'pydantic-ai-pax'
LEGACY_DISTRIBUTION = 'pydantic-clai2'
"""Pax's former name: a `uv tool` environment installed under it keeps updating under it, keeping `clai2`.

That needs a `pydantic-clai2` release for every version, so the shim cannot stop being released until `/update`
moves these environments to `pydantic-ai-pax`.
"""
_COMMANDS = {DISTRIBUTION: 'pax', LEGACY_DISTRIBUTION: 'clai2'}


def tool_distribution() -> str:
    """The package this `uv tool` environment was installed from, which uv names the environment after."""
    return LEGACY_DISTRIBUTION if Path(sys.prefix).name == LEGACY_DISTRIBUTION else DISTRIBUTION


def command_name() -> str:
    """The command this install provides: `clai2` from a `pydantic-clai2` environment, otherwise `pax`."""
    return _COMMANDS[tool_distribution()]
