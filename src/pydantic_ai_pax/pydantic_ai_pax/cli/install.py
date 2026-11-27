"""Which package installed the running Pax, and so which command starts it."""

import sys
from pathlib import Path

DISTRIBUTION = 'pydantic-ai-pax'
COMMAND = 'pax'
"""The command `DISTRIBUTION` provides: what `/update` installs, from any environment."""
LEGACY_DISTRIBUTION = 'pydantic-clai2'
"""Pax's former name: `/update` in a `uv tool` environment installed under it installs `pydantic-ai-pax` as its own tool.

That leaves the old environment and its `clai2` command in place until the person removes them, so no
`pydantic-clai2` release is needed after the last one that moves its environments this way.
"""
_COMMANDS = {DISTRIBUTION: COMMAND, LEGACY_DISTRIBUTION: 'clai2'}


def tool_distribution() -> str:
    """The package this `uv tool` environment was installed from, which uv names the environment after."""
    return LEGACY_DISTRIBUTION if Path(sys.prefix).name == LEGACY_DISTRIBUTION else DISTRIBUTION


def command_name() -> str:
    """The command this install provides: `clai2` from a `pydantic-clai2` environment, otherwise `pax`."""
    return _COMMANDS[tool_distribution()]
