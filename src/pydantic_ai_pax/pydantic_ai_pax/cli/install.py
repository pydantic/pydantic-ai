"""Which package installed the running Pax, with which extras, and so which command starts it."""

import sys
import tomllib
from pathlib import Path
from typing import Required

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

DISTRIBUTION = 'pydantic-ai-pax'
LEGACY_DISTRIBUTION = 'pydantic-clai2'
"""Pax's former name: a `uv tool` environment installed under it keeps updating under it, keeping `clai2`.

That needs a `pydantic-clai2` release for every version, so the shim cannot stop being released until `/update`
moves these environments to `pydantic-ai-pax`.
"""
_COMMANDS = {DISTRIBUTION: 'pax', LEGACY_DISTRIBUTION: 'clai2'}
RECEIPT = 'uv-receipt.toml'
"""The file `uv tool install` writes in the environment, recording the requirements it was asked for."""


class _ReceiptRequirement(TypedDict, total=False):
    name: Required[str]
    extras: list[str]


class _ReceiptTool(TypedDict, total=False):
    requirements: list[_ReceiptRequirement]


class _Receipt(TypedDict, total=False):
    tool: _ReceiptTool


_RECEIPT = TypeAdapter(_Receipt)


def tool_distribution() -> str:
    """The package this `uv tool` environment was installed from, which uv names the environment after."""
    return LEGACY_DISTRIBUTION if Path(sys.prefix).name == LEGACY_DISTRIBUTION else DISTRIBUTION


def command_name() -> str:
    """The command this install provides: `clai2` from a `pydantic-clai2` environment, otherwise `pax`."""
    return _COMMANDS[tool_distribution()]


def tool_extras() -> tuple[str, ...]:
    """The extras this `uv tool` environment installed Pax with, such as `groq` from `pydantic-ai-pax[groq]`.

    Read from uv's receipt; none outside a `uv tool` environment or when the receipt cannot be read.
    """
    try:
        receipt = _RECEIPT.validate_python(tomllib.loads(Path(sys.prefix, RECEIPT).read_text(encoding='utf-8')))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError):
        return ()
    distribution = tool_distribution()
    requirements = receipt.get('tool', {}).get('requirements', [])
    found = {
        extra
        for requirement in requirements
        if requirement['name'] == distribution
        for extra in requirement.get('extras', [])
    }
    return tuple(sorted(found))


def with_extras(name: str, extras: tuple[str, ...]) -> str:
    """`name[a,b]`, or `name` without extras."""
    return f'{name}[{",".join(extras)}]' if extras else name


def install_hint(extra: str) -> str:
    """The `uv tool install` command that adds `extra` to Pax and keeps the extras it was installed with."""
    return f"uv tool install '{with_extras(DISTRIBUTION, tuple(sorted({*tool_extras(), extra})))}'"
