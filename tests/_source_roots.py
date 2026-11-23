"""The packages the workspace ships, read from `[tool.uv.workspace].members` and each member's wheel targets.

A check that scans shipped source or configures a tool per package reads its roots from here, so a new
workspace member falls in its scope unless the check exempts it with a reason.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).parents[1]


@dataclass(frozen=True)
class ShippedPackage:
    """An importable package a workspace member builds into its wheel."""

    member: str
    """The member's directory, as `[tool.uv.workspace].members` lists it."""
    name: str
    """The import name."""

    @property
    def path(self) -> PurePosixPath:
        """The package directory, relative to the repository root."""
        return PurePosixPath(self.member, self.name)


def shipped_packages() -> list[ShippedPackage]:
    """Every package the workspace members ship, in member order."""
    packages: list[ShippedPackage] = []
    workspace = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    for member in workspace['tool']['uv']['workspace']['members']:
        project = tomllib.loads((REPO_ROOT / member / 'pyproject.toml').read_text(encoding='utf-8'))
        wheel = project['tool']['hatch']['build']['targets']['wheel']
        packages += [ShippedPackage(member, name) for name in wheel['packages']]
    return packages
