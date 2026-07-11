"""Keep the README honest about what ships.

Every capability package must document itself with a `README.md` and be linked
from the top-level `README.md`. A capability cannot land without showing up in
the docs, so the "what's available today" tables cannot silently fall behind the
code. This is the mechanical half of docs parity; the semantic half (does the
prose match the code as written) is a review-time concern, not a unit test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_PACKAGE = _ROOT / 'pydantic_ai_harness'

# The `experimental` package is a namespace/warning shim, not a capability, so it
# has no standalone README and is not listed in the top-level tables.
_NAMESPACE_PACKAGES = {_PACKAGE / 'experimental'}


def _is_deprecation_shim(package: Path) -> bool:
    """A package left at a moved capability's old path re-exports it and calls `warn_moved`.

    Such shims carry no docs of their own, so they are excluded from the capability tables.
    """
    return 'warn_moved(' in (package / '__init__.py').read_text(encoding='utf-8')


def _capability_packages() -> list[Path]:
    """Directories that are importable packages and represent a capability's public surface."""
    candidates: list[Path] = []
    for parent in (_PACKAGE, _PACKAGE / 'experimental'):
        for child in sorted(parent.iterdir()):
            if not child.is_dir() or child.name.startswith(('_', '.')):
                continue
            # A non-package dir under the capability roots does not occur in a clean tree, so this guard stays uncovered.
            if not (child / '__init__.py').exists():  # pragma: no cover
                continue
            if child in _NAMESPACE_PACKAGES or _is_deprecation_shim(child):
                continue
            candidates.append(child)
    return candidates


_CAPABILITY_PACKAGES = _capability_packages()


def test_capability_packages_discovered() -> None:
    # Guard against the discovery silently finding nothing (e.g. a moved package root),
    # which would make the parametrized checks below vacuously pass.
    assert len(_CAPABILITY_PACKAGES) >= 10


@pytest.mark.parametrize('package', _CAPABILITY_PACKAGES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_capability_has_readme(package: Path) -> None:
    readme = package / 'README.md'
    assert readme.exists(), (
        f'{package.relative_to(_ROOT)} is an importable capability package but has no README.md. '
        'Add one (start from an existing capability README) so its public surface is documented.'
    )


@pytest.mark.parametrize('package', _CAPABILITY_PACKAGES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_capability_linked_from_top_readme(package: Path) -> None:
    top_readme = (_ROOT / 'README.md').read_text(encoding='utf-8')
    link_target = f'{package.relative_to(_ROOT).as_posix()}/'
    assert link_target in top_readme, (
        f'{package.relative_to(_ROOT)} is not linked from the top-level README.md. '
        f'Add a row for it (linking `{link_target}`) to the "What\'s available today" or "Roadmap" tables '
        'so the README stays in step with the code.'
    )
