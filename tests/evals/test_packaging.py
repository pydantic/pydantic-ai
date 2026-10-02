"""`pydantic-evals` packaging checks that hold on a clean install.

The repository environment contains every transitive dependency, so these
checks read the installed `Requires-Dist` metadata directly instead of
importing `pydantic_evals`.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from ..conftest import try_import

with try_import() as evals_metadata_available:
    importlib.metadata.metadata('pydantic-evals')

pytestmark = pytest.mark.skipif(not evals_metadata_available(), reason='pydantic-evals not installed')


def _requires_dist() -> list[str]:
    return importlib.metadata.metadata('pydantic-evals').get_all('Requires-Dist') or []


def test_sniffio_declared_in_base_dependencies() -> None:
    base_requirements = [req for req in _requires_dist() if 'extra ==' not in req]
    assert any(req.startswith('sniffio') for req in base_requirements), (
        'sniffio must be a base dependency of pydantic-evals, not extras-only'
    )
