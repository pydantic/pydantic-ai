"""Regression tests for the gh-aw engine pin and its CI read path.

`src/pydantic_ai_harness/gh-aw/pydantic.md` pins the `pydantic-ai-harness` release a
generated gh-aw workflow installs at run time. Consumers compile the definition from
the default branch, so the pin must be a valid PEP 440 string naming a release PyPI
serves. Both checks belong to the CI extractor script, and these tests invoke it
rather than copy its logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EXTRACTOR = Path(__file__).parents[2] / 'src' / 'pydantic_ai_harness' / 'scripts' / 'gh_aw_engine_version.py'


def _load_extractor() -> ModuleType:
    spec = importlib.util.spec_from_file_location('gh_aw_engine_version', EXTRACTOR)
    assert spec is not None and spec.loader is not None, f'missing extractor script: {EXTRACTOR}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


extractor = _load_extractor()


def test_extractor_reads_engine_version() -> None:
    """The read path the CI pin job calls returns the release the committed definition pins."""
    assert extractor.engine_version() == '0.53.0'


def test_engine_pin_served_by_pypi() -> None:
    """The pinned release resolves on PyPI, so installs of generated workflows cannot break."""
    assert extractor.unpublished_reason(extractor.engine_version()) is None
