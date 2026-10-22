"""The Pylon integration ships as the optional `pylon` extra.

Only metadata checks belong here: this file stays collected on base (no-extras)
installs, so it must not import `pydantic_ai_harness.pylon`.
"""

from __future__ import annotations

import importlib.metadata


def _requires_dist() -> list[str]:
    return importlib.metadata.metadata('pydantic-ai-harness').get_all('Requires-Dist') or []


def test_pylon_extra_is_advertised() -> None:
    provides = importlib.metadata.metadata('pydantic-ai-harness').get_all('Provides-Extra') or []
    assert 'pylon' in provides


def test_mcp_is_an_optional_pylon_dependency() -> None:
    pylon_requirements = [req for req in _requires_dist() if 'extra == "pylon"' in req or "extra == 'pylon'" in req]
    assert pylon_requirements, 'the pylon extra must be declared'
    assert any('pydantic-ai-slim' in req and 'mcp' in req for req in pylon_requirements)
