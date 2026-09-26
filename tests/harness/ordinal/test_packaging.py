"""The Ordinal integration ships as the optional `ordinal` extra.

Only metadata checks belong here: this file stays collected on base (no-extras)
installs, so it must not import `pydantic_ai_harness.ordinal`.
"""

from __future__ import annotations

import importlib.metadata


def _requires_dist() -> list[str]:
    return importlib.metadata.metadata('pydantic-ai-harness').get_all('Requires-Dist') or []


def test_ordinal_extra_is_advertised() -> None:
    provides = importlib.metadata.metadata('pydantic-ai-harness').get_all('Provides-Extra') or []
    assert 'ordinal' in provides


def test_mcp_is_an_optional_ordinal_dependency() -> None:
    ordinal_requirements = [
        req for req in _requires_dist() if 'extra == "ordinal"' in req or "extra == 'ordinal'" in req
    ]
    assert ordinal_requirements, 'the ordinal extra must be declared'
    assert any('pydantic-ai-slim' in req and 'mcp' in req for req in ordinal_requirements)
