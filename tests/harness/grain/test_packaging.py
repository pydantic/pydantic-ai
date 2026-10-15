"""The Grain integration ships as the optional `grain` extra.

Only metadata checks belong here: this file stays collected on base (no-extras)
installs, so it must not import `pydantic_ai_harness.grain`.
"""

from __future__ import annotations

import importlib.metadata


def _requires_dist() -> list[str]:
    return importlib.metadata.metadata('pydantic-ai-harness').get_all('Requires-Dist') or []


def test_grain_extra_is_advertised() -> None:
    provides = importlib.metadata.metadata('pydantic-ai-harness').get_all('Provides-Extra') or []
    assert 'grain' in provides


def test_mcp_is_an_optional_grain_dependency() -> None:
    grain_requirements = [req for req in _requires_dist() if 'extra == "grain"' in req or "extra == 'grain'" in req]
    assert grain_requirements, 'the grain extra must be declared'
    assert any('pydantic-ai-slim' in req and 'mcp' in req for req in grain_requirements)
