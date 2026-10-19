"""The Day AI integration ships as the optional `day-ai` extra.

Only metadata checks belong here: this file stays collected on base (no-extras)
installs, so it must not import `pydantic_ai_harness.day_ai`.
"""

from __future__ import annotations

import importlib.metadata


def _requires_dist() -> list[str]:
    return importlib.metadata.metadata('pydantic-ai-harness').get_all('Requires-Dist') or []


def test_day_ai_extra_is_advertised() -> None:
    provides = importlib.metadata.metadata('pydantic-ai-harness').get_all('Provides-Extra') or []
    assert 'day-ai' in provides


def test_mcp_is_an_optional_day_ai_dependency() -> None:
    day_ai_requirements = [req for req in _requires_dist() if 'extra == "day-ai"' in req or "extra == 'day-ai'" in req]
    assert day_ai_requirements, 'the day-ai extra must be declared'
    assert any('pydantic-ai-slim' in req and 'mcp' in req for req in day_ai_requirements)
