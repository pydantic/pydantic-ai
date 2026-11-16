"""The PostHog integration ships as the optional `posthog` extra.

Only metadata checks belong here: this file stays collected on base (no-extras)
installs, so it must not import `pydantic_ai_harness.posthog`.
"""

from __future__ import annotations

import importlib.metadata


def _requires_dist() -> list[str]:
    return importlib.metadata.metadata('pydantic-ai-harness').get_all('Requires-Dist') or []


def test_posthog_extra_is_advertised() -> None:
    provides = importlib.metadata.metadata('pydantic-ai-harness').get_all('Provides-Extra') or []
    assert 'posthog' in provides


def test_mcp_is_an_optional_posthog_dependency() -> None:
    posthog_requirements = [
        req for req in _requires_dist() if 'extra == "posthog"' in req or "extra == 'posthog'" in req
    ]
    assert posthog_requirements, 'the posthog extra must be declared'
    assert any('pydantic-ai-slim' in req and 'mcp' in req for req in posthog_requirements)
