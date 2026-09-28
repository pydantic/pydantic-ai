"""The ACP integration ships as the optional `acp` extra: `pip install pydantic-ai-harness[acp]`.

Needed because a clean install during review could not import `pydantic_ai_harness.experimental.acp` at all:
`agent-client-protocol` was missing from the package metadata, which only an isolated install
(not the dev environment) reveals. It is declared as an optional extra so base installs stay
lean; these checks keep it in the metadata. The full clean-install import
(`uv run --isolated --with '.[acp]' ...`) is verified manually until a CI job covers it.

Only metadata checks belong here: this file stays collected on base (no-extras) installs, so it
must not import `pydantic_ai_harness.experimental.acp` or the `acp` SDK.
"""

from __future__ import annotations

import importlib.metadata


def _requires_dist() -> list[str]:
    return importlib.metadata.metadata('pydantic-ai-harness').get_all('Requires-Dist') or []


def test_acp_extra_is_advertised() -> None:
    provides = importlib.metadata.metadata('pydantic-ai-harness').get_all('Provides-Extra') or []
    assert 'acp' in provides


def test_agent_client_protocol_is_an_optional_acp_dependency() -> None:
    acp_requirements = [req for req in _requires_dist() if req.startswith('agent-client-protocol')]
    assert acp_requirements, 'agent-client-protocol must be declared'
    # Gated on the `acp` extra (an optional dependency), so base installs stay lean.
    assert all('extra ==' in req and 'acp' in req for req in acp_requirements)
